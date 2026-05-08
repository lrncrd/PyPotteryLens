/**
 * fewshot-tab.js — SAM2+DINOv2 direct labeling UI
 *
 * Workflow
 * --------
 *  1. Select an image -> image drawn on canvas; backend loads it for SAM2 (fast).
 *  2. POINT mode: left-click = positive (green+), right-click/Shift = negative (red x).
 *     BOX mode:  drag to draw rectangle; SAM2 preview on mouseup.
 *  3. Select or create a class, click "Add to Class".
 *  4. Repeat for all objects on the page.
 *  5. "Save Masks" -> save labeled masks to disk.
 *  6. "Analyze for Batch" -> run SAM2 auto-mask + DINOv2 for batch expansion.
 */

const FewShotTab = (() => {

    // state
    let projectId   = null;
    let images      = [];
    let currentImg  = null;
    let imageId     = null;
    let naturalW    = 0;
    let naturalH    = 0;

    let classes     = {};
    let activeClass = null;
    let prepared    = new Set();
    let analyzed    = new Set();
    let threshold   = 0.5;
    let batchPollId = null;

    // per-image data caches
    let imageClassData   = {};  // { imageId: { className: [example, ...] } }
    let ghostPredictions = {};  // { className: [prediction, ...] } for current image
    let hoveredGhost     = null; // { className, pred, predIdx } | null
    let ghostPopup       = null; // { div, backdrop } | null

    // prompt state
    let promptMode     = 'point';
    let pendingPoints  = [];
    let pendingBox     = null;
    let boxDragStart   = null;
    let previewContour = null;

    // canvas refs
    let canvas, overlay, promptCanvas, ctx, octx, pctx;

    // helpers
    function setStatus(msg, type) {
        var el = document.getElementById('fs-status');
        if (!el) return;
        el.textContent = msg;
        el.className = 'status-message status-' + (type || 'info');
    }

    function hexToRgba(hex, alpha) {
        var r = parseInt(hex.slice(1,3),16);
        var g = parseInt(hex.slice(3,5),16);
        var b = parseInt(hex.slice(5,7),16);
        return 'rgba('+r+','+g+','+b+','+alpha+')';
    }

    function toCanvasX(v) { return canvas ? v * canvas.width  / naturalW : v; }
    function toCanvasY(v) { return canvas ? v * canvas.height / naturalH : v; }

    // class list
    var PALETTE = ['#3b82f6','#10b981','#f59e0b','#ef4444','#8b5cf6',
                   '#ec4899','#14b8a6','#f97316','#84cc16','#06b6d4'];

    function renderClassList() {
        var container = document.getElementById('fs-class-list');
        if (!container) return;
        container.innerHTML = '';
        var names = Object.keys(classes);
        if (!names.length) {
            container.innerHTML = '<p style="font-size:0.8rem;color:#94a3b8;">No classes yet.</p>';
            return;
        }
        names.forEach(function(name) {
            var info = classes[name];
            var isActive = name === activeClass;
            var chip = document.createElement('div');
            chip.style.cssText =
                'display:flex;align-items:center;gap:6px;padding:5px 8px;margin-bottom:4px;' +
                'border-radius:6px;cursor:pointer;font-size:0.82rem;' +
                'background:' + (isActive ? hexToRgba(info.color,0.12) : '#f8fafc') + ';' +
                'border:1.5px solid ' + (isActive ? info.color : '#e2e8f0') + ';';

            var swatch = document.createElement('span');
            swatch.style.cssText = 'width:13px;height:13px;border-radius:50%;flex-shrink:0;background:'+info.color+';';

            var lbl = document.createElement('span');
            lbl.style.flex = '1';
            lbl.textContent = name + ' (' + (info.examples ? info.examples.length : 0) + ')';

            var del = document.createElement('button');
            del.textContent = 'x';
            del.title = 'Remove class';
            del.style.cssText = 'background:none;border:none;cursor:pointer;color:#94a3b8;padding:0 2px;font-size:0.75rem;';
            del.onclick = function(e) { e.stopPropagation(); removeClass(name); };

            chip.append(swatch, lbl, del);
            chip.onclick = function() { activeClass = name; renderClassList(); updateAddBtn(); drawPromptLayer(); };
            container.appendChild(chip);
        });
    }

    function addNewClass(name) {
        name = (name || '').trim();
        if (!name) return;
        if (!classes[name]) {
            var used = new Set(Object.values(classes).map(function(c) { return c.color; }));
            var color = PALETTE.find(function(c) { return !used.has(c); }) || PALETTE[Object.keys(classes).length % PALETTE.length];
            classes[name] = { color: color, examples: [] };
        }
        activeClass = name;
        renderClassList();
        updateAddBtn();
        drawPromptLayer();
    }

    function removeClass(name) {
        delete classes[name];
        if (activeClass === name) activeClass = Object.keys(classes)[0] || null;
        renderClassList();
        drawOverlay();
        updateAddBtn();
    }

    function updateAddBtn() {
        var btn = document.getElementById('fs-add-btn');
        if (!btn) return;
        var hasPrompt = pendingPoints.length > 0 || pendingBox !== null;
        btn.disabled = !activeClass || !hasPrompt || !prepared.has(imageId);
    }

    // image list
    function renderImageList() {
        var list  = document.getElementById('fs-image-list');
        var count = document.getElementById('fs-image-count');
        if (!list) return;
        list.innerHTML = '';
        if (count) count.textContent = images.length;
        images.forEach(function(filename) {
            var stem = filename.replace(/\.[^.]+$/, '');
            var icon = analyzed.has(stem) ? '\u{1F535}' : prepared.has(stem) ? '\u{1F7E2}' : '\u26AA';
            var div = document.createElement('div');
            div.style.cssText = 'padding:5px 8px;cursor:pointer;border-radius:4px;font-size:0.82rem;' +
                (filename === currentImg ? 'background:#eff6ff;color:#1e40af;font-weight:600;' : '');
            div.textContent = icon + ' ' + filename;
            div.onclick = function() { selectImage(filename); };
            list.appendChild(div);
        });
    }

    async function selectImage(filename) {
        saveExamplesForImage(imageId);   // persist before switching
        ghostPredictions = {};           // clear ghosts from previous image
        currentImg = filename;
        imageId    = filename.replace(/\.[^.]+$/, '');
        restoreExamplesForImage(imageId);
        resetPrompt(false);
        renderImageList();

        var img = new Image();
        img.crossOrigin = 'anonymous';
        img.src = '/api/projects/' + projectId + '/image/' + encodeURIComponent(filename);
        img.onload = function() {
            naturalW = img.naturalWidth;
            naturalH = img.naturalHeight;

            var wrapper = document.getElementById('fs-canvas-wrapper');
            wrapper.style.display = 'block';
            document.getElementById('fs-empty-msg').style.display = 'none';
            document.getElementById('fs-click-hint').style.display = 'block';

            var parentRect = wrapper.parentElement.getBoundingClientRect();
            var maxW = (parentRect.width || 880) - 8;
            var scale = Math.min(1, maxW / naturalW);
            var w = Math.round(naturalW * scale);
            var h = Math.round(naturalH * scale);

            [canvas, overlay, promptCanvas].forEach(function(c) {
                c.width  = w; c.height = h;
                c.style.width  = w + 'px';
                c.style.height = h + 'px';
            });

            ctx.drawImage(img, 0, 0, w, h);
            drawOverlay();
            resetPrompt(false);

            if (!prepared.has(imageId)) {
                setStatus('Loading image for SAM2...', 'info');
                apiRequest('/api/projects/' + projectId + '/fewshot/load-image', {
                    method: 'POST', body: JSON.stringify({ image_filename: filename })
                }).then(function(data) {
                    prepared.add(imageId);
                    renderImageList();
                    if (data && data.classes) { mergeServerClasses(data.classes); drawOverlay(); renderClassList(); }
                    setStatus('Ready - click or drag to segment.', 'success');
                    updateAddBtn();
                }).catch(function(e) {
                    setStatus('Load failed: ' + e.message, 'error');
                });
            } else {
                setStatus('Ready - click or drag to segment.', 'success');
            }
        };
        img.onerror = function() { setStatus('Failed to load: ' + filename, 'error'); };
    }

    // drawing
    function drawContour(ctx2d, contour, fill, stroke, lw) {
        if (!contour || contour.length < 3) return;
        ctx2d.beginPath();
        ctx2d.moveTo(toCanvasX(contour[0][0]), toCanvasY(contour[0][1]));
        for (var i = 1; i < contour.length; i++)
            ctx2d.lineTo(toCanvasX(contour[i][0]), toCanvasY(contour[i][1]));
        ctx2d.closePath();
        ctx2d.fillStyle   = fill;
        ctx2d.fill();
        ctx2d.strokeStyle = stroke;
        ctx2d.lineWidth   = lw || 1.5;
        ctx2d.stroke();
    }

    function drawOverlay() {
        if (!octx || !overlay) return;
        octx.clearRect(0, 0, overlay.width, overlay.height);
        // Confirmed labeled examples (solid color)
        Object.keys(classes).forEach(function(name) {
            var info = classes[name];
            (info.examples || []).forEach(function(ex) {
                if (ex.contour && ex.contour.length >= 3)
                    drawContour(octx, ex.contour, hexToRgba(info.color, 0.45), info.color, 2);
            });
        });
        // Ghost predictions (dashed, awaiting accept/reject)
        Object.keys(ghostPredictions).forEach(function(name) {
            var info  = classes[name];
            var color = info ? info.color : '#94a3b8';
            (ghostPredictions[name] || []).forEach(function(pred, idx) {
                if (!pred.contour || pred.contour.length < 3) return;
                var isHovered = hoveredGhost && hoveredGhost.className === name && hoveredGhost.predIdx === idx;
                octx.save();
                octx.beginPath();
                octx.moveTo(toCanvasX(pred.contour[0][0]), toCanvasY(pred.contour[0][1]));
                for (var i = 1; i < pred.contour.length; i++)
                    octx.lineTo(toCanvasX(pred.contour[i][0]), toCanvasY(pred.contour[i][1]));
                octx.closePath();
                octx.fillStyle = isHovered ? hexToRgba(color, 0.40) : hexToRgba(color, 0.18);
                octx.fill();
                if (!isHovered) octx.setLineDash([7, 4]);
                octx.strokeStyle = isHovered ? '#fff' : color;
                octx.lineWidth   = isHovered ? 2.5 : 1.5;
                octx.stroke();
                octx.setLineDash([]);
                if (isHovered) {
                    // label badge
                    var xs = pred.contour.map(function(p) { return toCanvasX(p[0]); });
                    var ys = pred.contour.map(function(p) { return toCanvasY(p[1]); });
                    var bx = (Math.min.apply(null,xs) + Math.max.apply(null,xs)) / 2;
                    var by = Math.min.apply(null,ys) - 8;
                    octx.fillStyle = '#1e293b';
                    octx.font = 'bold 11px sans-serif';
                    octx.textAlign = 'center'; octx.textBaseline = 'bottom';
                    octx.fillText('click to accept/reject', bx, by);
                }
                octx.restore();
            });
        });
    }

    function drawPromptLayer() {
        if (!pctx || !promptCanvas) return;
        pctx.clearRect(0, 0, promptCanvas.width, promptCanvas.height);

        if (previewContour && previewContour.length >= 3) {
            if (activeClass && classes[activeClass]) {
                // Classified — draw in class colour
                var color = classes[activeClass].color;
                drawContour(pctx, previewContour, hexToRgba(color, 0.45), color, 2.5);
            } else {
                // Floating / unclassified — grey with dashed outline
                pctx.save();
                pctx.beginPath();
                pctx.moveTo(toCanvasX(previewContour[0][0]), toCanvasY(previewContour[0][1]));
                for (var _i = 1; _i < previewContour.length; _i++)
                    pctx.lineTo(toCanvasX(previewContour[_i][0]), toCanvasY(previewContour[_i][1]));
                pctx.closePath();
                pctx.fillStyle = 'rgba(148,163,184,0.30)';
                pctx.fill();
                pctx.setLineDash([6, 4]);
                pctx.strokeStyle = '#94a3b8';
                pctx.lineWidth = 2;
                pctx.stroke();
                pctx.setLineDash([]);
                // '?' badge in centre
                var _xs = previewContour.map(function(p) { return toCanvasX(p[0]); });
                var _ys = previewContour.map(function(p) { return toCanvasY(p[1]); });
                var _cx = (Math.min.apply(null,_xs) + Math.max.apply(null,_xs)) / 2;
                var _cy = (Math.min.apply(null,_ys) + Math.max.apply(null,_ys)) / 2;
                pctx.fillStyle = 'rgba(255,255,255,0.88)';
                pctx.beginPath(); pctx.arc(_cx, _cy, 13, 0, 2*Math.PI); pctx.fill();
                pctx.fillStyle = '#64748b';
                pctx.font = 'bold 13px sans-serif';
                pctx.textAlign = 'center';
                pctx.textBaseline = 'middle';
                pctx.fillText('?', _cx, _cy);
                pctx.restore();
            }
        }

        if (promptMode === 'box' && pendingBox) {
            var b = pendingBox;
            pctx.strokeStyle = '#6366f1'; pctx.lineWidth = 2;
            pctx.setLineDash([6,3]);
            pctx.strokeRect(toCanvasX(b.x1), toCanvasY(b.y1),
                            toCanvasX(b.x2-b.x1), toCanvasY(b.y2-b.y1));
            pctx.setLineDash([]);
        }

        pendingPoints.forEach(function(p) {
            var cx = toCanvasX(p.x), cy = toCanvasY(p.y);
            pctx.beginPath(); pctx.arc(cx, cy, 8, 0, 2*Math.PI);
            pctx.fillStyle   = p.label === 1 ? '#22c55e' : '#ef4444';
            pctx.fill();
            pctx.strokeStyle = '#fff'; pctx.lineWidth = 2; pctx.stroke();
            pctx.fillStyle   = '#fff';
            pctx.font        = 'bold 11px sans-serif';
            pctx.textAlign   = 'center';
            pctx.textBaseline = 'middle';
            if (p.label === 1) {
                pctx.fillText('+', cx, cy);
            } else {
                pctx.strokeStyle = '#fff'; pctx.lineWidth = 1.5;
                pctx.beginPath(); pctx.moveTo(cx-4,cy-4); pctx.lineTo(cx+4,cy+4); pctx.stroke();
                pctx.beginPath(); pctx.moveTo(cx+4,cy-4); pctx.lineTo(cx-4,cy+4); pctx.stroke();
            }
        });
    }

    function resetPrompt(redraw) {
        pendingPoints  = [];
        pendingBox     = null;
        boxDragStart   = null;
        previewContour = null;
        if (redraw !== false && pctx)
            pctx.clearRect(0, 0, promptCanvas.width, promptCanvas.height);
        _hidePromptToolbar();
        _hideClassPopover();
        updateAddBtn();
    }

    // ── per-image examples cache ──────────────────────────────────────────────

    function saveExamplesForImage(id) {
        if (!id) return;
        imageClassData[id] = {};
        Object.keys(classes).forEach(function(name) {
            imageClassData[id][name] = (classes[name].examples || []).slice();
        });
    }

    function restoreExamplesForImage(id) {
        var saved = imageClassData[id];
        Object.keys(classes).forEach(function(name) {
            classes[name].examples = (saved && saved[name]) ? saved[name].slice() : [];
        });
    }

    function mergeServerClasses(serverClasses) {
        if (!serverClasses) return;
        Object.keys(serverClasses).forEach(function(name) {
            var srv = serverClasses[name];
            if (!classes[name]) {
                classes[name] = { color: srv.color, examples: [] };
            } else {
                classes[name].color = srv.color || classes[name].color;
            }
            // Restore examples for this image from server if not already cached locally
            if (!imageClassData[imageId] || !imageClassData[imageId][name]) {
                classes[name].examples = srv.examples || [];
            }
        });
        // Update local cache for current image
        imageClassData[imageId] = imageClassData[imageId] || {};
        Object.keys(classes).forEach(function(name) {
            if (!imageClassData[imageId][name]) {
                imageClassData[imageId][name] = (classes[name].examples || []).slice();
            }
        });
    }

    // ── prompt toolbar (shown after SAM2 preview) ─────────────────────────────

    function _showPromptToolbar() {
        if (!previewContour || previewContour.length < 3 || !promptCanvas) return;
        var xs   = previewContour.map(function(p) { return toCanvasX(p[0]); });
        var ys   = previewContour.map(function(p) { return toCanvasY(p[1]); });
        var minY = Math.min.apply(null, ys);
        var cx   = (Math.min.apply(null, xs) + Math.max.apply(null, xs)) / 2;
        var rect = promptCanvas.getBoundingClientRect();
        var scX  = rect.width  / promptCanvas.width;
        var scY  = rect.height / promptCanvas.height;
        var sx   = rect.left + cx   * scX;
        var sy   = rect.top  + minY * scY;
        var tb   = document.getElementById('fs-prompt-toolbar');
        if (!tb) return;
        tb.style.display = 'flex';
        var tbW  = tb.offsetWidth || 250;
        tb.style.left = Math.min(Math.max(sx - tbW / 2, 8), window.innerWidth - tbW - 8) + 'px';
        tb.style.top  = Math.max(sy - 50, 8) + 'px';
    }

    function _hidePromptToolbar() {
        var tb = document.getElementById('fs-prompt-toolbar');
        if (tb) tb.style.display = 'none';
    }

    // ── class assignment popover (shown after Accept) ─────────────────────────

    function _showClassPopover() {
        _hidePromptToolbar();
        var pop = document.getElementById('fs-class-popover');
        if (!pop) return;
        var chips = document.getElementById('fs-popover-class-chips');
        if (chips) {
            chips.innerHTML = '';
            Object.keys(classes).forEach(function(name) {
                var info = classes[name];
                var btn  = document.createElement('button');
                btn.style.cssText =
                    'padding:5px 12px;font-size:0.8rem;border:2px solid ' + info.color + ';' +
                    'color:' + info.color + ';background:transparent;border-radius:6px;cursor:pointer;';
                btn.textContent = name;
                btn.onclick = function() {
                    _hideClassPopover();
                    activeClass = name;
                    renderClassList();
                    commitExample();
                };
                chips.appendChild(btn);
            });
        }
        pop.style.display = 'block';
        if (!promptCanvas) return;
        var rect = promptCanvas.getBoundingClientRect();
        var popW = pop.offsetWidth || 220;
        var popH = pop.offsetHeight || 120;
        var left = Math.min(Math.max(rect.left + rect.width  / 2 - popW / 2, 8), window.innerWidth - popW - 8);
        var top  = Math.max(rect.top  + rect.height / 2 - popH / 2, 8);
        pop.style.left = left + 'px';
        pop.style.top  = top  + 'px';
    }

    function _hideClassPopover() {
        var pop = document.getElementById('fs-class-popover');
        if (pop) pop.style.display = 'none';
    }

    // ── ghost hit-testing helpers ─────────────────────────────────────────────

    function _pointInContourNatural(nx, ny, contour) {
        var inside = false;
        for (var i = 0, j = contour.length - 1; i < contour.length; j = i++) {
            var xi = contour[i][0], yi = contour[i][1];
            var xj = contour[j][0], yj = contour[j][1];
            if (((yi > ny) !== (yj > ny)) && (nx < (xj - xi) * (ny - yi) / (yj - yi) + xi))
                inside = !inside;
        }
        return inside;
    }

    function _findGhostAtNatural(nx, ny) {
        for (var name in ghostPredictions) {
            var preds = ghostPredictions[name];
            for (var i = 0; i < preds.length; i++) {
                var c = preds[i].contour;
                if (c && c.length >= 3 && _pointInContourNatural(nx, ny, c))
                    return { className: name, pred: preds[i], predIdx: i };
            }
        }
        return null;
    }

    // ── ghost popup (per-mask accept/reject) ──────────────────────────────────

    function _showGhostPopup(ghost, clientX, clientY) {
        _closeGhostPopup();
        var info  = classes[ghost.className];
        var color = info ? info.color : '#94a3b8';
        var sim   = ghost.pred.similarity !== undefined ? ' · ' + (ghost.pred.similarity * 100).toFixed(0) + '%' : '';
        var div   = document.createElement('div');
        div.style.cssText =
            'position:fixed;z-index:10001;background:#1e293b;color:#fff;border-radius:10px;' +
            'padding:10px 14px;box-shadow:0 6px 24px rgba(0,0,0,0.4);min-width:190px;';
        div.innerHTML =
            '<div style="display:flex;align-items:center;gap:6px;margin-bottom:8px;">' +
            '<span style="width:10px;height:10px;border-radius:50%;background:' + color + ';display:inline-block;flex-shrink:0;"></span>' +
            '<strong style="font-size:0.82rem;">' + ghost.className + '</strong>' +
            '<span style="font-size:0.72rem;color:#94a3b8;margin-left:auto;">' + sim + '</span>' +
            '</div>' +
            '<div style="display:flex;gap:6px;margin-bottom:6px;">' +
            '<button id="gp-accept-single" style="flex:1;padding:5px 0;font-size:0.78rem;background:#22c55e;color:#fff;border:none;border-radius:5px;cursor:pointer;font-weight:600;">✓ Accept</button>' +
            '<button id="gp-reject-single" style="flex:1;padding:5px 0;font-size:0.78rem;background:#ef4444;color:#fff;border:none;border-radius:5px;cursor:pointer;">&#10005; Reject</button>' +
            '</div>' +
            '<button id="gp-accept-all" style="width:100%;padding:4px 8px;font-size:0.72rem;background:#334155;color:#94a3b8;border:none;border-radius:5px;cursor:pointer;">Accept all for \"' + ghost.className + '\"</button>';
        var popW = 210;
        div.style.left = Math.min(Math.max(clientX + 14, 8), window.innerWidth - popW - 8) + 'px';
        div.style.top  = Math.max(clientY - 50, 8) + 'px';
        var backdrop = document.createElement('div');
        backdrop.style.cssText = 'position:fixed;inset:0;z-index:10000;';
        backdrop.onclick = function() { _closeGhostPopup(); };
        document.body.appendChild(backdrop);
        document.body.appendChild(div);
        ghostPopup = { div: div, backdrop: backdrop };
        document.getElementById('gp-accept-single').onclick = async function(e) {
            e.stopPropagation(); _closeGhostPopup();
            await _acceptSingleGhost(ghost.className, ghost.predIdx);
        };
        document.getElementById('gp-reject-single').onclick = function(e) {
            e.stopPropagation(); _closeGhostPopup();
            _rejectSingleGhost(ghost.className, ghost.predIdx);
        };
        document.getElementById('gp-accept-all').onclick = async function(e) {
            e.stopPropagation(); _closeGhostPopup();
            await _acceptGhosts(ghost.className);
        };
    }

    function _closeGhostPopup() {
        if (ghostPopup) {
            ghostPopup.div.remove();
            ghostPopup.backdrop.remove();
            ghostPopup = null;
        }
        if (hoveredGhost) { hoveredGhost = null; drawOverlay(); }
    }

    function _addConfirmedContour(className, pred) {
        if (!classes[className] || !pred.contour || pred.contour.length < 3) return;
        classes[className].examples.push({
            id: classes[className].examples.length,
            contour: pred.contour,
            area: pred.area || 0,
            x: 0, y: 0, sam_score: 1.0,
            mask_index: pred.index,
            _confirmed_ghost: true,
        });
        saveExamplesForImage(imageId);
    }

    async function _acceptSingleGhost(className, predIdx) {
        var preds = ghostPredictions[className];
        if (!preds || predIdx < 0 || predIdx >= preds.length) return;
        var pred = preds[predIdx];
        var confirmations = {}; confirmations[className] = [pred.index];
        try {
            await apiRequest('/api/projects/' + projectId + '/fewshot/confirm', {
                method: 'POST',
                body: JSON.stringify({ image_id: imageId, image_filename: currentImg, confirmations: confirmations })
            });
            _addConfirmedContour(className, pred);
            preds.splice(predIdx, 1);
            if (!preds.length) delete ghostPredictions[className];
            drawOverlay(); renderGhostPanel(); renderClassList();
            setStatus('Accepted 1 mask for \"' + className + '\" ✓', 'success');
        } catch(e) { setStatus('Confirm failed: ' + e.message, 'error'); }
    }

    function _rejectSingleGhost(className, predIdx) {
        var preds = ghostPredictions[className];
        if (!preds || predIdx < 0 || predIdx >= preds.length) return;
        preds.splice(predIdx, 1);
        if (!preds.length) delete ghostPredictions[className];
        drawOverlay(); renderGhostPanel();
    }

    // ── ghost prediction management ───────────────────────────────────────────

    function renderGhostPanel() {
        var panel = document.getElementById('fs-ghost-panel');
        if (!panel) return;
        var names = Object.keys(ghostPredictions).filter(function(n) {
            return ghostPredictions[n] && ghostPredictions[n].length > 0;
        });
        if (!names.length) { panel.style.display = 'none'; return; }
        panel.style.display = 'block';
        var html = '<p style="font-size:0.78rem;font-weight:600;color:#475569;margin:0 0 8px;">Ghost Predictions (accept \u2714 or reject \u2715):</p>';
        names.forEach(function(name) {
            var info  = classes[name];
            var color = info ? info.color : '#94a3b8';
            var n     = ghostPredictions[name].length;
            html +=
                '<div style="display:flex;align-items:center;gap:8px;margin-bottom:5px;">' +
                '<span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:' + color + ';flex-shrink:0;"></span>' +
                '<span style="flex:1;font-size:0.82rem;">' + name + ' \u00b7 ' + n + ' prediction' + (n !== 1 ? 's' : '') + '</span>' +
                '<button onclick="FewShotTab._acceptGhosts(\'' + name + '\')" title="Save masks" ' +
                'style="padding:2px 10px;font-size:0.75rem;background:#22c55e;color:#fff;border:none;border-radius:4px;cursor:pointer;">\u2714 Save</button>' +
                '<button onclick="FewShotTab._rejectGhosts(\'' + name + '\')" title="Discard" ' +
                'style="padding:2px 10px;font-size:0.75rem;background:#ef4444;color:#fff;border:none;border-radius:4px;cursor:pointer;">\u2715 Discard</button>' +
                '</div>';
        });
        panel.innerHTML = html;
    }

    async function _acceptGhosts(className) {
        var preds = ghostPredictions[className];
        if (!preds || !preds.length) return;
        var indices = preds.map(function(p) { return p.index; });
        var confirmations = {}; confirmations[className] = indices;
        try {
            await apiRequest('/api/projects/' + projectId + '/fewshot/confirm', {
                method: 'POST',
                body: JSON.stringify({ image_id: imageId, image_filename: currentImg, confirmations: confirmations })
            });
            preds.forEach(function(pred) { _addConfirmedContour(className, pred); });
            delete ghostPredictions[className];
            drawOverlay(); renderGhostPanel(); renderClassList();
            setStatus('Saved ' + indices.length + ' mask(s) for "' + className + '" \u2713', 'success');
        } catch(e) { setStatus('Confirm failed: ' + e.message, 'error'); }
    }

    function _rejectGhosts(className) {
        delete ghostPredictions[className];
        drawOverlay(); renderGhostPanel();
    }

    // coordinate mapping: CSS click -> natural image pixels
    function naturalCoords(e) {
        var rect = promptCanvas.getBoundingClientRect();
        return {
            x: Math.round((e.clientX - rect.left) / rect.width  * naturalW),
            y: Math.round((e.clientY - rect.top)  / rect.height * naturalH),
        };
    }

    // backend calls
    async function callPreview() {
        if (!imageId || !prepared.has(imageId)) return;
        var body = { image_id: imageId };
        if (pendingBox) body.box = [pendingBox.x1, pendingBox.y1, pendingBox.x2, pendingBox.y2];
        if (pendingPoints.length) {
            body.points = pendingPoints.map(function(p) { return [p.x, p.y]; });
            body.labels = pendingPoints.map(function(p) { return p.label; });
        }
        if (!body.box && !body.points) return;
        try {
            var data = await apiRequest(
                '/api/projects/' + projectId + '/fewshot/preview-prompt',
                { method: 'POST', body: JSON.stringify(body) }
            );
            previewContour = data.contour || null;
            drawPromptLayer();
            updateAddBtn();
            if (previewContour && previewContour.length >= 3) _showPromptToolbar();
        } catch(e) { console.warn('preview-prompt:', e); }
    }

    async function commitExample(classNameOverride) {
        var cls = classNameOverride !== undefined ? classNameOverride : activeClass;
        if (!cls) { setStatus('Select or create a class first.', 'error'); return; }
        if (!prepared.has(imageId)) { setStatus('Image not ready.', 'error'); return; }
        if (!pendingPoints.length && !pendingBox) { setStatus('Nothing to add.', 'error'); return; }

        var firstPt = pendingPoints[0] || { x:0, y:0 };
        var body = { image_id: imageId, class_name: cls,
                     x: firstPt.x, y: firstPt.y, threshold: threshold };
        if (pendingPoints.length) {
            body.points = pendingPoints.map(function(p) { return [p.x, p.y]; });
            body.labels = pendingPoints.map(function(p) { return p.label; });
        }
        if (pendingBox) body.box = [pendingBox.x1, pendingBox.y1, pendingBox.x2, pendingBox.y2];

        _hideClassPopover();
        setStatus('Segmenting…', 'info');
        try {
            var data = await apiRequest(
                '/api/projects/' + projectId + '/fewshot/add-example',
                { method: 'POST', body: JSON.stringify(body) }
            );
            if (data.classes) {
                Object.keys(data.classes).forEach(function(name) {
                    var srv = data.classes[name];
                    if (!classes[name]) classes[name] = { color: srv.color, examples: [] };
                    classes[name].color = srv.color || classes[name].color;
                    // Preserve ghost-confirmed entries (they live only in memory, server doesn't know)
                    var ghosts = (classes[name].examples || []).filter(function(ex) { return ex._confirmed_ghost; });
                    classes[name].examples = (srv.examples || []).concat(ghosts);
                });
                saveExamplesForImage(imageId);
            }
            drawOverlay();
            resetPrompt();
            renderClassList();
            setStatus('Added to “' + cls + '” ✓', 'success');
        } catch(e) { setStatus('Add failed: ' + e.message, 'error'); }
    }

    async function saveMasks() {
        if (!imageId || !currentImg) { setStatus('No image loaded.', 'error'); return; }
        if (!Object.keys(classes).some(function(n) { return (classes[n].examples||[]).length>0; })) {
            setStatus('No labeled objects to save.', 'error'); return;
        }
        setStatus('Saving masks...', 'info');
        try {
            var data = await apiRequest(
                '/api/projects/' + projectId + '/fewshot/save-masks',
                { method: 'POST', body: JSON.stringify({ image_id: imageId, image_filename: currentImg }) }
            );
            var n = Object.keys(data.saved || {}).length;
            setStatus(n + ' mask(s) saved \u2713', 'success');
        } catch(e) { setStatus('Save failed: ' + e.message, 'error'); }
    }

    async function analyzeForBatch() {
        if (!projectId || !currentImg) { setStatus('No image loaded.', 'error'); return; }

        // ── show loading overlay ──────────────────────────────────────────────
        var loadingDiv   = document.getElementById('fs-canvas-loading');
        var loadingLabel = document.getElementById('fs-loading-label');
        var analyzeBtn   = document.getElementById('fs-analyze-btn');
        if (loadingDiv)   { loadingDiv.style.display = 'flex'; }
        if (analyzeBtn)   { analyzeBtn.disabled = true; analyzeBtn.textContent = '⏳ Analyzing…'; }

        // animated status pulse
        var dots = 0;
        var dotsInterval = setInterval(function() {
            dots = (dots + 1) % 4;
            var label = 'SAM2 auto-mask + DINOv2' + '.'.repeat(dots + 1);
            setStatus(label, 'info');
            if (loadingLabel) loadingLabel.textContent = label;
        }, 600);

        try {
            var data = await apiRequest(
                '/api/projects/' + projectId + '/fewshot/process-image-sync',
                { method: 'POST', body: JSON.stringify({ image_filename: currentImg, threshold: threshold }) }
            );
            analyzed.add(imageId); prepared.add(imageId);
            renderImageList();
            // Load ghost predictions from response
            ghostPredictions = {};
            var totalPreds = 0;
            if (data.classes) {
                Object.keys(data.classes).forEach(function(name) {
                    var srv = data.classes[name];
                    if (!classes[name]) classes[name] = { color: srv.color, examples: [] };
                    var preds = srv.predictions || [];
                    if (preds.length) { ghostPredictions[name] = preds; totalPreds += preds.length; }
                });
                renderClassList();
            }
            drawOverlay();
            renderGhostPanel();
            if (totalPreds > 0) {
                setStatus(totalPreds + ' ghost prediction(s) — accept or discard below.', 'success');
            } else {
                setStatus('Analysis done: ' + data.num_masks + ' masks, none matched above threshold.', 'info');
            }
        } catch(e) {
            setStatus('Analysis failed: ' + e.message, 'error');
        } finally {
            clearInterval(dotsInterval);
            if (loadingDiv) loadingDiv.style.display = 'none';
            if (analyzeBtn) { analyzeBtn.disabled = false; analyzeBtn.textContent = '🔬 Analyze for Batch'; }
        }
    }

    async function applyToAll() {
        if (!projectId) { setStatus('No project.', 'error'); return; }
        if (!Object.keys(classes).length) { setStatus('No classes defined.', 'error'); return; }
        var progressDiv  = document.getElementById('fs-batch-progress');
        var progressBar  = document.getElementById('fs-batch-progress-bar');
        var progressInfo = document.getElementById('fs-batch-progress-info');
        if (progressDiv) progressDiv.style.display = 'block';
        setStatus('Starting batch...', 'info');
        try {
            await apiRequest('/api/projects/' + projectId + '/fewshot/apply-to-all', {
                method: 'POST', body: JSON.stringify({ threshold: threshold, excluded_images: [] }),
            });
            if (batchPollId) clearInterval(batchPollId);
            batchPollId = setInterval(async function() {
                try {
                    var prog = await apiRequest('/api/operation-progress');
                    if (progressBar && prog.total > 0) {
                        var pct = Math.round(prog.current / prog.total * 100);
                        progressBar.style.width = pct + '%';
                        progressBar.textContent = pct + '%';
                        if (progressInfo) progressInfo.textContent = prog.message || '';
                    }
                    if (prog.current >= prog.total && prog.total > 0) {
                        clearInterval(batchPollId); batchPollId = null;
                        setStatus('Batch complete \u2713', 'success');
                    }
                } catch(e) {}
            }, 1500);
        } catch(e) { setStatus('Batch failed: ' + e.message, 'error'); }
    }

    // floating tooltip near cursor
    var _tooltip = null;
    function _getTooltip() {
        if (!_tooltip) {
            _tooltip = document.createElement('div');
            _tooltip.style.cssText =
                'position:fixed;z-index:9999;pointer-events:none;display:none;' +
                'background:rgba(15,23,42,0.88);color:#fff;padding:4px 10px;' +
                'border-radius:6px;font-size:0.76rem;white-space:nowrap;' +
                'box-shadow:0 2px 8px rgba(0,0,0,0.25);';
            document.body.appendChild(_tooltip);
        }
        return _tooltip;
    }
    function _showTooltip(e, text) {
        var t = _getTooltip();
        t.textContent = text;
        t.style.display = 'block';
        t.style.left = (e.clientX + 14) + 'px';
        t.style.top  = (e.clientY - 30) + 'px';
    }
    function _hideTooltip() {
        var t = _getTooltip();
        t.style.display = 'none';
    }

    // canvas events
    function setupCanvasEvents() {
        promptCanvas.addEventListener('contextmenu', function(e) { e.preventDefault(); });

        promptCanvas.addEventListener('mousedown', function(e) {
            e.preventDefault();
            if (!imageId || !naturalW) return;
            // ghost hit-test — intercept click before normal point/box logic
            if (Object.keys(ghostPredictions).length > 0) {
                var gpos = naturalCoords(e);
                var found = _findGhostAtNatural(gpos.x, gpos.y);
                if (found) { _showGhostPopup(found, e.clientX, e.clientY); return; }
            }
            var pos = naturalCoords(e);
            if (promptMode === 'box') {
                boxDragStart   = { x: pos.x, y: pos.y };
                pendingBox     = null;
                previewContour = null;
                drawPromptLayer();
            } else {
                var isNeg = e.button === 2 || e.shiftKey;
                pendingPoints.push({ x: pos.x, y: pos.y, label: isNeg ? 0 : 1 });
                drawPromptLayer();
                callPreview();
            }
            updateAddBtn();
        });

        promptCanvas.addEventListener('mousemove', function(e) {
            // box rubber-band
            if (promptMode === 'box' && boxDragStart) {
                var pos = naturalCoords(e);
                pendingBox = {
                    x1: Math.min(boxDragStart.x, pos.x), y1: Math.min(boxDragStart.y, pos.y),
                    x2: Math.max(boxDragStart.x, pos.x), y2: Math.max(boxDragStart.y, pos.y),
                };
                previewContour = null;
                drawPromptLayer();
            }
            // ghost hover hit-test
            if (!ghostPopup && Object.keys(ghostPredictions).length > 0) {
                var gp = naturalCoords(e);
                var gfound = _findGhostAtNatural(gp.x, gp.y);
                if (gfound) {
                    if (!hoveredGhost || hoveredGhost.className !== gfound.className || hoveredGhost.predIdx !== gfound.predIdx) {
                        hoveredGhost = gfound;
                        drawOverlay();
                    }
                    _showTooltip(e, gfound.className + ' \u00b7 click to accept/reject');
                    return;
                } else if (hoveredGhost) {
                    hoveredGhost = null;
                    drawOverlay();
                }
            }
            // normal tooltip
            var hasPrompt = pendingPoints.length > 0 || pendingBox || previewContour;
            if (hasPrompt && !activeClass) {
                _showTooltip(e, '\u2190 Select or create a class');
            } else if (activeClass) {
                _showTooltip(e, activeClass);
            } else {
                _hideTooltip();
            }
        });

        promptCanvas.addEventListener('mouseleave', function() { _hideTooltip(); });

        promptCanvas.addEventListener('mouseup', function(e) {
            if (promptMode !== 'box' || !boxDragStart) return;
            boxDragStart = null;
            if (pendingBox && (pendingBox.x2-pendingBox.x1) > 5 && (pendingBox.y2-pendingBox.y1) > 5) {
                callPreview();
            }
            updateAddBtn();
        });
    }

    function updatePromptModeUI() {
        var btnPt  = document.getElementById('fs-mode-point');
        var btnBox = document.getElementById('fs-mode-box');
        var hint   = document.getElementById('fs-neg-hint');
        if (btnPt)  btnPt.classList.toggle('active',  promptMode === 'point');
        if (btnBox) btnBox.classList.toggle('active',  promptMode === 'box');
        if (hint)   hint.style.display = promptMode === 'point' ? '' : 'none';
        resetPrompt();
    }

    function init() {
        canvas       = document.getElementById('fs-canvas');
        overlay      = document.getElementById('fs-overlay');
        promptCanvas = document.getElementById('fs-prompt');
        if (!canvas || !overlay || !promptCanvas) {
            console.error('FewShotTab: canvas elements not found');
            return;
        }
        ctx  = canvas.getContext('2d');
        octx = overlay.getContext('2d');
        pctx = promptCanvas.getContext('2d');

        var slider    = document.getElementById('fs-threshold');
        var sliderVal = document.getElementById('fs-threshold-value');
        if (slider) slider.addEventListener('input', function() {
            threshold = parseFloat(slider.value);
            if (sliderVal) sliderVal.textContent = threshold.toFixed(2);
        });

        var classInput  = document.getElementById('fs-new-class-name');
        var addClassBtn = document.getElementById('fs-add-class-btn');
        if (addClassBtn) addClassBtn.addEventListener('click', function() {
            if (classInput && classInput.value.trim()) { addNewClass(classInput.value); classInput.value = ''; }
        });
        if (classInput) classInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter' && classInput.value.trim()) { addNewClass(classInput.value); classInput.value = ''; }
        });

        var btnPt  = document.getElementById('fs-mode-point');
        var btnBox = document.getElementById('fs-mode-box');
        if (btnPt)  btnPt.addEventListener('click',  function() { promptMode = 'point'; updatePromptModeUI(); });
        if (btnBox) btnBox.addEventListener('click',  function() { promptMode = 'box';   updatePromptModeUI(); });

        var resetBtn = document.getElementById('fs-reset-prompt');
        if (resetBtn) resetBtn.addEventListener('click', function() { resetPrompt(); });

        var addBtn = document.getElementById('fs-add-btn');
        if (addBtn) { addBtn.disabled = true; addBtn.addEventListener('click', commitExample); }

        var saveBtn    = document.getElementById('fs-save-btn');
        var analyzeBtn = document.getElementById('fs-analyze-btn');
        var batchBtn   = document.getElementById('fs-batch-btn');
        if (saveBtn)    saveBtn.addEventListener('click',    saveMasks);
        if (analyzeBtn) analyzeBtn.addEventListener('click', analyzeForBatch);
        if (batchBtn)   batchBtn.addEventListener('click',   applyToAll);

        setupCanvasEvents();

        // ── Prompt toolbar handlers ────────────────────────────────────────────
        var tbAccept = document.getElementById('fs-tb-accept');
        var tbUndo   = document.getElementById('fs-tb-undo');
        var tbCancel = document.getElementById('fs-tb-cancel');
        if (tbAccept) tbAccept.addEventListener('click', function() { _showClassPopover(); });
        if (tbUndo) tbUndo.addEventListener('click', function() {
            if (pendingPoints.length > 0) {
                pendingPoints.pop();
                previewContour = null;
                _hidePromptToolbar();
                if (pendingPoints.length > 0) callPreview();
                else drawPromptLayer();
            } else if (pendingBox) {
                pendingBox     = null;
                previewContour = null;
                _hidePromptToolbar();
                drawPromptLayer();
            }
            updateAddBtn();
        });
        if (tbCancel) tbCancel.addEventListener('click', function() { resetPrompt(); });

        // ── Class popover handlers ─────────────────────────────────────────────
        var popoverCreate = document.getElementById('fs-popover-create-btn');
        var popoverInput  = document.getElementById('fs-popover-new-name');
        function _commitFromPopover() {
            var name = popoverInput ? popoverInput.value.trim() : '';
            if (!name) return;
            _hideClassPopover();
            addNewClass(name);
            if (popoverInput) popoverInput.value = '';
            commitExample();
        }
        if (popoverCreate) popoverCreate.addEventListener('click', _commitFromPopover);
        if (popoverInput) popoverInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') _commitFromPopover();
        });

        console.log('FewShotTab init OK');
    }

    function onProjectSelected(id) {
        projectId = id;
        classes = {}; activeClass = null;
        prepared = new Set(); analyzed = new Set();
        images = []; currentImg = null; imageId = null;
        naturalW = 0; naturalH = 0;
        imageClassData   = {};
        ghostPredictions = {};
        resetPrompt(false); renderClassList(); loadImages();
    }

    async function loadImages() {
        if (!projectId) return;
        try {
            var data = await apiRequest('/api/projects/' + projectId + '/images');
            images = (data.images || []).sort();
            renderImageList();
        } catch(e) { console.warn('loadImages:', e); }
    }

    return { init: init, onProjectSelected: onProjectSelected, loadImages: loadImages,
             _acceptGhosts: _acceptGhosts, _rejectGhosts: _rejectGhosts };
})();

document.addEventListener('DOMContentLoaded', function() {
    FewShotTab.init();
    window.addEventListener('projectChanged', function(e) {
        var pid = e.detail && e.detail.project && e.detail.project.project_id;
        if (pid) FewShotTab.onProjectSelected(pid);
    });
});
