// Annotation Tab - Rebuilt from scratch
// Simple implementation for reviewing and editing masks

const annotationState = {
    currentProject: null,
    images: [],
    currentIndex: -1,
    canvas: null,
    ctx: null,
    maskCanvas: null,
    maskCtx: null,
    backgroundImage: null,
    originalBackgroundImage: null,  // Full resolution image
    originalWidth: 0,
    originalHeight: 0,
    displayWidth: 0,
    displayHeight: 0,
    currentTool: 'brush',
    brushSize: 20,
    isDrawing: false,
    isModified: false,
    maxDisplayWidth: 1200,  // Max width for display
    maxDisplayHeight: 800,   // Max height for display
    polygons: [],            // committed vessel polygons (ORIGINAL coords)
    currentPolygon: [],      // in-progress polygon (ORIGINAL coords)
    mousePreview: null,      // {x, y} display coords for rubber-band line
    vesselsSummary: {},      // base -> count of drawn vessels
    canvasZoom: 1,           // CSS zoom multiplier over the canvas buffer
    brushCursor: null,       // {x, y} buffer coords for brush/eraser size ring
    colorize: false,         // colour each separate mask differently
    colorizedCanvas: null,   // cached offscreen canvas with the coloured masks
    // Scale calibration
    scales: [],              // committed scale entries for current page (original coords)
    scaleDraftP1: null,      // [ox, oy] first ruler endpoint, null if not started
    scaleDraftP2: null,      // [ox, oy] second endpoint while popup is open
    scaleStep: null,         // null = ruler mode | 'zone' = zone-drag mode
    scaleZoneTargetIdx: null,// which scale entry gets the zone
    scaleZoneDragStart: null,// {x, y} original coords at drag start
    scaleCursorPos: null     // {x, y} display coords of cursor (live preview)
};

const POLYGON_CLOSE_THRESHOLD = 12; // display px to snap-close onto first point

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    console.log('[Annotation] Initializing...');
    
    initializeCanvas();
    initializeToolButtons();
    initializeNavigationButtons();
    initializeSaveButton();
    initializeExtractButton();
    initializeZoomControls();
    initializeScalePopup();

    window.addEventListener('projectChanged', handleProjectChanged);
    loadCurrentProject();
});

function initializeCanvas() {
    const canvas = document.getElementById('annotation-canvas');
    if (!canvas) return;
    
    annotationState.canvas = canvas;
    annotationState.ctx = canvas.getContext('2d');
    
    annotationState.maskCanvas = document.createElement('canvas');
    annotationState.maskCtx = annotationState.maskCanvas.getContext('2d');
    
    canvas.addEventListener('mousedown', startDrawing);
    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mouseup', stopDrawing);
    canvas.addEventListener('mouseleave', onCanvasMouseLeave);
    canvas.addEventListener('dblclick', onCanvasDblClick);
    document.addEventListener('keydown', onPolygonKeyDown);
}

// Convert a mouse event to display-space canvas coordinates
function eventToDisplayXY(e) {
    const canvas = annotationState.canvas;
    const rect = canvas.getBoundingClientRect();
    return {
        x: (e.clientX - rect.left) * (canvas.width / rect.width),
        y: (e.clientY - rect.top) * (canvas.height / rect.height)
    };
}

function displayToOriginal(x, y) {
    return [
        Math.round(x * annotationState.originalWidth / annotationState.displayWidth),
        Math.round(y * annotationState.originalHeight / annotationState.displayHeight)
    ];
}

function originalToDisplay(x, y) {
    return [
        x * annotationState.displayWidth / annotationState.originalWidth,
        y * annotationState.displayHeight / annotationState.originalHeight
    ];
}

function initializeToolButtons() {
    // Only real tools carry data-tool (brush/eraser/polygon/clear); zoom &
    // colorize are toggles handled separately.
    document.querySelectorAll('.btn-tool[data-tool]').forEach(btn => {
        btn.addEventListener('click', () => {
            const tool = btn.dataset.tool;
            if (tool === 'clear') clearMask();
            else selectTool(tool);
        });
    });

    document.getElementById('colorize-toggle')?.addEventListener('click', toggleColorize);

    const slider = document.getElementById('brush-size');
    if (slider) {
        slider.addEventListener('input', (e) => {
            annotationState.brushSize = parseInt(e.target.value);
        });
        annotationState.brushSize = parseInt(slider.value);
    }
}

function initializeNavigationButtons() {
    const prevBtn = document.getElementById('annotation-prev-btn');
    const nextBtn = document.getElementById('annotation-next-btn');
    if (prevBtn) prevBtn.addEventListener('click', () => navigateImage(-1));
    if (nextBtn) nextBtn.addEventListener('click', () => navigateImage(1));
}

function initializeSaveButton() {
    const btn = document.getElementById('annotation-save');
    if (btn) btn.addEventListener('click', saveMask);
}

function initializeExtractButton() {
    const btn = document.getElementById('extract-masks-btn');
    if (btn) btn.addEventListener('click', extractCards);
}

function initializeScalePopup() {
    document.getElementById('scale-confirm-btn')?.addEventListener('click', confirmScaleInput);
    document.getElementById('scale-cancel-btn')?.addEventListener('click', () => { cancelScaleDraft(); redrawCanvas(); });
    document.getElementById('scale-cm-input')?.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') { e.preventDefault(); confirmScaleInput(); }
        else if (e.key === 'Escape') { e.preventDefault(); cancelScaleDraft(); redrawCanvas(); }
    });
}

function initializeZoomControls() {
    document.getElementById('zoom-in-btn')?.addEventListener('click', () => zoomBy(1.25));
    document.getElementById('zoom-out-btn')?.addEventListener('click', () => zoomBy(0.8));
    document.getElementById('zoom-fit-btn')?.addEventListener('click', fitZoom);

    const container = document.getElementById('annotation-canvas-container');
    if (container) {
        // Ctrl/Cmd + wheel zooms toward the cursor; plain wheel scrolls normally
        container.addEventListener('wheel', (e) => {
            if (!e.ctrlKey && !e.metaKey) return;
            e.preventDefault();
            const factor = e.deltaY < 0 ? 1.15 : 0.87;
            zoomBy(factor, e.clientX, e.clientY);
        }, { passive: false });
    }

    // Re-fit when the annotation tab becomes visible (it may have loaded hidden)
    const tabBtn = document.querySelector('.tab-button[data-tab="annotation"]');
    if (tabBtn) {
        tabBtn.addEventListener('click', () => {
            if (annotationState.currentIndex >= 0) setTimeout(fitZoom, 50);
        });
    }
}

const ZOOM_MIN = 0.1, ZOOM_MAX = 8;

function applyZoom() {
    const canvas = annotationState.canvas;
    if (!canvas || !annotationState.displayWidth) return;
    const z = annotationState.canvasZoom;
    canvas.style.width = (annotationState.displayWidth * z) + 'px';
    canvas.style.height = (annotationState.displayHeight * z) + 'px';
    const label = document.getElementById('zoom-level');
    if (label) label.textContent = Math.round(z * 100) + '%';
}

function zoomBy(factor, anchorClientX, anchorClientY) {
    const container = document.getElementById('annotation-canvas-container');
    const prev = annotationState.canvasZoom;
    let next = Math.min(ZOOM_MAX, Math.max(ZOOM_MIN, prev * factor));
    if (next === prev) return;

    // Keep the point under the cursor stationary while zooming
    let relX = 0.5, relY = 0.5, anchored = false;
    if (container && anchorClientX != null) {
        const rect = container.getBoundingClientRect();
        const cx = anchorClientX - rect.left + container.scrollLeft;
        const cy = anchorClientY - rect.top + container.scrollTop;
        relX = cx / (annotationState.displayWidth * prev);
        relY = cy / (annotationState.displayHeight * prev);
        anchored = true;
    }

    annotationState.canvasZoom = next;
    applyZoom();

    if (container && anchored) {
        const newX = relX * annotationState.displayWidth * next;
        const newY = relY * annotationState.displayHeight * next;
        const rect = container.getBoundingClientRect();
        container.scrollLeft = newX - (anchorClientX - rect.left);
        container.scrollTop = newY - (anchorClientY - rect.top);
    }
}

function fitZoom() {
    const container = document.getElementById('annotation-canvas-container');
    if (!container || !annotationState.displayWidth) return;
    const availW = container.clientWidth - 8;
    const availH = container.clientHeight - 8;
    // Container may have no size yet (tab hidden) — default to 1:1 in that case
    if (availW <= 20 || availH <= 20) {
        annotationState.canvasZoom = 1;
    } else {
        // Fit the WHOLE page inside the viewport (contain): limit by both axes
        annotationState.canvasZoom = Math.min(ZOOM_MAX, Math.max(ZOOM_MIN,
            Math.min(availW / annotationState.displayWidth,
                     availH / annotationState.displayHeight)));
    }
    applyZoom();
}

function handleProjectChanged(event) {
    console.log('[Annotation] Project changed:', event.detail);
    if (event.detail && event.detail.project) {
        annotationState.currentProject = event.detail.project;
        loadProjectImages();
    } else {
        resetAnnotationTab();
    }
}

function loadCurrentProject() {
    let project = null;
    if (window.projectManager && window.projectManager.getCurrentProject) {
        project = window.projectManager.getCurrentProject();
    } else {
        const id = localStorage.getItem('currentProjectId');
        const name = localStorage.getItem('currentProjectName');
        if (id) project = { project_id: id, project_name: name };
    }
    
    if (project) {
        annotationState.currentProject = project;
        loadProjectImages();
    } else {
        showEmptyState('No project selected', 'Select a project from the Project Manager tab');
    }
}

function resetAnnotationTab() {
    annotationState.currentProject = null;
    annotationState.images = [];
    annotationState.currentIndex = -1;
    annotationState.polygons = [];
    annotationState.currentPolygon = [];
    annotationState.mousePreview = null;
    annotationState.vesselsSummary = {};
    renderVesselsPanel();
    updateImageCount(0);
    clearImageList();
    hideEditor();
    showEmptyState('No project selected', 'Select a project');
}

async function loadProjectImages() {
    if (!annotationState.currentProject) return;
    
    const projectId = annotationState.currentProject.project_id;
    console.log('[Annotation] Loading images for:', projectId);
    
    try {
        showLoading();
        
        const [imagesRes, masksRes, projectRes, vesselsRes] = await Promise.all([
            fetch(`/api/projects/${projectId}/images`).then(r => r.json()),
            fetch(`/api/projects/${projectId}/masks`).then(r => r.json()),
            fetch(`/api/projects/${projectId}`).then(r => r.json()),
            fetch(`/api/projects/${projectId}/vessels-summary`).then(r => r.json()).catch(() => ({}))
        ]);

        hideLoading();

        annotationState.vesselsSummary = (vesselsRes && vesselsRes.success) ? (vesselsRes.summary || {}) : {};
        
        if (!imagesRes.success) throw new Error('Failed to load images');
        
        const imageUrls = imagesRes.images || [];
        const maskUrls = masksRes.success ? (masksRes.masks || []) : [];
        
        // Get excluded images from project settings
        const excludedImages = new Set();
        if (projectRes.success && projectRes.project && projectRes.project.settings) {
            const excludedList = projectRes.project.settings.excluded_images || [];
            console.log('[Annotation] Excluded images:', excludedList);
            excludedList.forEach(img => excludedImages.add(img));
        }
        
        if (imageUrls.length === 0) {
            showEmptyState('No images found', 'Upload a PDF first');
            return;
        }
        
        const maskMap = {};
        maskUrls.forEach(url => {
            const filename = url.split('/').pop();
            const base = filename.replace(/_mask_layer\.png$/i, '');
            maskMap[base] = url;
        });
        
        // Filter out excluded images
        annotationState.images = imageUrls
            .filter(url => !excludedImages.has(url))
            .map(url => {
                const filename = url.split('/').pop();
                const base = filename.replace(/\.(jpg|jpeg|png|bmp)$/i, '');
                return {
                    imageUrl: url,
                    maskUrl: maskMap[base] || null,
                    filename: filename,
                    baseName: base,
                    hasMask: !!maskMap[base]
                };
            });
        
        console.log('[Annotation] Loaded', annotationState.images.length, 'images (after filtering excluded)');
        updateImageCount(annotationState.images.length);
        renderImageList();
        if (annotationState.images.length > 0) selectImage(0);
        
    } catch (error) {
        hideLoading();
        console.error('[Annotation] Error:', error);
        showEmptyState('Error loading images', error.message);
    }
}

function renderImageList() {
    const container = document.getElementById('annotation-image-list');
    if (!container) return;
    
    if (annotationState.images.length === 0) {
        container.innerHTML = '<div class="empty-list">No images</div>';
        return;
    }
    
    const html = annotationState.images.map((img, i) => {
        const icon = img.hasMask ? '✅' : '⚪';
        const active = i === annotationState.currentIndex ? 'active' : '';
        const vCount = annotationState.vesselsSummary[img.baseName];
        const badge = vCount
            ? `<span class="vessels-badge" title="${vCount} manually drawn vessel(s)">📐${vCount}</span>`
            : '';
        return `
            <div class="annotation-image-item ${active}" data-index="${i}">
                <span class="image-number">${icon}</span>
                <span class="image-name" title="${img.filename}">${img.filename}</span>
                ${badge}
            </div>
        `;
    }).join('');
    
    container.innerHTML = html;
    container.querySelectorAll('.annotation-image-item').forEach((item, i) => {
        item.addEventListener('click', () => selectImage(i));
    });
}

async function selectImage(index) {
    if (index < 0 || index >= annotationState.images.length) return;
    
    if (annotationState.isModified && annotationState.currentIndex >= 0) {
        await saveMask();
    }
    
    annotationState.currentIndex = index;
    const img = annotationState.images[index];
    
    document.querySelectorAll('.annotation-image-item').forEach((el, i) => {
        el.classList.toggle('active', i === index);
    });
    
    const label = document.getElementById('annotation-current-image');
    if (label) label.textContent = `${index + 1}/${annotationState.images.length} - ${img.filename}`;
    
    try {
        // Load original image
        const originalImg = await loadImage(img.imageUrl);
        annotationState.originalBackgroundImage = originalImg;
        annotationState.originalWidth = originalImg.width;
        annotationState.originalHeight = originalImg.height;
        
        // Calculate display size (maintain aspect ratio, max 1200x800)
        let displayWidth = originalImg.width;
        let displayHeight = originalImg.height;
        
        const aspectRatio = originalImg.width / originalImg.height;
        
        if (displayWidth > annotationState.maxDisplayWidth) {
            displayWidth = annotationState.maxDisplayWidth;
            displayHeight = displayWidth / aspectRatio;
        }
        
        if (displayHeight > annotationState.maxDisplayHeight) {
            displayHeight = annotationState.maxDisplayHeight;
            displayWidth = displayHeight * aspectRatio;
        }
        
        annotationState.displayWidth = Math.round(displayWidth);
        annotationState.displayHeight = Math.round(displayHeight);
        
        // Create resized version for display
        const resizedCanvas = document.createElement('canvas');
        resizedCanvas.width = annotationState.displayWidth;
        resizedCanvas.height = annotationState.displayHeight;
        const resizedCtx = resizedCanvas.getContext('2d');
        resizedCtx.drawImage(originalImg, 0, 0, annotationState.displayWidth, annotationState.displayHeight);
        
        // Convert to image for canvas use
        const resizedImg = await new Promise((resolve) => {
            const img = new Image();
            img.onload = () => resolve(img);
            img.src = resizedCanvas.toDataURL();
        });
        
        annotationState.backgroundImage = resizedImg;
        
        // Setup canvases with display size
        annotationState.canvas.width = annotationState.displayWidth;
        annotationState.canvas.height = annotationState.displayHeight;
        annotationState.maskCanvas.width = annotationState.displayWidth;
        annotationState.maskCanvas.height = annotationState.displayHeight;
        
        annotationState.maskCtx.clearRect(0, 0, annotationState.displayWidth, annotationState.displayHeight);
        
        // Load existing mask if present
        if (img.maskUrl) {
            const maskImg = await loadImage(img.maskUrl);
            // Resize mask to display size
            annotationState.maskCtx.drawImage(maskImg, 0, 0, annotationState.displayWidth, annotationState.displayHeight);
        }
        
        annotationState.isModified = false;
        await loadVessels(img.baseName);
        await loadScales(img.baseName);
        if (annotationState.colorize) computeColorized();
        redrawCanvas();
        showEditor();
        fitZoom();
        updateNavigationButtons();
        
        console.log(`[Annotation] Image loaded: ${annotationState.originalWidth}x${annotationState.originalHeight} -> ${annotationState.displayWidth}x${annotationState.displayHeight}`);
        
    } catch (error) {
        console.error('[Annotation] Error loading:', error);
        alert('Error: ' + error.message);
    }
}

function loadImage(url) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.crossOrigin = 'anonymous';
        img.onload = () => resolve(img);
        img.onerror = () => reject(new Error('Failed to load'));
        img.src = url;
    });
}

function navigateImage(dir) {
    const newIndex = annotationState.currentIndex + dir;
    if (newIndex >= 0 && newIndex < annotationState.images.length) {
        selectImage(newIndex);
    }
}

function updateNavigationButtons() {
    const prev = document.getElementById('annotation-prev-btn');
    const next = document.getElementById('annotation-next-btn');
    if (prev) prev.disabled = annotationState.currentIndex <= 0;
    if (next) next.disabled = annotationState.currentIndex >= annotationState.images.length - 1;
}

function redrawCanvas() {
    if (!annotationState.canvas || !annotationState.backgroundImage) return;
    const ctx = annotationState.ctx;
    const canvas = annotationState.canvas;
    
    // Direct rendering without requestAnimationFrame for immediate feedback
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(annotationState.backgroundImage, 0, 0);
    // Coloured connected-components view (except while actively painting, where
    // we show the live red strokes and recolour on mouse-up).
    if (annotationState.colorize && annotationState.colorizedCanvas && !annotationState.isDrawing) {
        ctx.globalAlpha = 0.75;
        ctx.drawImage(annotationState.colorizedCanvas, 0, 0);
    } else {
        ctx.globalAlpha = 0.5;
        ctx.drawImage(annotationState.maskCanvas, 0, 0);
    }
    ctx.globalAlpha = 1.0;
    drawPolygons(ctx);
    drawScaleLines(ctx);
    drawBrushCursor(ctx);
}

// Toggle the "political-map" colouring of separate masks
function toggleColorize() {
    annotationState.colorize = !annotationState.colorize;
    const btn = document.getElementById('colorize-toggle');
    if (btn) btn.classList.toggle('active', annotationState.colorize);
    if (annotationState.colorize) computeColorized();
    redrawCanvas();
}

// Label connected components of the current mask and paint each a distinct
// colour into an offscreen canvas. Two touching (fused) masks share one colour,
// which is exactly the anomaly the operator wants to spot.
function computeColorized() {
    const w = annotationState.displayWidth, h = annotationState.displayHeight;
    if (!w || !h) { annotationState.colorizedCanvas = null; return; }

    let src;
    try {
        src = annotationState.maskCtx.getImageData(0, 0, w, h);
    } catch (e) {
        console.warn('[Annotation] colorize failed:', e);
        return;
    }
    const data = src.data;
    const n = w * h;
    const fg = new Uint8Array(n);
    for (let i = 0; i < n; i++) fg[i] = data[i * 4 + 3] > 16 ? 1 : 0;

    const labels = new Int32Array(n);
    const stack = new Int32Array(n);
    const colors = [null];
    let label = 0;

    for (let s = 0; s < n; s++) {
        if (!fg[s] || labels[s]) continue;
        label++;
        const hue = (label * 137.508) % 360; // golden-angle → well-spread hues
        colors.push(hslToRgb(hue / 360, 0.7, 0.5));
        let sp = 0;
        stack[sp++] = s;
        labels[s] = label;
        while (sp > 0) {
            const p = stack[--sp];
            const x = p % w;
            if (x > 0)      { const q = p - 1; if (fg[q] && !labels[q]) { labels[q] = label; stack[sp++] = q; } }
            if (x < w - 1)  { const q = p + 1; if (fg[q] && !labels[q]) { labels[q] = label; stack[sp++] = q; } }
            if (p >= w)     { const q = p - w; if (fg[q] && !labels[q]) { labels[q] = label; stack[sp++] = q; } }
            if (p < n - w)  { const q = p + w; if (fg[q] && !labels[q]) { labels[q] = label; stack[sp++] = q; } }
        }
    }

    const out = new ImageData(w, h);
    const od = out.data;
    for (let i = 0; i < n; i++) {
        const l = labels[i];
        if (l) {
            const c = colors[l];
            od[i * 4] = c[0]; od[i * 4 + 1] = c[1]; od[i * 4 + 2] = c[2]; od[i * 4 + 3] = 255;
        }
    }

    let cc = annotationState.colorizedCanvas;
    if (!cc || cc.width !== w || cc.height !== h) {
        cc = document.createElement('canvas');
        cc.width = w; cc.height = h;
        annotationState.colorizedCanvas = cc;
    }
    cc.getContext('2d').putImageData(out, 0, 0);
}

function hslToRgb(h, s, l) {
    let r, g, b;
    if (s === 0) {
        r = g = b = l;
    } else {
        const hue2rgb = (p, q, t) => {
            if (t < 0) t += 1;
            if (t > 1) t -= 1;
            if (t < 1 / 6) return p + (q - p) * 6 * t;
            if (t < 1 / 2) return q;
            if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6;
            return p;
        };
        const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
        const p = 2 * l - q;
        r = hue2rgb(p, q, h + 1 / 3);
        g = hue2rgb(p, q, h);
        b = hue2rgb(p, q, h - 1 / 3);
    }
    return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)];
}

// Show the brush/eraser footprint as a ring so the user sees what is affected
function drawBrushCursor(ctx) {
    const tool = annotationState.currentTool;
    const c = annotationState.brushCursor;
    if (!c || (tool !== 'brush' && tool !== 'eraser')) return;
    const r = annotationState.brushSize;

    ctx.save();
    ctx.beginPath();
    ctx.arc(c.x, c.y, r, 0, Math.PI * 2);
    if (tool === 'eraser') {
        // Eraser: hollow ring with a subtle white fill = "this will be removed"
        ctx.fillStyle = 'rgba(255,255,255,0.35)';
        ctx.fill();
        ctx.lineWidth = 1.5;
        ctx.setLineDash([6, 4]);
        ctx.strokeStyle = '#1e293b';
    } else {
        // Brush: red footprint matching the painted colour
        ctx.fillStyle = 'rgba(255,0,0,0.25)';
        ctx.fill();
        ctx.lineWidth = 1.5;
        ctx.strokeStyle = '#dc2626';
    }
    ctx.stroke();
    ctx.restore();
}

// Draw committed polygons (green) + the in-progress one (orange)
function drawPolygons(ctx) {
    // Committed vessel polygons
    (annotationState.polygons || []).forEach((poly, idx) => {
        if (poly.length < 2) return;
        ctx.beginPath();
        poly.forEach(([ox, oy], k) => {
            const [px, py] = originalToDisplay(ox, oy);
            if (k === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
        });
        ctx.closePath();
        ctx.fillStyle = 'rgba(22,163,74,0.15)';
        ctx.strokeStyle = '#16a34a';
        ctx.lineWidth = 2;
        ctx.fill();
        ctx.stroke();
        // number label at first vertex
        const [lx, ly] = originalToDisplay(poly[0][0], poly[0][1]);
        ctx.fillStyle = '#16a34a';
        ctx.font = 'bold 14px sans-serif';
        ctx.fillText(`#${idx + 1}`, lx + 3, ly - 4);
    });

    // In-progress polygon
    const cur = annotationState.currentPolygon || [];
    if (cur.length > 0) {
        ctx.beginPath();
        cur.forEach(([ox, oy], k) => {
            const [px, py] = originalToDisplay(ox, oy);
            if (k === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
        });
        if (annotationState.mousePreview) {
            ctx.lineTo(annotationState.mousePreview.x, annotationState.mousePreview.y);
        }
        ctx.strokeStyle = '#ea580c';
        ctx.lineWidth = 2;
        ctx.stroke();
        // vertices
        cur.forEach(([ox, oy], k) => {
            const [px, py] = originalToDisplay(ox, oy);
            ctx.beginPath();
            ctx.arc(px, py, k === 0 ? 6 : 4, 0, Math.PI * 2);
            ctx.fillStyle = k === 0 ? '#16a34a' : '#ea580c';
            ctx.fill();
        });
    }
}

function addPolygonVertex(e) {
    if (!annotationState.backgroundImage) return;
    const { x, y } = eventToDisplayXY(e);
    const cur = annotationState.currentPolygon;

    // Snap-close if clicking near the first vertex
    if (cur.length >= 3) {
        const [fx, fy] = originalToDisplay(cur[0][0], cur[0][1]);
        if (Math.hypot(x - fx, y - fy) <= POLYGON_CLOSE_THRESHOLD) {
            finishPolygon();
            return;
        }
    }
    // Ignore near-duplicate clicks (e.g. the two clicks of a double-click)
    if (cur.length > 0) {
        const [lx, ly] = originalToDisplay(cur[cur.length - 1][0], cur[cur.length - 1][1]);
        if (Math.hypot(x - lx, y - ly) < 5) return;
    }
    cur.push(displayToOriginal(x, y));
    redrawCanvas();
}

function finishPolygon() {
    const cur = annotationState.currentPolygon;
    if (cur.length >= 3) {
        annotationState.polygons.push(cur);
        annotationState.currentPolygon = [];
        annotationState.mousePreview = null;
        updateVesselsSummaryForCurrent();
        renderVesselsPanel();
        renderImageList();
        redrawCanvas();
        persistVessels();
    } else {
        cancelPolygon();
    }
}

function cancelPolygon() {
    annotationState.currentPolygon = [];
    annotationState.mousePreview = null;
    redrawCanvas();
}

function onCanvasDblClick(e) {
    if (annotationState.currentTool !== 'polygon') return;
    e.preventDefault();
    finishPolygon();
}

function onPolygonKeyDown(e) {
    if (annotationState.currentTool === 'scale') {
        if (e.key === 'Escape') { e.preventDefault(); cancelScaleDraft(); redrawCanvas(); }
        return;
    }
    if (annotationState.currentTool !== 'polygon') return;
    if (e.key === 'Enter') { e.preventDefault(); finishPolygon(); }
    else if (e.key === 'Escape') { e.preventDefault(); cancelPolygon(); }
    else if (e.key === 'Backspace' && annotationState.currentPolygon.length > 0) {
        e.preventDefault();
        annotationState.currentPolygon.pop();
        redrawCanvas();
    }
}

function startDrawing(e) {
    if (annotationState.currentTool === 'polygon') {
        addPolygonVertex(e);
        return;
    }
    if (annotationState.currentTool === 'scale') {
        handleScaleMouseDown(e);
        return;
    }
    annotationState.isDrawing = true;
    draw(e);
}

function stopDrawing(e) {
    if (annotationState.currentTool === 'scale' && annotationState.scaleStep === 'zone' && annotationState.isDrawing) {
        handleScaleZoneEnd(e);
        return;
    }
    if (!annotationState.isDrawing) return;
    annotationState.isDrawing = false;
    if (annotationState.colorize) {
        computeColorized();
        redrawCanvas();
    }
}

function draw(e) {
    if (annotationState.currentTool === 'polygon') {
        // Rubber-band preview to the cursor while building a polygon
        if (annotationState.currentPolygon.length > 0) {
            annotationState.mousePreview = eventToDisplayXY(e);
            redrawCanvas();
        }
        return;
    }

    if (annotationState.currentTool === 'scale') {
        annotationState.scaleCursorPos = eventToDisplayXY(e);
        redrawCanvas();
        return;
    }

    // Brush / eraser: always track the cursor so we can show the size ring
    const { x, y } = eventToDisplayXY(e);
    annotationState.brushCursor = { x, y };

    if (annotationState.isDrawing) {
        const ctx = annotationState.maskCtx;
        annotationState.isModified = true;

        if (annotationState.currentTool === 'eraser') {
            ctx.globalCompositeOperation = 'destination-out';
            ctx.fillStyle = 'rgba(0,0,0,1)';
        } else {
            ctx.globalCompositeOperation = 'source-over';
            ctx.fillStyle = 'rgba(255, 0, 0, 1)';
        }

        ctx.beginPath();
        ctx.arc(x, y, annotationState.brushSize, 0, Math.PI * 2);
        ctx.fill();
    }

    // Immediate redraw for responsiveness + cursor ring
    redrawCanvas();
}

function onCanvasMouseLeave() {
    const wasDrawing = annotationState.isDrawing;
    annotationState.isDrawing = false;
    annotationState.brushCursor = null;
    annotationState.scaleCursorPos = null;
    if (wasDrawing && annotationState.colorize) computeColorized();
    redrawCanvas();
}

function selectTool(tool) {
    // Abandon any half-drawn polygon when switching tools
    if (annotationState.currentTool === 'polygon' && tool !== 'polygon') {
        cancelPolygon();
    }
    // Cancel in-progress scale when switching away
    if (annotationState.currentTool === 'scale' && tool !== 'scale') {
        cancelScaleDraft();
    }
    annotationState.currentTool = tool;
    document.querySelectorAll('.btn-tool').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.tool === tool);
    });
    const canvas = annotationState.canvas;
    if (canvas) canvas.style.cursor = 'crosshair';
}

// ---- Manually drawn vessels (polygons) ----------------------------------

async function loadVessels(baseName) {
    annotationState.polygons = [];
    annotationState.currentPolygon = [];
    annotationState.mousePreview = null;
    if (!annotationState.currentProject) { renderVesselsPanel(); return; }
    const projectId = annotationState.currentProject.project_id;
    try {
        const res = await fetch(`/api/projects/${projectId}/vessels/${encodeURIComponent(baseName)}`);
        const data = await res.json();
        if (data.success) annotationState.polygons = data.polygons || [];
    } catch (e) {
        console.warn('[Annotation] Could not load vessels:', e);
    }
    renderVesselsPanel();
}

function renderVesselsPanel() {
    const panel = document.getElementById('vessels-panel');
    const list = document.getElementById('vessels-list');
    const count = document.getElementById('vessels-count');
    if (!panel || !list) return;

    const polys = annotationState.polygons || [];
    if (count) count.textContent = polys.length;

    if (polys.length === 0) {
        panel.style.display = 'none';
        list.innerHTML = '';
        return;
    }
    panel.style.display = 'block';
    list.innerHTML = polys.map((p, i) => `
        <div class="vessel-item" data-index="${i}">
            <span class="vessel-item-label">#${i + 1} — polygon (${p.length} points)</span>
            <button class="vessel-delete" data-index="${i}">🗑️ Delete</button>
        </div>
    `).join('');

    list.querySelectorAll('.vessel-delete').forEach(btn => {
        btn.addEventListener('click', () => deleteVessel(parseInt(btn.dataset.index)));
    });
}

function deleteVessel(index) {
    annotationState.polygons.splice(index, 1);
    updateVesselsSummaryForCurrent();
    renderVesselsPanel();
    renderImageList();
    redrawCanvas();
    persistVessels();
}

function updateVesselsSummaryForCurrent() {
    const img = annotationState.images[annotationState.currentIndex];
    if (!img) return;
    const n = annotationState.polygons.length;
    if (n > 0) annotationState.vesselsSummary[img.baseName] = n;
    else delete annotationState.vesselsSummary[img.baseName];
}

async function persistVessels() {
    if (!annotationState.currentProject) return;
    const img = annotationState.images[annotationState.currentIndex];
    if (!img) return;
    const projectId = annotationState.currentProject.project_id;
    try {
        await fetch(`/api/projects/${projectId}/vessels/${encodeURIComponent(img.baseName)}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ polygons: annotationState.polygons })
        });
    } catch (e) {
        console.error('[Annotation] Failed to persist vessels:', e);
    }
}

// ---- Scale calibration --------------------------------------------------

function computeScaleRatio(s) {
    const dx = s.p2[0] - s.p1[0];
    const dy = s.p2[1] - s.p1[1];
    const distPx = Math.hypot(dx, dy);
    return s.real_cm > 0 ? distPx / s.real_cm : null;
}

async function loadScales(baseName) {
    annotationState.scales = [];
    annotationState.scaleDraftP1 = null;
    annotationState.scaleDraftP2 = null;
    annotationState.scaleStep = null;
    if (!annotationState.currentProject) { renderScalesPanel(); return; }
    const projectId = annotationState.currentProject.project_id;
    try {
        const res = await fetch(`/api/projects/${projectId}/scale/${encodeURIComponent(baseName)}`);
        const data = await res.json();
        if (data.success) annotationState.scales = data.scales || [];
    } catch (e) {
        console.warn('[Annotation] Could not load scales:', e);
    }
    renderScalesPanel();
}

function renderScalesPanel() {
    const panel = document.getElementById('scales-panel');
    const list = document.getElementById('scales-list');
    const badge = document.getElementById('scales-count');
    if (!panel || !list) return;

    const scales = annotationState.scales || [];
    if (badge) badge.textContent = scales.length;

    if (scales.length === 0) {
        panel.style.display = 'none';
        list.innerHTML = '';
        return;
    }
    panel.style.display = 'block';
    list.innerHTML = scales.map((s, i) => {
        const ratio = computeScaleRatio(s);
        const ratioStr = ratio ? ratio.toFixed(1) + ' px/cm' : '—';
        const zoneStr = s.zone ? '✓ zone set' : 'global';
        return `
            <div class="vessel-item scale-line-item">
                <span class="vessel-item-label">#${i + 1} · ${s.real_cm}cm → <code>${ratioStr}</code> <em>(${zoneStr})</em></span>
                <div class="scale-item-actions">
                    <button class="btn-scale-zone" data-idx="${i}" title="Draw a zone rectangle on canvas to limit this scale to a page region">📍</button>
                    <button class="vessel-delete btn-scale-del" data-idx="${i}">🗑️</button>
                </div>
            </div>
        `;
    }).join('');

    list.querySelectorAll('.btn-scale-zone').forEach(btn => {
        btn.addEventListener('click', () => startScaleZone(parseInt(btn.dataset.idx)));
    });
    list.querySelectorAll('.btn-scale-del').forEach(btn => {
        btn.addEventListener('click', () => deleteScale(parseInt(btn.dataset.idx)));
    });
}

function startScaleZone(idx) {
    annotationState.scaleZoneTargetIdx = idx;
    annotationState.scaleStep = 'zone';
    annotationState.scaleZoneDragStart = null;
    selectTool('scale');
    if (window.PyPotteryUtils) window.PyPotteryUtils.showToast('Drag a rectangle on the canvas to define the zone for this scale', 'info');
}

function deleteScale(idx) {
    annotationState.scales.splice(idx, 1);
    renderScalesPanel();
    redrawCanvas();
    persistScales();
}

async function persistScales() {
    if (!annotationState.currentProject) return;
    const img = annotationState.images[annotationState.currentIndex];
    if (!img) return;
    const projectId = annotationState.currentProject.project_id;
    try {
        await fetch(`/api/projects/${projectId}/scale/${encodeURIComponent(img.baseName)}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ scales: annotationState.scales })
        });
    } catch (e) {
        console.error('[Annotation] Failed to persist scales:', e);
    }
}

// Mouse handlers for the scale tool

function handleScaleMouseDown(e) {
    const dxy = eventToDisplayXY(e);
    const [ox, oy] = displayToOriginal(dxy.x, dxy.y);

    if (annotationState.scaleStep === 'zone') {
        annotationState.scaleZoneDragStart = { x: ox, y: oy };
        annotationState.isDrawing = true;
        return;
    }

    // Ruler mode: first click = p1, second click = p2 → show popup
    if (annotationState.scaleDraftP1 === null) {
        annotationState.scaleDraftP1 = [ox, oy];
    } else {
        const p1 = annotationState.scaleDraftP1;
        const distPx = Math.hypot(ox - p1[0], oy - p1[1]);
        if (distPx < 5) return; // too close, ignore
        annotationState.scaleDraftP2 = [ox, oy];
        showScalePopup();
    }
    redrawCanvas();
}

function handleScaleZoneEnd(e) {
    const dxy = eventToDisplayXY(e);
    const [ox, oy] = displayToOriginal(dxy.x, dxy.y);
    const start = annotationState.scaleZoneDragStart;
    annotationState.isDrawing = false;
    annotationState.scaleZoneDragStart = null;

    if (!start || Math.abs(ox - start.x) < 5 || Math.abs(oy - start.y) < 5) {
        annotationState.scaleStep = null;
        annotationState.scaleZoneTargetIdx = null;
        redrawCanvas();
        return;
    }

    const idx = annotationState.scaleZoneTargetIdx;
    if (idx !== null && annotationState.scales[idx]) {
        annotationState.scales[idx].zone = [
            Math.min(start.x, ox), Math.min(start.y, oy),
            Math.max(start.x, ox), Math.max(start.y, oy)
        ];
        persistScales();
        renderScalesPanel();
    }
    annotationState.scaleStep = null;
    annotationState.scaleZoneTargetIdx = null;
    redrawCanvas();
}

function showScalePopup() {
    const popup = document.getElementById('scale-input-popup');
    if (!popup) return;
    popup.style.display = 'flex';
    const input = document.getElementById('scale-cm-input');
    if (input) { input.value = ''; input.focus(); }
}

function hideScalePopup() {
    const popup = document.getElementById('scale-input-popup');
    if (popup) popup.style.display = 'none';
}

function cancelScaleDraft() {
    annotationState.scaleDraftP1 = null;
    annotationState.scaleDraftP2 = null;
    annotationState.scaleStep = null;
    annotationState.scaleZoneTargetIdx = null;
    annotationState.scaleZoneDragStart = null;
    hideScalePopup();
}

function confirmScaleInput() {
    const input = document.getElementById('scale-cm-input');
    const val = parseFloat(input?.value);
    if (!val || val <= 0) { input?.focus(); return; }
    const p1 = annotationState.scaleDraftP1;
    const p2 = annotationState.scaleDraftP2;
    if (!p1 || !p2) return;
    annotationState.scales.push({ p1, p2, real_cm: val, zone: null });
    persistScales();
    renderScalesPanel();
    cancelScaleDraft();
    redrawCanvas();
}

// Draw committed scale lines + zones + in-progress drafts
function drawScaleLines(ctx) {
    const scales = annotationState.scales || [];

    scales.forEach((s) => {
        const [dx1, dy1] = originalToDisplay(s.p1[0], s.p1[1]);
        const [dx2, dy2] = originalToDisplay(s.p2[0], s.p2[1]);
        const len = Math.hypot(dx2 - dx1, dy2 - dy1);
        if (len < 1) return;

        ctx.save();
        ctx.strokeStyle = '#f59e0b';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(dx1, dy1);
        ctx.lineTo(dx2, dy2);
        ctx.stroke();

        // Endpoint ticks (perpendicular to line)
        const nx = (dy2 - dy1) / len, ny = -(dx2 - dx1) / len;
        const tk = 6;
        ctx.beginPath();
        ctx.moveTo(dx1 - nx * tk, dy1 - ny * tk); ctx.lineTo(dx1 + nx * tk, dy1 + ny * tk);
        ctx.moveTo(dx2 - nx * tk, dy2 - ny * tk); ctx.lineTo(dx2 + nx * tk, dy2 + ny * tk);
        ctx.stroke();

        // Label
        const ratio = computeScaleRatio(s);
        const label = ratio ? `${s.real_cm}cm · ${ratio.toFixed(1)} px/cm` : `${s.real_cm}cm`;
        const mx = (dx1 + dx2) / 2, my = (dy1 + dy2) / 2;
        ctx.fillStyle = '#f59e0b';
        ctx.font = 'bold 11px sans-serif';
        ctx.fillText(label, mx + 4, my - 4);

        // Zone rectangle
        if (s.zone) {
            const [zx1, zy1] = originalToDisplay(s.zone[0], s.zone[1]);
            const [zx2, zy2] = originalToDisplay(s.zone[2], s.zone[3]);
            ctx.setLineDash([5, 3]);
            ctx.globalAlpha = 0.35;
            ctx.fillStyle = '#fef3c7';
            ctx.fillRect(zx1, zy1, zx2 - zx1, zy2 - zy1);
            ctx.globalAlpha = 1;
            ctx.strokeRect(zx1, zy1, zx2 - zx1, zy2 - zy1);
            ctx.setLineDash([]);
        }
        ctx.restore();
    });

    // Draft ruler: p1 placed, waiting for p2
    if (annotationState.scaleDraftP1 && !annotationState.scaleDraftP2 && annotationState.scaleCursorPos) {
        const [dx1, dy1] = originalToDisplay(annotationState.scaleDraftP1[0], annotationState.scaleDraftP1[1]);
        const { x: cx, y: cy } = annotationState.scaleCursorPos;
        ctx.save();
        ctx.setLineDash([5, 3]);
        ctx.strokeStyle = '#f59e0b';
        ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.moveTo(dx1, dy1); ctx.lineTo(cx, cy); ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = '#f59e0b';
        ctx.beginPath(); ctx.arc(dx1, dy1, 4, 0, Math.PI * 2); ctx.fill();
        ctx.restore();
    }

    // Draft zone rect
    if (annotationState.scaleStep === 'zone' && annotationState.isDrawing &&
        annotationState.scaleZoneDragStart && annotationState.scaleCursorPos) {
        const [zx1, zy1] = originalToDisplay(annotationState.scaleZoneDragStart.x, annotationState.scaleZoneDragStart.y);
        const { x: cx, y: cy } = annotationState.scaleCursorPos;
        ctx.save();
        ctx.setLineDash([5, 3]);
        ctx.strokeStyle = '#f59e0b';
        ctx.lineWidth = 1.5;
        ctx.globalAlpha = 0.25;
        ctx.fillStyle = '#fef3c7';
        ctx.fillRect(zx1, zy1, cx - zx1, cy - zy1);
        ctx.globalAlpha = 1;
        ctx.strokeRect(zx1, zy1, cx - zx1, cy - zy1);
        ctx.setLineDash([]);
        ctx.restore();
    }
}

function clearMask() {
    if (!confirm('Clear all annotations?')) return;
    const canvas = annotationState.maskCanvas;
    annotationState.maskCtx.clearRect(0, 0, canvas.width, canvas.height);
    annotationState.isModified = true;
    if (annotationState.colorize) computeColorized();
    redrawCanvas();
}

async function saveMask() {
    if (!annotationState.currentProject || annotationState.currentIndex < 0) return;
    if (!annotationState.isModified) return;
    
    const img = annotationState.images[annotationState.currentIndex];
    const projectId = annotationState.currentProject.project_id;
    
    try {
        // Create full-resolution mask canvas
        const fullResMaskCanvas = document.createElement('canvas');
        fullResMaskCanvas.width = annotationState.originalWidth;
        fullResMaskCanvas.height = annotationState.originalHeight;
        const fullResMaskCtx = fullResMaskCanvas.getContext('2d');
        
        // Scale up the mask to original resolution
        fullResMaskCtx.drawImage(
            annotationState.maskCanvas, 
            0, 0, annotationState.displayWidth, annotationState.displayHeight,
            0, 0, annotationState.originalWidth, annotationState.originalHeight
        );
        
        const blob = await new Promise(resolve => {
            fullResMaskCanvas.toBlob(resolve, 'image/png');
        });
        
        const formData = new FormData();
        formData.append('mask', blob, `${img.baseName}_mask_layer.png`);
        
        const res = await fetch(`/api/projects/${projectId}/masks/save`, {
            method: 'POST',
            body: formData
        });
        
        const result = await res.json();
        if (result.success) {
            annotationState.isModified = false;
            img.hasMask = true;
            // Update maskUrl so it can be reloaded
            img.maskUrl = result.mask_url || `/api/projects/${projectId}/mask/${img.baseName}_mask_layer.png`;
            renderImageList();
            // Force redraw of the main canvas to show the newly saved mask
            redrawCanvas();
            if (window.PyPotteryUtils) {
                window.PyPotteryUtils.showToast('Mask saved!', 'success');
            }
            console.log(`[Annotation] Mask saved at original resolution: ${annotationState.originalWidth}x${annotationState.originalHeight}`);
        } else {
            throw new Error(result.error || 'Save failed');
        }
    } catch (error) {
        console.error('[Annotation] Save error:', error);
        alert('Error: ' + error.message);
    }
}

async function extractCards() {
    if (!annotationState.currentProject) return;
    if (annotationState.isModified) await saveMask();

    // Show custom confirmation dialog instead of native confirm()
    const confirmed = await showExtractConfirmDialog();
    if (!confirmed) return;

    const projectId = annotationState.currentProject.project_id;
    const btn = document.getElementById('extract-masks-btn');
    
    try {
        if (btn) {
            btn.disabled = true;
            btn.textContent = '⏳ Extracting...';
        }
        
        // Use progress tracking with status and progress bar
        await window.PyPotteryUtils.executeWithProgress(
            'extract_masks',
            async () => {
                const res = await fetch(`/api/projects/${projectId}/masks/extract`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({})
                });
                
                const result = await res.json();
                
                if (!result.success) {
                    throw new Error(result.error || 'Extract failed');
                }
                
                return result;
            },
            'annotation-status',
            'extraction-progress-bar'
        );
        
        if (btn) {
            btn.disabled = false;
            btn.textContent = '📤 Extract Cards';
        }
        
        window.PyPotteryUtils.showStatus('annotation-status', 'Cards extracted successfully!', 'success');
        window.PyPotteryUtils.showToast('Cards extracted successfully!', 'success');
        
        if (window.projectManager && window.projectManager.loadProjects) {
            window.projectManager.loadProjects();
        }
        
    } catch (error) {
        console.error('[Annotation] Extract error:', error);
        window.PyPotteryUtils.showStatus('annotation-status', 'Error: ' + error.message, 'error');
        window.PyPotteryUtils.showToast('Error: ' + error.message, 'error');
        if (btn) {
            btn.disabled = false;
            btn.textContent = '📤 Extract Cards';
        }
    }
}

function updateImageCount(count) {
    const el = document.getElementById('annotation-image-count');
    if (el) el.textContent = count;
}

function clearImageList() {
    const el = document.getElementById('annotation-image-list');
    if (el) el.innerHTML = '<div class="empty-list">No images</div>';
}

function showEditor() {
    const editor = document.getElementById('annotation-editor');
    const empty = document.getElementById('annotation-empty-msg');
    if (editor) editor.style.display = 'block';
    if (empty) empty.style.display = 'none';
}

function hideEditor() {
    const editor = document.getElementById('annotation-editor');
    const empty = document.getElementById('annotation-empty-msg');
    if (editor) editor.style.display = 'none';
    if (empty) empty.style.display = 'flex';
}

function showEmptyState(title, msg) {
    const el = document.getElementById('annotation-empty-msg');
    if (el) {
        el.innerHTML = `<h3>${title}</h3><p>${msg}</p>`;
        el.style.display = 'flex';
    }
    hideEditor();
}

function showLoading() {
    if (window.PyPotteryUtils) window.PyPotteryUtils.showLoading('Loading...');
}

function hideLoading() {
    if (window.PyPotteryUtils) window.PyPotteryUtils.hideLoading();
}

// Export with unique name to avoid conflicts with model-tab
window.loadAnnotationProjectImages = loadProjectImages;

// Custom dialog for extract confirmation — returns a Promise<boolean>
function showExtractConfirmDialog() {
    return new Promise((resolve) => {
        const dialog = document.getElementById('extract-confirm-dialog');
        const okBtn = document.getElementById('extract-confirm-ok');
        const cancelBtn = document.getElementById('extract-confirm-cancel');

        if (!dialog || !okBtn || !cancelBtn) {
            resolve(true);
            return;
        }

        dialog.style.display = 'flex';
        // Trigger reflow for CSS transition
        void dialog.offsetWidth;
        dialog.classList.add('show');

        let isCleaningUp = false;
        function cleanup(result) {
            if (isCleaningUp) return;
            isCleaningUp = true;
            dialog.classList.remove('show');
            okBtn.removeEventListener('click', onOk);
            cancelBtn.removeEventListener('click', onCancel);
            dialog.removeEventListener('click', onBackdrop);
            setTimeout(() => {
                dialog.style.display = 'none';
                resolve(result);
            }, 250);
        }

        function onOk(e) { e.stopPropagation(); cleanup(true); }
        function onCancel(e) { e.stopPropagation(); cleanup(false); }
        function onBackdrop(e) { if (e.target === dialog) cleanup(false); }

        okBtn.addEventListener('click', onOk);
        cancelBtn.addEventListener('click', onCancel);
        dialog.addEventListener('click', onBackdrop);
    });
}

console.log('[Annotation] Ready');