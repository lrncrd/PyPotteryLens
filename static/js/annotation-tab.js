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
    brushCursor: null        // {x, y} buffer coords for brush/eraser size ring
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
    document.querySelectorAll('.btn-tool').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const tool = e.target.dataset.tool;
            if (tool === 'clear') clearMask();
            else selectTool(tool);
        });
    });
    
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
    ctx.globalAlpha = 0.5;
    ctx.drawImage(annotationState.maskCanvas, 0, 0);
    ctx.globalAlpha = 1.0;
    drawPolygons(ctx);
    drawBrushCursor(ctx);
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
    annotationState.isDrawing = true;
    draw(e);
}

function stopDrawing() {
    annotationState.isDrawing = false;
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
    annotationState.isDrawing = false;
    if (annotationState.brushCursor) {
        annotationState.brushCursor = null;
        redrawCanvas();
    }
}

function selectTool(tool) {
    // Abandon any half-drawn polygon when switching tools
    if (annotationState.currentTool === 'polygon' && tool !== 'polygon') {
        cancelPolygon();
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

function clearMask() {
    if (!confirm('Clear all annotations?')) return;
    const canvas = annotationState.maskCanvas;
    annotationState.maskCtx.clearRect(0, 0, canvas.width, canvas.height);
    annotationState.isModified = true;
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

        function cleanup(result) {
            dialog.style.display = 'none';
            okBtn.removeEventListener('click', onOk);
            cancelBtn.removeEventListener('click', onCancel);
            resolve(result);
        }

        function onOk() { cleanup(true); }
        function onCancel() { cleanup(false); }

        okBtn.addEventListener('click', onOk);
        cancelBtn.addEventListener('click', onCancel);
        dialog.style.display = 'flex';
    });
}

console.log('[Annotation] Ready');