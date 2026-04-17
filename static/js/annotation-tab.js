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
    maxDisplayHeight: 800   // Max height for display
};

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    console.log('[Annotation] Initializing...');
    
    initializeCanvas();
    initializeToolButtons();
    initializeNavigationButtons();
    initializeSaveButton();
    initializeExtractButton();
    
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
    canvas.addEventListener('mouseout', stopDrawing);
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
        
        const [imagesRes, masksRes, projectRes] = await Promise.all([
            fetch(`/api/projects/${projectId}/images`).then(r => r.json()),
            fetch(`/api/projects/${projectId}/masks`).then(r => r.json()),
            fetch(`/api/projects/${projectId}`).then(r => r.json())
        ]);
        
        hideLoading();
        
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
        return `
            <div class="annotation-image-item ${active}" data-index="${i}">
                <span class="image-number">${icon}</span>
                <span class="image-name" title="${img.filename}">${img.filename}</span>
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
        redrawCanvas();
        showEditor();
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
}

function startDrawing(e) {
    annotationState.isDrawing = true;
    draw(e);
}

function stopDrawing() {
    annotationState.isDrawing = false;
}

function draw(e) {
    if (!annotationState.isDrawing) return;
    const canvas = annotationState.canvas;
    const ctx = annotationState.maskCtx;
    const rect = canvas.getBoundingClientRect();
    const x = (e.clientX - rect.left) * (canvas.width / rect.width);
    const y = (e.clientY - rect.top) * (canvas.height / rect.height);
    
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
    
    // Immediate redraw for better responsiveness
    redrawCanvas();
}

function selectTool(tool) {
    annotationState.currentTool = tool;
    document.querySelectorAll('.btn-tool').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.tool === tool);
    });
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