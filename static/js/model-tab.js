// Model Application Tab JavaScript - Project-aware version

// State
let modelState = {
    currentProject: null,
    images: [],
    excludedImages: new Set(),
    // Drag selection state
    isDragging: false,
    dragStartX: 0,
    dragStartY: 0,
    selectionBox: null
};

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    console.log('Model tab initialized (project-aware)');

    if (!window.PyPotteryUtils) {
        console.error('PyPotteryUtils not loaded!');
        return;
    }

    // Confidence slider
    const confidenceSlider = document.getElementById('confidence');
    const confidenceValue = document.getElementById('confidence-value');

    if (confidenceSlider && confidenceValue) {
        confidenceSlider.addEventListener('input', (e) => {
            confidenceValue.textContent = parseFloat(e.target.value).toFixed(2);
        });
        confidenceValue.textContent = parseFloat(confidenceSlider.value).toFixed(2);
    }

    // Apply model button
    const applyBtn = document.getElementById('apply-model-btn');
    if (applyBtn) {
        applyBtn.addEventListener('click', handleApplyModel);
    }

    // Selection buttons
    const selectAllBtn = document.getElementById('select-all-btn');
    const deselectAllBtn = document.getElementById('deselect-all-btn');

    if (selectAllBtn) {
        selectAllBtn.addEventListener('click', handleSelectAll);
    }
    if (deselectAllBtn) {
        deselectAllBtn.addEventListener('click', handleDeselectAll);
    }

    // Setup drag selection
    setupDragSelection();

    // Image modal
    setupImageModal();

    // Load current project
    loadCurrentProject();

    // Listen for project changes
    window.addEventListener('projectChanged', (e) => {
        console.log('[Model] Project changed event:', e.detail);
        const project = e.detail && e.detail.project ? e.detail.project : null;
        modelState.currentProject = project;
        loadProjectImages();
    });
});

// Export for use by main.js with unique name
window.loadModelProjectImages = loadProjectImages;

function loadCurrentProject() {
    console.log('[Model] Loading current project...');
    // Try to get current project from project manager
    if (window.projectManager && window.projectManager.getCurrentProject) {
        modelState.currentProject = window.projectManager.getCurrentProject();
        console.log('[Model] Project from projectManager:', modelState.currentProject);
    } else {
        // Fallback to localStorage
        const pid = localStorage.getItem('currentProjectId');
        const pname = localStorage.getItem('currentProjectName');
        console.log('[Model] localStorage projectId:', pid);
        if (pid) {
            modelState.currentProject = { project_id: pid, project_name: pname || 'Unnamed' };
        }
    }
    
    if (modelState.currentProject) {
        console.log('[Model] Loading images for project:', modelState.currentProject.project_id);
        loadProjectImages();
    } else {
        console.log('[Model] No current project found');
    }
}

async function loadProjectImages() {
    console.log('[Model] loadProjectImages called');
    console.log('[Model] Current project:', modelState.currentProject);
    
    const emptyMsg = document.getElementById('model-empty-msg');
    const gallery = document.getElementById('model-gallery');
    
    if (!modelState.currentProject || !modelState.currentProject.project_id) {
        console.log('[Model] No project - showing empty state');
        if (emptyMsg) {
            emptyMsg.innerHTML = '<h3>📁 No project selected</h3><p>Select a project from the Project Manager tab</p>';
            emptyMsg.style.display = 'flex';
        }
        if (gallery) gallery.style.display = 'none';
        modelState.images = [];
        modelState.excludedImages.clear();
        return;
    }
    
    try {
        console.log('[Model] Fetching images for project:', modelState.currentProject.project_id);
        window.PyPotteryUtils.showLoading('Loading project images...');
        
        // Load both images and project metadata (to get excluded images)
        const [imagesResponse, projectResponse] = await Promise.all([
            window.PyPotteryUtils.apiRequest(`/api/projects/${modelState.currentProject.project_id}/images`),
            window.PyPotteryUtils.apiRequest(`/api/projects/${modelState.currentProject.project_id}`)
        ]);
        
        console.log('[Model] Images API response:', imagesResponse);
        console.log('[Model] Project API response:', projectResponse);
        window.PyPotteryUtils.hideLoading();
        
        if (imagesResponse.success && imagesResponse.images) {
            console.log('[Model] Loaded', imagesResponse.images.length, 'images');
            modelState.images = imagesResponse.images;
            
            // Load excluded images from project settings
            modelState.excludedImages.clear();
            if (projectResponse.success && projectResponse.project && projectResponse.project.settings) {
                const excludedList = projectResponse.project.settings.excluded_images || [];
                console.log('[Model] Restored excluded images:', excludedList);
                excludedList.forEach(img => modelState.excludedImages.add(img));
            }
            
            displayGallery(imagesResponse.images);
            
            if (emptyMsg) emptyMsg.style.display = 'none';
            if (gallery) gallery.style.display = 'grid';
        } else {
            console.log('[Model] No images found in response');
            if (emptyMsg) {
                emptyMsg.innerHTML = '<h3>📁 No images found</h3><p>Upload a PDF in the PDF tab to generate images</p>';
                emptyMsg.style.display = 'flex';
            }
            if (gallery) gallery.style.display = 'none';
        }
    } catch (error) {
        window.PyPotteryUtils.hideLoading();
        console.error('[Model] Error loading project images:', error);
        if (emptyMsg) {
            emptyMsg.innerHTML = `<h3>❌ Error</h3><p>${error.message}</p>`;
            emptyMsg.style.display = 'flex';
        }
        if (gallery) gallery.style.display = 'none';
    }
}

function displayGallery(images) {
    const gallery = document.getElementById('model-gallery');
    if (!gallery) {
        console.error('[Model] Gallery element not found!');
        return;
    }
    
    console.log('[Model] Displaying gallery with', images.length, 'images');
    console.log('[Model] Excluded images:', Array.from(modelState.excludedImages));
    gallery.innerHTML = '';
    
    if (!images || images.length === 0) {
        gallery.innerHTML = '<div class="empty-list">No images in project</div>';
        return;
    }
    
    images.forEach((imageUrl, index) => {
        console.log('[Model] Adding image', index + 1, ':', imageUrl);
        const itemDiv = document.createElement('div');
        itemDiv.className = 'gallery-item';
        itemDiv.dataset.imageUrl = imageUrl;

        // Check if this image is excluded
        const isExcluded = modelState.excludedImages.has(imageUrl);
        if (isExcluded) {
            itemDiv.classList.add('excluded');
        }

        const img = document.createElement('img');
        // Use thumbnail for gallery display
        const thumbnailUrl = imageUrl.replace('/image/', '/thumbnail/');
        img.src = thumbnailUrl;
        img.alt = imageUrl.split('/').pop();
        img.title = imageUrl.split('/').pop() + ' (click to view full size)';

        img.addEventListener('click', (e) => {
            e.stopPropagation();
            showImageModal(imageUrl);  // Still show full-size in modal
        });

        const deleteBtn = document.createElement('button');
        deleteBtn.className = 'delete-btn';
        deleteBtn.innerHTML = isExcluded ? '✓' : '×';
        deleteBtn.title = isExcluded ? 'Include in processing' : 'Exclude from processing';

        deleteBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            toggleImageExclusion(imageUrl, itemDiv, deleteBtn);
        });

        itemDiv.appendChild(img);
        itemDiv.appendChild(deleteBtn);
        gallery.appendChild(itemDiv);
    });

    console.log('[Model] Gallery rendered with', gallery.children.length, 'items');

    // Show selection controls and update info
    showSelectionControls(true);
    updateSelectionInfo();
}

function toggleImageExclusion(imageUrl, itemDiv, deleteBtn) {
    if (modelState.excludedImages.has(imageUrl)) {
        modelState.excludedImages.delete(imageUrl);
        itemDiv.classList.remove('excluded');
        deleteBtn.innerHTML = '×';
        deleteBtn.title = 'Exclude from processing';
    } else {
        modelState.excludedImages.add(imageUrl);
        itemDiv.classList.add('excluded');
        deleteBtn.innerHTML = '✓';
        deleteBtn.title = 'Include in processing';
    }
    
    // Save excluded images to project settings
    saveExcludedImages();
}

async function saveExcludedImages() {
    if (!modelState.currentProject || !modelState.currentProject.project_id) return;

    try {
        const excludedArray = Array.from(modelState.excludedImages);
        console.log('[Model] Saving excluded images:', excludedArray);

        await window.PyPotteryUtils.apiRequest(
            `/api/projects/${modelState.currentProject.project_id}/excluded-images`,
            {
                method: 'POST',
                body: JSON.stringify({
                    excluded_images: excludedArray
                })
            }
        );

        console.log('[Model] Excluded images saved successfully');
    } catch (error) {
        console.error('[Model] Error saving excluded images:', error);
    }
}

// Select All - exclude all images
function handleSelectAll() {
    const gallery = document.getElementById('model-gallery');
    if (!gallery) return;

    const items = gallery.querySelectorAll('.gallery-item');
    items.forEach(item => {
        const imageUrl = item.dataset.imageUrl;
        if (imageUrl && !modelState.excludedImages.has(imageUrl)) {
            modelState.excludedImages.add(imageUrl);
            item.classList.add('excluded');
            const btn = item.querySelector('.delete-btn');
            if (btn) {
                btn.innerHTML = '✓';
                btn.title = 'Include in processing';
            }
        }
    });

    updateSelectionInfo();
    saveExcludedImages();
}

// Deselect All - include all images
function handleDeselectAll() {
    const gallery = document.getElementById('model-gallery');
    if (!gallery) return;

    const items = gallery.querySelectorAll('.gallery-item');
    items.forEach(item => {
        const imageUrl = item.dataset.imageUrl;
        if (imageUrl && modelState.excludedImages.has(imageUrl)) {
            modelState.excludedImages.delete(imageUrl);
            item.classList.remove('excluded');
            const btn = item.querySelector('.delete-btn');
            if (btn) {
                btn.innerHTML = '×';
                btn.title = 'Exclude from processing';
            }
        }
    });

    updateSelectionInfo();
    saveExcludedImages();
}

// Update selection info text
function updateSelectionInfo() {
    const infoSpan = document.getElementById('selection-info');
    if (infoSpan) {
        const count = modelState.excludedImages.size;
        infoSpan.textContent = `${count} image${count !== 1 ? 's' : ''} excluded`;
    }
}

// Show/hide selection controls
function showSelectionControls(show) {
    const controls = document.getElementById('model-selection-controls');
    if (controls) {
        controls.style.display = show ? 'flex' : 'none';
    }
}

// Setup drag selection on gallery
function setupDragSelection() {
    const gallery = document.getElementById('model-gallery');
    if (!gallery) return;

    // Create selection box element
    const selectionBox = document.createElement('div');
    selectionBox.className = 'selection-box';
    selectionBox.style.cssText = `
        position: absolute;
        border: 2px dashed #007bff;
        background: rgba(0, 123, 255, 0.1);
        pointer-events: none;
        display: none;
        z-index: 1000;
    `;
    gallery.style.position = 'relative';
    gallery.appendChild(selectionBox);
    modelState.selectionBox = selectionBox;

    // Mouse down - start drag
    gallery.addEventListener('mousedown', (e) => {
        // Only start drag if clicking on gallery background, not on items
        if (e.target === gallery || e.target.classList.contains('selection-box')) {
            modelState.isDragging = true;
            const rect = gallery.getBoundingClientRect();
            modelState.dragStartX = e.clientX - rect.left + gallery.scrollLeft;
            modelState.dragStartY = e.clientY - rect.top + gallery.scrollTop;

            selectionBox.style.left = modelState.dragStartX + 'px';
            selectionBox.style.top = modelState.dragStartY + 'px';
            selectionBox.style.width = '0px';
            selectionBox.style.height = '0px';
            selectionBox.style.display = 'block';

            e.preventDefault();
        }
    });

    // Mouse move - update selection box
    gallery.addEventListener('mousemove', (e) => {
        if (!modelState.isDragging) return;

        const rect = gallery.getBoundingClientRect();
        const currentX = e.clientX - rect.left + gallery.scrollLeft;
        const currentY = e.clientY - rect.top + gallery.scrollTop;

        const left = Math.min(modelState.dragStartX, currentX);
        const top = Math.min(modelState.dragStartY, currentY);
        const width = Math.abs(currentX - modelState.dragStartX);
        const height = Math.abs(currentY - modelState.dragStartY);

        selectionBox.style.left = left + 'px';
        selectionBox.style.top = top + 'px';
        selectionBox.style.width = width + 'px';
        selectionBox.style.height = height + 'px';

        // Highlight items that intersect with selection box
        highlightSelectedItems(left, top, width, height);
    });

    // Mouse up - finish selection
    gallery.addEventListener('mouseup', (e) => {
        if (!modelState.isDragging) return;

        const rect = gallery.getBoundingClientRect();
        const currentX = e.clientX - rect.left + gallery.scrollLeft;
        const currentY = e.clientY - rect.top + gallery.scrollTop;

        const left = Math.min(modelState.dragStartX, currentX);
        const top = Math.min(modelState.dragStartY, currentY);
        const width = Math.abs(currentX - modelState.dragStartX);
        const height = Math.abs(currentY - modelState.dragStartY);

        // Only process if we actually dragged (not just clicked)
        if (width > 10 && height > 10) {
            selectItemsInBox(left, top, width, height, e.shiftKey);
        }

        modelState.isDragging = false;
        selectionBox.style.display = 'none';

        // Remove temporary highlights
        gallery.querySelectorAll('.gallery-item.drag-hover').forEach(item => {
            item.classList.remove('drag-hover');
        });
    });

    // Mouse leave - cancel drag
    gallery.addEventListener('mouseleave', () => {
        if (modelState.isDragging) {
            modelState.isDragging = false;
            selectionBox.style.display = 'none';
            gallery.querySelectorAll('.gallery-item.drag-hover').forEach(item => {
                item.classList.remove('drag-hover');
            });
        }
    });
}

// Highlight items during drag
function highlightSelectedItems(boxLeft, boxTop, boxWidth, boxHeight) {
    const gallery = document.getElementById('model-gallery');
    if (!gallery) return;

    const items = gallery.querySelectorAll('.gallery-item');
    items.forEach(item => {
        const itemRect = item.getBoundingClientRect();
        const galleryRect = gallery.getBoundingClientRect();

        // Convert to gallery-relative coordinates
        const itemLeft = itemRect.left - galleryRect.left + gallery.scrollLeft;
        const itemTop = itemRect.top - galleryRect.top + gallery.scrollTop;
        const itemRight = itemLeft + itemRect.width;
        const itemBottom = itemTop + itemRect.height;

        const boxRight = boxLeft + boxWidth;
        const boxBottom = boxTop + boxHeight;

        // Check intersection
        const intersects = !(itemRight < boxLeft || itemLeft > boxRight ||
                           itemBottom < boxTop || itemTop > boxBottom);

        if (intersects) {
            item.classList.add('drag-hover');
        } else {
            item.classList.remove('drag-hover');
        }
    });
}

// Select/toggle items in selection box
function selectItemsInBox(boxLeft, boxTop, boxWidth, boxHeight, addToSelection) {
    const gallery = document.getElementById('model-gallery');
    if (!gallery) return;

    const items = gallery.querySelectorAll('.gallery-item');
    let changed = false;

    items.forEach(item => {
        const itemRect = item.getBoundingClientRect();
        const galleryRect = gallery.getBoundingClientRect();

        const itemLeft = itemRect.left - galleryRect.left + gallery.scrollLeft;
        const itemTop = itemRect.top - galleryRect.top + gallery.scrollTop;
        const itemRight = itemLeft + itemRect.width;
        const itemBottom = itemTop + itemRect.height;

        const boxRight = boxLeft + boxWidth;
        const boxBottom = boxTop + boxHeight;

        const intersects = !(itemRight < boxLeft || itemLeft > boxRight ||
                           itemBottom < boxTop || itemTop > boxBottom);

        if (intersects) {
            const imageUrl = item.dataset.imageUrl;
            const btn = item.querySelector('.delete-btn');

            if (addToSelection) {
                // Shift+drag: add to exclusion
                if (!modelState.excludedImages.has(imageUrl)) {
                    modelState.excludedImages.add(imageUrl);
                    item.classList.add('excluded');
                    if (btn) {
                        btn.innerHTML = '✓';
                        btn.title = 'Include in processing';
                    }
                    changed = true;
                }
            } else {
                // Normal drag: toggle
                if (modelState.excludedImages.has(imageUrl)) {
                    modelState.excludedImages.delete(imageUrl);
                    item.classList.remove('excluded');
                    if (btn) {
                        btn.innerHTML = '×';
                        btn.title = 'Exclude from processing';
                    }
                } else {
                    modelState.excludedImages.add(imageUrl);
                    item.classList.add('excluded');
                    if (btn) {
                        btn.innerHTML = '✓';
                        btn.title = 'Include in processing';
                    }
                }
                changed = true;
            }
        }
    });

    if (changed) {
        updateSelectionInfo();
        saveExcludedImages();
    }
}

function setupImageModal() {
    const modal = document.getElementById('image-modal');
    const modalImg = document.getElementById('modal-image');
    const closeBtn = modal?.querySelector('.close-modal');
    
    if (!modal || !modalImg || !closeBtn) return;
    
    closeBtn.addEventListener('click', () => {
        modal.classList.remove('active');
    });
    
    modal.addEventListener('click', (e) => {
        if (e.target === modal) {
            modal.classList.remove('active');
        }
    });
    
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && modal.classList.contains('active')) {
            modal.classList.remove('active');
        }
    });
}

function showImageModal(imageUrl) {
    const modal = document.getElementById('image-modal');
    const modalImg = document.getElementById('modal-image');
    
    if (modal && modalImg) {
        modalImg.src = imageUrl;
        modal.classList.add('active');
    }
}

async function handleApplyModel() {
    if (!modelState.currentProject || !modelState.currentProject.project_id) {
        window.PyPotteryUtils.showToast('No project selected', 'warning');
        return;
    }
    
    const model = document.getElementById('model-select').value;
    const confidence = parseFloat(document.getElementById('confidence').value);
    const kernelSize = parseInt(document.getElementById('kernel-size').value);
    const iterations = parseInt(document.getElementById('iterations').value);
    const diagnostic = document.getElementById('diagnostic-mode').checked;
    
    if (!model) {
        window.PyPotteryUtils.showToast('Please select a model', 'warning');
        return;
    }
    
    const excludedImagesArray = Array.from(modelState.excludedImages);
    const totalImages = modelState.images.length;
    const imagesToProcess = totalImages - excludedImagesArray.length;
    
    if (excludedImagesArray.length > 0) {
        const confirmed = confirm(
            `You are excluding ${excludedImagesArray.length} image(s).\n` +
            `${imagesToProcess} image(s) will be processed.\n\n` +
            `Continue?`
        );
        
        if (!confirmed) return;
    }
    
    try {
        const progressContainer = document.getElementById('model-progress');
        const progressBar = document.getElementById('model-progress-bar');
        const progressInfo = document.getElementById('model-progress-info');
        
        if (progressContainer) {
            progressContainer.classList.add('active');
            progressBar.style.width = '0%';
            progressBar.textContent = '0%';
            progressInfo.textContent = `Processing 0/${imagesToProcess} images...`;
        }
        
        const applyBtn = document.getElementById('apply-model-btn');
        if (applyBtn) {
            applyBtn.disabled = true;
            applyBtn.textContent = '⏳ Processing...';
        }
        
        const response = await window.PyPotteryUtils.apiRequest('/api/model/apply', {
            method: 'POST',
            body: JSON.stringify({
                project_id: modelState.currentProject.project_id,
                model: model,
                confidence: confidence,
                kernel_size: kernelSize,
                iterations: iterations,
                diagnostic: diagnostic,
                excluded_images: excludedImagesArray
            })
        });
        
        // Poll for real-time progress
        if (response.success) {
            await pollModelProgress();
        }
        
        if (progressContainer) {
            progressContainer.classList.remove('active');
        }
        
        if (applyBtn) {
            applyBtn.disabled = false;
            applyBtn.textContent = '🚀 Apply Model to Project';
        }
        
        if (response.success) {
            window.PyPotteryUtils.showStatus('model-status', 'Model applied successfully!', 'success');
            window.PyPotteryUtils.showToast('Model applied successfully!', 'success');
            
            modelState.excludedImages.clear();
            loadProjectImages();
            
            // Refresh project list to show updated status
            if (window.projectManager && window.projectManager.loadProjects) {
                window.projectManager.loadProjects();
            }
        } else {
            window.PyPotteryUtils.showStatus('model-status', response.error || 'Failed to apply model', 'error');
            window.PyPotteryUtils.showToast('Failed to apply model', 'error');
        }
        
    } catch (error) {
        console.error('Error applying model:', error);
        
        const progressContainer = document.getElementById('model-progress');
        if (progressContainer) {
            progressContainer.classList.remove('active');
        }
        
        const applyBtn = document.getElementById('apply-model-btn');
        if (applyBtn) {
            applyBtn.disabled = false;
            applyBtn.textContent = '🚀 Apply Model to Project';
        }
        
        window.PyPotteryUtils.showStatus('model-status', error.message, 'error');
        window.PyPotteryUtils.showToast(error.message, 'error');
    }
}

async function pollModelProgress() {
    const progressBar = document.getElementById('model-progress-bar');
    const progressInfo = document.getElementById('model-progress-info');
    
    if (!progressBar || !progressInfo) return;
    
    let isActive = true;
    
    while (isActive) {
        try {
            const response = await fetch('/api/model/progress');
            const progress = await response.json();
            
            if (progress.total > 0) {
                const percentage = Math.round((progress.current / progress.total) * 100);
                progressBar.style.width = `${percentage}%`;
                progressBar.textContent = `${percentage}%`;
                progressInfo.textContent = `Processing ${progress.current}/${progress.total} - ${progress.message}`;
            }
            
            // Check if processing is complete
            if (!progress.active) {
                isActive = false;
                progressBar.style.width = '100%';
                progressBar.textContent = '100%';
                progressInfo.textContent = progress.message || 'Complete';
            }
            
            // Wait before next poll
            if (isActive) {
                await new Promise(resolve => setTimeout(resolve, 500));
            }
            
        } catch (error) {
            console.error('Error polling progress:', error);
            isActive = false;
        }
    }
}

console.log('[Model] Module loaded');
