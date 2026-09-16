// Model Application Tab JavaScript - Project-aware version

// State
let modelState = {
    currentProject: null,
    images: [],
    excludedImages: new Set()
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
    
    // Execution mode radio cards
    const modeCards = document.querySelectorAll('.mode-radio-card');
    modeCards.forEach(card => {
        card.addEventListener('click', () => {
            const radio = card.querySelector('input[type="radio"]');
            if (radio) {
                radio.checked = true;
                modeCards.forEach(c => c.classList.remove('active'));
                card.classList.add('active');
                
                // Sync legacy checkbox if present
                const legacyCheckbox = document.getElementById('diagnostic-mode');
                if (legacyCheckbox) {
                    legacyCheckbox.checked = (radio.value === 'diagnostic');
                }
            }
        });
    });

    // Gallery Toolbar buttons
    const deselectAllBtn = document.getElementById('deselect-all-images-btn');
    if (deselectAllBtn) {
        deselectAllBtn.addEventListener('click', deselectAllImages);
    }

    const selectAllBtn = document.getElementById('select-all-images-btn');
    if (selectAllBtn) {
        selectAllBtn.addEventListener('click', selectAllImages);
    }

    // Apply model button
    const applyBtn = document.getElementById('apply-model-btn');
    if (applyBtn) {
        applyBtn.addEventListener('click', handleApplyModel);
    }
    
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
    const galleryContainer = document.getElementById('model-gallery-container');
    const gallery = document.getElementById('model-gallery');
    
    if (!modelState.currentProject || !modelState.currentProject.project_id) {
        console.log('[Model] No project - showing empty state');
        if (emptyMsg) {
            emptyMsg.innerHTML = '<h3><i class="bi bi-folder-x"></i> No project selected</h3><p>Select a project from the Project Manager tab</p>';
            emptyMsg.style.display = 'flex';
        }
        if (galleryContainer) galleryContainer.style.display = 'none';
        if (gallery) gallery.style.display = 'none';
        modelState.images = [];
        modelState.excludedImages.clear();
        updateGalleryStats();
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
            if (galleryContainer) galleryContainer.style.display = 'block';
            if (gallery) gallery.style.display = 'grid';
        } else {
            console.log('[Model] No images found in response');
            if (emptyMsg) {
                emptyMsg.innerHTML = '<h3><i class="bi bi-images"></i> No images found</h3><p>Upload a PDF in the PDF tab to generate images</p>';
                emptyMsg.style.display = 'flex';
            }
            if (galleryContainer) galleryContainer.style.display = 'none';
            if (gallery) gallery.style.display = 'none';
            updateGalleryStats();
        }
    } catch (error) {
        window.PyPotteryUtils.hideLoading();
        console.error('[Model] Error loading project images:', error);
        if (emptyMsg) {
            emptyMsg.innerHTML = `<h3><i class="bi bi-exclamation-triangle"></i> Error</h3><p>${error.message}</p>`;
            emptyMsg.style.display = 'flex';
        }
        if (galleryContainer) galleryContainer.style.display = 'none';
        if (gallery) gallery.style.display = 'none';
    }
}

function updateGalleryStats() {
    const totalCount = modelState.images ? modelState.images.length : 0;
    const excludedCount = modelState.excludedImages ? modelState.excludedImages.size : 0;
    const selectedCount = Math.max(0, totalCount - excludedCount);
    
    const countBadge = document.getElementById('model-gallery-count-badge');
    const selectedBadge = document.getElementById('model-gallery-selected-badge');
    
    if (countBadge) {
        countBadge.textContent = `${totalCount} ${totalCount === 1 ? 'figure' : 'figures'}`;
    }
    
    if (selectedBadge) {
        if (selectedCount === 0) {
            selectedBadge.textContent = 'None selected (all excluded)';
            selectedBadge.classList.add('gallery-badge-warning');
        } else {
            selectedBadge.textContent = `${selectedCount} to process`;
            selectedBadge.classList.remove('gallery-badge-warning');
        }
    }
}

function deselectAllImages() {
    if (!modelState.images || modelState.images.length === 0) return;
    
    modelState.images.forEach(imageUrl => {
        modelState.excludedImages.add(imageUrl);
    });
    
    const gallery = document.getElementById('model-gallery');
    if (gallery) {
        gallery.querySelectorAll('.gallery-item').forEach(item => {
            item.classList.add('excluded');
            const btn = item.querySelector('.delete-btn');
            if (btn) {
                btn.innerHTML = '<i class="bi bi-check-lg"></i>';
                btn.title = 'Include in processing';
            }
        });
    }
    
    updateGalleryStats();
    saveExcludedImages();
    window.PyPotteryUtils.showToast('All figures excluded. Click figures you wish to process.', 'info');
}

function selectAllImages() {
    if (!modelState.images || modelState.images.length === 0) return;
    
    modelState.excludedImages.clear();
    
    const gallery = document.getElementById('model-gallery');
    if (gallery) {
        gallery.querySelectorAll('.gallery-item').forEach(item => {
            item.classList.remove('excluded');
            const btn = item.querySelector('.delete-btn');
            if (btn) {
                btn.innerHTML = '<i class="bi bi-x-lg"></i>';
                btn.title = 'Exclude from processing';
            }
        });
    }
    
    updateGalleryStats();
    saveExcludedImages();
    window.PyPotteryUtils.showToast('All figures selected for model processing.', 'success');
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
        updateGalleryStats();
        return;
    }
    
    images.forEach((imageUrl, index) => {
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
            showImageModal(imageUrl);
        });
        
        const deleteBtn = document.createElement('button');
        deleteBtn.className = 'delete-btn';
        deleteBtn.innerHTML = isExcluded ? '<i class="bi bi-check-lg"></i>' : '<i class="bi bi-x-lg"></i>';
        deleteBtn.title = isExcluded ? 'Include in processing' : 'Exclude from processing';
        
        deleteBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            toggleImageExclusion(imageUrl, itemDiv, deleteBtn);
        });
        
        itemDiv.appendChild(img);
        itemDiv.appendChild(deleteBtn);
        
        const pageLabel = document.createElement('span');
        pageLabel.className = 'gallery-item-label';
        pageLabel.textContent = imageUrl.split('/').pop().replace(/\.[^/.]+$/, '');
        itemDiv.appendChild(pageLabel);
        
        gallery.appendChild(itemDiv);
    });
    
    updateGalleryStats();
    console.log('[Model] Gallery rendered with', gallery.children.length, 'items');
}

function toggleImageExclusion(imageUrl, itemDiv, deleteBtn) {
    if (modelState.excludedImages.has(imageUrl)) {
        modelState.excludedImages.delete(imageUrl);
        itemDiv.classList.remove('excluded');
        deleteBtn.innerHTML = '<i class="bi bi-x-lg"></i>';
        deleteBtn.title = 'Exclude from processing';
    } else {
        modelState.excludedImages.add(imageUrl);
        itemDiv.classList.add('excluded');
        deleteBtn.innerHTML = '<i class="bi bi-check-lg"></i>';
        deleteBtn.title = 'Include in processing';
    }
    
    updateGalleryStats();
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
    
    const modeRadio = document.querySelector('input[name="model-execution-mode"]:checked') || document.querySelector('input[name="processing-mode"]:checked');
    const diagnostic = modeRadio ? (modeRadio.value === 'diagnostic') : false;
    
    if (!model) {
        window.PyPotteryUtils.showToast('Please select a vision model first', 'warning');
        return;
    }
    
    const excludedImagesArray = Array.from(modelState.excludedImages);
    const totalImages = modelState.images.length;
    let imagesToProcess = totalImages - excludedImagesArray.length;
    if (diagnostic && imagesToProcess > 25) {
        imagesToProcess = 25;
    }
    
    if (imagesToProcess <= 0) {
        window.PyPotteryUtils.showToast('No figures selected for processing! Click figures in the gallery to include them.', 'warning');
        return;
    }
    
    if (excludedImagesArray.length > 0 && !diagnostic) {
        const confirmed = await window.PyPotteryUtils.showConfirmDialog({
            title: 'Confirm Model Inference',
            subtitle: `You have excluded <strong>${excludedImagesArray.length}</strong> image(s) from this run.`,
            icon: 'bi-exclamation-triangle-fill',
            iconColor: '#d97706',
            iconBg: '#fef3c7',
            detailsLabel: 'Processing plan:',
            details: [
                `${imagesToProcess} image(s) will be processed`,
                `${excludedImagesArray.length} image(s) will be skipped`
            ],
            note: 'Model inference may take some time depending on your hardware.',
            confirmText: 'Continue Inference',
            cancelText: 'Cancel',
            confirmClass: 'btn-primary'
        });
        
        if (!confirmed) return;
    }

    // Full-Screen Processing Overlay Elements
    const overlay = document.getElementById('model-processing-overlay');
    const percentageEl = document.getElementById('fullscreen-percentage');
    const progressFill = document.getElementById('fullscreen-progress-fill');
    const currentFileEl = document.getElementById('fullscreen-current-file');
    const countEl = document.getElementById('fullscreen-image-count');
    const modelBadge = document.getElementById('processing-model-badge');
    const cancelBtn = document.getElementById('model-cancel-btn');
    const applyBtn = document.getElementById('apply-model-btn');
    
    // Show and initialize Full-Screen Overlay
    if (overlay) {
        overlay.style.display = 'flex';
        if (percentageEl) percentageEl.textContent = '0%';
        if (progressFill) progressFill.style.width = '0%';
        if (currentFileEl) currentFileEl.textContent = 'Initializing inference pipeline...';
        if (countEl) countEl.textContent = `0 / ${imagesToProcess}`;
        if (modelBadge) modelBadge.textContent = `${model} (${diagnostic ? 'Diagnostic 25' : 'Full'})`;
        if (cancelBtn) {
            cancelBtn.disabled = false;
            cancelBtn.innerHTML = '<i class="bi bi-stop-circle-fill"></i> Stop Processing';
        }
    }
    
    if (applyBtn) {
        applyBtn.disabled = true;
        applyBtn.innerHTML = '<i class="bi bi-hourglass-split"></i> Processing...';
    }

    // Cancel Button click handler
    let isCancelled = false;
    if (cancelBtn) {
        cancelBtn.onclick = async () => {
            if (isCancelled) return;
            isCancelled = true;
            cancelBtn.disabled = true;
            cancelBtn.innerHTML = '<i class="bi bi-hourglass-split"></i> Stopping...';
            if (currentFileEl) {
                currentFileEl.textContent = 'Stopping process at current image boundary...';
            }
            try {
                await fetch('/api/model/cancel', { method: 'POST' });
            } catch (err) {
                console.error('[Model] Error requesting cancel:', err);
            }
        };
    }
    
    try {
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
        
        let finalResult = { success: false, cancelled: false, message: '' };
        
        // Poll for real-time progress while overlay is visible
        if (response.success) {
            finalResult = await pollModelProgress(imagesToProcess, {
                percentageEl,
                progressFill,
                currentFileEl,
                countEl
            });
        } else {
            finalResult.message = response.error || 'Failed to start inference';
        }
        
        // Brief pause to allow user to see 100% or final status
        await new Promise(resolve => setTimeout(resolve, 700));
        
        if (overlay) {
            overlay.style.display = 'none';
        }
        
        if (applyBtn) {
            applyBtn.disabled = false;
            applyBtn.innerHTML = '<i class="bi bi-play-circle-fill"></i> Apply Model to Project';
        }
        
        if (finalResult.cancelled) {
            window.PyPotteryUtils.showStatus('model-status', 'Model processing was stopped by user.', 'info');
            window.PyPotteryUtils.showToast('Model processing stopped.', 'info');
            loadProjectImages();
        } else if (response.success && finalResult.success) {
            window.PyPotteryUtils.showStatus('model-status', 'Model applied successfully!', 'success');
            window.PyPotteryUtils.showToast('Model applied successfully!', 'success');
            
            modelState.excludedImages.clear();
            loadProjectImages();
            
            // Refresh project list to show updated status
            if (window.projectManager && window.projectManager.loadProjects) {
                window.projectManager.loadProjects();
            }
        } else {
            const errMsg = finalResult.message || response.error || 'Failed to apply model';
            window.PyPotteryUtils.showStatus('model-status', errMsg, 'error');
            window.PyPotteryUtils.showToast(errMsg, 'error');
        }
        
    } catch (error) {
        console.error('[Model] Error applying model:', error);
        
        if (overlay) {
            overlay.style.display = 'none';
        }
        
        if (applyBtn) {
            applyBtn.disabled = false;
            applyBtn.innerHTML = '<i class="bi bi-play-circle-fill"></i> Apply Model to Project';
        }
        
        window.PyPotteryUtils.showStatus('model-status', error.message, 'error');
        window.PyPotteryUtils.showToast(error.message, 'error');
    }
}

async function pollModelProgress(totalExpected, elements) {
    const { percentageEl, progressFill, currentFileEl, countEl } = elements;
    let isActive = true;
    let finalResult = { success: true, cancelled: false, message: '' };
    
    while (isActive) {
        try {
            const response = await fetch('/api/model/progress');
            const progress = await response.json();
            
            const total = progress.total > 0 ? progress.total : totalExpected;
            const current = progress.current || 0;
            const percentage = total > 0 ? Math.min(100, Math.round((current / total) * 100)) : 0;
            
            if (percentageEl) percentageEl.textContent = `${percentage}%`;
            if (progressFill) progressFill.style.width = `${percentage}%`;
            if (countEl) countEl.textContent = `${current} / ${total}`;
            if (currentFileEl && progress.message) {
                currentFileEl.textContent = progress.message;
            }
            
            if (progress.cancelled) {
                finalResult.cancelled = true;
                finalResult.message = progress.message || 'Processing cancelled by user';
            }

            // Check if processing is complete
            if (!progress.active) {
                isActive = false;
                if (progress.error) {
                    finalResult.success = false;
                    finalResult.message = progress.message || 'Failed to apply model';
                } else if (!progress.cancelled) {
                    if (percentageEl) percentageEl.textContent = '100%';
                    if (progressFill) progressFill.style.width = '100%';
                    finalResult.message = progress.message || 'Inference complete!';
                }
            } else {
                await new Promise(resolve => setTimeout(resolve, 400));
            }
            
        } catch (error) {
            console.error('[Model] Error polling progress:', error);
            isActive = false;
            finalResult.success = false;
            finalResult.message = error.message;
        }
    }
    
    return finalResult;
}

console.log('[Model] Module loaded');
