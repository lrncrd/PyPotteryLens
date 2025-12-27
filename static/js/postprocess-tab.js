// Post Processing Tab JavaScript - Project-aware version
// Uses window.PyPotteryUtils.* functions directly

// State
let postprocessState = {
    currentProject: null,
    cards: [],
    currentIndex: 0,
    totalImages: 0,
    excludedImages: new Set(),
    cropMode: null  // null, 'selecting', 'ready'
};

// Crop state
let cropState = {
    active: false,
    startX: 0,
    startY: 0,
    endX: 0,
    endY: 0,
    canvas: null,
    ctx: null
};

document.addEventListener('DOMContentLoaded', () => {
    setupPostprocessListeners();
    loadCurrentProject();
    
    // Listen for project changes
    window.addEventListener('projectChanged', (e) => {
        const project = e.detail && e.detail.project ? e.detail.project : null;
        postprocessState.currentProject = project;
        loadProjectCards();
    });
});

function loadCurrentProject() {
    if (window.projectManager && window.projectManager.getCurrentProject) {
        postprocessState.currentProject = window.projectManager.getCurrentProject();
    } else {
        const pid = localStorage.getItem('currentProjectId');
        const pname = localStorage.getItem('currentProjectName');
        if (pid) {
            postprocessState.currentProject = { project_id: pid, project_name: pname || 'Unnamed' };
        }
    }
    
    if (postprocessState.currentProject) {
        loadProjectCards();
    }
}

function setupPostprocessListeners() {
    // Navigation
    document.getElementById('postprocess-prev')?.addEventListener('click', () => navigatePostprocess(-1));
    document.getElementById('postprocess-next')?.addEventListener('click', () => navigatePostprocess(1));

    // Process button
    document.getElementById('process-all-btn')?.addEventListener('click', handleProcessAll);

    // Flip buttons
    document.getElementById('flip-vertical-btn')?.addEventListener('click', () => handleFlip('vertical'));
    document.getElementById('flip-horizontal-btn')?.addEventListener('click', () => handleFlip('horizontal'));

    // Type selection
    document.getElementById('type-select')?.addEventListener('change', handleTypeChange);

    // Export button
    document.getElementById('export-btn')?.addEventListener('click', showExportDialog);

    // Export dialog
    document.getElementById('export-cancel')?.addEventListener('click', hideExportDialog);
    document.getElementById('export-confirm')?.addEventListener('click', handleExportConfirm);

    // Crop buttons
    document.getElementById('auto-crop-preview-btn')?.addEventListener('click', handleAutoCropPreview);
    document.getElementById('manual-crop-btn')?.addEventListener('click', toggleManualCropMode);
    document.getElementById('freehand-crop-btn')?.addEventListener('click', toggleFreehandCropMode);
    document.getElementById('crop-to-content-btn')?.addEventListener('click', handleCropToContent);
    document.getElementById('cancel-crop-btn')?.addEventListener('click', cancelCropMode);
    document.getElementById('apply-crop-btn')?.addEventListener('click', applyCrop);

    // Preview modal buttons
    document.getElementById('keep-left-btn')?.addEventListener('click', () => applyAutoCrop('left'));
    document.getElementById('keep-right-btn')?.addEventListener('click', () => applyAutoCrop('right'));
    document.getElementById('cancel-preview-btn')?.addEventListener('click', hidePreviewModal);

    // Listen for exclusion changes from other tabs
    window.addEventListener('exclusionChanged', (e) => {
        if (e.detail && e.detail.allExcluded) {
            postprocessState.excludedImages = new Set(e.detail.allExcluded);
            updateExclusionVisuals();
        } else if (e.detail && e.detail.filename !== undefined) {
            // Single file exclusion change
            if (e.detail.excluded) {
                postprocessState.excludedImages.add(e.detail.filename);
            } else {
                postprocessState.excludedImages.delete(e.detail.filename);
            }
            updateExclusionVisuals();
        }
    });
}

async function loadProjectCards() {
    if (!postprocessState.currentProject || !postprocessState.currentProject.project_id) {
        showEmptyState('No project selected', 'Select a project from the Project Manager tab');
        return;
    }

    try {
        window.PyPotteryUtils.showLoading('Loading project cards...');
        
        const response = await window.PyPotteryUtils.apiRequest(
            `/api/projects/${postprocessState.currentProject.project_id}/cards`
        );
        
        window.PyPotteryUtils.hideLoading();
        
        if (response.success) {
            postprocessState.cards = response.cards || [];
            postprocessState.totalImages = response.total || 0;

            if (postprocessState.totalImages === 0) {
                showEmptyState('No cards found', 'Extract cards from masks in the Annotation tab first');
                return;
            }

            // Load exclusions
            await loadExclusions();

            // Populate file list
            populatePostprocessFileList(postprocessState.cards);

            // Load first card
            await loadPostprocessImage(0);
        } else {
            showEmptyState('Error loading cards', response.error);
        }
        
    } catch (error) {
        window.PyPotteryUtils.hideLoading();
        console.error('Error loading project cards:', error);
        showEmptyState('Error', error.message);
    }
}

async function loadPostprocessImage(imgNum) {
    if (!postprocessState.currentProject || postprocessState.cards.length === 0) return;
    
    if (imgNum < 0 || imgNum >= postprocessState.totalImages) return;

    try {
        postprocessState.currentIndex = imgNum;
        
        // Update active item in list
        const listItems = document.querySelectorAll('#postprocess-image-list .image-list-item');
        listItems.forEach((item, index) => {
            item.classList.toggle('active', index === imgNum);
        });
        
        // Get card data
        const cardData = postprocessState.cards[imgNum];
        const cardUrl = cardData.url || cardData;  // Support both old and new format
        const cardType = cardData.type || 'ENT';
        const cardFilename = cardData.filename || cardUrl.split('/').pop();
        
        // Display original card
        document.getElementById('postprocess-original').src = cardUrl;
        
        // Update type select
        const typeSelect = document.getElementById('type-select');
        if (typeSelect) {
            typeSelect.value = cardType;
        }
        
        // Check if transformed version exists
        const transformedUrl = `/api/projects/${postprocessState.currentProject.project_id}/card-modified/${cardFilename}`;
        
        // Try to load transformed version
        const transformedImg = document.getElementById('postprocess-transformed');
        const placeholder = document.getElementById('postprocess-placeholder');
        
        // First, check if the file exists with a HEAD request
        try {
            const checkResponse = await fetch(transformedUrl, { method: 'HEAD' });
            if (checkResponse.ok) {
                // Transformed version exists, load it
                transformedImg.src = transformedUrl;
                transformedImg.style.display = 'block';
                if (placeholder) placeholder.style.display = 'none';
            } else {
                // Transformed version doesn't exist, show placeholder
                transformedImg.src = '';
                transformedImg.style.display = 'none';
                if (placeholder) placeholder.style.display = 'flex';
            }
        } catch (error) {
            // Error checking, assume doesn't exist
            transformedImg.src = '';
            transformedImg.style.display = 'none';
            if (placeholder) placeholder.style.display = 'flex';
        }
        
    } catch (error) {
        console.error('Error loading image:', error);
        window.PyPotteryUtils.showToast(error.message, 'error');
    }
}

async function handleProcessAll() {
    if (!postprocessState.currentProject) {
        window.PyPotteryUtils.showToast('Please select a project first', 'warning');
        return;
    }

    const flipVertical = document.getElementById('auto-flip-vertical').checked;
    const flipHorizontal = document.getElementById('auto-flip-horizontal').checked;

    try {
        window.PyPotteryUtils.showStatus('postprocess-status', 'Starting processing...', 'info');

        // Use progress tracking with progress bar
        const response = await window.PyPotteryUtils.executeWithProgress(
            'postprocess',
            async () => {
                const res = await window.PyPotteryUtils.apiRequest(`/api/projects/${postprocessState.currentProject.project_id}/postprocess`, {
                    method: 'POST',
                    body: JSON.stringify({
                        flip_vertical: flipVertical,
                        flip_horizontal: flipHorizontal
                    })
                });
                return res;
            },
            'postprocess-status',
            'postprocess-progress-bar'
        );

        if (response.success) {
            window.PyPotteryUtils.showStatus('postprocess-status', response.message, 'success');
            window.PyPotteryUtils.showToast(`Processed ${response.count} images!`, 'success');
            // Reload cards to get updated classifications
            await loadProjectCards();
            // Reload current image
            await loadPostprocessImage(postprocessState.currentIndex);
        } else {
            window.PyPotteryUtils.showStatus('postprocess-status', response.error, 'error');
            window.PyPotteryUtils.showToast('Processing failed', 'error');
        }
    } catch (error) {
        console.error('Error processing:', error);
        window.PyPotteryUtils.showStatus('postprocess-status', error.message, 'error');
        window.PyPotteryUtils.showToast(error.message, 'error');
    }
}

async function handleFlip(flipType) {
    if (!postprocessState.currentProject) return;

    try {
        window.PyPotteryUtils.showLoading(`Flipping image ${flipType}...`);

        const response = await window.PyPotteryUtils.apiRequest(`/api/projects/${postprocessState.currentProject.project_id}/postprocess/flip`, {
            method: 'POST',
            body: JSON.stringify({
                img_num: postprocessState.currentIndex,
                flip_type: flipType
            })
        });

        window.PyPotteryUtils.hideLoading();

        if (response.success) {
            document.getElementById('postprocess-transformed').src = response.image;
            window.PyPotteryUtils.showToast('Image flipped successfully', 'success');
        } else {
            window.PyPotteryUtils.showToast('Failed to flip image', 'error');
        }
    } catch (error) {
        window.PyPotteryUtils.hideLoading();
        console.error('Error flipping:', error);
        window.PyPotteryUtils.showToast(error.message, 'error');
    }
}

async function handleTypeChange(e) {
    if (!postprocessState.currentProject) {
        console.error('No project selected');
        return;
    }

    const newType = e.target.value;
    
    // Get current card filename
    const cardData = postprocessState.cards[postprocessState.currentIndex];
    const filename = cardData.filename || (cardData.url || cardData).split('/').pop();

    console.log('Updating type:', {
        filename: filename,
        newType: newType,
        currentIndex: postprocessState.currentIndex,
        cardData: cardData
    });

    try {
        const response = await window.PyPotteryUtils.apiRequest(`/api/projects/${postprocessState.currentProject.project_id}/postprocess/update-type`, {
            method: 'POST',
            body: JSON.stringify({
                filename: filename,
                type: newType
            })
        });

        console.log('Update response:', response);

        if (response.success) {
            window.PyPotteryUtils.showToast('Type updated successfully', 'success');
            // Update local state
            if (cardData.type !== undefined) {
                cardData.type = newType;
            }
        } else {
            console.error('Update failed:', response.error);
            window.PyPotteryUtils.showToast(response.error || 'Failed to update type', 'error');
        }
    } catch (error) {
        console.error('Error updating type:', error);
        window.PyPotteryUtils.showToast(error.message, 'error');
    }
}

function navigatePostprocess(direction) {
    const newIndex = postprocessState.currentIndex + direction;
    if (newIndex >= 0 && newIndex < postprocessState.totalImages) {
        loadPostprocessImage(newIndex);
    }
}

function showEmptyState(title, message) {
    const originalImg = document.getElementById('postprocess-original');
    const transformedImg = document.getElementById('postprocess-transformed');
    if (originalImg) originalImg.src = '';
    if (transformedImg) transformedImg.src = '';
    window.PyPotteryUtils.showStatus('postprocess-status', `${title}: ${message}`, 'info');
}

// Export dialog functions
function showExportDialog() {
    if (!postprocessState.currentProject) {
        window.PyPotteryUtils.showToast('Please select a project first', 'warning');
        return;
    }
    document.getElementById('export-dialog').style.display = 'flex';
}

function hideExportDialog() {
    document.getElementById('export-dialog').style.display = 'none';
    document.getElementById('export-acronym').value = '';
    document.getElementById('export-dialog-status').textContent = '';
    document.getElementById('export-dialog-status').className = 'status-message';
}

async function handleExportConfirm() {
    const acronym = document.getElementById('export-acronym').value.trim();

    // Validation
    if (!acronym) {
        window.PyPotteryUtils.showStatus('export-dialog-status', 'Please enter an acronym', 'error');
        return;
    }

    if (!/^[a-zA-Z0-9_]+$/.test(acronym)) {
        window.PyPotteryUtils.showStatus('export-dialog-status', 'Acronym can only contain letters, numbers, and underscores', 'error');
        return;
    }

    try {
        window.PyPotteryUtils.showLoading('Exporting and creating ZIP...');
        window.PyPotteryUtils.showStatus('export-dialog-status', 'Exporting...', 'info');

        // Make request to export endpoint
        const response = await fetch(`/api/projects/${postprocessState.currentProject.project_id}/export`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                acronym: acronym
            })
        });

        window.PyPotteryUtils.hideLoading();

        if (response.ok) {
            // Get the blob
            const blob = await response.blob();
            
            // Create download link
            const url = window.URL.createObjectURL(blob);
            const link = document.createElement('a');
            link.href = url;
            link.download = `${acronym}.zip`;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            window.URL.revokeObjectURL(url);
            
            window.PyPotteryUtils.showStatus('export-dialog-status', 'Export completed! ZIP downloaded.', 'success');
            window.PyPotteryUtils.showToast('Export completed successfully!', 'success');
            
            // Close dialog after 2 seconds
            setTimeout(() => {
                hideExportDialog();
                window.PyPotteryUtils.showStatus('postprocess-status', 'Export completed', 'success');
            }, 2000);
        } else {
            const errorData = await response.json();
            window.PyPotteryUtils.showStatus('export-dialog-status', errorData.error || 'Export failed', 'error');
        }
    } catch (error) {
        window.PyPotteryUtils.hideLoading();
        console.error('Error exporting:', error);
        window.PyPotteryUtils.showStatus('export-dialog-status', error.message, 'error');
    }
}

function populatePostprocessFileList(cards) {
    const listContainer = document.getElementById('postprocess-image-list');
    const countDisplay = document.getElementById('postprocess-image-count');
    
    if (!listContainer || !countDisplay) return;
    
    countDisplay.textContent = cards.length;
    
    if (cards.length === 0) {
        listContainer.innerHTML = '<div class="empty-message" style="padding: 2rem; text-align: center; color: var(--text-secondary);">No cards available</div>';
        return;
    }
    
    listContainer.innerHTML = '';
    
    cards.forEach((cardData, index) => {
        // Extract filename from URL or data object
        const filename = cardData.filename || (cardData.url || cardData).split('/').pop();
        const isExcluded = postprocessState.excludedImages.has(filename);

        const item = document.createElement('div');
        item.className = `image-list-item${isExcluded ? ' excluded' : ''}`;
        item.dataset.index = index;
        item.dataset.filename = filename;

        item.innerHTML = `
            <input type="checkbox" class="exclude-checkbox" title="Exclude from export" ${isExcluded ? 'checked' : ''}>
            <span class="image-number">${index}</span>
            <span class="image-name" title="${filename}">${filename}</span>
            <span class="exclude-indicator">EXCLUDED</span>
        `;

        // Checkbox click handler
        const checkbox = item.querySelector('.exclude-checkbox');
        checkbox.addEventListener('click', (e) => {
            e.stopPropagation();
            handleExclusionToggle(filename, e.target.checked);
        });

        // Item click handler (for selection)
        item.addEventListener('click', (e) => {
            if (e.target.classList.contains('exclude-checkbox')) return;
            loadPostprocessImage(index);
            // Update active state
            document.querySelectorAll('#postprocess-image-list .image-list-item').forEach(i => i.classList.remove('active'));
            item.classList.add('active');
        });

        listContainer.appendChild(item);
    });
    
    // Mark first item as active
    const firstItem = listContainer.querySelector('.image-list-item');
    if (firstItem) firstItem.classList.add('active');
}

// ============ EXCLUSION FUNCTIONS ============

async function loadExclusions() {
    if (!postprocessState.currentProject) return;

    try {
        const response = await window.PyPotteryUtils.apiRequest(
            `/api/projects/${postprocessState.currentProject.project_id}/cards/exclusions`
        );

        if (response.success && response.excluded) {
            postprocessState.excludedImages = new Set(response.excluded);
        }
    } catch (error) {
        console.error('Error loading exclusions:', error);
    }
}

async function handleExclusionToggle(filename, excluded) {
    if (!postprocessState.currentProject) return;

    try {
        const response = await window.PyPotteryUtils.apiRequest(
            `/api/projects/${postprocessState.currentProject.project_id}/cards/exclude`,
            {
                method: 'POST',
                body: JSON.stringify({ filename, excluded })
            }
        );

        if (response.success) {
            if (excluded) {
                postprocessState.excludedImages.add(filename);
            } else {
                postprocessState.excludedImages.delete(filename);
            }

            updateExclusionVisuals();

            // Notify other tabs
            window.dispatchEvent(new CustomEvent('exclusionChanged', {
                detail: { excluded: Array.from(postprocessState.excludedImages) }
            }));

            window.PyPotteryUtils.showToast(
                excluded ? 'Image excluded from export' : 'Image included in export',
                'success'
            );
        }
    } catch (error) {
        console.error('Error toggling exclusion:', error);
        window.PyPotteryUtils.showToast('Error updating exclusion', 'error');
    }
}

function updateExclusionVisuals() {
    const listItems = document.querySelectorAll('#postprocess-image-list .image-list-item');
    listItems.forEach(item => {
        const filename = item.querySelector('.image-name')?.textContent;
        const isExcluded = postprocessState.excludedImages.has(filename);
        item.classList.toggle('excluded', isExcluded);

        // Update checkbox if present
        const checkbox = item.querySelector('.exclude-checkbox');
        if (checkbox) {
            checkbox.checked = isExcluded;
        }
    });

    // Update count
    const countEl = document.getElementById('excluded-count');
    if (countEl) {
        countEl.textContent = postprocessState.excludedImages.size;
    }
}

// ============ CROP FUNCTIONS ============

async function handleAutoCrop() {
    if (!postprocessState.currentProject || postprocessState.cards.length === 0) return;

    const cardData = postprocessState.cards[postprocessState.currentIndex];
    const filename = cardData.filename || (cardData.url || cardData).split('/').pop();

    try {
        window.PyPotteryUtils.showLoading('Auto-cropping image...');

        const response = await window.PyPotteryUtils.apiRequest(
            `/api/projects/${postprocessState.currentProject.project_id}/cards/crop`,
            {
                method: 'POST',
                body: JSON.stringify({
                    filename: filename,
                    mode: 'auto',
                    keep_side: 'auto'
                })
            }
        );

        window.PyPotteryUtils.hideLoading();

        if (response.success) {
            // Update the transformed image
            document.getElementById('postprocess-transformed').src = response.cropped_image;
            document.getElementById('postprocess-transformed').style.display = 'block';
            document.getElementById('postprocess-placeholder')?.style.setProperty('display', 'none');

            window.PyPotteryUtils.showToast('Section removed successfully', 'success');
        } else {
            window.PyPotteryUtils.showToast(response.error || 'Auto-crop failed', 'error');
        }
    } catch (error) {
        window.PyPotteryUtils.hideLoading();
        console.error('Error auto-cropping:', error);
        window.PyPotteryUtils.showToast(error.message, 'error');
    }
}

async function handleCropToContent() {
    if (!postprocessState.currentProject || postprocessState.cards.length === 0) return;

    const cardData = postprocessState.cards[postprocessState.currentIndex];
    const filename = cardData.filename || (cardData.url || cardData).split('/').pop();

    try {
        window.PyPotteryUtils.showLoading('Cropping to content...');

        const response = await window.PyPotteryUtils.apiRequest(
            `/api/projects/${postprocessState.currentProject.project_id}/cards/crop`,
            {
                method: 'POST',
                body: JSON.stringify({
                    filename: filename,
                    mode: 'content'
                })
            }
        );

        window.PyPotteryUtils.hideLoading();

        if (response.success) {
            document.getElementById('postprocess-transformed').src = response.cropped_image;
            document.getElementById('postprocess-transformed').style.display = 'block';
            document.getElementById('postprocess-placeholder')?.style.setProperty('display', 'none');

            window.PyPotteryUtils.showToast('Cropped to content', 'success');
        } else {
            window.PyPotteryUtils.showToast(response.error || 'Crop failed', 'error');
        }
    } catch (error) {
        window.PyPotteryUtils.hideLoading();
        console.error('Error cropping to content:', error);
        window.PyPotteryUtils.showToast(error.message, 'error');
    }
}

function toggleManualCropMode() {
    // Get the processed image wrapper (where we'll add the crop overlay)
    const imageWrapper = document.querySelector('.processed-image-wrapper');
    if (!imageWrapper) {
        console.error('Could not find .processed-image-wrapper');
        return;
    }

    if (postprocessState.cropMode === 'selecting') {
        cancelManualCrop();
        return;
    }

    postprocessState.cropMode = 'selecting';

    // Use the transformed (processed) image for cropping
    const targetImg = document.getElementById('postprocess-transformed');
    if (!targetImg || targetImg.style.display === 'none') {
        // Fallback to original if no processed image
        const originalImg = document.getElementById('postprocess-original');
        if (!originalImg) {
            window.PyPotteryUtils.showToast('No image available to crop', 'error');
            return;
        }
        // Use original image container
        const originalContainer = originalImg.parentElement;
        if (!originalImg.complete) {
            originalImg.onload = () => initCropCanvas(originalContainer, originalImg);
        } else {
            initCropCanvas(originalContainer, originalImg);
        }
    } else {
        // Use processed image
        if (!targetImg.complete) {
            targetImg.onload = () => initCropCanvas(imageWrapper, targetImg);
        } else {
            initCropCanvas(imageWrapper, targetImg);
        }
    }

    // Update button states
    document.getElementById('manual-crop-btn')?.classList.add('active');
    document.getElementById('apply-crop-btn').style.display = 'inline-block';
    document.getElementById('cancel-crop-btn').style.display = 'inline-block';

    window.PyPotteryUtils.showToast('Click and drag to select crop area', 'info');
}

function initCropCanvas(container, img) {
    // Remove existing canvas
    const existing = document.getElementById('crop-canvas');
    if (existing) existing.remove();

    const canvas = document.createElement('canvas');
    canvas.id = 'crop-canvas';
    canvas.width = img.offsetWidth;
    canvas.height = img.offsetHeight;
    canvas.style.cssText = `
        position: absolute;
        top: 0;
        left: 0;
        cursor: crosshair;
        z-index: 10;
    `;

    container.style.position = 'relative';
    container.appendChild(canvas);

    cropState.canvas = canvas;
    cropState.ctx = canvas.getContext('2d');

    // Add event listeners
    canvas.addEventListener('mousedown', startCrop);
    canvas.addEventListener('mousemove', updateCropSelection);
    canvas.addEventListener('mouseup', endCropSelection);
}

function startCrop(e) {
    if (postprocessState.cropMode !== 'selecting') return;

    const rect = cropState.canvas.getBoundingClientRect();
    cropState.startX = e.clientX - rect.left;
    cropState.startY = e.clientY - rect.top;
    cropState.active = true;
}

function updateCropSelection(e) {
    if (!cropState.active) return;

    const rect = cropState.canvas.getBoundingClientRect();
    cropState.endX = e.clientX - rect.left;
    cropState.endY = e.clientY - rect.top;

    drawCropRect();
}

function endCropSelection(e) {
    if (!cropState.active) return;

    cropState.active = false;
    postprocessState.cropMode = 'ready';

    // Show apply/cancel buttons
    document.getElementById('apply-crop-btn')?.classList.add('visible');
}

function drawCropRect() {
    const ctx = cropState.ctx;
    const canvas = cropState.canvas;

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw semi-transparent overlay
    ctx.fillStyle = 'rgba(0, 0, 0, 0.5)';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Calculate selection rectangle
    const x = Math.min(cropState.startX, cropState.endX);
    const y = Math.min(cropState.startY, cropState.endY);
    const w = Math.abs(cropState.endX - cropState.startX);
    const h = Math.abs(cropState.endY - cropState.startY);

    // Clear selection area (make it visible)
    ctx.clearRect(x, y, w, h);

    // Draw selection border
    ctx.strokeStyle = '#4CAF50';
    ctx.lineWidth = 2;
    ctx.setLineDash([5, 5]);
    ctx.strokeRect(x, y, w, h);
}

function cancelManualCrop() {
    postprocessState.cropMode = null;
    cropState.active = false;

    // Remove canvas
    const canvas = document.getElementById('crop-canvas');
    if (canvas) canvas.remove();

    // Update button states
    document.getElementById('manual-crop-btn')?.classList.remove('active');
    document.getElementById('apply-crop-btn').style.display = 'none';
    document.getElementById('cancel-crop-btn').style.display = 'none';

    // Update status
    const statusEl = document.getElementById('crop-status');
    if (statusEl) {
        statusEl.textContent = '';
        statusEl.classList.remove('active');
    }
}

async function applyManualCrop() {
    if (!postprocessState.currentProject || postprocessState.cards.length === 0) return;

    const cardData = postprocessState.cards[postprocessState.currentIndex];
    const filename = cardData.filename || (cardData.url || cardData).split('/').pop();

    // Get the original image dimensions
    const originalImg = document.getElementById('postprocess-original');
    const displayWidth = originalImg.offsetWidth;
    const displayHeight = originalImg.offsetHeight;
    const naturalWidth = originalImg.naturalWidth;
    const naturalHeight = originalImg.naturalHeight;

    // Calculate scale
    const scaleX = naturalWidth / displayWidth;
    const scaleY = naturalHeight / displayHeight;

    // Calculate crop rectangle in original image coordinates
    const rect = {
        x: Math.round(Math.min(cropState.startX, cropState.endX) * scaleX),
        y: Math.round(Math.min(cropState.startY, cropState.endY) * scaleY),
        width: Math.round(Math.abs(cropState.endX - cropState.startX) * scaleX),
        height: Math.round(Math.abs(cropState.endY - cropState.startY) * scaleY)
    };

    try {
        window.PyPotteryUtils.showLoading('Applying crop...');

        const response = await window.PyPotteryUtils.apiRequest(
            `/api/projects/${postprocessState.currentProject.project_id}/cards/crop`,
            {
                method: 'POST',
                body: JSON.stringify({
                    filename: filename,
                    mode: 'manual',
                    rect: rect
                })
            }
        );

        window.PyPotteryUtils.hideLoading();

        if (response.success) {
            // Update the transformed image
            document.getElementById('postprocess-transformed').src = response.cropped_image;
            document.getElementById('postprocess-transformed').style.display = 'block';
            document.getElementById('postprocess-placeholder')?.style.setProperty('display', 'none');

            cancelManualCrop();
            window.PyPotteryUtils.showToast('Image cropped successfully', 'success');
        } else {
            window.PyPotteryUtils.showToast(response.error || 'Crop failed', 'error');
        }
    } catch (error) {
        window.PyPotteryUtils.hideLoading();
        console.error('Error applying crop:', error);
        window.PyPotteryUtils.showToast(error.message, 'error');
    }
}

// ============================================================================
// Auto Crop Preview Functions
// ============================================================================

async function handleAutoCropPreview() {
    if (!postprocessState.currentProject || postprocessState.cards.length === 0) {
        window.PyPotteryUtils.showToast('No image selected', 'error');
        return;
    }

    const cardData = postprocessState.cards[postprocessState.currentIndex];
    const filename = cardData.filename || (cardData.url || cardData).split('/').pop();

    try {
        window.PyPotteryUtils.showLoading('Analyzing image...');

        const response = await window.PyPotteryUtils.apiRequest(
            `/api/projects/${postprocessState.currentProject.project_id}/cards/crop-preview`,
            {
                method: 'POST',
                body: JSON.stringify({ filename })
            }
        );

        window.PyPotteryUtils.hideLoading();

        if (response.success) {
            showPreviewModal(response);
        } else {
            window.PyPotteryUtils.showToast(response.error || 'Preview failed', 'error');
        }
    } catch (error) {
        window.PyPotteryUtils.hideLoading();
        console.error('Error generating preview:', error);
        window.PyPotteryUtils.showToast(error.message, 'error');
    }
}

function showPreviewModal(previewData) {
    const modal = document.getElementById('section-preview-modal');
    if (!modal) return;

    // Set images
    document.getElementById('preview-left').src = previewData.left_preview;
    document.getElementById('preview-right').src = previewData.right_preview;

    // Set complexity scores
    document.getElementById('left-complexity').textContent =
        `(complexity: ${previewData.complexity_left.toFixed(2)})`;
    document.getElementById('right-complexity').textContent =
        `(complexity: ${previewData.complexity_right.toFixed(2)})`;

    // Highlight recommended side
    const leftHalf = document.querySelector('.preview-half[data-side="left"]');
    const rightHalf = document.querySelector('.preview-half[data-side="right"]');

    leftHalf.classList.remove('recommended');
    rightHalf.classList.remove('recommended');

    if (previewData.recommended_side === 'left') {
        leftHalf.classList.add('recommended');
    } else {
        rightHalf.classList.add('recommended');
    }

    document.getElementById('recommended-side').textContent =
        `✨ Recommended: Keep ${previewData.recommended_side} side (silhouette)`;

    // Store preview data for later use
    postprocessState.previewData = previewData;

    modal.style.display = 'flex';
}

function hidePreviewModal() {
    const modal = document.getElementById('section-preview-modal');
    if (modal) {
        modal.style.display = 'none';
    }
    postprocessState.previewData = null;
}

async function applyAutoCrop(side) {
    if (!postprocessState.currentProject || postprocessState.cards.length === 0) return;

    const cardData = postprocessState.cards[postprocessState.currentIndex];
    const filename = cardData.filename || (cardData.url || cardData).split('/').pop();

    try {
        window.PyPotteryUtils.showLoading('Applying crop...');
        hidePreviewModal();

        const response = await window.PyPotteryUtils.apiRequest(
            `/api/projects/${postprocessState.currentProject.project_id}/cards/crop`,
            {
                method: 'POST',
                body: JSON.stringify({
                    filename: filename,
                    mode: 'auto',
                    keep_side: side
                })
            }
        );

        window.PyPotteryUtils.hideLoading();

        if (response.success) {
            // Update the transformed image
            document.getElementById('postprocess-transformed').src = response.cropped_image;
            document.getElementById('postprocess-transformed').style.display = 'block';
            document.getElementById('postprocess-placeholder')?.style.setProperty('display', 'none');

            window.PyPotteryUtils.showToast(`Kept ${side} side successfully`, 'success');
        } else {
            window.PyPotteryUtils.showToast(response.error || 'Crop failed', 'error');
        }
    } catch (error) {
        window.PyPotteryUtils.hideLoading();
        console.error('Error applying auto crop:', error);
        window.PyPotteryUtils.showToast(error.message, 'error');
    }
}

// ============================================================================
// Freehand Crop Functions
// ============================================================================

let freehandState = {
    active: false,
    points: [],
    canvas: null,
    ctx: null
};

function toggleFreehandCropMode() {
    const imageWrapper = document.querySelector('.processed-image-wrapper');
    if (!imageWrapper) {
        console.error('Could not find .processed-image-wrapper');
        return;
    }

    if (postprocessState.cropMode === 'freehand') {
        cancelCropMode();
        return;
    }

    // Cancel any existing crop mode first
    cancelCropMode();

    postprocessState.cropMode = 'freehand';
    freehandState.points = [];

    const targetImg = document.getElementById('postprocess-transformed');
    const originalImg = document.getElementById('postprocess-original');
    const img = (targetImg && targetImg.style.display !== 'none') ? targetImg : originalImg;

    if (!img) {
        window.PyPotteryUtils.showToast('No image available to crop', 'error');
        return;
    }

    const container = img.parentElement;
    initFreehandCanvas(container, img);

    // Update button states
    document.getElementById('freehand-crop-btn')?.classList.add('active');
    document.querySelector('.crop-actions').style.display = 'flex';

    updateCropStatus('Draw freehand selection, close the path to finish');
    window.PyPotteryUtils.showToast('Draw freehand selection', 'info');
}

function initFreehandCanvas(container, img) {
    // Remove existing canvas
    const existing = document.getElementById('freehand-canvas');
    if (existing) existing.remove();

    const canvas = document.createElement('canvas');
    canvas.id = 'freehand-canvas';
    canvas.className = 'freehand-canvas';
    canvas.width = img.offsetWidth;
    canvas.height = img.offsetHeight;
    canvas.style.width = img.offsetWidth + 'px';
    canvas.style.height = img.offsetHeight + 'px';

    // Position over the image
    const imgRect = img.getBoundingClientRect();
    const containerRect = container.getBoundingClientRect();
    canvas.style.left = (imgRect.left - containerRect.left) + 'px';
    canvas.style.top = (imgRect.top - containerRect.top) + 'px';

    container.style.position = 'relative';
    container.appendChild(canvas);

    freehandState.canvas = canvas;
    freehandState.ctx = canvas.getContext('2d');
    freehandState.active = false;
    freehandState.points = [];

    canvas.addEventListener('mousedown', startFreehand);
    canvas.addEventListener('mousemove', drawFreehand);
    canvas.addEventListener('mouseup', endFreehand);
    canvas.addEventListener('mouseleave', endFreehand);

    // Touch support
    canvas.addEventListener('touchstart', (e) => {
        e.preventDefault();
        const touch = e.touches[0];
        const rect = canvas.getBoundingClientRect();
        startFreehand({ offsetX: touch.clientX - rect.left, offsetY: touch.clientY - rect.top });
    });
    canvas.addEventListener('touchmove', (e) => {
        e.preventDefault();
        const touch = e.touches[0];
        const rect = canvas.getBoundingClientRect();
        drawFreehand({ offsetX: touch.clientX - rect.left, offsetY: touch.clientY - rect.top });
    });
    canvas.addEventListener('touchend', endFreehand);
}

function startFreehand(e) {
    freehandState.active = true;
    freehandState.points = [{ x: e.offsetX, y: e.offsetY }];
}

function drawFreehand(e) {
    if (!freehandState.active || !freehandState.ctx) return;

    freehandState.points.push({ x: e.offsetX, y: e.offsetY });
    renderFreehandPath();
}

function endFreehand() {
    if (!freehandState.active) return;
    freehandState.active = false;

    // Close the path
    if (freehandState.points.length > 2) {
        freehandState.points.push(freehandState.points[0]);
        renderFreehandPath();
        updateCropStatus(`Selection complete (${freehandState.points.length} points). Click Apply to crop.`);
    }
}

function renderFreehandPath() {
    const ctx = freehandState.ctx;
    const canvas = freehandState.canvas;
    if (!ctx || !canvas) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (freehandState.points.length < 2) return;

    ctx.beginPath();
    ctx.moveTo(freehandState.points[0].x, freehandState.points[0].y);

    for (let i = 1; i < freehandState.points.length; i++) {
        ctx.lineTo(freehandState.points[i].x, freehandState.points[i].y);
    }

    ctx.closePath();
    ctx.fillStyle = 'rgba(37, 99, 235, 0.2)';
    ctx.fill();
    ctx.strokeStyle = '#2563eb';
    ctx.lineWidth = 2;
    ctx.setLineDash([5, 5]);
    ctx.stroke();
}

// ============================================================================
// Unified Crop Functions
// ============================================================================

function cancelCropMode() {
    postprocessState.cropMode = null;
    cropState.active = false;
    freehandState.active = false;
    freehandState.points = [];

    // Remove any crop canvas
    document.getElementById('crop-canvas')?.remove();
    document.getElementById('freehand-canvas')?.remove();

    // Update button states
    document.getElementById('manual-crop-btn')?.classList.remove('active');
    document.getElementById('freehand-crop-btn')?.classList.remove('active');
    document.querySelector('.crop-actions').style.display = 'none';

    updateCropStatus('');
}

function updateCropStatus(message) {
    const statusEl = document.getElementById('crop-status');
    if (statusEl) {
        statusEl.textContent = message;
        statusEl.classList.toggle('active', !!message);
    }
}

async function applyCrop() {
    if (postprocessState.cropMode === 'selecting') {
        await applyManualCrop();
    } else if (postprocessState.cropMode === 'freehand') {
        await applyFreehandCrop();
    }
}

async function applyFreehandCrop() {
    if (!postprocessState.currentProject || postprocessState.cards.length === 0) return;
    if (freehandState.points.length < 3) {
        window.PyPotteryUtils.showToast('Please draw a selection first', 'error');
        return;
    }

    const cardData = postprocessState.cards[postprocessState.currentIndex];
    const filename = cardData.filename || (cardData.url || cardData).split('/').pop();

    // Get image for coordinate scaling
    const targetImg = document.getElementById('postprocess-transformed');
    const originalImg = document.getElementById('postprocess-original');
    const img = (targetImg && targetImg.style.display !== 'none') ? targetImg : originalImg;

    const displayWidth = img.offsetWidth;
    const displayHeight = img.offsetHeight;
    const naturalWidth = img.naturalWidth;
    const naturalHeight = img.naturalHeight;

    const scaleX = naturalWidth / displayWidth;
    const scaleY = naturalHeight / displayHeight;

    // Scale points to original image coordinates
    const scaledPoints = freehandState.points.map(p => [
        Math.round(p.x * scaleX),
        Math.round(p.y * scaleY)
    ]);

    try {
        window.PyPotteryUtils.showLoading('Applying freehand crop...');

        const response = await window.PyPotteryUtils.apiRequest(
            `/api/projects/${postprocessState.currentProject.project_id}/cards/crop-freehand`,
            {
                method: 'POST',
                body: JSON.stringify({
                    filename: filename,
                    points: scaledPoints,
                    smoothing: 3
                })
            }
        );

        window.PyPotteryUtils.hideLoading();

        if (response.success) {
            document.getElementById('postprocess-transformed').src = response.cropped_image;
            document.getElementById('postprocess-transformed').style.display = 'block';
            document.getElementById('postprocess-placeholder')?.style.setProperty('display', 'none');

            cancelCropMode();
            window.PyPotteryUtils.showToast('Freehand crop applied successfully', 'success');
        } else {
            window.PyPotteryUtils.showToast(response.error || 'Crop failed', 'error');
        }
    } catch (error) {
        window.PyPotteryUtils.hideLoading();
        console.error('Error applying freehand crop:', error);
        window.PyPotteryUtils.showToast(error.message, 'error');
    }
}

// Export for use by main.js
window.loadPostprocessCards = loadProjectCards;
window.handleExclusionToggle = handleExclusionToggle;

