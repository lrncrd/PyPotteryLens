// Post Processing Tab JavaScript - Project-aware version
// Uses window.PyPotteryUtils.* functions directly

// State
let postprocessState = {
    currentProject: null,
    cards: [],
    currentIndex: 0,
    totalImages: 0
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
        
        const item = document.createElement('div');
        item.className = 'image-list-item';
        item.dataset.index = index;
        item.dataset.tooltip = filename;
        
        item.innerHTML = `
            <span class="image-number">${index}</span>
            <span class="image-name" title="${filename}">${filename}</span>
        `;
        
        item.addEventListener('click', () => {
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

// Export for use by main.js
window.loadPostprocessCards = loadProjectCards;

