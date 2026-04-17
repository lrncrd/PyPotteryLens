// Tabular Tab JavaScript - Project-aware version
// Uses window.PyPotteryUtils.* functions directly

// State
let tabularState = {
    currentProject: null,
    cards: [],
    currentIndex: 0,
    totalCards: 0,
    tableData: [],
    columns: [],
    currentImageName: null,  // Track current image for saving
    imageList: [],  // List of all images with reviewed status
    isReviewed: false,  // Current image reviewed status
    fullImageUrl: null  // Full resolution image URL for zoom
};

document.addEventListener('DOMContentLoaded', () => {
    setupTabularListeners();
    loadCurrentProject();
    
    // Listen for project changes
    window.addEventListener('projectChanged', (e) => {
        const project = e.detail && e.detail.project ? e.detail.project : null;
        tabularState.currentProject = project;
        loadProjectCards();
    });
});

function loadCurrentProject() {
    if (window.projectManager && window.projectManager.getCurrentProject) {
        tabularState.currentProject = window.projectManager.getCurrentProject();
    } else {
        const pid = localStorage.getItem('currentProjectId');
        const pname = localStorage.getItem('currentProjectName');
        if (pid) {
            tabularState.currentProject = { project_id: pid, project_name: pname || 'Unnamed' };
        }
    }
    
    if (tabularState.currentProject) {
        loadProjectCards();
    }
}

function setupTabularListeners() {
    // Navigation
    document.getElementById('tabular-prev')?.addEventListener('click', () => navigateTabular(-1));
    document.getElementById('tabular-next')?.addEventListener('click', () => navigateTabular(1));
    document.getElementById('tabular-goto-btn')?.addEventListener('click', handleTabularGoto);

    // Add column
    document.getElementById('add-column-btn')?.addEventListener('click', handleAddColumn);
    
    // AI bibliographic extraction
    document.getElementById('ai-bibliographic-btn')?.addEventListener('click', handleAiBibliographic);
    document.getElementById('ai-bibliographic-batch-btn')?.addEventListener('click', handleAiBibliographicBatch);

    // Export to CSV - use combined export endpoint
    document.getElementById('export-csv-btn')?.addEventListener('click', exportCombinedCSV);
    
    // Mark as reviewed
    document.getElementById('tabular-mark-reviewed-btn')?.addEventListener('click', markAsReviewed);
    
    // Setup magnifying glass zoom on hover
    setupMagnifyingGlass();
}

async function loadProjectCards() {
    if (!tabularState.currentProject || !tabularState.currentProject.project_id) {
        showEmptyState('No project selected', 'Select a project from the Project Manager tab');
        return;
    }
    
    try {
        window.PyPotteryUtils.showLoading('Loading project cards...');
        
        const response = await window.PyPotteryUtils.apiRequest(
            `/api/projects/${tabularState.currentProject.project_id}/cards`
        );
        
        window.PyPotteryUtils.hideLoading();
        
        if (response.success) {
            tabularState.cards = response.cards || [];
            tabularState.totalCards = response.total || 0;
            
            if (tabularState.totalCards === 0) {
                showEmptyState('No cards found', 'Extract cards from masks in the Annotation tab first');
                return;
            }
            
            // Load first card data
            await loadTabularData(0);
        } else {
            showEmptyState('Error loading cards', response.error);
        }
        
    } catch (error) {
        window.PyPotteryUtils.hideLoading();
        console.error('Error loading project cards:', error);
        showEmptyState('Error', error.message);
    }
}

function showEmptyState(title, message) {
    const canvas = document.getElementById('tabular-canvas');
    const tableContainer = document.getElementById('tabular-table-container');
    
    if (canvas) {
        const ctx = canvas.getContext('2d');
        canvas.width = 400;
        canvas.height = 300;
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = '#64748b';
        ctx.font = '16px Arial';
        ctx.textAlign = 'center';
        ctx.fillText(title, canvas.width / 2, canvas.height / 2 - 20);
        ctx.fillText(message, canvas.width / 2, canvas.height / 2 + 20);
    }
    
    if (tableContainer) {
        tableContainer.innerHTML = `
            <div style="padding: 2rem; text-align: center; color: #64748b;">
                <h3>${title}</h3>
                <p>${message}</p>
            </div>
        `;
    }
}

async function loadTabularData(imgNum) {
    if (!tabularState.currentProject || !tabularState.currentProject.project_id) {
        return;
    }

    try {
        window.PyPotteryUtils.showLoading('Loading card data...');
        
        const response = await window.PyPotteryUtils.apiRequest(
            `/api/projects/${tabularState.currentProject.project_id}/tabular/load`,
            {
                method: 'POST',
                body: JSON.stringify({
                    img_num: imgNum
                })
            }
        );

        window.PyPotteryUtils.hideLoading();

        if (response.success) {
            tabularState.currentIndex = response.current;
            tabularState.totalCards = response.total;
            tabularState.tableData = response.table;
            tabularState.columns = response.columns;

            displayTabularData(response);
        } else {
            window.PyPotteryUtils.showToast('Failed to load data', 'error');
        }
    } catch (error) {
        window.PyPotteryUtils.hideLoading();
        console.error('Error loading tabular data:', error);
        window.PyPotteryUtils.showToast(error.message, 'error');
    }
}

function displayTabularData(data) {
    // Store image_name and other metadata
    tabularState.currentImageName = data.image_name;
    tabularState.imageList = data.image_list || [];
    tabularState.isReviewed = data.is_reviewed || false;
    tabularState.fullImageUrl = data.full_image_url;
    tabularState.currentIndex = data.current;
    tabularState.totalCards = data.total;

    // Update reviewed status button
    updateReviewedButton();
    
    // Display image list sidebar
    displayImageList();

    // Display image with annotations
    if (data.image) {
        displayAnnotatedImage(data.image, data.annotations);
    }

    // Display table
    if (data.table && data.columns) {
        displayTable(data.table, data.columns);
    }
}

function displayAnnotatedImage(imageData, annotations) {
    const canvas = document.getElementById('tabular-canvas');
    if (!canvas) {
        console.error('[Tabular] Canvas not found!');
        return;
    }

    const ctx = canvas.getContext('2d');
    const img = new Image();

    img.onload = () => {
        canvas.width = img.width;
        canvas.height = img.height;
        ctx.drawImage(img, 0, 0);

        // Draw annotations (bounding boxes)
        if (annotations && annotations.length > 0) {
            ctx.strokeStyle = '#2563eb';
            ctx.lineWidth = 3;
            ctx.font = 'bold 16px Arial';
            ctx.fillStyle = '#2563eb';

            annotations.forEach(annot => {
                const bbox = annot.bbox;  // [x1, y1, x2, y2]
                const label = annot.label;
                
                const [x1, y1, x2, y2] = bbox;
                const width = x2 - x1;
                const height = y2 - y1;
                
                // Draw rectangle
                ctx.strokeRect(x1, y1, width, height);
                
                // Draw label background
                const labelText = `ID: ${label}`;
                const textMetrics = ctx.measureText(labelText);
                const textWidth = textMetrics.width;
                const textHeight = 20;
                
                ctx.fillStyle = '#2563eb';
                ctx.fillRect(x1, y1 - textHeight - 4, textWidth + 8, textHeight + 4);
                
                // Draw label text
                ctx.fillStyle = '#ffffff';
                ctx.fillText(labelText, x1 + 4, y1 - 8);
                
                // Reset fill style for next annotation
                ctx.fillStyle = '#2563eb';
            });
        }
    };

    img.onerror = (error) => {
        console.error('[Tabular] Image load error:', error);
    };

    img.src = imageData;
}

function displayTable(data, columns) {
    const headerEl = document.getElementById('table-header');
    const bodyEl = document.getElementById('table-body');

    if (!headerEl || !bodyEl) return;

    // Clear existing content
    headerEl.innerHTML = '';
    bodyEl.innerHTML = '';

    // Create header
    const headerRow = document.createElement('tr');
    columns.forEach(col => {
        const th = document.createElement('th');
        th.textContent = col;
        headerRow.appendChild(th);
    });
    headerEl.appendChild(headerRow);

    // Create body
    data.forEach((row, rowIndex) => {
        const tr = document.createElement('tr');
        columns.forEach(col => {
            const td = document.createElement('td');
            const input = document.createElement('input');
            input.type = 'text';
            input.value = row[col] || '';
            input.dataset.row = rowIndex;
            input.dataset.col = col;
            input.addEventListener('change', handleCellChange);
            td.appendChild(input);
            tr.appendChild(td);
        });
        bodyEl.appendChild(tr);
    });
}

async function handleCellChange(e) {
    const rowIndex = parseInt(e.target.dataset.row);
    const column = e.target.dataset.col;
    const value = e.target.value;

    // Update local state
    if (tabularState.tableData[rowIndex]) {
        tabularState.tableData[rowIndex][column] = value;
    }

    // Auto-save
    await saveTabularData();
}

async function saveTabularData() {
    if (!tabularState.currentProject || !tabularState.tableData.length) return;

    try {
        await window.PyPotteryUtils.apiRequest(
            `/api/projects/${tabularState.currentProject.project_id}/tabular/save`,
            {
                method: 'POST',
                body: JSON.stringify({
                    table: tabularState.tableData,
                    image_name: tabularState.currentImageName  // Include current image name
                })
            }
        );
        console.log('Table auto-saved');
    } catch (error) {
        console.error('Error saving table:', error);
    }
}

async function handleAddColumn() {
    const input = document.getElementById('new-column-name');
    const columnName = input.value.trim();

    if (!columnName) {
        window.PyPotteryUtils.showToast('Please enter a column name', 'warning');
        return;
    }

    try {
        const response = await window.PyPotteryUtils.apiRequest('/api/tabular/add-column', {
            method: 'POST',
            body: JSON.stringify({
                column_name: columnName,
                table: tabularState.tableData
            })
        });

        if (response.success) {
            tabularState.tableData = response.table;
            tabularState.columns = response.columns;
            displayTable(tabularState.tableData, tabularState.columns);
            input.value = '';
            window.PyPotteryUtils.showToast('Column added successfully', 'success');
            await saveTabularData();
        } else {
            window.PyPotteryUtils.showToast('Failed to add column', 'error');
        }
    } catch (error) {
        console.error('Error adding column:', error);
        window.PyPotteryUtils.showToast(error.message, 'error');
    }
}

function navigateTabular(direction) {
    const newIndex = tabularState.currentIndex + direction;
    if (newIndex >= 0 && newIndex < tabularState.totalCards) {
        loadTabularData(newIndex);
    }
}

function handleTabularGoto() {
    const input = document.getElementById('tabular-goto');
    if (!input) return;
    
    const index = parseInt(input.value);

    if (!isNaN(index) && index >= 0 && index < tabularState.totalCards) {
        loadTabularData(index);
        input.value = '';
    } else {
        window.PyPotteryUtils.showToast('Invalid card number', 'warning');
    }
}

function exportToCSV() {
    if (!tabularState.tableData || tabularState.tableData.length === 0) {
        window.PyPotteryUtils.showToast('No data to export', 'warning');
        return;
    }
    
    try {
        // Convert table data to CSV
        const columns = tabularState.columns;
        const rows = tabularState.tableData;
        
        // Create CSV header
        let csv = columns.join(',') + '\n';
        
        // Add rows
        rows.forEach(row => {
            const values = columns.map(col => {
                const value = row[col] || '';
                // Escape quotes and wrap in quotes if contains comma
                return value.includes(',') ? `"${value.replace(/"/g, '""')}"` : value;
            });
            csv += values.join(',') + '\n';
        });
        
        // Create download link
        const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
        const link = document.createElement('a');
        const url = URL.createObjectURL(blob);
        
        const projectName = tabularState.currentProject ? tabularState.currentProject.project_name : 'project';
        link.setAttribute('href', url);
        link.setAttribute('download', `${projectName}_tabular_data.csv`);
        link.style.visibility = 'hidden';
        
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        
        window.PyPotteryUtils.showToast('CSV exported successfully', 'success');
        
    } catch (error) {
        console.error('Error exporting CSV:', error);
        window.PyPotteryUtils.showToast('Failed to export CSV', 'error');
    }
}

async function exportCombinedCSV() {
    if (!tabularState.currentProject || !tabularState.currentProject.project_id) {
        window.PyPotteryUtils.showToast('No project selected', 'warning');
        return;
    }
    
    try {
        window.PyPotteryUtils.showLoading('Saving combined CSV to project folder...');
        
        const response = await window.PyPotteryUtils.apiRequest(
            `/api/projects/${tabularState.currentProject.project_id}/tabular/export`,
            {
                method: 'POST',
                body: JSON.stringify({})
            }
        );
        
        window.PyPotteryUtils.hideLoading();
        
        if (response.success) {
            window.PyPotteryUtils.showToast(`CSV saved: ${response.path}`, 'success');
        } else {
            window.PyPotteryUtils.showToast(response.error || 'Export failed', 'error');
        }
    } catch (error) {
        window.PyPotteryUtils.hideLoading();
        console.error('Error exporting CSV:', error);
        window.PyPotteryUtils.showToast('Failed to export CSV', 'error');
    }
}

function displayImageList() {
    const listContainer = document.getElementById('tabular-image-list');
    if (!listContainer) return;
    
    listContainer.innerHTML = '';
    
    if (!tabularState.imageList || tabularState.imageList.length === 0) {
        listContainer.innerHTML = '<div style="padding: 1rem; color: #64748b;">No images</div>';
        return;
    }
    
    tabularState.imageList.forEach((item, index) => {
        const div = document.createElement('div');
        div.className = 'tabular-image-item';
        if (index === tabularState.currentIndex) {
            div.classList.add('active');
        }
        if (item.reviewed) {
            div.classList.add('reviewed');
        }
        
        div.innerHTML = `
            <span class="image-name">${item.image_name}</span>
            <span class="status-icon">${item.reviewed ? '✅' : '⚪'}</span>
        `;
        
        div.addEventListener('click', () => {
            loadTabularData(index);
        });
        
        listContainer.appendChild(div);
    });
}

function updateReviewedButton() {
    const btn = document.getElementById('tabular-mark-reviewed-btn');
    if (!btn) return;
    
    if (tabularState.isReviewed) {
        btn.textContent = '✅ Reviewed';
        btn.disabled = true;
        btn.style.opacity = '0.6';
    } else {
        btn.textContent = '👁️ Mark as Reviewed';
        btn.disabled = false;
        btn.style.opacity = '1';
    }
}

async function markAsReviewed() {
    if (!tabularState.currentProject || !tabularState.currentImageName) return;
    
    try {
        const response = await window.PyPotteryUtils.apiRequest(
            `/api/projects/${tabularState.currentProject.project_id}/reviewed`,
            {
                method: 'POST',
                body: JSON.stringify({
                    image_name: tabularState.currentImageName
                })
            }
        );
        
        if (response.success) {
            tabularState.isReviewed = true;
            updateReviewedButton();
            
            // Update image list
            const item = tabularState.imageList.find(i => i.image_name === tabularState.currentImageName);
            if (item) {
                item.reviewed = true;
                displayImageList();
            }
            
            window.PyPotteryUtils.showToast('Marked as reviewed', 'success');
        }
    } catch (error) {
        console.error('Error marking as reviewed:', error);
        window.PyPotteryUtils.showToast('Failed to mark as reviewed', 'error');
    }
}

function openZoomModal() {
    if (!tabularState.fullImageUrl) {
        window.PyPotteryUtils.showToast('Full resolution image not available', 'warning');
        return;
    }
    
    // Create modal
    const modal = document.createElement('div');
    modal.id = 'zoom-modal';
    modal.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.9);
        z-index: 10000;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: zoom-out;
    `;
    
    const img = document.createElement('img');
    img.src = tabularState.fullImageUrl;
    img.style.cssText = `
        max-width: 95%;
        max-height: 95%;
        object-fit: contain;
    `;
    
    modal.appendChild(img);
    
    // Close on click
    modal.addEventListener('click', () => {
        document.body.removeChild(modal);
    });
    
    document.body.appendChild(modal);
}

function setupMagnifyingGlass() {
    const canvas = document.getElementById('tabular-canvas');
    const zoomHint = document.querySelector('.zoom-hint');
    if (!canvas) return;
    
    // Simple hover zoom: click to toggle between normal and zoomed view
    canvas.style.cursor = 'zoom-in';
    canvas.style.transition = 'transform 0.3s ease';
    canvas.style.transformOrigin = 'center center';
    
    let isZoomed = false;
    
    canvas.addEventListener('click', (e) => {
        if (isZoomed) {
            // Zoom out
            canvas.style.transform = 'scale(1)';
            canvas.style.cursor = 'zoom-in';
            canvas.style.position = 'relative';
            canvas.style.zIndex = '1';
            if (zoomHint) zoomHint.textContent = 'Click to zoom';
            isZoomed = false;
        } else {
            // Zoom in
            const rect = canvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            
            // Calculate zoom origin as percentage
            const originX = (x / rect.width) * 100;
            const originY = (y / rect.height) * 100;
            
            canvas.style.transformOrigin = `${originX}% ${originY}%`;
            canvas.style.transform = 'scale(2)';
            canvas.style.cursor = 'zoom-out';
            canvas.style.position = 'relative';
            canvas.style.zIndex = '100';
            if (zoomHint) zoomHint.textContent = 'Click to zoom out';
            isZoomed = true;
        }
    });
    
    // Reset zoom when changing image
    const observer = new MutationObserver(() => {
        if (isZoomed) {
            canvas.style.transform = 'scale(1)';
            canvas.style.cursor = 'zoom-in';
            canvas.style.position = 'relative';
            canvas.style.zIndex = '1';
            if (zoomHint) zoomHint.textContent = 'Click to zoom';
            isZoomed = false;
        }
    });
    
    observer.observe(canvas, { attributes: true, attributeFilter: ['src'] });
}

// Export for use by main.js
window.refreshTabular = loadProjectCards;

/* =========================================================
 * GPU / download confirmation dialog
 * ========================================================= */
async function checkAiRequirements() {
    const res = await window.PyPotteryUtils.apiRequest('/api/check-ai-requirements');
    return res;
}

function showAiConfirmDialog(requirements, onConfirm) {
    // Remove any existing dialog
    document.getElementById('ai-requirements-dialog')?.remove();

    const { cuda_available, vram_gb, gpu_name, model_cached, meets_requirements } = requirements;

    const gpuLine = cuda_available
        ? `<p>GPU detected: <strong>${gpu_name}</strong> (${vram_gb.toFixed(1)} GB VRAM)</p>`
        : `<p style="color:#ef4444;">No CUDA GPU detected on this system.</p>`;

    const downloadNote = model_cached
        ? `<p style="color:#22c55e;">✅ Model already cached locally — no download needed.</p>`
        : `<p style="color:#f59e0b;">⚠️ The Gemma 4 E2B model (~10 GB) will be downloaded the first time. Make sure you have a stable internet connection and enough disk space.</p>`;

    const blocker = !meets_requirements
        ? `<p style="color:#ef4444; font-weight:600;">This feature requires a CUDA GPU with at least 6 GB of VRAM. Your system does not meet this requirement.</p>`
        : '';

    const overlay = document.createElement('div');
    overlay.id = 'ai-requirements-dialog';
    overlay.style.cssText = `
        position:fixed; inset:0; background:rgba(0,0,0,0.6); z-index:20000;
        display:flex; align-items:center; justify-content:center;
    `;
    overlay.innerHTML = `
        <div style="background:#1e293b; color:#e2e8f0; border-radius:12px; padding:2rem;
                    max-width:480px; width:90%; box-shadow:0 20px 60px rgba(0,0,0,0.5);">
            <h3 style="margin:0 0 1rem; font-size:1.2rem;">🤖 AI Bibliographic Extraction</h3>
            ${gpuLine}
            ${downloadNote}
            ${blocker}
            <p style="color:#94a3b8; font-size:0.85rem; margin-top:0.5rem;">
                The model uses the Gemma 4 E2B multimodal architecture from Google and runs
                entirely on your local machine — no data is sent to the cloud.
            </p>
            <div style="display:flex; justify-content:flex-end; gap:0.75rem; margin-top:1.5rem;">
                <button id="ai-dialog-cancel" class="btn btn-secondary">Cancel</button>
                <button id="ai-dialog-confirm" class="btn btn-primary"
                    ${meets_requirements ? '' : 'disabled'}>
                    ${model_cached ? 'Run Extraction' : 'Download & Run'}
                </button>
            </div>
            <div id="ai-download-progress-wrapper" style="display:none; margin-top:1rem;">
                <p id="ai-download-progress-label" style="font-size:0.85rem; color:#94a3b8; margin:0 0 0.4rem;"></p>
                <div style="background:#334155; border-radius:6px; overflow:hidden; height:12px;">
                    <div id="ai-download-progress-bar"
                         style="height:100%; background:#6366f1; transition:width 0.4s; width:0%"></div>
                </div>
            </div>
        </div>
    `;

    document.body.appendChild(overlay);

    document.getElementById('ai-dialog-cancel').addEventListener('click', () => overlay.remove());
    document.getElementById('ai-dialog-confirm').addEventListener('click', () => {
        document.getElementById('ai-dialog-confirm').disabled = true;
        document.getElementById('ai-dialog-cancel').disabled = true;
        if (!requirements.model_cached) {
            document.getElementById('ai-download-progress-wrapper').style.display = 'block';
        }
        onConfirm(overlay);
    });
}

function startProgressPolling(labelEl, barEl, stopSignal) {
    const interval = setInterval(async () => {
        if (stopSignal.stopped) { clearInterval(interval); return; }
        try {
            const prog = await window.PyPotteryUtils.apiRequest('/api/progress');
            if (prog && prog.active) {
                labelEl.textContent = prog.message || '';
                barEl.style.width = (prog.percent || 0) + '%';
            }
        } catch (_) { /* ignore polling errors */ }
    }, 800);
    return interval;
}

async function handleAiBibliographic() {
    if (!tabularState.currentProject || !tabularState.currentProject.project_id) {
        window.PyPotteryUtils.showToast('No project selected', 'warning');
        return;
    }

    const statusEl = document.getElementById('ai-bibliographic-status');
    const btn = document.getElementById('ai-bibliographic-btn');

    // Check GPU requirements first
    let requirements;
    try {
        requirements = await checkAiRequirements();
    } catch (e) {
        window.PyPotteryUtils.showToast('Could not check system requirements', 'error');
        return;
    }

    showAiConfirmDialog(requirements, async (overlay) => {
        const labelEl = document.getElementById('ai-download-progress-label');
        const barEl = document.getElementById('ai-download-progress-bar');
        const stopSignal = { stopped: false };
        const pollInterval = startProgressPolling(labelEl, barEl, stopSignal);

        btn.disabled = true;
        if (statusEl) statusEl.textContent = '⏳ Loading model and analysing...';
        window.PyPotteryUtils.showLoading('Extracting references with Gemma 4 AI...');

        try {
            const response = await window.PyPotteryUtils.apiRequest(
                `/api/projects/${tabularState.currentProject.project_id}/tabular/ai-bibliographic`,
                {
                    method: 'POST',
                    body: JSON.stringify({ img_num: tabularState.currentIndex })
                }
            );

            stopSignal.stopped = true;
            clearInterval(pollInterval);
            window.PyPotteryUtils.hideLoading();
            overlay.remove();

            if (response.success) {
                tabularState.tableData = response.table;
                tabularState.columns = response.columns;
                displayTable(response.table, response.columns);
                if (statusEl) statusEl.textContent = '✅ References extracted successfully';
                window.PyPotteryUtils.showToast('Bibliographic references extracted!', 'success');
            } else {
                if (statusEl) statusEl.textContent = '❌ Error: ' + (response.error || 'unknown');
                window.PyPotteryUtils.showToast(response.error || 'AI Error', 'error');
            }
        } catch (error) {
            stopSignal.stopped = true;
            clearInterval(pollInterval);
            window.PyPotteryUtils.hideLoading();
            overlay.remove();
            if (statusEl) statusEl.textContent = '❌ ' + error.message;
            window.PyPotteryUtils.showToast(error.message, 'error');
            console.error('[AI Bibliographic] Error:', error);
        } finally {
            btn.disabled = false;
        }
    });
}

async function handleAiBibliographicBatch() {
    if (!tabularState.currentProject || !tabularState.currentProject.project_id) {
        window.PyPotteryUtils.showToast('No project selected', 'warning');
        return;
    }

    const statusEl = document.getElementById('ai-bibliographic-status');
    const btn = document.getElementById('ai-bibliographic-batch-btn');

    // Check GPU requirements first
    let requirements;
    try {
        requirements = await checkAiRequirements();
    } catch (e) {
        window.PyPotteryUtils.showToast('Could not check system requirements', 'error');
        return;
    }

    // Modify dialog title for batch mode
    const originalTitle = '🤖 AI Bibliographic Extraction';
    requirements._batchMode = true;

    showAiConfirmDialog(requirements, async (overlay) => {
        const labelEl = document.getElementById('ai-download-progress-label');
        const barEl = document.getElementById('ai-download-progress-bar');
        // Show progress wrapper for batch mode even if model is cached
        document.getElementById('ai-download-progress-wrapper').style.display = 'block';
        const stopSignal = { stopped: false };
        const pollInterval = startProgressPolling(labelEl, barEl, stopSignal);

        btn.disabled = true;
        if (statusEl) statusEl.textContent = '⏳ Running batch extraction...';
        window.PyPotteryUtils.showLoading('Batch AI extraction in progress...');

        try {
            const response = await window.PyPotteryUtils.apiRequest(
                `/api/projects/${tabularState.currentProject.project_id}/tabular/ai-bibliographic-batch`,
                { method: 'POST', body: JSON.stringify({}) }
            );

            stopSignal.stopped = true;
            clearInterval(pollInterval);
            window.PyPotteryUtils.hideLoading();
            overlay.remove();

            if (response.success) {
                const errMsg = response.errors && response.errors.length
                    ? ` (${response.errors.length} errors)` : '';
                if (statusEl) statusEl.textContent = `✅ Batch complete: ${response.processed} images${errMsg}`;
                window.PyPotteryUtils.showToast(`Batch extraction done: ${response.processed} images${errMsg}`, 'success');
                // Reload current card to reflect new data
                await loadTabularData(tabularState.currentIndex);
            } else {
                if (statusEl) statusEl.textContent = '❌ Batch error: ' + (response.error || 'unknown');
                window.PyPotteryUtils.showToast(response.error || 'Batch AI Error', 'error');
            }
        } catch (error) {
            stopSignal.stopped = true;
            clearInterval(pollInterval);
            window.PyPotteryUtils.hideLoading();
            overlay.remove();
            if (statusEl) statusEl.textContent = '❌ ' + error.message;
            window.PyPotteryUtils.showToast(error.message, 'error');
            console.error('[AI Batch] Error:', error);
        } finally {
            btn.disabled = false;
        }
    });
}
