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

    // AI backend toggle panel
    document.getElementById('ai-backend-toggle-btn')?.addEventListener('click', () => {
        const panel = document.getElementById('ai-backend-panel');
        if (panel) panel.style.display = panel.style.display === 'none' ? 'block' : 'none';
    });

    // Show/hide OpenRouter config based on radio selection
    document.querySelectorAll('input[name="ai-backend-choice"]').forEach(radio => {
        radio.addEventListener('change', () => {
            const isOpenRouter = document.getElementById('ai-backend-openrouter')?.checked;
            const configEl = document.getElementById('ai-openrouter-config');
            if (configEl) configEl.style.display = isOpenRouter ? 'block' : 'none';
            localStorage.setItem('pypottery_ai_backend', isOpenRouter ? 'openrouter' : 'local');
        });
    });

    // Restore saved backend choice from localStorage
    const _savedBackend = localStorage.getItem('pypottery_ai_backend');
    if (_savedBackend === 'openrouter') {
        const radioEl = document.getElementById('ai-backend-openrouter');
        if (radioEl) {
            radioEl.checked = true;
            const configEl = document.getElementById('ai-openrouter-config');
            if (configEl) configEl.style.display = 'block';
        }
    }

    // Persist OpenRouter API key and model in sessionStorage (not localStorage for security)
    const _orKey = document.getElementById('ai-openrouter-apikey');
    const _orModel = document.getElementById('ai-openrouter-model');
    if (_orKey) {
        const _savedKey = sessionStorage.getItem('pypottery_or_apikey');
        if (_savedKey) _orKey.value = _savedKey;
        _orKey.addEventListener('input', () => sessionStorage.setItem('pypottery_or_apikey', _orKey.value));
    }
    if (_orModel) {
        const _savedModel = localStorage.getItem('pypottery_or_model');
        if (_savedModel) _orModel.value = _savedModel;
        _orModel.addEventListener('input', () => localStorage.setItem('pypottery_or_model', _orModel.value));
    }

    // Prompt customisation panel toggle
    document.getElementById('ai-prompt-toggle-btn')?.addEventListener('click', () => {
        const panel = document.getElementById('ai-prompt-panel');
        if (panel) panel.style.display = panel.style.display === 'none' ? 'block' : 'none';
    });
    document.getElementById('ai-prompt-reset-btn')?.addEventListener('click', () => {
        const ta = document.getElementById('ai-prompt-suffix');
        if (ta) ta.value = '';
        localStorage.removeItem('pypottery_ai_prompt');
        _showPromptSaveIndicator('Reset');
    });

    // Load saved prompt from localStorage and auto-save on change
    const _promptTa = document.getElementById('ai-prompt-suffix');
    if (_promptTa) {
        const _saved = localStorage.getItem('pypottery_ai_prompt');
        if (_saved) _promptTa.value = _saved;

        let _promptSaveTimer = null;
        _promptTa.addEventListener('input', () => {
            clearTimeout(_promptSaveTimer);
            _promptSaveTimer = setTimeout(() => {
                localStorage.setItem('pypottery_ai_prompt', _promptTa.value);
                _showPromptSaveIndicator('Saved');
            }, 600);
        });
    }

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
    // Close any open bbox editor when switching images
    closeBboxEditor();

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

function _drawAnnotations(ctx, img, annotations, hoveredLabel) {
    ctx.drawImage(img, 0, 0);
    if (!annotations || annotations.length === 0) return;

    ctx.font = 'bold 16px Arial';

    annotations.forEach(annot => {
        const [x1, y1, x2, y2] = annot.bbox;
        const label = annot.label;
        const hovered = (label === hoveredLabel);
        const color = hovered ? '#f97316' : '#2563eb';   // orange when hovered, blue otherwise

        // Semi-transparent fill on hover
        if (hovered) {
            ctx.fillStyle = 'rgba(249, 115, 22, 0.18)';
            ctx.fillRect(x1, y1, x2 - x1, y2 - y1);
        }

        // Box stroke (thicker on hover)
        ctx.strokeStyle = color;
        ctx.lineWidth = hovered ? 5 : 3;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

        // Label tag
        const labelText = `ID: ${label}`;
        const textWidth = ctx.measureText(labelText).width;
        const textHeight = 20;
        ctx.fillStyle = color;
        ctx.fillRect(x1, y1 - textHeight - 4, textWidth + 8, textHeight + 4);
        ctx.fillStyle = '#ffffff';
        ctx.fillText(labelText, x1 + 4, y1 - 8);
    });
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

        // Store image reference so hover redraws can use it
        tabularState._bboxImg = img;
        tabularState.annotations = annotations || [];

        _drawAnnotations(ctx, img, tabularState.annotations, null);
        _setupBboxInteraction(canvas, tabularState.annotations);
    };

    img.onerror = (error) => {
        console.error('[Tabular] Image load error:', error);
    };

    img.src = imageData;
}

function _setupBboxInteraction(canvas, annotations) {
    // Remove previous handlers to avoid accumulation
    if (tabularState._bboxClickHandler) {
        canvas.removeEventListener('click', tabularState._bboxClickHandler, true);
    }
    if (tabularState._bboxMoveHandler) {
        canvas.removeEventListener('mousemove', tabularState._bboxMoveHandler);
    }

    if (!annotations || annotations.length === 0) return;

    // Use capture=true so our handler fires before the zoom handler;
    // if a bbox is hit we stop propagation to prevent zoom.
    tabularState._bboxClickHandler = (e) => {
        const hit = _hitTestBbox(e, canvas, annotations);
        if (!hit) { closeBboxEditor(); return; }
        e.stopImmediatePropagation();
        const rowIndex = tabularState.tableData.findIndex(r => String(r.ID) === String(hit.label));
        if (rowIndex === -1) return;
        highlightTableRow(String(hit.label));
        showBboxEditor(rowIndex, hit.label, e.clientX, e.clientY);
    };
    canvas.addEventListener('click', tabularState._bboxClickHandler, true);

    // Mousemove: change cursor + redraw with hover highlight
    let _lastHovered = null;
    const ctx = canvas.getContext('2d');
    tabularState._bboxMoveHandler = (e) => {
        const hit = _hitTestBbox(e, canvas, annotations);
        const hLabel = hit ? hit.label : null;
        canvas.style.cursor = hit ? 'pointer' : 'zoom-in';
        if (hLabel !== _lastHovered) {
            _lastHovered = hLabel;
            if (tabularState._bboxImg) {
                _drawAnnotations(ctx, tabularState._bboxImg, annotations, hLabel);
            }
        }
    };
    canvas.addEventListener('mousemove', tabularState._bboxMoveHandler);
}

function _hitTestBbox(e, canvas, annotations) {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / canvas.clientWidth;
    const scaleY = canvas.height / canvas.clientHeight;
    const cx = (e.clientX - rect.left) * scaleX;
    const cy = (e.clientY - rect.top) * scaleY;
    for (const annot of annotations) {
        const [x1, y1, x2, y2] = annot.bbox;
        if (cx >= x1 && cx <= x2 && cy >= y1 && cy <= y2) return annot;
    }
    return null;
}

function showBboxEditor(rowIndex, label, clientX, clientY) {
    closeBboxEditor();

    const row = tabularState.tableData[rowIndex];
    if (!row) return;

    const editableCols = tabularState.columns.filter(c => c !== 'ID');

    const el = document.createElement('div');
    el.id = 'bbox-editor';
    el.className = 'bbox-editor';

    // Header
    const title = document.createElement('div');
    title.className = 'bbox-editor-title';
    title.innerHTML = `<span>🏺 ID: ${label}</span>`;
    const closeBtn = document.createElement('button');
    closeBtn.className = 'bbox-editor-close';
    closeBtn.textContent = '✕';
    closeBtn.addEventListener('click', closeBboxEditor);
    title.appendChild(closeBtn);
    el.appendChild(title);

    // Fields
    const fieldsDiv = document.createElement('div');
    fieldsDiv.className = 'bbox-editor-fields';
    editableCols.forEach(col => {
        const fieldDiv = document.createElement('div');
        fieldDiv.className = 'bbox-editor-field';
        const lbl = document.createElement('label');
        lbl.textContent = col;
        const inp = document.createElement('input');
        inp.type = 'text';
        inp.value = row[col] || '';
        inp.dataset.col = col;
        inp.dataset.row = rowIndex;
        // Live update state on change
        inp.addEventListener('change', (e) => {
            if (tabularState.tableData[rowIndex]) {
                tabularState.tableData[rowIndex][col] = e.target.value;
                // Sync the main table input if visible
                const tableInput = document.querySelector(
                    `#table-body input[data-row="${rowIndex}"][data-col="${col}"]`
                );
                if (tableInput) tableInput.value = e.target.value;
            }
        });
        fieldDiv.appendChild(lbl);
        fieldDiv.appendChild(inp);
        fieldsDiv.appendChild(fieldDiv);
    });
    el.appendChild(fieldsDiv);

    // Save button
    const saveBtn = document.createElement('button');
    saveBtn.className = 'bbox-editor-save';
    saveBtn.textContent = '✓ Save';
    saveBtn.addEventListener('click', async () => {
        await saveTabularData();
        closeBboxEditor();
    });
    el.appendChild(saveBtn);

    // Position (viewport-relative, clamped to stay visible)
    document.body.appendChild(el);
    const pw = el.offsetWidth, ph = el.offsetHeight;
    let left = clientX + 12, top = clientY - 20;
    if (left + pw > window.innerWidth - 8) left = clientX - pw - 12;
    if (top + ph > window.innerHeight - 8) top = window.innerHeight - ph - 8;
    if (top < 8) top = 8;
    el.style.left = `${left}px`;
    el.style.top = `${top}px`;

    // Close on outside click
    setTimeout(() => {
        document.addEventListener('click', _outsideEditorClick, true);
    }, 0);
}

function _outsideEditorClick(e) {
    const el = document.getElementById('bbox-editor');
    if (el && !el.contains(e.target)) {
        closeBboxEditor();
    }
}

function closeBboxEditor() {
    document.removeEventListener('click', _outsideEditorClick, true);
    const el = document.getElementById('bbox-editor');
    if (el) el.remove();
    // Clear table row highlight
    document.querySelectorAll('.data-table tr.bbox-highlighted').forEach(tr => {
        tr.classList.remove('bbox-highlighted');
    });
}

function highlightTableRow(rowId) {
    // Remove previous highlight
    document.querySelectorAll('.data-table tr.bbox-highlighted').forEach(tr => {
        tr.classList.remove('bbox-highlighted');
    });
    const tr = document.querySelector(`#table-body tr[data-row-id="${rowId}"]`);
    if (tr) {
        tr.classList.add('bbox-highlighted');
        tr.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
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
        tr.dataset.rowId = row.ID || rowIndex;
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

/** Return the user-defined prompt context, or empty string if not set. */
function getPromptSuffix() {
    const ta = document.getElementById('ai-prompt-suffix');
    return ta ? ta.value.trim() : '';
}

/** Return current AI backend params to include in every AI request body. */
function getAiBackendParams() {
    const isOpenRouter = document.getElementById('ai-backend-openrouter')?.checked;
    if (!isOpenRouter) {
        return { ai_backend: 'local' };
    }
    return {
        ai_backend: 'openrouter',
        openrouter_api_key: document.getElementById('ai-openrouter-apikey')?.value.trim() || '',
        openrouter_model: document.getElementById('ai-openrouter-model')?.value.trim() || 'google/gemini-flash-1.5',
    };
}

function showVisionUnsupportedDialog(modelName) {
    document.getElementById('ai-vision-unsupported-dialog')?.remove();
    const overlay = document.createElement('div');
    overlay.id = 'ai-vision-unsupported-dialog';
    overlay.style.cssText = `
        position:fixed; inset:0; background:rgba(0,0,0,0.6); z-index:20000;
        display:flex; align-items:center; justify-content:center;
    `;
    overlay.innerHTML = `
        <div style="background:#1e293b; color:#e2e8f0; border-radius:12px; padding:2rem;
                    max-width:460px; width:90%; box-shadow:0 20px 60px rgba(0,0,0,0.5);">
            <h3 style="margin:0 0 1rem; font-size:1.2rem; color:#f87171;">⚠️ Model does not support vision</h3>
            <p style="margin:0 0 0.75rem;">
                <code style="color:#f59e0b; background:#0f172a; padding:0.15rem 0.4rem; border-radius:4px;">${modelName}</code>
                does not support image input on OpenRouter.
            </p>
            <p style="margin:0 0 1.25rem; color:#94a3b8; font-size:0.85rem;">
                Please choose a vision-capable model. Browse available models at
                <a href="https://openrouter.ai/models" target="_blank" style="color:#6366f1;">openrouter.ai/models</a>
                and filter by image input support.
            </p>
            <div style="display:flex; justify-content:flex-end;">
                <button id="ai-vision-dialog-ok" class="btn btn-primary">OK, change model</button>
            </div>
        </div>
    `;
    document.body.appendChild(overlay);
    document.getElementById('ai-vision-dialog-ok').addEventListener('click', () => {
        overlay.remove();
        // Open the AI Backend panel so the user can change the model immediately
        const panel = document.getElementById('ai-backend-panel');
        if (panel) panel.style.display = 'block';
    });
}

/** Flash a brief "Saved" / "Reset" badge next to the prompt reset button. */
function _showPromptSaveIndicator(text) {
    const btn = document.getElementById('ai-prompt-reset-btn');
    if (!btn) return;
    let badge = document.getElementById('ai-prompt-save-badge');
    if (!badge) {
        badge = document.createElement('span');
        badge.id = 'ai-prompt-save-badge';
        badge.style.cssText = 'font-size:0.72rem;color:#22c55e;margin-left:0.5rem;opacity:1;transition:opacity 1s ease;';
        btn.parentNode.insertBefore(badge, btn.nextSibling);
    }
    badge.textContent = text === 'Reset' ? '✓ Reset' : '✓ Saved';
    badge.style.color = text === 'Reset' ? '#f59e0b' : '#22c55e';
    badge.style.opacity = '1';
    clearTimeout(badge._hideTimer);
    badge._hideTimer = setTimeout(() => { badge.style.opacity = '0'; }, 2000);
}

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
            const prog = await window.PyPotteryUtils.apiRequest('/api/operation-progress');
            if (prog && prog.active) {
                labelEl.textContent = prog.message || '';
                barEl.style.width = (prog.percent || 0) + '%';
            }
        } catch (_) { /* ignore polling errors */ }
    }, 800);
    return interval;
}

function showBatchProgressOverlay() {
    document.getElementById('ai-batch-progress-overlay')?.remove();
    const overlay = document.createElement('div');
    overlay.id = 'ai-batch-progress-overlay';
    overlay.style.cssText = 'position:fixed; inset:0; background:rgba(0,0,0,0.6); z-index:20000; display:flex; align-items:center; justify-content:center;';
    overlay.innerHTML = `
        <div style="background:#1e293b; color:#e2e8f0; border-radius:12px; padding:2rem;
                    max-width:480px; width:90%; box-shadow:0 20px 60px rgba(0,0,0,0.5);">
            <h3 style="margin:0 0 1rem; font-size:1.2rem;">🤖 Batch AI Extraction</h3>
            <p id="ai-batch-progress-label" style="font-size:0.85rem; color:#94a3b8; margin:0 0 0.4rem;">Starting...</p>
            <div style="background:#334155; border-radius:6px; overflow:hidden; height:12px;">
                <div id="ai-batch-progress-bar"
                     style="height:100%; background:#6366f1; transition:width 0.4s; width:0%"></div>
            </div>
        </div>
    `;
    document.body.appendChild(overlay);
    return overlay;
}

async function handleAiBibliographic() {
    if (!tabularState.currentProject || !tabularState.currentProject.project_id) {
        window.PyPotteryUtils.showToast('No project selected', 'warning');
        return;
    }

    const statusEl = document.getElementById('ai-bibliographic-status');
    const btn = document.getElementById('ai-bibliographic-btn');
    const backendParams = getAiBackendParams();

    // For OpenRouter, skip GPU check entirely and call directly
    if (backendParams.ai_backend === 'openrouter') {
        if (!backendParams.openrouter_api_key) {
            window.PyPotteryUtils.showToast('Please enter your OpenRouter API key in the AI Backend panel', 'warning');
            document.getElementById('ai-backend-panel').style.display = 'block';
            return;
        }
        btn.disabled = true;
        if (statusEl) statusEl.textContent = '⏳ Analysing via OpenRouter...';
        window.PyPotteryUtils.showLoading('Extracting references via OpenRouter...');
        try {
            const response = await window.PyPotteryUtils.apiRequest(
                `/api/projects/${tabularState.currentProject.project_id}/tabular/ai-bibliographic`,
                { method: 'POST', body: JSON.stringify({ img_num: tabularState.currentIndex, prompt_suffix: getPromptSuffix(), ...backendParams }) }
            );
            window.PyPotteryUtils.hideLoading();
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
            window.PyPotteryUtils.hideLoading();
            if (statusEl) statusEl.textContent = '❌ ' + error.message;
            window.PyPotteryUtils.showToast(error.message, 'error');
            console.error('[AI Bibliographic] Error:', error);
        } finally {
            btn.disabled = false;
        }
        return;
    }

    // Local backend: check GPU requirements first
    let requirements;
    try {
        requirements = await checkAiRequirements();
    } catch (e) {
        window.PyPotteryUtils.showToast('Could not check system requirements', 'error');
        return;
    }

    // If model is already cached, skip confirm dialog and run directly
    if (requirements.model_cached) {
        btn.disabled = true;
        if (statusEl) statusEl.textContent = '⏳ Analysing with Gemma 4 AI...';
        window.PyPotteryUtils.showLoading('Extracting references with Gemma 4 AI...');
        try {
            const response = await window.PyPotteryUtils.apiRequest(
                `/api/projects/${tabularState.currentProject.project_id}/tabular/ai-bibliographic`,
                { method: 'POST', body: JSON.stringify({ img_num: tabularState.currentIndex, prompt_suffix: getPromptSuffix(), ...backendParams }) }
            );
            window.PyPotteryUtils.hideLoading();
            if (response.success) {
                tabularState.tableData = response.table;
                tabularState.columns = response.columns;
                displayTable(response.table, response.columns);
                if (statusEl) statusEl.textContent = '✅ References extracted successfully';
                window.PyPotteryUtils.showToast('Bibliographic references extracted!', 'success');
            } else if (response.vision_unsupported) {
                if (statusEl) statusEl.textContent = '';
                showVisionUnsupportedDialog(backendParams.openrouter_model);
            } else {
                if (statusEl) statusEl.textContent = '❌ Error: ' + (response.error || 'unknown');
                window.PyPotteryUtils.showToast(response.error || 'AI Error', 'error');
            }
        } catch (error) {
            window.PyPotteryUtils.hideLoading();
            if (statusEl) statusEl.textContent = '❌ ' + error.message;
            window.PyPotteryUtils.showToast(error.message, 'error');
            console.error('[AI Bibliographic] Error:', error);
        } finally {
            btn.disabled = false;
        }
        return;
    }

    // Model not yet cached: show confirm dialog with download progress bar
    showAiConfirmDialog(requirements, async (overlay) => {
        const labelEl = document.getElementById('ai-download-progress-label');
        const barEl = document.getElementById('ai-download-progress-bar');
        const stopSignal = { stopped: false };
        const pollInterval = startProgressPolling(labelEl, barEl, stopSignal);

        btn.disabled = true;
        if (statusEl) statusEl.textContent = '⏳ Downloading model and analysing...';
        window.PyPotteryUtils.showLoading('Downloading Gemma 4 AI model (~10 GB)...');

        try {
            const response = await window.PyPotteryUtils.apiRequest(
                `/api/projects/${tabularState.currentProject.project_id}/tabular/ai-bibliographic`,
                {
                    method: 'POST',
                    body: JSON.stringify({ img_num: tabularState.currentIndex, prompt_suffix: getPromptSuffix(), ...backendParams })
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
            } else if (response.vision_unsupported) {
                if (statusEl) statusEl.textContent = '';
                showVisionUnsupportedDialog(backendParams.openrouter_model);
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
    const backendParams = getAiBackendParams();

    // Helper: run the batch request with a given progress label/bar and overlay
    async function runBatch(overlay, labelEl, barEl) {
        const stopSignal = { stopped: false };
        const pollInterval = startProgressPolling(labelEl, barEl, stopSignal);
        btn.disabled = true;
        if (statusEl) statusEl.textContent = '⏳ Running batch extraction...';
        try {
            const response = await window.PyPotteryUtils.apiRequest(
                `/api/projects/${tabularState.currentProject.project_id}/tabular/ai-bibliographic-batch`,
                { method: 'POST', body: JSON.stringify({ prompt_suffix: getPromptSuffix(), ...backendParams }) }
            );
            stopSignal.stopped = true;
            clearInterval(pollInterval);
            overlay.remove();
            if (response.success) {
                const errMsg = response.errors && response.errors.length
                    ? ` (${response.errors.length} errors)` : '';
                if (statusEl) statusEl.textContent = `✅ Batch complete: ${response.processed} images${errMsg}`;
                window.PyPotteryUtils.showToast(`Batch extraction done: ${response.processed} images${errMsg}`, 'success');
                await loadTabularData(tabularState.currentIndex);
            } else if (response.vision_unsupported) {
                if (statusEl) statusEl.textContent = '';
                showVisionUnsupportedDialog(backendParams.openrouter_model);
            } else {
                if (statusEl) statusEl.textContent = '❌ Batch error: ' + (response.error || 'unknown');
                window.PyPotteryUtils.showToast(response.error || 'Batch AI Error', 'error');
            }
        } catch (error) {
            stopSignal.stopped = true;
            clearInterval(pollInterval);
            overlay.remove();
            if (statusEl) statusEl.textContent = '❌ ' + error.message;
            window.PyPotteryUtils.showToast(error.message, 'error');
            console.error('[AI Batch] Error:', error);
        } finally {
            btn.disabled = false;
        }
    }

    // For OpenRouter, skip GPU check and run batch directly with progress overlay
    if (backendParams.ai_backend === 'openrouter') {
        if (!backendParams.openrouter_api_key) {
            window.PyPotteryUtils.showToast('Please enter your OpenRouter API key in the AI Backend panel', 'warning');
            document.getElementById('ai-backend-panel').style.display = 'block';
            return;
        }
        const overlay = showBatchProgressOverlay();
        const labelEl = document.getElementById('ai-batch-progress-label');
        const barEl = document.getElementById('ai-batch-progress-bar');
        await runBatch(overlay, labelEl, barEl);
        return;
    }

    // Local backend: check GPU requirements first
    let requirements;
    try {
        requirements = await checkAiRequirements();
    } catch (e) {
        window.PyPotteryUtils.showToast('Could not check system requirements', 'error');
        return;
    }

    // If model is already cached, skip confirm dialog and show progress overlay directly
    if (requirements.model_cached) {
        const overlay = showBatchProgressOverlay();
        const labelEl = document.getElementById('ai-batch-progress-label');
        const barEl = document.getElementById('ai-batch-progress-bar');
        await runBatch(overlay, labelEl, barEl);
        return;
    }

    // Model not yet cached: show confirm dialog with download note
    showAiConfirmDialog(requirements, async (overlay) => {
        const labelEl = document.getElementById('ai-download-progress-label');
        const barEl = document.getElementById('ai-download-progress-bar');
        // Always show the progress bar inside the dialog for batch mode
        document.getElementById('ai-download-progress-wrapper').style.display = 'block';
        await runBatch(overlay, labelEl, barEl);
    });
}
