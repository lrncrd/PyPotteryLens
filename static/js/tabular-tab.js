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
    fullImageUrl: null,  // Full resolution image URL for zoom
    excludedImages: new Set()  // Excluded images set
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

    // Listen for exclusion changes from other tabs (cross-tab sync)
    window.addEventListener('exclusionChanged', (e) => {
        const { filename, excluded, allExcluded } = e.detail;

        // Update local state
        if (excluded) {
            tabularState.excludedImages.add(filename);
        } else {
            tabularState.excludedImages.delete(filename);
        }

        // Update UI for this specific item
        updateExclusionVisualInList(filename, excluded);
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

    // Export to CSV - use combined export endpoint
    document.getElementById('export-csv-btn')?.addEventListener('click', exportCombinedCSV);

    // Mark as reviewed
    document.getElementById('tabular-mark-reviewed-btn')?.addEventListener('click', markAsReviewed);

    // Extract metadata from PDF
    document.getElementById('extract-metadata-btn')?.addEventListener('click', extractMetadata);

    // Setup magnifying glass zoom on hover
    setupMagnifyingGlass();

    // File browser for PDF
    setupFileBrowser();

    // AI Settings panel
    setupAISettings();
}

// ============================================================================
// File Browser for Reference PDF
// ============================================================================

function setupFileBrowser() {
    const browseBtn = document.getElementById('browse-pdf-btn');
    const fileInput = document.getElementById('pdf-file-input');
    const pathInput = document.getElementById('reference-pdf-path');

    if (browseBtn && fileInput && pathInput) {
        browseBtn.addEventListener('click', () => {
            fileInput.click();
        });

        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                const file = e.target.files[0];
                // Store the file object for potential upload
                pathInput.dataset.file = file.name;
                pathInput.value = file.name;

                // Show info about the selected file
                window.PyPotteryUtils.showStatus('tabular-status',
                    `Selected: ${file.name} (${(file.size / 1024 / 1024).toFixed(2)} MB)`, 'info');
            }
        });
    }
}

// ============================================================================
// AI Settings Panel
// ============================================================================

function setupAISettings() {
    // Load saved settings on init
    loadAISettings();

    // Toggle panel visibility
    window.toggleAISettings = function() {
        const content = document.getElementById('ai-settings-content');
        const toggle = document.getElementById('ai-settings-toggle');
        if (content && toggle) {
            content.classList.toggle('collapsed');
            toggle.textContent = content.classList.contains('collapsed') ? '▶' : '▼';
        }
    };

    // Save Anthropic key
    document.getElementById('save-anthropic-key')?.addEventListener('click', () => {
        saveAPIKey('anthropic');
    });

    // Save OpenAI key
    document.getElementById('save-openai-key')?.addEventListener('click', () => {
        saveAPIKey('openai');
    });

    // Toggle key visibility
    document.getElementById('toggle-anthropic-key')?.addEventListener('click', () => {
        toggleKeyVisibility('anthropic-api-key');
    });

    document.getElementById('toggle-openai-key')?.addEventListener('click', () => {
        toggleKeyVisibility('openai-api-key');
    });

    // AI Extract Metadata button
    document.getElementById('ai-extract-metadata-btn')?.addEventListener('click', aiExtractMetadata);

    // Calibration buttons
    document.getElementById('set-project-calibration')?.addEventListener('click', setProjectCalibration);
    document.getElementById('detect-scale-bar-btn')?.addEventListener('click', detectScaleBar);
}

async function loadAISettings() {
    try {
        const response = await fetch('/api/settings');
        const data = await response.json();

        if (data.success && data.settings) {
            const settings = data.settings;

            // Update provider dropdown
            const providerSelect = document.getElementById('ai-provider-select');
            if (providerSelect && settings.default_ai_provider) {
                providerSelect.value = settings.default_ai_provider;
            }

            // Update key status indicators
            updateKeyStatus('anthropic', settings.has_anthropic_key, settings.anthropic_api_key);
            updateKeyStatus('openai', settings.has_openai_key, settings.openai_api_key);
        }
    } catch (error) {
        console.error('Failed to load AI settings:', error);
    }
}

function updateKeyStatus(provider, hasKey, maskedKey) {
    const statusEl = document.getElementById(`${provider}-key-status`);
    if (statusEl) {
        if (hasKey) {
            statusEl.textContent = `Configured: ${maskedKey}`;
            statusEl.className = 'key-status configured';
        } else {
            statusEl.textContent = 'Not configured';
            statusEl.className = 'key-status not-configured';
        }
    }
}

async function saveAPIKey(provider) {
    const input = document.getElementById(`${provider}-api-key`);
    const key = input?.value?.trim();

    if (!key) {
        window.PyPotteryUtils.showStatus('tabular-status', `Please enter a ${provider} API key`, 'error');
        return;
    }

    try {
        const response = await fetch('/api/settings/api-key', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ provider, key })
        });

        const data = await response.json();

        if (data.success) {
            window.PyPotteryUtils.showStatus('tabular-status', data.message, 'success');
            input.value = ''; // Clear input after saving
            updateKeyStatus(provider, true, data.masked_key);
        } else {
            window.PyPotteryUtils.showStatus('tabular-status', data.error || 'Failed to save key', 'error');
        }
    } catch (error) {
        console.error('Failed to save API key:', error);
        window.PyPotteryUtils.showStatus('tabular-status', 'Failed to save API key', 'error');
    }
}

function toggleKeyVisibility(inputId) {
    const input = document.getElementById(inputId);
    if (input) {
        input.type = input.type === 'password' ? 'text' : 'password';
    }
}

// Progress polling state
let progressPollingInterval = null;

function showAIProgress(show = true) {
    const container = document.getElementById('ai-progress-container');
    if (container) {
        container.style.display = show ? 'block' : 'none';
    }
}

function updateAIProgress(percent, message, title = 'Processing...') {
    const progressBar = document.getElementById('ai-progress-bar');
    const progressPercent = document.getElementById('ai-progress-percent');
    const progressMessage = document.getElementById('ai-progress-message');
    const progressTitle = document.getElementById('ai-progress-title');

    if (progressBar) progressBar.style.width = `${percent}%`;
    if (progressPercent) progressPercent.textContent = `${percent}%`;
    if (progressMessage) progressMessage.textContent = message;
    if (progressTitle) progressTitle.textContent = title;
}

function startProgressPolling() {
    // Clear any existing interval
    if (progressPollingInterval) {
        clearInterval(progressPollingInterval);
    }

    progressPollingInterval = setInterval(async () => {
        try {
            const response = await fetch('/api/operation-progress');
            const data = await response.json();

            if (data.success && data.progress && data.progress.active) {
                const p = data.progress;
                updateAIProgress(
                    p.percent,
                    p.message,
                    `🧠 AI Extraction (${p.current}/${p.total})`
                );
            }
        } catch (error) {
            console.error('Progress polling error:', error);
        }
    }, 500); // Poll every 500ms
}

function stopProgressPolling() {
    if (progressPollingInterval) {
        clearInterval(progressPollingInterval);
        progressPollingInterval = null;
    }
}

async function aiExtractMetadata() {
    if (!tabularState.currentProject || !tabularState.currentProject.project_id) {
        window.PyPotteryUtils.showStatus('tabular-status', 'No project selected', 'error');
        return;
    }

    const provider = document.getElementById('ai-provider-select')?.value || 'anthropic';
    const btn = document.getElementById('ai-extract-metadata-btn');
    const originalText = btn?.innerHTML;

    try {
        if (btn) {
            btn.disabled = true;
            btn.innerHTML = '⏳ AI Processing...';
        }

        // Show progress bar and start polling
        showAIProgress(true);
        updateAIProgress(0, 'Initializing AI extraction...', '🧠 Starting AI Extraction');
        startProgressPolling();

        // Get PDF path from widget if provided
        const pdfPath = document.getElementById('reference-pdf-path')?.value || '';

        window.PyPotteryUtils.showStatus('tabular-status',
            `Using ${provider === 'anthropic' ? 'Claude' : 'GPT'} to extract metadata...`, 'info');

        const response = await fetch(`/api/projects/${tabularState.currentProject.project_id}/metadata/ai-extract`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ provider, pdf_path: pdfPath })
        });

        const data = await response.json();

        // Stop polling and update final state
        stopProgressPolling();

        if (data.success) {
            updateAIProgress(100, 'Extraction complete!', '✅ AI Extraction Complete');
            window.PyPotteryUtils.showStatus('tabular-status',
                `AI extraction complete! Processed ${data.processed || 0} images (${data.successful || 0} successful).`, 'success');
            // Reload table data
            loadProjectCards();

            // Hide progress bar after 3 seconds
            setTimeout(() => showAIProgress(false), 3000);
        } else {
            updateAIProgress(0, data.error || 'Extraction failed', '❌ AI Extraction Failed');
            window.PyPotteryUtils.showStatus('tabular-status',
                data.error || 'AI extraction failed', 'error');
            // Hide progress bar after 3 seconds
            setTimeout(() => showAIProgress(false), 3000);
        }
    } catch (error) {
        console.error('AI extraction error:', error);
        stopProgressPolling();
        updateAIProgress(0, error.message, '❌ AI Extraction Failed');
        window.PyPotteryUtils.showStatus('tabular-status',
            'AI extraction failed: ' + error.message, 'error');
        setTimeout(() => showAIProgress(false), 3000);
    } finally {
        if (btn) {
            btn.disabled = false;
            btn.innerHTML = originalText;
        }
    }
}

async function setProjectCalibration() {
    const pixelsPerCm = parseFloat(document.getElementById('calibration-pixels-per-cm')?.value);

    if (!pixelsPerCm || pixelsPerCm <= 0) {
        window.PyPotteryUtils.showStatus('tabular-status', 'Please enter a valid pixels per cm value', 'error');
        return;
    }

    try {
        const response = await fetch('/api/settings/calibration', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ pixels_per_cm: pixelsPerCm })
        });

        const data = await response.json();

        if (data.success) {
            window.PyPotteryUtils.showStatus('tabular-status', 'Calibration saved', 'success');
        } else {
            window.PyPotteryUtils.showStatus('tabular-status', data.error || 'Failed to save calibration', 'error');
        }
    } catch (error) {
        console.error('Calibration error:', error);
        window.PyPotteryUtils.showStatus('tabular-status', 'Failed to save calibration', 'error');
    }
}

async function detectScaleBar() {
    if (!tabularState.currentProject || !tabularState.currentImageName) {
        window.PyPotteryUtils.showStatus('tabular-status', 'No image selected', 'error');
        return;
    }

    const statusEl = document.getElementById('scale-detection-status');
    const btn = document.getElementById('detect-scale-bar-btn');
    const originalText = btn?.innerHTML;

    try {
        if (btn) {
            btn.disabled = true;
            btn.innerHTML = '⏳ Detecting...';
        }
        if (statusEl) {
            statusEl.textContent = 'Analyzing...';
            statusEl.className = 'detection-status';
        }

        const response = await fetch(`/api/projects/${tabularState.currentProject.project_id}/scale-bar/detect`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image_name: tabularState.currentImageName })
        });

        const data = await response.json();

        if (data.success && data.result) {
            const result = data.result;
            if (statusEl) {
                statusEl.textContent = `Detected: ${result.pixels}px (${result.unit_text || 'unknown scale'})`;
                statusEl.className = 'detection-status success';
            }
            // Auto-fill calibration input if we can parse the scale
            if (result.pixels && result.cm) {
                const calibrationInput = document.getElementById('calibration-pixels-per-cm');
                if (calibrationInput) {
                    calibrationInput.value = (result.pixels / result.cm).toFixed(2);
                }
            }
        } else {
            if (statusEl) {
                statusEl.textContent = 'No scale bar detected';
                statusEl.className = 'detection-status error';
            }
        }
    } catch (error) {
        console.error('Scale bar detection error:', error);
        if (statusEl) {
            statusEl.textContent = 'Detection failed';
            statusEl.className = 'detection-status error';
        }
    } finally {
        if (btn) {
            btn.disabled = false;
            btn.innerHTML = originalText;
        }
    }
}

async function extractMetadata() {
    if (!tabularState.currentProject || !tabularState.currentProject.project_id) {
        window.PyPotteryUtils.showStatus('tabular-status', 'No project selected', 'error');
        return;
    }

    // Check if there's a reference PDF for period extraction
    const referencePdfInput = document.getElementById('reference-pdf-path');
    const referencePdfPath = referencePdfInput ? referencePdfInput.value.trim() : '';

    const btn = document.getElementById('extract-metadata-btn');
    const originalText = btn.innerHTML;
    try {
        btn.disabled = true;
        btn.innerHTML = '⏳ Extracting...';

        let statusMsg = 'Extracting metadata from PDF (may take a while for OCR)...';
        if (referencePdfPath) {
            statusMsg += ' Also extracting period info from reference PDF.';
        }
        window.PyPotteryUtils.showStatus('tabular-status', statusMsg, 'info');

        const requestBody = {};
        if (referencePdfPath) {
            requestBody.reference_pdf_path = referencePdfPath;
        }

        const response = await window.PyPotteryUtils.apiRequest(
            `/api/projects/${tabularState.currentProject.project_id}/metadata/extract`,
            {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(requestBody)
            }
        );
        if (response.success) {
            window.PyPotteryUtils.showStatus('tabular-status', response.message || 'Metadata extracted!', 'success');
            await loadTabularData();
        } else {
            window.PyPotteryUtils.showStatus('tabular-status', response.error || 'Failed to extract metadata', 'error');
        }
    } catch (error) {
        console.error('Error extracting metadata:', error);
        window.PyPotteryUtils.showStatus('tabular-status', `Error: ${error.message}`, 'error');
    } finally {
        btn.disabled = false;
        btn.innerHTML = originalText;
    }
}

async function loadProjectCards() {
    if (!tabularState.currentProject || !tabularState.currentProject.project_id) {
        showEmptyState('No project selected', 'Select a project from the Project Manager tab');
        return;
    }

    try {
        window.PyPotteryUtils.showLoading('Loading project cards...');

        // Initialize exclusion manager for this project
        await window.PyPotteryUtils.exclusionManager.init(tabularState.currentProject.project_id);

        // Sync local state with exclusion manager
        tabularState.excludedImages = new Set(window.PyPotteryUtils.exclusionManager.getExcluded());

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
        div.dataset.filename = item.image_name;

        if (index === tabularState.currentIndex) {
            div.classList.add('active');
        }
        if (item.reviewed) {
            div.classList.add('reviewed');
        }

        // Check exclusion state
        const isExcluded = tabularState.excludedImages.has(item.image_name);
        if (isExcluded) {
            div.classList.add('excluded');
        }

        div.innerHTML = `
            <input type="checkbox" class="exclude-checkbox" title="Exclude from export" ${isExcluded ? 'checked' : ''}>
            <span class="image-name" title="${item.image_name}">${item.image_name}</span>
            <span class="exclude-indicator">EXCLUDED</span>
            <span class="status-icon">${item.reviewed ? '✅' : '⚪'}</span>
        `;

        // Handle exclusion checkbox click (stop propagation to prevent navigation)
        const checkbox = div.querySelector('.exclude-checkbox');
        checkbox.addEventListener('click', (e) => {
            e.stopPropagation();
        });
        checkbox.addEventListener('change', async (e) => {
            e.stopPropagation();
            const excluded = e.target.checked;
            await handleTabularExclusionToggle(item.image_name, excluded);
        });

        // Navigation click (on the rest of the item)
        div.addEventListener('click', (e) => {
            // Don't navigate if clicking on checkbox
            if (e.target.classList.contains('exclude-checkbox')) return;
            loadTabularData(index);
        });

        listContainer.appendChild(div);
    });

    // Update excluded count display
    updateExcludedCountDisplay();
}

// Handle exclusion toggle in tabular tab
async function handleTabularExclusionToggle(filename, excluded) {
    const success = await window.PyPotteryUtils.exclusionManager.toggle(filename, excluded);

    if (success) {
        // Update local state
        if (excluded) {
            tabularState.excludedImages.add(filename);
        } else {
            tabularState.excludedImages.delete(filename);
        }

        // Update visual
        updateExclusionVisualInList(filename, excluded);
        updateExcludedCountDisplay();
    } else {
        // Revert checkbox if failed
        const item = document.querySelector(`.tabular-image-item[data-filename="${filename}"] .exclude-checkbox`);
        if (item) {
            item.checked = !excluded;
        }
        window.PyPotteryUtils.showToast('Failed to update exclusion', 'error');
    }
}

// Update visual state of a single item in the list
function updateExclusionVisualInList(filename, excluded) {
    const item = document.querySelector(`.tabular-image-item[data-filename="${filename}"]`);
    if (!item) return;

    const checkbox = item.querySelector('.exclude-checkbox');
    if (checkbox) {
        checkbox.checked = excluded;
    }

    if (excluded) {
        item.classList.add('excluded');
    } else {
        item.classList.remove('excluded');
    }
}

// Update excluded count display
function updateExcludedCountDisplay() {
    const countEl = document.getElementById('tabular-excluded-count');
    if (countEl) {
        const count = tabularState.excludedImages.size;
        countEl.textContent = count > 0 ? `(${count} excluded)` : '';
    }
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
