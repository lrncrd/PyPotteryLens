// Post Processing Tab JavaScript - grid of pieces at real relative size
// Uses window.PyPotteryUtils.* functions directly

// State
let postprocessState = {
    currentProject: null,
    cards: []
};

// Grid sizing: largest piece maps to MAX px, smallest clamped to MIN px.
// MIN must stay wide enough for the hover controls (flip V/H + exclude + ENT/FRAG pill).
const PP_MAX_PX = 300;
const PP_MIN_PX = 130;

document.addEventListener('DOMContentLoaded', () => {
    setupPostprocessListeners();
    loadCurrentProject();

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
    document.getElementById('process-all-btn')?.addEventListener('click', handleProcessAll);
    document.getElementById('export-btn')?.addEventListener('click', showExportDialog);

    // Export dialog
    document.getElementById('export-cancel')?.addEventListener('click', hideExportDialog);
    document.getElementById('export-confirm')?.addEventListener('click', handleExportConfirm);

    // Lightbox (zoom on click)
    document.getElementById('postprocess-lightbox-close')?.addEventListener('click', closeLightbox);
    document.getElementById('postprocess-lightbox')?.addEventListener('click', (e) => {
        if (e.target.id === 'postprocess-lightbox') closeLightbox();
    });
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') closeLightbox();
    });
}

function projectId() {
    return postprocessState.currentProject && postprocessState.currentProject.project_id;
}

async function loadProjectCards() {
    if (!projectId()) {
        showEmptyState('Select a project from the Project Manager tab.');
        return;
    }
    try {
        window.PyPotteryUtils.showLoading('Loading project cards...');
        const response = await window.PyPotteryUtils.apiRequest(
            `/api/projects/${projectId()}/cards`
        );
        window.PyPotteryUtils.hideLoading();

        if (response.success) {
            postprocessState.cards = response.cards || [];
            if (postprocessState.cards.length === 0) {
                showEmptyState('No cards found. Extract cards from masks in the Annotation tab first.');
                return;
            }
            renderGrid(postprocessState.cards);
        } else {
            showEmptyState('Error loading cards: ' + (response.error || 'unknown'));
        }
    } catch (error) {
        window.PyPotteryUtils.hideLoading();
        console.error('Error loading project cards:', error);
        showEmptyState('Error: ' + error.message);
    }
}

// Build the grid; each card is sized proportionally to its real pixel size
function renderGrid(cards) {
    const grid = document.getElementById('postprocess-grid');
    const count = document.getElementById('postprocess-image-count');
    if (count) count.textContent = cards.length;
    if (!grid) return;

    if (!cards.length) {
        showEmptyState('No cards available.');
        return;
    }

    const pid = projectId();
    const maxDim = Math.max(1, ...cards.map(c => Math.max(c.width || 0, c.height || 0)));
    const scale = PP_MAX_PX / maxDim;

    grid.innerHTML = '';
    cards.forEach(card => {
        const filename = card.filename;
        const realW = card.width || 0;
        let dispW = Math.round((realW || PP_MIN_PX) * scale);
        dispW = Math.max(PP_MIN_PX, Math.min(PP_MAX_PX, dispW));

        const item = document.createElement('div');
        const typeClass = (card.type === 'FRAG') ? 'type-frag' : 'type-ent';
        item.className = `pp-card ${typeClass}` + (card.excluded ? ' excluded' : '');
        item.style.width = dispW + 'px';
        item.dataset.filename = filename;

        const base = card.has_modified
            ? `/api/projects/${pid}/card-modified/${encodeURIComponent(filename)}`
            : `/api/projects/${pid}/card/${encodeURIComponent(filename)}`;
        const imgSrc = `${base}?v=${Date.now()}`;

        item.innerHTML = `
            <img class="pp-card-img" loading="lazy" src="${imgSrc}" alt="${filename}">
            <div class="pp-card-badge">EXCLUDED</div>
            <div class="pp-card-overlay">
                <div class="pp-overlay-row">
                    <button class="pp-btn" data-act="flipv" title="Flip vertical">↕</button>
                    <button class="pp-btn" data-act="fliph" title="Flip horizontal">↔</button>
                    <button class="pp-btn pp-exclude" data-act="exclude" title="Exclude / include from export">✕</button>
                </div>
                <div class="pp-type-pill" data-act="type" title="Toggle ENT / FRAG">
                    <span class="pp-type-ent ${card.type === 'ENT' ? 'active' : ''}">ENT</span>
                    <span class="pp-type-frag ${card.type === 'FRAG' ? 'active' : ''}">FRAG</span>
                </div>
            </div>
        `;

        const img = item.querySelector('.pp-card-img');
        img.addEventListener('click', () => openLightbox(img.src));
        item.querySelector('[data-act="flipv"]').addEventListener('click', (e) => { e.stopPropagation(); flipCard(card, item, 'vertical'); });
        item.querySelector('[data-act="fliph"]').addEventListener('click', (e) => { e.stopPropagation(); flipCard(card, item, 'horizontal'); });
        item.querySelector('[data-act="exclude"]').addEventListener('click', (e) => { e.stopPropagation(); toggleExclude(card, item); });
        item.querySelector('[data-act="type"]').addEventListener('click', (e) => { e.stopPropagation(); toggleType(card, item); });

        grid.appendChild(item);
    });
}

async function flipCard(card, item, flipType) {
    if (!projectId()) return;
    try {
        const res = await window.PyPotteryUtils.apiRequest(`/api/projects/${projectId()}/postprocess/flip`, {
            method: 'POST',
            body: JSON.stringify({ card_filename: card.filename, flip_type: flipType })
        });
        if (res.success) {
            card.has_modified = true;
            const img = item.querySelector('.pp-card-img');
            // Returned base64 reflects the flip immediately (avoids cache issues)
            img.src = res.image || `/api/projects/${projectId()}/card-modified/${encodeURIComponent(card.filename)}?v=${Date.now()}`;
        } else {
            window.PyPotteryUtils.showToast(res.error || 'Flip failed', 'error');
        }
    } catch (error) {
        window.PyPotteryUtils.showToast(error.message, 'error');
    }
}

async function toggleType(card, item) {
    if (!projectId()) return;
    const newType = (card.type === 'ENT') ? 'FRAG' : 'ENT';
    try {
        const res = await window.PyPotteryUtils.apiRequest(`/api/projects/${projectId()}/postprocess/update-type`, {
            method: 'POST',
            body: JSON.stringify({ filename: card.filename, type: newType })
        });
        if (res.success) {
            card.type = newType;
            item.querySelector('.pp-type-ent').classList.toggle('active', newType === 'ENT');
            item.querySelector('.pp-type-frag').classList.toggle('active', newType === 'FRAG');
            item.classList.toggle('type-ent', newType === 'ENT');
            item.classList.toggle('type-frag', newType === 'FRAG');
        } else {
            window.PyPotteryUtils.showToast(res.error || 'Run "Process All Images" first to set types', 'error');
        }
    } catch (error) {
        window.PyPotteryUtils.showToast(error.message, 'error');
    }
}

async function toggleExclude(card, item) {
    if (!projectId()) return;
    const newExcluded = !card.excluded;
    try {
        const res = await window.PyPotteryUtils.apiRequest(`/api/projects/${projectId()}/postprocess/exclude`, {
            method: 'POST',
            body: JSON.stringify({ filename: card.filename, excluded: newExcluded })
        });
        if (res.success) {
            card.excluded = newExcluded;
            item.classList.toggle('excluded', newExcluded);
        } else {
            window.PyPotteryUtils.showToast(res.error || 'Failed to exclude', 'error');
        }
    } catch (error) {
        window.PyPotteryUtils.showToast(error.message, 'error');
    }
}

function openLightbox(src) {
    const lb = document.getElementById('postprocess-lightbox');
    const img = document.getElementById('postprocess-lightbox-img');
    if (!lb || !img) return;
    img.src = src;
    lb.style.display = 'flex';
}

function closeLightbox() {
    const lb = document.getElementById('postprocess-lightbox');
    if (lb) lb.style.display = 'none';
}

async function handleProcessAll() {
    if (!projectId()) {
        window.PyPotteryUtils.showToast('Please select a project first', 'warning');
        return;
    }
    const flipVertical = document.getElementById('auto-flip-vertical').checked;
    const flipHorizontal = document.getElementById('auto-flip-horizontal').checked;

    try {
        window.PyPotteryUtils.showStatus('postprocess-status', 'Starting processing...', 'info');
        const response = await window.PyPotteryUtils.executeWithProgress(
            'postprocess',
            async () => {
                return await window.PyPotteryUtils.apiRequest(`/api/projects/${projectId()}/postprocess`, {
                    method: 'POST',
                    body: JSON.stringify({ flip_vertical: flipVertical, flip_horizontal: flipHorizontal })
                });
            },
            'postprocess-status',
            'postprocess-progress-bar'
        );

        if (response.success) {
            window.PyPotteryUtils.showStatus('postprocess-status', response.message || 'Done', 'success');
            window.PyPotteryUtils.showToast(`Processed ${response.count || ''} images!`, 'success');
            await loadProjectCards();  // reloads cards (now has_modified) and re-renders the grid
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

function showEmptyState(message) {
    const grid = document.getElementById('postprocess-grid');
    if (grid) grid.innerHTML = `<div class="pp-empty">${message}</div>`;
    const count = document.getElementById('postprocess-image-count');
    if (count) count.textContent = '0';
}

// Export dialog functions
function showExportDialog() {
    if (!projectId()) {
        window.PyPotteryUtils.showToast('Please select a project first', 'warning');
        return;
    }
    document.getElementById('export-dialog').style.display = 'flex';
}

function hideExportDialog() {
    document.getElementById('export-dialog').style.display = 'none';
    document.getElementById('export-acronym').value = '';
    const st = document.getElementById('export-dialog-status');
    if (st) { st.textContent = ''; st.className = 'status-message'; }
}

async function handleExportConfirm() {
    const acronym = document.getElementById('export-acronym').value.trim();
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

        const response = await fetch(`/api/projects/${projectId()}/export`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ acronym: acronym })
        });

        window.PyPotteryUtils.hideLoading();

        if (response.ok) {
            const blob = await response.blob();
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

// Export for use by main.js
window.loadPostprocessCards = loadProjectCards;
