// Helper: Check if a project is currently active
function hasActiveProject() {
    if (window.projectManager && typeof window.projectManager.getCurrentProject === 'function') {
        const p = window.projectManager.getCurrentProject();
        if (p && p.project_id) return true;
    }
    const savedId = localStorage.getItem('currentProjectId');
    return Boolean(savedId && savedId !== 'null' && savedId !== 'undefined');
}

// Tab Management
class TabManager {
    constructor() {
        this.tabs = document.querySelectorAll('.tab-button');
        this.contents = document.querySelectorAll('.tab-content');
        console.log('TabManager initialized with', this.tabs.length, 'tabs');
        this.init();
    }

    init() {
        this.tabs.forEach(tab => {
            tab.addEventListener('click', (e) => {
                const tabId = tab.dataset.tab;
                if (tabId !== 'projects' && !hasActiveProject()) {
                    e.preventDefault();
                    if (typeof showToast === 'function') {
                        showToast('Select or create a project first to access other sections.', 'warning');
                    }
                    return;
                }
                this.switchTab(tab);
            });
        });
    }

    switchTab(clickedTab) {
        const tabId = clickedTab.dataset.tab;
        if (tabId !== 'projects' && !hasActiveProject()) {
            if (typeof showToast === 'function') {
                showToast('Select or create a project first to access other sections.', 'warning');
            }
            return;
        }
        console.log('Switching to tab:', tabId);

        // Update tab buttons
        this.tabs.forEach(tab => tab.classList.remove('active'));
        clickedTab.classList.add('active');

        // Update tab contents
        this.contents.forEach(content => {
            content.classList.remove('active');
        });

        const targetContent = document.getElementById(`${tabId}-tab`);
        if (targetContent) {
            targetContent.classList.add('active');
            console.log('Tab content shown:', tabId);
        } else {
            console.error('Tab content not found:', `${tabId}-tab`);
        }

        // Refresh data when switching to certain tabs
        this.onTabSwitch(tabId);
    }

    async onTabSwitch(tabId) {
        try {
            switch (tabId) {
                case 'projects':
                    await refreshProjectsTab();
                    break;
                case 'model':
                    await refreshModelTab();
                    break;
                case 'annotation':
                    await refreshAnnotationTab();
                    break;
                case 'tabular':
                    await refreshTabularTab();
                    break;
                case 'postprocess':
                    await refreshPostprocessTab();
                    break;
            }
        } catch (error) {
            console.error('Error refreshing tab:', error);
        }
    }
}

// Refresh functions for each tab
async function refreshProjectsTab() {
    console.log('Refreshing projects tab...');
    if (window.projectManager && window.projectManager.loadProjects) {
        await window.projectManager.loadProjects();
    }
}
async function refreshModelTab() {
    console.log('Refreshing model tab (project-aware)...');
    if (!window.PyPotteryUtils) {
        console.error('PyPotteryUtils not available!');
        return;
    }
    
    try {
        // Refresh models list
        console.log('Fetching models...');
        const modelsResponse = await window.PyPotteryUtils.apiRequest('/api/models');
        console.log('Models response:', modelsResponse);
        
        if (modelsResponse.success) {
            const select = document.getElementById('model-select');
            if (select) {
                select.innerHTML = '<option value="">Select model...</option>';
                modelsResponse.models.forEach(model => {
                    const option = document.createElement('option');
                    option.value = model;
                    option.textContent = model;
                    select.appendChild(option);
                });
                console.log('Populated', modelsResponse.models.length, 'models');
            }
        }
        
        // Load project images if model-tab.js has the function
        if (window.loadModelProjectImages) {
            console.log('Loading project images for model tab...');
            await window.loadModelProjectImages();
        }
    } catch (error) {
        console.error('Error refreshing model tab:', error);
    }
}

async function refreshAnnotationTab() {
    console.log('Refreshing annotation tab (project-aware)...');
    if (!window.PyPotteryUtils) {
        console.error('PyPotteryUtils not available!');
        return;
    }
    
    try {
        // Load project masks if annotation-tab.js has the function
        if (window.loadAnnotationProjectImages) {
            console.log('Loading annotation project images...');
            await window.loadAnnotationProjectImages();
        }
    } catch (error) {
        console.error('Error refreshing annotation tab:', error);
    }
}

async function refreshTabularTab() {
    console.log('Refreshing tabular tab...');
    if (!window.PyPotteryUtils) {
        console.error('PyPotteryUtils not available!');
        return;
    }
    
    // Load project tabular data if function is available
    if (window.refreshTabular) {
        console.log('Loading tabular data for current project');
        await window.refreshTabular();
    } else {
        console.warn('refreshTabular function not available');
    }
}

async function refreshPostprocessTab() {
    console.log('Refreshing postprocess tab...');
    if (!window.PyPotteryUtils) {
        console.error('PyPotteryUtils not available!');
        return;
    }
    
    // Load postprocess cards if function is available
    if (window.loadPostprocessCards) {
        console.log('Loading postprocess cards for current project');
        await window.loadPostprocessCards();
    } else {
        console.warn('loadPostprocessCards function not available');
    }
}

// Initialize application
document.addEventListener('DOMContentLoaded', () => {
    console.log('PyPotteryLens Flask App initialized');

    // Initialize tab manager
    const tabManager = new TabManager();

    // Add CSS slideOut animation if not exists
    if (!document.querySelector('style#toast-animations')) {
        const style = document.createElement('style');
        style.id = 'toast-animations';
        style.textContent = `
            @keyframes slideOut {
                to {
                    transform: translateX(400px);
                    opacity: 0;
                }
            }
        `;
        document.head.appendChild(style);
    }

    // Listen for project changes to update UI
    window.addEventListener('projectChanged', (event) => {
        const project = event.detail.project;
        console.log('Project changed:', project);
        
        // Enable/disable tabs based on project state
        updateTabsState(project);
    });

    // Initial check of tab lock state
    updateTabsState(window.projectManager ? window.projectManager.getCurrentProject() : null);

    // Initial refresh of first tab (now projects tab)
    refreshProjectsTab().catch(console.error);
});

/**
 * Update tabs state based on current project
 */
function updateTabsState(project) {
    const hasProject = Boolean(project && project.project_id) || hasActiveProject();
    const otherTabs = document.querySelectorAll('.tab-button:not([data-tab="projects"])');
    
    otherTabs.forEach(tab => {
        tab.classList.toggle('tab-locked', !hasProject);
        if (!hasProject) {
            tab.setAttribute('title', 'Select or create a project first to unlock this section');
        } else {
            tab.removeAttribute('title');
        }
    });

    if (!hasProject) {
        const activeTab = document.querySelector('.tab-button.active');
        if (activeTab && activeTab.dataset.tab !== 'projects') {
            const projectsTab = document.querySelector('.tab-button[data-tab="projects"]');
            if (projectsTab) {
                projectsTab.click();
            }
        }
    }
}

// Export functions
window.PyPotteryApp = {
    refreshProjectsTab,
    refreshModelTab,
    refreshAnnotationTab,
    refreshTabularTab,
    refreshPostprocessTab,
    updateTabsState
};

// Splash Screen Management
class SplashScreen {
    constructor() {
        this.splash = document.getElementById('splash-screen');
        this.progressBar = document.getElementById('splash-progress-bar');
        this.progressText = document.getElementById('splash-progress-text');
        this.message = document.getElementById('splash-message');
        this.mainContainer = document.getElementById('main-container');
        this.currentProgress = 0;
    }

    updateProgress(percent, message) {
        this.currentProgress = Math.min(100, Math.max(0, percent));
        this.progressBar.style.width = `${this.currentProgress}%`;
        this.progressText.textContent = `${Math.round(this.currentProgress)}%`;
        if (message) {
            this.message.textContent = message;
        }
    }

    async checkInitializationStatus() {
        try {
            const response = await fetch('/api/init-status');
            const data = await response.json();
            
            if (data.stage) {
                this.updateProgress(data.progress, data.message);
            }
            
            if (data.ready) {
                return true;
            }
            
            return false;
        } catch (error) {
            console.error('Error checking init status:', error);
            return false;
        }
    }

    async waitForInit() {
        this.updateProgress(10, 'Starting initialization...');
        
        let maxAttempts = 600; // 5 minutes max (600 * 500ms)
        let attempts = 0;
        
        while (attempts < maxAttempts) {
            const ready = await this.checkInitializationStatus();
            
            if (ready) {
                this.updateProgress(100, 'Ready!');
                await this.hideSplash();
                return;
            }
            
            await new Promise(resolve => setTimeout(resolve, 500));
            attempts++;
        }
        
        // Timeout - show error
        this.updateProgress(0, 'Initialization timeout. Please refresh the page.');
    }

    async hideSplash() {
        // Wait a moment to show 100%
        await new Promise(resolve => setTimeout(resolve, 500));
        
        // Fade out splash
        this.splash.classList.add('fade-out');
        
        // Show main container
        await new Promise(resolve => setTimeout(resolve, 500));
        this.mainContainer.classList.add('ready');
        this.mainContainer.style.display = 'block';
        
        // Remove splash from DOM
        setTimeout(() => {
            if (this.splash.parentNode) {
                this.splash.parentNode.removeChild(this.splash);
            }
        }, 1000);
    }
}

// Initialize splash screen on page load
if (document.getElementById('splash-screen')) {
    const splashScreen = new SplashScreen();
    splashScreen.waitForInit();
}

// Model Info Popup Management
function initializeModelInfoPopup() {
    const button = document.getElementById('model-info-button');
    const popup = document.getElementById('model-info-popup');
    const closeBtn = document.getElementById('model-info-close');

    if (!button || !popup) return;

    // Show popup when button is clicked
    button.addEventListener('click', async (e) => {
        e.preventDefault();
        popup.classList.add('show');
        // Load system info when popup opens
        await loadSystemInfo();
    });

    // Hide popup when close button is clicked
    if (closeBtn) {
        closeBtn.addEventListener('click', () => {
            popup.classList.remove('show');
        });
    }

    // Hide popup when clicking outside
    popup.addEventListener('click', (e) => {
        if (e.target === popup) {
            popup.classList.remove('show');
        }
    });

    // Hide popup on Escape key
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && popup.classList.contains('show')) {
            popup.classList.remove('show');
        }
    });
}

// Load system information
async function loadSystemInfo() {
    const container = document.getElementById('system-info-container');
    if (!container) return;

    try {
        const response = await fetch('/api/system-info');
        const data = await response.json();

        if (response.ok) {
            let html = '';

            // CPU info
            if (data.cpu) {
                const cores = data.cpu.cores || 1;
                html += `
                <div class="model-info-row">
                    <span class="info-label">CPU</span>
                    <span class="info-value">${cores} Cores</span>
                </div>`;
            }

            // GPU info
            if (data.gpu) {
                if (data.gpu.cuda_available) {
                    const gpuNames = (data.gpu.gpu_names && data.gpu.gpu_names.length > 0)
                        ? data.gpu.gpu_names.join(', ')
                        : `CUDA (${data.gpu.gpu_count || 1} Device)`;
                    const safeGpu = gpuNames.replace(/"/g, '&quot;');
                    html += `
                    <div class="model-info-row">
                        <span class="info-label">GPU</span>
                        <span class="info-value" title="${safeGpu}">${safeGpu}</span>
                    </div>`;
                } else {
                    html += `
                    <div class="model-info-row">
                        <span class="info-label">GPU</span>
                        <span class="info-value text-muted">CPU Only</span>
                    </div>`;
                }
            }

            // MPS info (Apple Silicon) - only shown when available
            if (data.mps && data.mps.mps_available) {
                html += `
                <div class="model-info-row">
                    <span class="info-label">Hardware Acceleration</span>
                    <span class="info-value">Apple Silicon (MPS)</span>
                </div>`;
            }

            container.innerHTML = html;
        } else {
            container.innerHTML = `
            <div class="model-info-row">
                <span class="info-label">Status</span>
                <span class="info-value text-muted">Unable to load</span>
            </div>`;
        }
    } catch (error) {
        console.error('Error loading system info:', error);
        container.innerHTML = `
        <div class="model-info-row">
            <span class="info-label">Status</span>
            <span class="info-value text-muted">Unable to load</span>
        </div>`;
    }
}

// Initialize model info popup after DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    initializeModelInfoPopup();
});

// ==========================================
// Auto-Shutdown Heartbeat & Beacon System
// ==========================================
(function initAutoShutdownBeacon() {
    // Generate a unique identifier for this tab instance
    const tabSessionId = 'tab_' + Math.random().toString(36).substring(2, 11) + '_' + Date.now();
    const HEARTBEAT_INTERVAL_MS = 2500;

    function sendHeartbeat() {
        fetch('/api/heartbeat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ tab_id: tabSessionId }),
            keepalive: true
        }).catch(() => {
            // Harmless if server is restarting or shutting down
        });
    }

    // Send first heartbeat immediately on script load
    sendHeartbeat();

    // Regular interval heartbeat
    const intervalId = setInterval(sendHeartbeat, HEARTBEAT_INTERVAL_MS);

    // Re-ping when tab regains focus or visibility
    document.addEventListener('visibilitychange', () => {
        if (document.visibilityState === 'visible') {
            sendHeartbeat();
        }
    });

    window.addEventListener('focus', () => {
        sendHeartbeat();
    });

    // Prompt user confirmation dialog when closing tab or browser
    window.addEventListener('beforeunload', (e) => {
        // Standard cross-browser way to trigger native confirmation modal
        e.preventDefault();
        e.returnValue = '';
        return '';
    });

    // Notify backend when tab/window is ACTUALLY being closed
    // pagehide only fires if the user confirmed the exit dialog (not if they clicked Cancel/Stay)
    let beaconSent = false;
    function sendShutdownBeacon() {
        if (beaconSent) return;
        beaconSent = true;
        clearInterval(intervalId);
        const payload = JSON.stringify({ tab_id: tabSessionId });

        if (navigator.sendBeacon) {
            const blob = new Blob([payload], { type: 'application/json' });
            navigator.sendBeacon('/api/beacon_shutdown', blob);
        } else {
            fetch('/api/beacon_shutdown', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: payload,
                keepalive: true
            }).catch(() => {});
        }
    }

    window.addEventListener('pagehide', sendShutdownBeacon);
    window.addEventListener('unload', sendShutdownBeacon);
})();


