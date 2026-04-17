// Utility functions for PyPotteryLens Flask App

// API request wrapper
async function apiRequest(url, options = {}) {
    try {
        const response = await fetch(url, {
            ...options,
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            }
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Request failed');
        }

        return data;
    } catch (error) {
        console.error('API Request Error:', error);
        throw error;
    }
}

// Show loading overlay
function showLoading(message = 'Processing...') {
    const overlay = document.getElementById('loading-overlay');
    const messageEl = document.getElementById('loading-message');
    if (overlay && messageEl) {
        messageEl.textContent = message;
        overlay.style.display = 'flex';
    }
}

// Hide loading overlay
function hideLoading() {
    const overlay = document.getElementById('loading-overlay');
    if (overlay) {
        overlay.style.display = 'none';
    }
}

// Show toast notification
function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    if (!container) return;

    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.innerHTML = `
        <span>${getToastIcon(type)}</span>
        <span>${message}</span>
    `;

    container.appendChild(toast);

    // Auto remove after 5 seconds
    setTimeout(() => {
        toast.style.animation = 'slideOut 0.3s ease-out';
        setTimeout(() => toast.remove(), 300);
    }, 5000);
}

function getToastIcon(type) {
    const icons = {
        success: '✅',
        error: '❌',
        info: 'ℹ️',
        warning: '⚠️'
    };
    return icons[type] || icons.info;
}

// Show status message
function showStatus(elementId, message, type = 'info') {
    const element = document.getElementById(elementId);
    if (!element) return;

    element.textContent = message;
    element.className = `status-message ${type}`;
    element.style.display = 'block';
}

// Hide status message
function hideStatus(elementId) {
    const element = document.getElementById(elementId);
    if (element) {
        element.style.display = 'none';
    }
}

// File upload helper
async function uploadFile(file, url, additionalData = {}) {
    console.log('uploadFile called with:', { file: file.name, url, additionalData });
    
    const formData = new FormData();
    formData.append('file', file);

    for (const [key, value] of Object.entries(additionalData)) {
        console.log(`Adding form data: ${key} = ${value}`);
        formData.append(key, value);
    }

    try {
        console.log('Sending request to:', url);
        const response = await fetch(url, {
            method: 'POST',
            body: formData
        });

        console.log('Response status:', response.status);
        console.log('Response headers:', response.headers);

        const data = await response.json();
        console.log('Response data:', data);

        if (!response.ok) {
            throw new Error(data.error || 'Upload failed');
        }

        return data;
    } catch (error) {
        console.error('Upload Error:', error);
        throw error;
    }
}

// Populate dropdown
function populateDropdown(selectId, items, placeholder = 'Select...') {
    const select = document.getElementById(selectId);
    if (!select) return;

    select.innerHTML = `<option value="">${placeholder}</option>`;
    items.forEach(item => {
        const option = document.createElement('option');
        option.value = item;
        option.textContent = item;
        select.appendChild(option);
    });
}

// Debounce function
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Image loader with cache
const imageCache = new Map();

async function loadImage(url) {
    if (imageCache.has(url)) {
        return imageCache.get(url);
    }

    return new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => {
            imageCache.set(url, img);
            resolve(img);
        };
        img.onerror = reject;
        img.src = url;
    });
}

// Format file size
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}

// Validate input
function validateInput(value, rules) {
    for (const rule of rules) {
        if (rule.type === 'required' && !value) {
            return rule.message || 'This field is required';
        }
        if (rule.type === 'pattern' && !rule.pattern.test(value)) {
            return rule.message || 'Invalid format';
        }
        if (rule.type === 'min' && value < rule.value) {
            return rule.message || `Minimum value is ${rule.value}`;
        }
        if (rule.type === 'max' && value > rule.value) {
            return rule.message || `Maximum value is ${rule.value}`;
        }
    }
    return null;
}

// Create canvas from image
function imageToCanvas(img, maxWidth = 800, maxHeight = 600) {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');

    let width = img.width;
    let height = img.height;

    // Calculate scaling
    if (width > maxWidth) {
        height = height * (maxWidth / width);
        width = maxWidth;
    }
    if (height > maxHeight) {
        width = width * (maxHeight / height);
        height = maxHeight;
    }

    canvas.width = width;
    canvas.height = height;
    ctx.drawImage(img, 0, 0, width, height);

    return canvas;
}

// Download file
function downloadFile(data, filename, mimeType = 'text/plain') {
    const blob = new Blob([data], { type: mimeType });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
}

// Poll operation progress and show in status element
async function pollOperationProgress(operation, statusElementId, progressBarId) {
    let attempts = 0;
    const maxAttempts = 600; // 5 minutes max
    let seenActive = false; // Track whether we've seen the operation start
    
    const statusEl = document.getElementById(statusElementId);
    const progressBar = document.getElementById(progressBarId);
    
    while (attempts < maxAttempts) {
        try {
            const response = await fetch('/api/operation-progress');
            const data = await response.json();
            
            if (data.active && data.operation === operation) {
                // Operation is running, update UI
                seenActive = true;
                if (statusEl) {
                    statusEl.textContent = `${data.message} (${data.percent}%)`;
                    statusEl.className = 'status-message info';
                }
                if (progressBar) {
                    progressBar.style.width = `${data.percent}%`;
                    progressBar.textContent = `${data.percent}%`;
                }
            } else if (!data.active && seenActive) {
                // Operation completed (was active, now finished)
                if (progressBar) {
                    progressBar.style.width = '100%';
                    progressBar.textContent = '100%';
                }
                return true;
            }
            
            await new Promise(resolve => setTimeout(resolve, 500));
            attempts++;
        } catch (error) {
            console.error('Error polling progress:', error);
            break;
        }
    }
    
    return false;
}

// Execute operation with progress tracking (no loading overlay)
async function executeWithProgress(operation, executeFunc, statusElementId, progressBarId) {
    try {
        const statusEl = document.getElementById(statusElementId);
        const progressBar = document.getElementById(progressBarId);
        
        // Reset progress bar
        if (progressBar) {
            progressBar.style.width = '0%';
            progressBar.textContent = '0%';
        }
        if (statusEl) {
            statusEl.textContent = 'Starting...';
            statusEl.className = 'status-message info';
        }
        
        // Start polling
        const pollPromise = pollOperationProgress(operation, statusElementId, progressBarId);
        
        // Execute the operation
        const resultPromise = executeFunc();
        
        // Wait for both
        const [pollResult, operationResult] = await Promise.all([pollPromise, resultPromise]);
        
        return operationResult;
        
    } catch (error) {
        const statusEl = document.getElementById(statusElementId);
        if (statusEl) {
            statusEl.textContent = `Error: ${error.message}`;
            statusEl.className = 'status-message error';
        }
        throw error;
    }
}

// Export utilities
window.PyPotteryUtils = {
    apiRequest,
    showLoading,
    hideLoading,
    showToast,
    showStatus,
    hideStatus,
    uploadFile,
    pollOperationProgress,
    executeWithProgress,
    populateDropdown,
    debounce,
    loadImage,
    formatFileSize,
    validateInput,
    imageToCanvas,
    downloadFile
};
