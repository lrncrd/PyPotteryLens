// PDF Processing Tab JavaScript

// Project-aware PDF Upload Handler
document.addEventListener('DOMContentLoaded', () => {
    console.log('PDF tab initialized');

    if (!window.PyPotteryUtils) {
        console.error('PyPotteryUtils not loaded!');
        return;
    }

    const uploadInput = document.getElementById('pdf-upload');
    const uploadBtn = document.getElementById('pdf-upload-btn');
    const splitPagesCheckbox = document.getElementById('split-pages');
    const pdfSelectedInfo = document.getElementById('pdf-selected-info');
    const pdfStatus = 'pdf-status';

    // Helper to display selected project in the PDF tab
    function updateProjectDisplay(project) {
        if (!pdfSelectedInfo) return;
        if (!project) {
            pdfSelectedInfo.innerHTML = '<em>No file uploaded yet — no project selected</em>';
            return;
        }
        pdfSelectedInfo.innerHTML = `<strong>Active project:</strong> ${escapeHtml(project.project_name)}`;
    }

    // Escape helper
    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    // Wire button to input
    if (uploadBtn && uploadInput) {
        uploadBtn.addEventListener('click', () => uploadInput.click());
    }

    // Update display from currentProject if available
    if (window.projectManager && window.projectManager.getCurrentProject) {
        updateProjectDisplay(window.projectManager.getCurrentProject());
    } else {
        // Try localStorage fallback
        const savedName = localStorage.getItem('currentProjectName');
        if (savedName && pdfSelectedInfo) {
            pdfSelectedInfo.innerHTML = `<strong>Active project:</strong> ${escapeHtml(savedName)}`;
        }
    }

    // Listen to project changes
    window.addEventListener('projectChanged', (e) => {
        const project = e.detail && e.detail.project ? e.detail.project : null;
        updateProjectDisplay(project);
    });

    if (!uploadInput) {
        console.error('PDF upload input not found!');
        return;
    }

    uploadInput.addEventListener('change', async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        if (!file.name.toLowerCase().endsWith('.pdf')) {
            window.PyPotteryUtils.showToast('Please select a PDF file', 'error');
            uploadInput.value = '';
            return;
        }

        // Get current project
        let project = null;
        if (window.projectManager && window.projectManager.getCurrentProject) {
            project = window.projectManager.getCurrentProject();
        }
        // fallback to localStorage
        if (!project) {
            const pid = localStorage.getItem('currentProjectId');
            const pname = localStorage.getItem('currentProjectName');
            if (pid) project = { project_id: pid, project_name: pname || 'Unnamed' };
        }

        if (!project || !project.project_id) {
            window.PyPotteryUtils.showToast('No project selected. Please select or create a project first.', 'error');
            uploadInput.value = '';
            return;
        }

        try {
            window.PyPotteryUtils.showLoading('Uploading and processing PDF...');

            const splitPages = splitPagesCheckbox ? splitPagesCheckbox.checked : false;

            // Use uploadFile helper: it will append fields to FormData
            const data = await window.PyPotteryUtils.uploadFile(file, '/api/pdf/upload', {
                split_pages: splitPages,
                project_id: project.project_id
            });

            window.PyPotteryUtils.hideLoading();

            if (data && data.success) {
                const msg = data.message || 'PDF processed successfully';
                window.PyPotteryUtils.showStatus(pdfStatus, msg, 'success');
                window.PyPotteryUtils.showToast('PDF uploaded to project: ' + project.project_name, 'success');
                // Update display with filename + project
                pdfSelectedInfo.innerHTML = `<strong>Project:</strong> ${escapeHtml(project.project_name)} &nbsp; • &nbsp; <strong>File:</strong> ${escapeHtml(file.name)}`;
                // Optionally trigger a refresh of project metadata
                if (window.projectManager && window.projectManager.loadProjects) {
                    window.projectManager.loadProjects();
                }
            } else {
                const err = data && data.error ? data.error : 'Failed to process PDF';
                window.PyPotteryUtils.showStatus(pdfStatus, err, 'error');
                window.PyPotteryUtils.showToast(err, 'error');
            }
        } catch (err) {
            window.PyPotteryUtils.hideLoading();
            console.error('PDF upload error:', err);
            window.PyPotteryUtils.showStatus(pdfStatus, err.message || String(err), 'error');
            window.PyPotteryUtils.showToast(err.message || String(err), 'error');
        } finally {
            uploadInput.value = '';
        }
    });
});
