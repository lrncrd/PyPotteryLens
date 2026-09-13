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
    const pdfDropzone = document.getElementById('pdf-dropzone');
    const pdfStatus = 'pdf-status';

    // State tracking
    let isDropzoneLocked = false;

    // Helper to display selected project in the PDF tab
    function updateProjectDisplay(project) {
        if (!pdfSelectedInfo) return;
        if (!project) {
            pdfSelectedInfo.innerHTML = '<span class="info-empty"><i class="bi bi-exclamation-circle"></i> No project selected — please select or create a project in Project Manager</span>';
            setDropzoneLocked(false);
            return;
        }

        const status = project.workflow_status || {};
        const hasPdf = Boolean(project.has_pdf || (status.pdf_count && status.pdf_count > 0) || project.pdf_filename);
        const pdfName = project.pdf_filename || (hasPdf ? 'Document.pdf' : null);

        if (hasPdf) {
            pdfSelectedInfo.innerHTML = `
                <div class="dossier-pill-row">
                    <span class="dossier-pill target-project"><i class="bi bi-folder2-open"></i> <strong>Target:</strong> ${escapeHtml(project.project_name)}</span>
                    <span class="dossier-pill target-file success"><i class="bi bi-file-earmark-check"></i> <strong>PDF Active:</strong> ${escapeHtml(pdfName)}</span>
                    <button type="button" id="delete-pdf-btn" class="btn btn-sm btn-outline-danger delete-pdf-btn" title="Remove current PDF to upload a new document">
                        <i class="bi bi-trash3"></i> Remove PDF
                    </button>
                </div>
            `;

            // Wire delete button
            const deleteBtn = document.getElementById('delete-pdf-btn');
            if (deleteBtn) {
                deleteBtn.addEventListener('click', async (e) => {
                    e.stopPropagation();
                    await handleDeletePdf(project.project_id, project.project_name, pdfName);
                });
            }

            setDropzoneLocked(true, pdfName);
        } else {
            pdfSelectedInfo.innerHTML = `
                <div class="dossier-pill-row">
                    <span class="dossier-pill target-project"><i class="bi bi-folder2-open"></i> <strong>Target:</strong> ${escapeHtml(project.project_name)}</span>
                    <span class="dossier-pill target-file"><i class="bi bi-file-earmark-pdf"></i> Awaiting document</span>
                </div>
            `;
            setDropzoneLocked(false);
        }
    }

    function setDropzoneLocked(locked, activePdfName = '') {
        isDropzoneLocked = locked;
        if (!pdfDropzone) return;

        const titleEl = pdfDropzone.querySelector('.dropzone-title');
        const subEl = pdfDropzone.querySelector('.dropzone-sub');

        if (locked) {
            pdfDropzone.classList.add('dropzone-locked');
            if (uploadBtn) {
                uploadBtn.disabled = true;
                uploadBtn.classList.add('btn-disabled');
                uploadBtn.innerHTML = '<i class="bi bi-lock-fill"></i> PDF Already Uploaded';
            }
            if (titleEl) titleEl.innerHTML = '<i class="bi bi-file-earmark-lock"></i> PDF Already Ingested';
            if (subEl) subEl.textContent = `Active: "${activePdfName || 'Document.pdf'}". A project can only contain 1 PDF. Click "Remove PDF" above if you wish to upload another document.`;
        } else {
            pdfDropzone.classList.remove('dropzone-locked');
            if (uploadBtn) {
                uploadBtn.disabled = false;
                uploadBtn.classList.remove('btn-disabled');
                uploadBtn.innerHTML = '<i class="bi bi-file-earmark-arrow-up"></i> Select PDF File';
            }
            if (titleEl) titleEl.textContent = 'Upload Ceramic Publication PDF';
            if (subEl) subEl.textContent = 'Drag and drop your report or monograph PDF here, or click to browse files';
        }
    }

    async function handleDeletePdf(projectId, projectName, pdfName) {
        const confirmed = await window.PyPotteryUtils.showConfirmDialog({
            title: 'Remove PDF Document',
            subtitle: `Are you sure you want to remove <strong>"${escapeHtml(pdfName || 'the PDF')}"</strong> from project <strong>"${escapeHtml(projectName)}"</strong>?`,
            icon: 'bi-trash3-fill',
            iconColor: '#dc2626',
            iconBg: '#fee2e2',
            detailsLabel: 'This action will remove:',
            details: [
                'The ingested PDF document file',
                'All extracted page images and spreads',
                'Pre-processed raster cache for this document'
            ],
            note: 'You will need to upload a new PDF to process ceramic illustrations.',
            confirmText: 'Remove PDF',
            cancelText: 'Cancel',
            confirmClass: 'btn-danger'
        });
        if (!confirmed) return;

        try {
            window.PyPotteryUtils.showLoading('Removing PDF and extracted images...');
            const response = await fetch(`/api/projects/${projectId}/pdf`, {
                method: 'DELETE'
            });
            const data = await response.json();
            window.PyPotteryUtils.hideLoading();

            if (data.success) {
                window.PyPotteryUtils.showToast('PDF removed successfully. Ready for new upload.', 'success');
                window.PyPotteryUtils.showStatus(pdfStatus, 'PDF document removed from project.', 'info');

                // Refresh current project from server
                const projRes = await fetch(`/api/projects/${projectId}`);
                const projData = await projRes.json();
                if (projData.success) {
                    updateProjectDisplay(projData.project);
                    window.dispatchEvent(new CustomEvent('projectChanged', { 
                        detail: { project: projData.project } 
                    }));
                }
                if (window.projectManager && window.projectManager.loadProjects) {
                    window.projectManager.loadProjects();
                }
            } else {
                const err = data.error || 'Failed to remove PDF';
                window.PyPotteryUtils.showToast(err, 'error');
                window.PyPotteryUtils.showStatus(pdfStatus, err, 'error');
            }
        } catch (err) {
            window.PyPotteryUtils.hideLoading();
            console.error('Delete PDF error:', err);
            window.PyPotteryUtils.showToast(err.message || String(err), 'error');
        }
    }

    // Escape helper
    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    // Wire button to input
    if (uploadBtn && uploadInput) {
        uploadBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            if (isDropzoneLocked) {
                window.PyPotteryUtils.showToast('This project already has a PDF. Click "Remove PDF" above to replace it.', 'info');
                return;
            }
            uploadInput.click();
        });
    }

    // Wire dropzone click & drag-and-drop
    if (pdfDropzone && uploadInput) {
        pdfDropzone.addEventListener('click', (e) => {
            if (isDropzoneLocked) {
                window.PyPotteryUtils.showToast('This project already has a PDF. Click "Remove PDF" above to replace it.', 'info');
                return;
            }
            if (e.target !== uploadBtn && !uploadBtn.contains(e.target)) {
                uploadInput.click();
            }
        });

        ['dragenter', 'dragover'].forEach(eventName => {
            pdfDropzone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                if (!isDropzoneLocked) {
                    pdfDropzone.classList.add('drag-active');
                }
            });
        });

        ['dragleave', 'drop'].forEach(eventName => {
            pdfDropzone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                pdfDropzone.classList.remove('drag-active');
            });
        });

        pdfDropzone.addEventListener('drop', (e) => {
            if (isDropzoneLocked) {
                window.PyPotteryUtils.showToast('This project already has a PDF. Click "Remove PDF" in Ingestion Status to replace it.', 'warning');
                return;
            }
            const dt = e.dataTransfer;
            const files = dt.files;
            if (files && files.length > 0) {
                uploadInput.files = files;
                uploadInput.dispatchEvent(new Event('change'));
            }
        });
    }

    // Update display from currentProject if available
    if (window.projectManager && window.projectManager.getCurrentProject) {
        updateProjectDisplay(window.projectManager.getCurrentProject());
    } else {
        const savedId = localStorage.getItem('currentProjectId');
        const savedName = localStorage.getItem('currentProjectName');
        if (savedId) {
            fetch(`/api/projects/${savedId}`)
                .then(r => r.json())
                .then(d => {
                    if (d.success && d.project) {
                        updateProjectDisplay(d.project);
                    } else if (savedName && pdfSelectedInfo) {
                        pdfSelectedInfo.innerHTML = `
                            <div class="dossier-pill-row">
                                <span class="dossier-pill target-project"><i class="bi bi-folder2-open"></i> <strong>Target:</strong> ${escapeHtml(savedName)}</span>
                                <span class="dossier-pill target-file"><i class="bi bi-file-earmark-pdf"></i> Awaiting document</span>
                            </div>
                        `;
                    }
                })
                .catch(() => {});
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

        if (isDropzoneLocked) {
            window.PyPotteryUtils.showToast('This project already has a PDF. Click "Remove PDF" above to replace it.', 'warning');
            uploadInput.value = '';
            return;
        }

        try {
            window.PyPotteryUtils.showLoading('Uploading and processing PDF...');

            const splitPages = splitPagesCheckbox ? splitPagesCheckbox.checked : false;

            const data = await window.PyPotteryUtils.uploadFile(file, '/api/pdf/upload', {
                split_pages: splitPages,
                project_id: project.project_id
            });

            window.PyPotteryUtils.hideLoading();

            if (data && data.success) {
                const msg = data.message || 'PDF processed successfully';
                window.PyPotteryUtils.showStatus(pdfStatus, msg, 'success');
                window.PyPotteryUtils.showToast('PDF uploaded to project: ' + project.project_name, 'success');

                // Reload project to update has_pdf / pdf_filename and dropzone lock
                const projRes = await fetch(`/api/projects/${project.project_id}`);
                const projData = await projRes.json();
                if (projData.success) {
                    updateProjectDisplay(projData.project);
                    window.dispatchEvent(new CustomEvent('projectChanged', { 
                        detail: { project: projData.project } 
                    }));
                }
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
