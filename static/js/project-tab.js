/**
 * Project Manager Tab Logic
 * Handles project creation, selection, and deletion
 */

// Current active project
let currentProject = null;
let selectedIcon = '1.png'; // Default icon

// DOM Elements
const newProjectNameInput = document.getElementById('new-project-name');
const newProjectDescInput = document.getElementById('new-project-description');
const createProjectBtn = document.getElementById('create-project-btn');
const refreshProjectsBtn = document.getElementById('refresh-projects-btn');
const projectsList = document.getElementById('projects-list');
const projectsStatus = document.getElementById('projects-status');
const currentProjectName = document.getElementById('current-project-name');
const iconSelector = document.getElementById('icon-selector');

/**
 * Initialize project tab
 */
function initProjectTab() {
    // Load icons
    loadIcons();
    
    // Load projects on tab load
    loadProjects();
    
    // Event listeners
    createProjectBtn.addEventListener('click', createNewProject);
    refreshProjectsBtn.addEventListener('click', loadProjects);
    
    // Enter key to create project
    newProjectNameInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            createNewProject();
        }
    });
}

/**
 * Load available icons from server
 */
async function loadIcons() {
    try {
        const response = await fetch('/api/icons');
        const data = await response.json();
        
        if (!data.success) {
            throw new Error(data.error || 'Error loading icons');
        }
        
        displayIcons(data.icons);
        
    } catch (error) {
        console.error('Error loading icons:', error);
        iconSelector.innerHTML = '<div class="error-message">Error loading icons</div>';
    }
}

/**
 * Display icons in selector
 */
function displayIcons(icons) {
    if (!icons || icons.length === 0) {
        iconSelector.innerHTML = '<div class="info-text">No icons available</div>';
        return;
    }
    
    iconSelector.innerHTML = icons.map(icon => `
        <div class="icon-option ${icon === selectedIcon ? 'selected' : ''}" 
             data-icon="${icon}"
             onclick="selectIcon('${icon}')">
            <img src="/api/icons/${icon}" alt="${icon}">
        </div>
    `).join('');
}

/**
 * Select an icon
 */
function selectIcon(icon) {
    selectedIcon = icon;
    
    // Update UI
    document.querySelectorAll('.icon-option').forEach(option => {
        if (option.dataset.icon === icon) {
            option.classList.add('selected');
        } else {
            option.classList.remove('selected');
        }
    });
}

/**
 * Load all projects from server
 */
async function loadProjects() {
    try {
        projectsList.innerHTML = '<div class="loading-message">Loading projects...</div>';
        
        const response = await fetch('/api/projects');
        const data = await response.json();
        
        if (!data.success) {
            throw new Error(data.error || 'Error loading projects');
        }
        
        displayProjects(data.projects);
        
    } catch (error) {
        console.error('Error loading projects:', error);
        showStatus('Error: ' + error.message, 'error');
        projectsList.innerHTML = '<div class="error-message">Error loading projects</div>';
    }
}

/**
 * Display projects in the grid
 */
function displayProjects(projects) {
    if (!projects || projects.length === 0) {
        projectsList.innerHTML = `
            <div class="empty-state">
                <p>📁 No projects found</p>
                <p class="info-text">Create your first project using the form above</p>
            </div>
        `;
        return;
    }
    
    projectsList.innerHTML = projects.map(project => createProjectCard(project)).join('');
    
    // Add event listeners to all buttons
    document.querySelectorAll('.project-select-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const projectId = e.target.dataset.projectId;
            selectProject(projectId);
        });
    });
    
    document.querySelectorAll('.project-delete-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            const projectId = e.target.dataset.projectId;
            const projectName = e.target.dataset.projectName;
            deleteProject(projectId, projectName);
        });
    });
}

/**
 * Create HTML for a project card
 */
function createProjectCard(project) {
    const isActive = currentProject && currentProject.project_id === project.project_id;
    const status = project.workflow_status;
    const icon = project.icon || '1.png';
    
    // Calculate progress
    const totalSteps = 5;
    let completedSteps = 0;
    if (status.pdf_processed) completedSteps++;
    if (status.images_extracted > 0) completedSteps++;
    if (status.model_applied) completedSteps++;
    if (status.masks_extracted > 0) completedSteps++;
    if (status.annotations_completed > 0) completedSteps++;
    const progressPercent = (completedSteps / totalSteps) * 100;
    
    // Format dates
    const createdDate = new Date(project.created_at).toLocaleDateString('it-IT');
    const modifiedDate = new Date(project.last_modified).toLocaleDateString('it-IT');
    
    return `
        <div class="project-card ${isActive ? 'project-active' : ''}" data-project-id="${project.project_id}">
            <div class="project-card-header">
                <img src="/api/icons/${icon}" alt="Project Icon" class="project-icon">
                <div class="project-header-content">
                    <h4 class="project-title">${escapeHtml(project.project_name)}</h4>
                    ${isActive ? '<span class="project-badge">✓ ACTIVE</span>' : ''}
                </div>
            </div>
            
            <div class="project-card-body">
                ${project.description ? `<p class="project-description">${escapeHtml(project.description)}</p>` : ''}
                
                <div class="project-stats">
                    <div class="project-stat">
                        <span class="stat-label">📄 PDF:</span>
                        <span class="stat-value">${status.pdf_count}</span>
                    </div>
                    <div class="project-stat">
                        <span class="stat-label">🖼️ Images:</span>
                        <span class="stat-value">${status.images_extracted}</span>
                    </div>
                    <div class="project-stat">
                        <span class="stat-label">🎭 Masks:</span>
                        <span class="stat-value">${status.masks_extracted}</span>
                    </div>
                    <div class="project-stat">
                        <span class="stat-label">✓ Annotations:</span>
                        <span class="stat-value">${status.annotations_completed}</span>
                    </div>
                </div>
                
                <div class="project-progress">
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: ${progressPercent}%"></div>
                    </div>
                    <span class="progress-text">${Math.round(progressPercent)}% completed</span>
                </div>
                
                <div class="project-dates">
                    <span class="project-date">Created: ${createdDate}</span>
                    <span class="project-date">Modified: ${modifiedDate}</span>
                </div>
            </div>
            
            <div class="project-card-footer">
                <button class="btn btn-primary project-select-btn" data-project-id="${project.project_id}">
                    ${isActive ? '✓ Selected' : '📂 Select'}
                </button>
                <button class="btn btn-danger project-delete-btn" 
                        data-project-id="${project.project_id}"
                        data-project-name="${escapeHtml(project.project_name)}">
                    🗑️ Delete
                </button>
            </div>
        </div>
    `;
}

/**
 * Create a new project
 */
async function createNewProject() {
    const projectName = newProjectNameInput.value.trim();
    const description = newProjectDescInput.value.trim();
    
    if (!projectName) {
        showStatus('Please enter a name for the project', 'error');
        newProjectNameInput.focus();
        return;
    }
    
    try {
        createProjectBtn.disabled = true;
        createProjectBtn.textContent = '⏳ Creating...';

        const response = await fetch('/api/projects', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                project_name: projectName,
                description: description,
                icon: selectedIcon
            })
        });
        
        const data = await response.json();
        
        if (!data.success) {
            throw new Error(data.error || 'Error creating project');
        }

        showStatus(`✓ Project "${projectName}" created successfully!`, 'success');

        // Clear form
        newProjectNameInput.value = '';
        newProjectDescInput.value = '';
        selectedIcon = '1.png'; // Reset to default
        
        // Reload projects
        await loadProjects();
        
        // Auto-select the new project
        selectProject(data.project.project_id);
        
    } catch (error) {
        console.error('Error creating project:', error);
        showStatus('Error: ' + error.message, 'error');
    } finally {
        createProjectBtn.disabled = false;
        createProjectBtn.textContent = '➕ Create Project';
    }
}

/**
 * Select a project as active
 */
async function selectProject(projectId) {
    try {
        const response = await fetch(`/api/projects/${projectId}`);
        const data = await response.json();
        
        if (!data.success) {
            throw new Error(data.error || 'Project not found');
        }
        
        currentProject = data.project;
        
        // Update UI
        currentProjectName.textContent = currentProject.project_name;
        
        // Save to localStorage
        localStorage.setItem('currentProjectId', projectId);
        localStorage.setItem('currentProjectName', currentProject.project_name);
        
        // Refresh project cards to show active state
        loadProjects();
        showStatus(`✓ Project "${currentProject.project_name}" selected`, 'success');
        
        // Notify other tabs about project change
        window.dispatchEvent(new CustomEvent('projectChanged', { 
            detail: { project: currentProject } 
        }));
        
    } catch (error) {
        console.error('Error selecting project:', error);
        showStatus('Error: ' + error.message, 'error');
    }
}

/**
 * Delete a project
 */
async function deleteProject(projectId, projectName) {
    const confirmed = confirm(
        `⚠️ WARNING!\n\nAre you sure you want to delete the project "${projectName}"?\n\n` +
        `This action will delete:\n` +
        `- All source PDFs\n` +
        `- All extracted images\n` +
        `- All masks and annotations\n` +
        `- All exported data\n\n` +
        `This action is IRREVERSIBLE!`
    );
    
    if (!confirmed) return;
    
    try {
        const response = await fetch(`/api/projects/${projectId}`, {
            method: 'DELETE'
        });
        
        const data = await response.json();
        
        if (!data.success) {
            throw new Error(data.error || 'Error deleting project');
        }
        
        showStatus(`✓ Project "${projectName}" deleted`, 'success');
        
        // If deleted project was active, clear it
        if (currentProject && currentProject.project_id === projectId) {
            currentProject = null;
            currentProjectName.textContent = 'No project selected';
            localStorage.removeItem('currentProjectId');
            localStorage.removeItem('currentProjectName');
            
            window.dispatchEvent(new CustomEvent('projectChanged', { 
                detail: { project: null } 
            }));
        }
        
        // Reload projects
        await loadProjects();
        
    } catch (error) {
        console.error('Error deleting project:', error);
        showStatus('Errore: ' + error.message, 'error');
    }
}

/**
 * Show status message
 */
function showStatus(message, type = 'info') {
    projectsStatus.textContent = message;
    projectsStatus.className = `status-message status-${type}`;
    projectsStatus.style.display = 'block';
    
    setTimeout(() => {
        projectsStatus.style.display = 'none';
    }, 5000);
}

/**
 * Escape HTML to prevent XSS
 */
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

/**
 * Get current project
 */
function getCurrentProject() {
    return currentProject;
}

/**
 * Load saved project from localStorage on page load
 */
function restoreSavedProject() {
    const savedProjectId = localStorage.getItem('currentProjectId');
    const savedProjectName = localStorage.getItem('currentProjectName');
    
    if (savedProjectId && savedProjectName) {
        currentProjectName.textContent = savedProjectName;
        // Set the current project without re-selecting (to avoid double loading)
        currentProject = {
            project_id: savedProjectId,
            project_name: savedProjectName
        };
        
        // Try to load full project data silently
        fetch(`/api/projects/${savedProjectId}`)
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    currentProject = data.project;
                    currentProjectName.textContent = currentProject.project_name;
                    
                    // Notify other tabs about project (only once)
                    window.dispatchEvent(new CustomEvent('projectChanged', { 
                        detail: { project: currentProject } 
                    }));
                } else {
                    throw new Error('Project not found');
                }
            })
            .catch(() => {
                // If project no longer exists, clear localStorage
                localStorage.removeItem('currentProjectId');
                localStorage.removeItem('currentProjectName');
                currentProjectName.textContent = 'No project selected';
                currentProject = null;
            });
    }
}

// Initialize on load
document.addEventListener('DOMContentLoaded', () => {
    initProjectTab();
    restoreSavedProject();
});

// Export for other modules
window.projectManager = {
    getCurrentProject,
    selectProject,
    loadProjects
};
