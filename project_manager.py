"""
Project Manager for PyPotteryLens
Handles creation, loading, and management of project workspaces
"""

import json
import re
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional


def _natural_sort_key(s: str):
    """Key function for natural (human) sort order: image_2 before image_10."""
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]

class ProjectManager:
    """Manages project workspaces with hierarchical folder structure"""
    
    def __init__(self, projects_root: str = "projects"):
        self.projects_root = Path(projects_root)
        self.projects_root.mkdir(exist_ok=True)
    
    def create_project(self, project_name: str, description: str = "", icon: str = "1.png") -> Dict:
        """
        Create a new project with folder structure and metadata
        
        Args:
            project_name: Name of the project
            description: Optional project description
            icon: Icon filename (default: "1.png")
            
        Returns:
            Dict with project metadata
        """
        # Sanitize project name for filesystem
        safe_name = "".join(c for c in project_name if c.isalnum() or c in (' ', '-', '_')).strip()
        safe_name = safe_name.replace(' ', '_')
        
        # Create unique ID based on timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        project_id = f"{safe_name}_{timestamp}"
        
        project_path = self.projects_root / project_id
        
        # Check if project already exists
        if project_path.exists():
            raise ValueError(f"Project already exists: {project_id}")
        
        # Create folder structure
        folders = [
            'pdf_source',      # Original PDF files
            'images',          # Extracted images from PDFs
            'masks',           # Mask images (_mask folder content)
            'cards',           # Card annotations (_card folder content)
            'cards_modified',  # Post-processed cards
            'exports',         # Final exports (CSV, visualizations)
            'models'           # Model files used in this project
        ]
        
        for folder in folders:
            (project_path / folder).mkdir(parents=True, exist_ok=True)
        
        # Create project metadata
        metadata = {
            'project_id': project_id,
            'project_name': project_name,
            'description': description,
            'icon': icon,
            'created_at': datetime.now().isoformat(),
            'last_modified': datetime.now().isoformat(),
            'workflow_status': {
                'pdf_processed': False,
                'pdf_count': 0,
                'images_extracted': 0,
                'model_applied': False,
                'masks_extracted': 0,
                'annotations_completed': 0,
                'total_images': 0,
                'reviewed_images': []  # List of reviewed image names
            },
            'settings': {
                'model_file': None,
                'confidence_threshold': 0.5,
                'excluded_images': []
            }
        }
        
        # Save metadata
        self._save_metadata(project_path, metadata)
        
        return metadata
    
    def update_excluded_images(self, project_id: str, excluded_images: list) -> bool:
        """
        Update list of excluded images for a project
        
        Args:
            project_id: ID of the project
            excluded_images: List of image filenames to exclude
            
        Returns:
            True if successful, False otherwise
        """
        project_path = self.projects_root / project_id
        
        if not project_path.exists():
            return False
        
        try:
            metadata = self._load_metadata(project_path)
            metadata['settings']['excluded_images'] = excluded_images
            metadata['last_modified'] = datetime.now().isoformat()
            self._save_metadata(project_path, metadata)
            return True
        except Exception as e:
            print(f"Error updating excluded images for {project_id}: {e}")
            return False
    
    def list_projects(self) -> List[Dict]:
        """
        List all available projects
        
        Returns:
            List of project metadata dictionaries
        """
        projects = []
        
        if not self.projects_root.exists():
            return projects
        
        for project_dir in self.projects_root.iterdir():
            if project_dir.is_dir():
                metadata_file = project_dir / 'project.json'
                if metadata_file.exists():
                    try:
                        metadata = self._load_metadata(project_dir)
                        projects.append(metadata)
                    except Exception as e:
                        print(f"Error loading project {project_dir.name}: {e}")
        
        # Sort by last modified (most recent first)
        projects.sort(key=lambda x: x.get('last_modified', ''), reverse=True)
        
        return projects
    
    def get_project(self, project_id: str) -> Optional[Dict]:
        """
        Get metadata for a specific project
        
        Args:
            project_id: ID of the project
            
        Returns:
            Project metadata or None if not found
        """
        project_path = self.projects_root / project_id
        
        if not project_path.exists():
            return None
        
        try:
            return self._load_metadata(project_path)
        except Exception as e:
            print(f"Error loading project {project_id}: {e}")
            return None
    
    def delete_project(self, project_id: str) -> bool:
        """
        Delete a project and all its contents
        
        Args:
            project_id: ID of the project to delete
            
        Returns:
            True if successful, False otherwise
        """
        project_path = self.projects_root / project_id
        
        if not project_path.exists():
            return False
        
        try:
            shutil.rmtree(project_path)
            return True
        except Exception as e:
            print(f"Error deleting project {project_id}: {e}")
            return False
    
    def update_workflow_status(self, project_id: str, status_updates: Dict) -> bool:
        """
        Update workflow status for a project
        
        Args:
            project_id: ID of the project
            status_updates: Dictionary of status fields to update
            
        Returns:
            True if successful, False otherwise
        """
        project_path = self.projects_root / project_id
        
        if not project_path.exists():
            return False
        
        try:
            metadata = self._load_metadata(project_path)
            metadata['workflow_status'].update(status_updates)
            metadata['last_modified'] = datetime.now().isoformat()
            self._save_metadata(project_path, metadata)
            return True
        except Exception as e:
            print(f"Error updating workflow status for {project_id}: {e}")
            return False
    
    def update_settings(self, project_id: str, settings: Dict) -> bool:
        """
        Update project settings
        
        Args:
            project_id: ID of the project
            settings: Dictionary of settings to update
            
        Returns:
            True if successful, False otherwise
        """
        project_path = self.projects_root / project_id
        
        if not project_path.exists():
            return False
        
        try:
            metadata = self._load_metadata(project_path)
            metadata['settings'].update(settings)
            metadata['last_modified'] = datetime.now().isoformat()
            self._save_metadata(project_path, metadata)
            return True
        except Exception as e:
            print(f"Error updating settings for {project_id}: {e}")
            return False
    
    def add_reviewed_image(self, project_id: str, image_name: str) -> bool:
        """
        Mark an image as reviewed
        
        Args:
            project_id: ID of the project
            image_name: Name of the reviewed image
            
        Returns:
            True if successful, False otherwise
        """
        project_path = self.projects_root / project_id
        
        if not project_path.exists():
            return False
        
        try:
            metadata = self._load_metadata(project_path)
            reviewed = metadata['workflow_status'].get('reviewed_images', [])
            
            if image_name not in reviewed:
                reviewed.append(image_name)
                metadata['workflow_status']['reviewed_images'] = reviewed
                metadata['workflow_status']['annotations_completed'] = len(reviewed)
                metadata['last_modified'] = datetime.now().isoformat()
                self._save_metadata(project_path, metadata)
            
            return True
        except Exception as e:
            print(f"Error adding reviewed image for {project_id}: {e}")
            return False
    
    def get_project_path(self, project_id: str, subfolder: str = None) -> Optional[Path]:
        """
        Get the filesystem path for a project or its subfolder
        
        Args:
            project_id: ID of the project
            subfolder: Optional subfolder name (images, masks, cards, etc.)
            
        Returns:
            Path object or None if project doesn't exist
        """
        project_path = self.projects_root / project_id
        
        if not project_path.exists():
            return None
        
        if subfolder:
            return project_path / subfolder
        
        return project_path
    
    def _load_metadata(self, project_path: Path) -> Dict:
        """Load project metadata from project.json"""
        metadata_file = project_path / 'project.json'
        with open(metadata_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _save_metadata(self, project_path: Path, metadata: Dict):
        """Save project metadata to project.json"""
        metadata_file = project_path / 'project.json'
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    def get_images_list(self, project_id: str, folder_type: str = 'images') -> List[str]:
        """
        Get list of images in a project folder
        
        Args:
            project_id: ID of the project
            folder_type: Type of folder (images, masks, cards)
            
        Returns:
            List of image filenames
        """
        folder_path = self.get_project_path(project_id, folder_type)
        
        if not folder_path or not folder_path.exists():
            return []
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
        images = []
        
        for file_path in folder_path.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                images.append(file_path.name)
        
        return sorted(images, key=_natural_sort_key)
    
    def count_files(self, project_id: str, folder_type: str = 'images') -> int:
        """
        Count files in a project folder
        
        Args:
            project_id: ID of the project
            folder_type: Type of folder
            
        Returns:
            Number of files
        """
        return len(self.get_images_list(project_id, folder_type))
