"""
Flask Application for PyPotteryLens
Migrated from Gradio to Flask with native HTML, CSS, and JavaScript
"""

from flask import Flask, render_template, request, jsonify, send_file, send_from_directory, Response
from pathlib import Path
import os
import re
import json
import base64
from io import BytesIO
import pandas as pd
import numpy as np
from werkzeug.utils import secure_filename
import torch
import gc
import threading
import time

from utils import (
    PDFProcessor,
    FewShotProcessor,
    MaskExtractor,
    TabularProcessor,
    ExportProcessor,
    PDFConfig,
    FewShotConfig,
    MaskExtractionConfig,
    TabularConfig,
    ExportConfig,
    _save_mask_rgba,
)

from project_manager import ProjectManager

app = Flask(__name__)
app.config['SECRET_KEY'] = 'pypotterylens-secret-key-2024'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
app.config['UPLOAD_FOLDER'] = Path('temp_uploads')
app.config['UPLOAD_FOLDER'].mkdir(exist_ok=True)

# Initialize Project Manager
project_manager = ProjectManager(projects_root="projects")

# === Gemma 4 AI model (lazy loaded for bibliographic extraction) ===
_gemma_model = None
_gemma_processor = None
_gemma_model_lock = threading.Lock()

def load_gemma_model(hf_token=None):
    """Lazy-load google/gemma-4-E2B-it for vision-based bibliographic extraction."""
    global _gemma_model, _gemma_processor
    with _gemma_model_lock:
        if _gemma_model is None:
            from transformers import AutoProcessor, AutoModelForMultimodalLM
            model_id = "google/gemma-4-E2B-it"
            token = hf_token or os.environ.get('HF_TOKEN', '') or None
            cache_dir = Path("models_llm")
            cache_dir.mkdir(exist_ok=True)
            print(f"[AI] Loading {model_id} into {cache_dir}...")
            _gemma_processor = AutoProcessor.from_pretrained(
                model_id, token=token, cache_dir=cache_dir
            )
            if torch.cuda.is_available():
                device_map = "auto"
            else:
                # MPS (Apple Silicon) has a max single-buffer limit (~4GB) that
                # prevents loading Gemma 4 E2B (5.1B params). Use CPU instead.
                device_map = {"": "cpu"}
            _gemma_model = AutoModelForMultimodalLM.from_pretrained(
                model_id,
                token=token,
                dtype="auto",
                device_map=device_map,
                cache_dir=cache_dir,
                low_cpu_mem_usage=True
            )
            print(f"[AI] Gemma 4 E2B-it loaded (device_map={device_map})")
    return _gemma_model, _gemma_processor

# Initialization status tracking
init_status = {
    'ready': False,
    'stage': 'starting',
    'progress': 0,
    'message': 'Initializing...'
}

def update_init_status(stage, progress, message):
    """Update initialization status"""
    global init_status
    init_status['stage'] = stage
    init_status['progress'] = progress
    init_status['message'] = message
    print(f"[Init] {progress}% - {message}")

# Global progress tracking
operation_progress = {
    'active': False,
    'operation': '',
    'total': 0,
    'current': 0,
    'message': '',
    'percent': 0
}

def update_operation_progress(operation, current, total, message=''):
    """Update operation progress for frontend polling"""
    global operation_progress
    operation_progress['active'] = True
    operation_progress['operation'] = operation
    operation_progress['current'] = current
    operation_progress['total'] = total
    operation_progress['message'] = message
    operation_progress['percent'] = int((current / total * 100)) if total > 0 else 0
    print(f"[{operation}] {operation_progress['percent']}% - {message}")

def clear_operation_progress():
    """Clear operation progress"""
    global operation_progress
    operation_progress['active'] = False
    operation_progress['operation'] = ''
    operation_progress['total'] = 0
    operation_progress['current'] = 0
    operation_progress['message'] = ''
    operation_progress['percent'] = 0

# Initialize directories
ROOT_DIR = Path(".")
PRED_OUTPUT_DIR = ROOT_DIR / "outputs"
PDFIMG_OUTPUT_DIR = ROOT_DIR / "pdf2img_outputs"
MODELS_DIR = ROOT_DIR / "models_vision"
MODELS_CLASSIFIER_DIR = ROOT_DIR / "models_classifier"
ASSETS_DIR = ROOT_DIR / "imgs"

# Create necessary directories
for directory in [PDFIMG_OUTPUT_DIR, MODELS_DIR, PRED_OUTPUT_DIR, MODELS_CLASSIFIER_DIR]:
    directory.mkdir(exist_ok=True)


# ==================== MODEL INITIALIZATION ====================

def download_model(url, destination, model_name, base_progress, progress_range):
    """Download a model file with progress tracking for splash screen"""
    import urllib.request
    import sys
    
    print(f"\n📥 Downloading {model_name}...")
    print(f"   URL: {url}")
    print(f"   Destination: {destination}")
    
    def show_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(downloaded * 100 / total_size, 100)
            
            # Update splash screen progress
            current_progress = base_progress + (percent / 100.0) * progress_range
            downloaded_mb = downloaded / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            update_init_status('downloading_models', current_progress, 
                             f'Downloading {model_name}: {downloaded_mb:.1f}/{total_mb:.1f} MB')
            
            # Console progress bar
            bar_length = 50
            filled_length = int(bar_length * percent / 100)
            bar = '█' * filled_length + '░' * (bar_length - filled_length)
            
            sys.stdout.write(f'\r   [{bar}] {percent:.1f}% ({downloaded_mb:.1f}/{total_mb:.1f} MB)')
            sys.stdout.flush()
    
    try:
        urllib.request.urlretrieve(url, destination, show_progress)
        print(f"\n   ✅ Successfully downloaded {model_name}")
        return True
    except Exception as e:
        print(f"\n   ❌ Error downloading {model_name}: {e}")
        return False


def initialize_models():
    """Download SAM2 and DINOv2 weights locally at startup with progress display."""
    update_init_status('checking_models', 10, 'Checking model weights...')

    # ── SAM2 ──────────────────────────────────────────────────────────────────
    sam2_path = MODELS_DIR / 'sam2.1_hiera_large.pt'
    sam2_url = 'https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt'

    if sam2_path.exists():
        print(f"✅ SAM2 checkpoint found at {sam2_path}")
    else:
        print(f"⬇️  Downloading SAM2 checkpoint (~1.2 GB) → {sam2_path}")
        update_init_status('downloading', 15, 'Downloading SAM2 Hiera Large (~1.2 GB)...')
        download_model(sam2_url, sam2_path, 'SAM2 Hiera Large', 15, 35)

    # ── DINOv2 ────────────────────────────────────────────────────────────────
    dinov2_hub        = MODELS_DIR / 'hub'
    dinov2_checkpoints = dinov2_hub / 'checkpoints'
    dinov2_checkpoints.mkdir(parents=True, exist_ok=True)

    dinov2_weights = dinov2_checkpoints / 'dinov2_vitl14_pretrain.pth'
    dinov2_url     = 'https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth'

    if dinov2_weights.exists():
        print(f"✅ DINOv2 weights found at {dinov2_weights}")
    else:
        print(f"⬇️  Downloading DINOv2 ViT-L/14 weights (~1.1 GB) → {dinov2_weights}")
        update_init_status('downloading', 50, 'Downloading DINOv2 ViT-L/14 (~1.1 GB)...')
        download_model(dinov2_url, dinov2_weights, 'DINOv2 ViT-L/14', 50, 30)

    update_init_status('models_ready', 80, 'Models ready, initializing processors...')


def initialize_processors():
    """Initialize all processors"""
    global pdf_processor, few_shot_processor, mask_extractor, tabular_processor, export_processor

    update_init_status('init_processors', 85, 'Initializing PDF processor...')
    pdf_processor = PDFProcessor(PDFConfig(output_dir=PDFIMG_OUTPUT_DIR))

    update_init_status('init_processors', 90, 'Initializing few-shot processor (SAM2+DINOv2 — lazy loaded)...')
    few_shot_processor = FewShotProcessor(FewShotConfig(
        sam2_checkpoint_path=MODELS_DIR / "sam2.1_hiera_large.pt",
        dinov2_hub_dir=MODELS_DIR / "hub",
    ))

    update_init_status('init_processors', 94, 'Initializing mask extractor...')
    mask_extractor = MaskExtractor(MaskExtractionConfig(
        pdfimg_output_dir=PDFIMG_OUTPUT_DIR,
        pred_output_dir=PRED_OUTPUT_DIR
    ))

    update_init_status('init_processors', 96, 'Initializing tabular processor...')
    tabular_processor = TabularProcessor(TabularConfig(
        pdfimg_output_dir=PDFIMG_OUTPUT_DIR,
        pred_output_dir=PRED_OUTPUT_DIR
    ))

    update_init_status('init_processors', 99, 'Initializing export processor...')
    export_processor = ExportProcessor(ExportConfig(
        pred_output_dir=PRED_OUTPUT_DIR
    ))

    # Mark as ready
    update_init_status('ready', 100, 'Application ready!')
    init_status['ready'] = True
    print("\n" + "="*80)
    print(" ✅ All processors initialized - Application ready!")
    print("="*80)


# Initialize processor variables as None
pdf_processor = None
few_shot_processor = None
mask_extractor = None
tabular_processor = None
export_processor = None


def background_initialization():
    """Run initialization in background thread"""
    import threading
    
    def init_thread():
        try:
            update_init_status('init_start', 5, 'Starting initialization...')
            initialize_models()
            initialize_processors()
        except Exception as e:
            print(f"ERROR during initialization: {e}")
            import traceback
            traceback.print_exc()
            update_init_status('error', 0, f'Initialization failed: {e}')
    
    thread = threading.Thread(target=init_thread, daemon=True)
    thread.start()
    print("🚀 Background initialization started...")


# Start background initialization
background_initialization()


# ==================== ROUTES ====================

@app.route('/api/init-status')
def get_init_status():
    """Get initialization status"""
    return jsonify(init_status)

@app.route('/api/operation-progress')
def get_operation_progress():
    """Get current operation progress for frontend polling"""
    return jsonify(operation_progress)

@app.route('/api/system-info')
def get_system_info():
    """Get system information including CPU, GPU, and MPS availability"""
    try:
        import os
        import torch
        
        system_info = {
            'cpu': {
                'cores': os.cpu_count() or 1,
                'available': True
            },
            'gpu': {
                'cuda_available': torch.cuda.is_available(),
                'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
                'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
                'gpu_names': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else []
            },
            'mps': {
                'mps_available': torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False
            }
        }
        
        return jsonify(system_info)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/check-ai-requirements')
def check_ai_requirements():
    """Check if the system meets requirements for AI bibliographic extraction:
    - CUDA GPU with at least 6 GB VRAM
    - Whether the Gemma model is already cached locally
    """
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        vram_gb = 0.0
        gpu_name = ''
        if cuda_available and torch.cuda.device_count() > 0:
            props = torch.cuda.get_device_properties(0)
            vram_gb = props.total_memory / (1024 ** 3)
            gpu_name = props.name

        # Check if model blobs exist in the local cache directory
        model_cache_dir = Path("models_llm") / "models--google--gemma-4-E2B-it"
        model_cached = model_cache_dir.exists() and any(model_cache_dir.rglob("*.safetensors"))

        meets_requirements = cuda_available and vram_gb >= 6.0

        return jsonify({
            'cuda_available': cuda_available,
            'vram_gb': round(vram_gb, 2),
            'gpu_name': gpu_name,
            'model_cached': model_cached,
            'meets_requirements': meets_requirements
        })
    except Exception as e:
        return jsonify({'error': str(e), 'cuda_available': False,
                        'vram_gb': 0, 'gpu_name': '', 'model_cached': False,
                        'meets_requirements': False}), 500


@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

# ============================================================================
# PROJECT MANAGEMENT API ROUTES
# ============================================================================

@app.route('/api/projects', methods=['GET'])
def list_projects():
    """Get list of all projects"""
    try:
        projects = project_manager.list_projects()
        return jsonify({'projects': projects, 'success': True})
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/api/projects', methods=['POST'])
def create_project():
    """Create a new project"""
    try:
        data = request.get_json()
        project_name = data.get('project_name', '').strip()
        description = data.get('description', '').strip()
        icon = data.get('icon', '1.png')
        
        if not project_name:
            return jsonify({'error': 'Project name is required', 'success': False}), 400
        
        metadata = project_manager.create_project(project_name, description, icon)
        return jsonify({'project': metadata, 'success': True})
    except ValueError as e:
        return jsonify({'error': str(e), 'success': False}), 400
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/api/projects/<project_id>', methods=['GET'])
def get_project(project_id):
    """Get project metadata"""
    try:
        metadata = project_manager.get_project(project_id)
        if metadata is None:
            return jsonify({'error': 'Project not found', 'success': False}), 404
        return jsonify({'project': metadata, 'success': True})
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/api/projects/<project_id>', methods=['DELETE'])
def delete_project(project_id):
    """Delete a project"""
    try:
        success = project_manager.delete_project(project_id)
        if not success:
            return jsonify({'error': 'Project not found or could not be deleted', 'success': False}), 404
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/api/icons', methods=['GET'])
def get_icons():
    """Get list of available project icons"""
    try:
        icons_path = Path('static/imgs/icons')
        if not icons_path.exists():
            return jsonify({'icons': [], 'success': True})

        # Get all PNG files in static/imgs/icons folder
        icons = [f.name for f in icons_path.iterdir()
                if f.is_file() and f.suffix.lower() == '.png' and f.name != 'LogoLens.png']
        
        # Sort icons numerically if they are numbered
        try:
            icons.sort(key=lambda x: int(x.replace('.png', '')))
        except:
            icons.sort()
        
        return jsonify({'icons': icons, 'success': True})
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/api/icons/<filename>')
def serve_icon(filename):
    """Serve an icon file"""
    try:
        icons_path = Path('static/imgs/icons')
        return send_from_directory(icons_path, filename)
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 404


@app.route('/api/projects/<project_id>/workflow', methods=['PATCH'])
def update_workflow_status(project_id):
    """Update workflow status for a project"""
    try:
        data = request.get_json()
        status_updates = data.get('status_updates', {})
        
        success = project_manager.update_workflow_status(project_id, status_updates)
        if not success:
            return jsonify({'error': 'Project not found', 'success': False}), 404
        
        # Return updated metadata
        metadata = project_manager.get_project(project_id)
        return jsonify({'project': metadata, 'success': True})
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/api/projects/<project_id>/settings', methods=['PATCH'])
def update_project_settings(project_id):
    """Update project settings"""
    try:
        data = request.get_json()
        settings = data.get('settings', {})
        
        success = project_manager.update_settings(project_id, settings)
        if not success:
            return jsonify({'error': 'Project not found', 'success': False}), 404
        
        # Return updated metadata
        metadata = project_manager.get_project(project_id)
        return jsonify({'project': metadata, 'success': True})
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/api/projects/<project_id>/excluded-images', methods=['POST'])
def update_excluded_images(project_id):
    """Update excluded images list for a project"""
    try:
        data = request.get_json()
        excluded_images = data.get('excluded_images', [])
        
        success = project_manager.update_excluded_images(project_id, excluded_images)
        if not success:
            return jsonify({'error': 'Project not found', 'success': False}), 404
        
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/api/projects/<project_id>/reviewed', methods=['POST'])
def add_reviewed_image(project_id):
    """Mark an image as reviewed"""
    try:
        data = request.get_json()
        image_name = data.get('image_name', '')
        
        if not image_name:
            return jsonify({'error': 'Image name is required', 'success': False}), 400
        
        success = project_manager.add_reviewed_image(project_id, image_name)
        if not success:
            return jsonify({'error': 'Project not found', 'success': False}), 404
        
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/api/projects/<project_id>/images', methods=['GET'])
def get_project_images(project_id):
    """Get list of images in project"""
    try:
        images = project_manager.get_images_list(project_id, 'images')
        
        if images is None:
            return jsonify({'error': 'Project not found', 'success': False}), 404
        
        return jsonify({
            'images': images,
            'count': len(images),
            'success': True
        })
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/api/projects/<project_id>/image/<filename>', methods=['GET'])
def serve_project_image(project_id, filename):
    """Serve an image from project's images folder"""
    try:
        images_path = project_manager.get_project_path(project_id, 'images')
        
        if not images_path or not images_path.exists():
            return jsonify({'error': 'Project or images folder not found', 'success': False}), 404
        
        return send_from_directory(images_path, filename)
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 404


@app.route('/api/projects/<project_id>/masks', methods=['GET'])
def get_project_masks(project_id):
    """Get list of mask images in project"""
    try:
        masks = project_manager.get_images_list(project_id, 'masks')
        
        if masks is None:
            return jsonify({'error': 'Project not found', 'success': False}), 404
        
        # Create URLs for each mask
        mask_urls = [f'/api/projects/{project_id}/mask/{img}' for img in masks]
        
        return jsonify({
            'masks': mask_urls,
            'count': len(masks),
            'success': True
        })
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/api/projects/<project_id>/mask/<filename>', methods=['GET'])
def serve_project_mask(project_id, filename):
    """Serve a mask image from project's masks folder"""
    try:
        masks_path = project_manager.get_project_path(project_id, 'masks')
        
        if not masks_path or not masks_path.exists():
            return jsonify({'error': 'Project or masks folder not found', 'success': False}), 404
        
        return send_from_directory(masks_path, filename)
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 404


@app.route('/api/projects/<project_id>/masks/extract', methods=['POST'])
def extract_project_masks(project_id):
    """Extract cards from masks in project with progress tracking"""
    try:
        # Verify project exists
        project_metadata = project_manager.get_project(project_id)
        if not project_metadata:
            return jsonify({'error': 'Project not found', 'success': False}), 404
        
        # Get project paths
        masks_path = project_manager.get_project_path(project_id, 'masks')
        cards_path = project_manager.get_project_path(project_id, 'cards')
        
        if not masks_path or not masks_path.exists():
            return jsonify({'error': 'Project masks folder not found', 'success': False}), 404
        
        # Count mask files for progress
        mask_files = [f for f in masks_path.iterdir() if f.name.endswith('_mask_layer.png')]
        total_masks = len(mask_files)
        
        if total_masks == 0:
            return jsonify({'error': 'No mask files found. Apply a model first.', 'success': False}), 404
        
        # Initialize progress
        update_operation_progress('extract_masks', 0, total_masks, 'Starting extraction...')
        
        # Run extraction in a way that allows progress updates
        # We'll monkey-patch the print function temporarily
        import builtins
        original_print = builtins.print
        
        def progress_print(*args, **kwargs):
            msg = ' '.join(str(arg) for arg in args)
            # Look for progress pattern "Processing mask X/Y"
            if 'Processing mask' in msg:
                try:
                    parts = msg.split()
                    idx = parts.index('mask') + 1
                    current = int(parts[idx].split('/')[0])
                    update_operation_progress('extract_masks', current, total_masks, 
                                             f'Extracting mask {current}/{total_masks}')
                except:
                    pass
            original_print(*args, **kwargs)
        
        builtins.print = progress_print
        
        try:
            # Extract masks using project paths
            result = mask_extractor.extract_masks_from_project(
                str(masks_path),
                str(cards_path)
            )
        finally:
            builtins.print = original_print
            clear_operation_progress()
        
        # Update project workflow status
        card_count = len(list(cards_path.glob('*.png'))) if cards_path.exists() else 0
        project_manager.update_workflow_status(project_id, {
            'cards_extracted': card_count
        })
        
        return jsonify({
            'message': result,
            'success': True
        })
        
    except Exception as e:
        clear_operation_progress()
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/api/projects/<project_id>/masks/save', methods=['POST'])
def save_project_mask(project_id):
    """Save edited mask for a project image"""
    try:
        # Verify project exists
        project_metadata = project_manager.get_project(project_id)
        if not project_metadata:
            return jsonify({'error': 'Project not found', 'success': False}), 404
        
        # Get uploaded mask file
        if 'mask' not in request.files:
            return jsonify({'error': 'No mask file provided', 'success': False}), 400
        
        mask_file = request.files['mask']
        if mask_file.filename == '':
            return jsonify({'error': 'Empty filename', 'success': False}), 400
        
        # Save to project masks folder
        masks_path = project_manager.get_project_path(project_id, 'masks')
        filename = secure_filename(mask_file.filename)
        mask_filepath = masks_path / filename
        
        mask_file.save(mask_filepath)
        
        # Generate mask URL for frontend
        mask_url = f'/api/projects/{project_id}/mask/{filename}'
        
        return jsonify({
            'message': f'Mask saved: {filename}',
            'filename': filename,
            'mask_url': mask_url,
            'success': True
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e), 'success': False}), 500


# ============================================================================
# LEGACY API ROUTES (will be refactored to use projects)
# ============================================================================

@app.route('/api/folders/images')
def get_image_folders():
    """Get list of image folders"""
    try:
        folders = [f for f in os.listdir(PDFIMG_OUTPUT_DIR) 
                  if os.path.isdir(PDFIMG_OUTPUT_DIR / f)]
        return jsonify({'folders': folders, 'success': True})
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/api/folders/masks')
def get_mask_folders():
    """Get list of folders with masks (for annotation tab)"""
    try:
        # Get folders ending with _mask
        folders = [f.replace('_mask', '') for f in os.listdir(PRED_OUTPUT_DIR) 
                  if f.endswith('_mask') and os.path.isdir(PRED_OUTPUT_DIR / f)]
        return jsonify({'folders': folders, 'success': True})
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/api/folders/results')
def get_results_folders():
    """Get list of result folders"""
    try:
        folders = [f for f in os.listdir(PRED_OUTPUT_DIR) 
                  if f.endswith('_card') and os.path.isdir(PRED_OUTPUT_DIR / f)]
        return jsonify({'folders': folders, 'success': True})
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/api/models')
def get_models():
    """Get list of available models"""
    try:
        models = [f for f in os.listdir(MODELS_DIR) 
                 if f.endswith('.pt')]
        return jsonify({'models': models, 'success': True})
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500


# ==================== PDF PROCESSING ====================

@app.route('/api/pdf/upload', methods=['POST'])
def upload_pdf():
    """Upload and process PDF file into a project"""
    try:
        print("PDF upload request received")
        print("Files:", request.files)
        print("Form data:", request.form)
        
        if 'file' not in request.files:
            print("Error: No file in request")
            return jsonify({'error': 'No file provided', 'success': False}), 400
        
        file = request.files['file']
        split_pages = request.form.get('split_pages', 'false').lower() == 'true'
        project_id = request.form.get('project_id', '').strip()
        
        print(f"File name: {file.filename}")
        print(f"Split pages: {split_pages}")
        print(f"Project ID: {project_id}")
        
        if not project_id:
            return jsonify({'error': 'No project selected', 'success': False}), 400
        
        # Verify project exists
        project_metadata = project_manager.get_project(project_id)
        if not project_metadata:
            return jsonify({'error': 'Project not found', 'success': False}), 404
        
        if file.filename == '':
            print("Error: Empty filename")
            return jsonify({'error': 'No file selected', 'success': False}), 400
        
        if not file.filename.lower().endswith('.pdf'):
            print("Error: Not a PDF file")
            return jsonify({'error': 'Only PDF files are allowed', 'success': False}), 400
        
        # Save PDF to project's pdf_source folder
        filename = secure_filename(file.filename)
        pdf_source_path = project_manager.get_project_path(project_id, 'pdf_source')
        pdf_filepath = pdf_source_path / filename
        print(f"Saving PDF to project: {pdf_filepath}")
        file.save(pdf_filepath)
        
        # Get project images folder for output
        images_output_path = project_manager.get_project_path(project_id, 'images')
        
        # Process PDF and save images to project (use project name for image naming)
        print(f"Processing PDF to: {images_output_path}")
        result = pdf_processor.process_pdf_to_folder(
            str(pdf_filepath), 
            str(images_output_path), 
            split_pages,
            project_name=project_metadata.get('project_name', 'project')
        )
        print(f"Processing result: {result}")
        
        # Update project metadata
        image_count = project_manager.count_files(project_id, 'images')
        project_manager.update_workflow_status(project_id, {
            'pdf_processed': True,
            'pdf_count': len(list(pdf_source_path.glob('*.pdf'))),
            'images_extracted': image_count,
            'total_images': image_count
        })
        
        return jsonify({
            'message': f'PDF processed successfully. {image_count} images extracted.',
            'images_count': image_count,
            'success': True
        })
        
    except Exception as e:
        print(f"Error in PDF upload: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e), 'success': False}), 500


# ==================== FEW-SHOT DETECTION ====================

@app.route('/api/projects/<project_id>/fewshot/process-image', methods=['POST'])
def fewshot_process_image(project_id):
    """
    Run SAM2 automask + DINOv2 on one image and cache the result.
    Body: { "image_filename": "page_001.jpg" }
    """
    try:
        data = request.json
        image_filename = data.get('image_filename')
        if not image_filename:
            return jsonify({'error': 'image_filename required', 'success': False}), 400

        project = project_manager.get_project(project_id)
        if not project:
            return jsonify({'error': 'Project not found', 'success': False}), 404

        images_path = project_manager.get_project_path(project_id, 'images')
        fewshot_path = project_manager.get_project_path(project_id, 'fewshot')
        image_path = images_path / image_filename

        if not image_path.exists():
            return jsonify({'error': f'Image not found: {image_filename}', 'success': False}), 404

        # Use filename stem as image_id
        image_id = Path(image_filename).stem

        def run():
            try:
                result = few_shot_processor.process_image(str(image_path), image_id, fewshot_path)
                update_operation_progress('fewshot_process', 1, 1, f"Done: {result['num_masks']} masks")
            except Exception as e:
                update_operation_progress('fewshot_process', 1, 1, f"Error: {e}")
                import traceback; traceback.print_exc()

        update_operation_progress('fewshot_process', 0, 1, f'Processing {image_filename}…')
        threading.Thread(target=run, daemon=True).start()

        return jsonify({'message': 'Processing started', 'image_id': image_id, 'success': True})

    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/api/projects/<project_id>/fewshot/process-image-sync', methods=['POST'])
def fewshot_process_image_sync(project_id):
    """Synchronous version — returns immediately with full mask data."""
    try:
        data = request.json
        image_filename = data.get('image_filename')
        if not image_filename:
            return jsonify({'error': 'image_filename required', 'success': False}), 400

        project = project_manager.get_project(project_id)
        if not project:
            return jsonify({'error': 'Project not found', 'success': False}), 404

        images_path = project_manager.get_project_path(project_id, 'images')
        fewshot_path = project_manager.get_project_path(project_id, 'fewshot')
        image_path = images_path / image_filename
        if not image_path.exists():
            return jsonify({'error': f'Image not found: {image_filename}', 'success': False}), 404

        image_id  = Path(image_filename).stem
        threshold = float(data.get('threshold', 0.5))
        result    = few_shot_processor.process_image(str(image_path), image_id, fewshot_path, threshold)
        data_store      = few_shot_processor._store.get(image_id, {})
        classes_summary = few_shot_processor._classes_summary(data_store.get("classes", {}))
        return jsonify({**result, 'classes': classes_summary, 'success': True})

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/api/projects/<project_id>/fewshot/load-image', methods=['POST'])
def fewshot_load_image(project_id):
    """
    Quickly load a project image into memory so SAM2 predictor can run on it.
    Does NOT run auto-mask generation.  Fast (< 1 s).
    Body: { image_filename }
    """
    try:
        data = request.json
        image_filename = data.get('image_filename')
        if not image_filename:
            return jsonify({'error': 'image_filename required', 'success': False}), 400

        project = project_manager.get_project(project_id)
        if not project:
            return jsonify({'error': 'Project not found', 'success': False}), 404

        images_path = project_manager.get_project_path(project_id, 'images')
        image_path  = images_path / image_filename
        if not image_path.exists():
            return jsonify({'error': f'Image not found: {image_filename}', 'success': False}), 404

        image_id     = Path(image_filename).stem
        fewshot_path = project_manager.get_project_path(project_id, 'fewshot')
        result       = few_shot_processor.load_for_prediction(str(image_path), image_id, fewshot_path)
        return jsonify({**result, 'success': True})

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/api/projects/<project_id>/fewshot/save-masks', methods=['POST'])
def fewshot_save_masks(project_id):
    """
    Save all labeled examples for the current image as RGBA mask PNGs.
    Works without auto-mask pre-processing.
    Body: { image_id, image_filename }
    """
    try:
        data           = request.json
        image_id       = data.get('image_id')
        image_filename = data.get('image_filename')
        if not image_id or not image_filename:
            return jsonify({'error': 'image_id and image_filename required', 'success': False}), 400

        project = project_manager.get_project(project_id)
        if not project:
            return jsonify({'error': 'Project not found', 'success': False}), 404

        masks_path = project_manager.get_project_path(project_id, 'masks')
        result     = few_shot_processor.save_labeled_masks(image_id, masks_path, image_filename)
        return jsonify({**result, 'success': True})

    except ValueError as e:
        return jsonify({'error': str(e), 'success': False}), 400
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/api/projects/<project_id>/fewshot/preview-prompt', methods=['POST'])
def fewshot_preview_prompt(project_id):
    """
    Run SAM2 predictor with given prompts and return the mask contour immediately.
    Does NOT modify any class — pure preview for live visual feedback.
    Body: {
      image_id,
      points (optional): [[x,y], ...],
      labels (optional): [1, 0, ...],   1=foreground 0=background
      box    (optional): [x1, y1, x2, y2]
    }
    """
    try:
        data = request.json
        image_id = data.get('image_id')
        if not image_id:
            return jsonify({'error': 'image_id required', 'success': False}), 400

        points = data.get('points')
        labels = data.get('labels')
        box    = data.get('box')

        result = few_shot_processor.preview_prompt(image_id, points, labels, box)
        return jsonify({**result, 'success': True})

    except ValueError as e:
        return jsonify({'error': str(e), 'success': False}), 400
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/api/projects/<project_id>/fewshot/add-example', methods=['POST'])
def fewshot_add_example(project_id):
    """
    Add a labelled example to a class.
    Body: {
      image_id, class_name, x, y, threshold,
      points (optional), labels (optional), box (optional)
    }
    """
    try:
        data = request.json
        image_id   = data.get('image_id')
        class_name = data.get('class_name')
        x          = int(data.get('x', 0))
        y          = int(data.get('y', 0))
        threshold  = float(data.get('threshold', 0.5))
        points     = data.get('points')    # [[x,y], ...]
        labels     = data.get('labels')    # [1, 0, ...]
        box        = data.get('box')       # [x1, y1, x2, y2]

        if not image_id or not class_name:
            return jsonify({'error': 'image_id and class_name required', 'success': False}), 400

        fewshot_path = project_manager.get_project_path(project_id, 'fewshot')

        summary = few_shot_processor.add_example(
            image_id, class_name, x, y, threshold, fewshot_path,
            points=points, labels=labels, box=box,
        )
        return jsonify({'classes': summary, 'success': True})

    except ValueError as e:
        return jsonify({'error': str(e), 'success': False}), 400
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/api/projects/<project_id>/fewshot/remove-example', methods=['POST'])
def fewshot_remove_example(project_id):
    """
    Remove a labelled example.
    Body: { image_id, class_name, example_index, threshold }
    """
    try:
        data = request.json
        image_id      = data.get('image_id')
        class_name    = data.get('class_name')
        example_index = int(data.get('example_index', 0))
        threshold     = float(data.get('threshold', 0.5))

        fewshot_path = project_manager.get_project_path(project_id, 'fewshot')
        summary = few_shot_processor.remove_example(image_id, class_name, example_index, threshold, fewshot_path)
        return jsonify({'classes': summary, 'success': True})

    except ValueError as e:
        return jsonify({'error': str(e), 'success': False}), 400
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/api/projects/<project_id>/fewshot/classes/<image_id>', methods=['GET'])
def fewshot_get_classes(project_id, image_id):
    """Return current class predictions for an image."""
    try:
        threshold = float(request.args.get('threshold', 0.5))
        summary = few_shot_processor.get_predictions(image_id, threshold)
        return jsonify({'classes': summary, 'success': True})
    except ValueError as e:
        return jsonify({'error': str(e), 'success': False}), 400
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/api/projects/<project_id>/fewshot/confirm', methods=['POST'])
def fewshot_confirm(project_id):
    """
    Confirm selected predictions and write RGBA mask PNGs.
    Body: {
      image_id, image_filename,
      confirmations: { class_name: [mask_index, ...], ... }
    }
    """
    try:
        data = request.json
        image_id       = data.get('image_id')
        image_filename = data.get('image_filename')
        confirmations  = data.get('confirmations', {})

        masks_path = project_manager.get_project_path(project_id, 'masks')
        messages = []
        for class_name, indices in confirmations.items():
            msg = few_shot_processor.confirm_predictions(
                image_id, class_name, indices, masks_path, image_filename
            )
            messages.append(msg)

        project_manager.update_workflow_status(project_id, {'fewshot_labeling_done': True})
        return jsonify({'messages': messages, 'success': True})

    except ValueError as e:
        return jsonify({'error': str(e), 'success': False}), 400
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/api/projects/<project_id>/fewshot/apply-to-all', methods=['POST'])
def fewshot_apply_to_all(project_id):
    """
    Batch: apply all class examples to every image in the project.
    Body: { threshold (float), excluded_images (list) }
    Runs in background; poll /api/operation-progress.
    """
    try:
        data = request.json
        threshold        = float(data.get('threshold', 0.5))
        excluded_images  = data.get('excluded_images', [])

        project = project_manager.get_project(project_id)
        if not project:
            return jsonify({'error': 'Project not found', 'success': False}), 404

        images_path  = project_manager.get_project_path(project_id, 'images')
        masks_path   = project_manager.get_project_path(project_id, 'masks')
        fewshot_path = project_manager.get_project_path(project_id, 'fewshot')

        def run():
            try:
                result = few_shot_processor.apply_to_all(
                    images_path, masks_path, fewshot_path,
                    threshold, excluded_images,
                    progress_callback=lambda c, t, m: update_operation_progress('batch_fewshot', c, t, m),
                )
                mask_count = sum(1 for f in masks_path.iterdir() if f.suffix == '.png')
                project_manager.update_workflow_status(project_id, {
                    'batch_applied': True,
                    'masks_generated': mask_count,
                })
                clear_operation_progress()
            except Exception as e:
                clear_operation_progress()
                import traceback; traceback.print_exc()

        update_operation_progress('batch_fewshot', 0, 1, 'Starting batch processing…')
        threading.Thread(target=run, daemon=True).start()

        return jsonify({'message': 'Batch processing started', 'success': True})

    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

# ==================== TABULAR ====================

@app.route('/api/tabular/load', methods=['POST'])
def load_tabular_data():
    """Load tabular data for an image"""
    try:
        data = request.json
        folder = data.get('folder')
        img_num = int(data.get('img_num', 0))
        
        if not folder:
            return jsonify({'error': 'Folder is required', 'success': False}), 400
        
        # Get image and table data
        image_data, current_num, table_df, max_imgs = tabular_processor.image_selection(folder, img_num)
        
        # Convert image to base64 if exists
        img_base64 = None
        annotations = []
        if image_data and hasattr(image_data, 'value'):
            # Handle AnnotatedImage value
            img_array, annot_list = image_data.value
            from PIL import Image
            import io
            
            img = Image.fromarray(img_array)
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            img_base64 = base64.b64encode(buffer.getvalue()).decode()
            annotations = annot_list
        
        # Convert DataFrame to dict
        table_data = table_df.to_dict('records') if not table_df.empty else []
        
        return jsonify({
            'image': f'data:image/png;base64,{img_base64}' if img_base64 else None,
            'annotations': annotations,
            'table': table_data,
            'columns': list(table_df.columns) if not table_df.empty else [],
            'current': current_num,
            'total': max_imgs,
            'success': True
        })
        
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/api/tabular/save', methods=['POST'])
def save_tabular_data():
    """Save tabular data"""
    try:
        data = request.json
        folder = data.get('folder')
        table_data = data.get('table')
        
        if not folder or not table_data:
            return jsonify({'error': 'Missing required data', 'success': False}), 400
        
        # Convert to DataFrame
        df = pd.DataFrame(table_data)
        
        # Save
        tabular_processor.save_table(df, folder)
        
        return jsonify({
            'message': 'Table saved successfully',
            'success': True
        })
        
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/api/tabular/add-column', methods=['POST'])
def add_column():
    """Add a new column to the table"""
    try:
        data = request.json
        column_name = data.get('column_name')
        table_data = data.get('table')
        
        if not column_name:
            return jsonify({'error': 'Column name is required', 'success': False}), 400
        
        df = pd.DataFrame(table_data)
        if column_name not in df.columns:
            df[column_name] = ""
        
        return jsonify({
            'table': df.to_dict('records'),
            'columns': list(df.columns),
            'success': True
        })
        
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/api/projects/<project_id>/cards')
def get_project_cards(project_id):
    """Get list of card images for a project"""
    try:
        # Verify project exists
        project_metadata = project_manager.get_project(project_id)
        if not project_metadata:
            return jsonify({'error': 'Project not found', 'success': False}), 404
        
        # Get cards path
        cards_path = project_manager.get_project_path(project_id, 'cards')
        
        if not cards_path or not cards_path.exists():
            return jsonify({
                'cards': [],
                'total': 0,
                'success': True
            })
        
        # Get all card images
        def _natural_key(s):
            return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]

        card_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        cards = sorted([f.name for f in cards_path.iterdir() 
                       if f.is_file() and f.suffix.lower() in card_extensions],
                      key=_natural_key)
        
        # Load classifications if available (check both cards and cards_modified)
        classifications = {}
        
        # Try cards_modified first (where classifications.csv is usually saved after processing)
        project_base_path = project_manager.get_project_path(project_id, 'cards')
        if project_base_path:
            project_root = project_base_path.parent
            cards_modified_path = project_root / 'cards_modified'
            classifications_csv = cards_modified_path / 'classifications.csv'
            
            if not classifications_csv.exists():
                # Fallback to cards folder
                classifications_csv = project_base_path / 'classifications.csv'
        else:
            classifications_csv = None
        
        if classifications_csv and classifications_csv.exists():
            try:
                import pandas as pd
                df = pd.read_csv(classifications_csv)
                print(f"Loaded classifications from {classifications_csv}, columns: {df.columns.tolist()}")
                
                # Create mapping from filename to type with normalization
                for _, row in df.iterrows():
                    # Try different column names for filename
                    filename = row.get('filename') or row.get('Filename') or row.get('mask_file') or row.get('id')
                    type_val = row.get('type') or row.get('Type')
                    
                    if filename and type_val:
                        # Normalize filename by removing path and keeping just the name
                        from pathlib import Path
                        filename_clean = Path(filename).name
                        classifications[filename_clean] = type_val
                        print(f"Mapped {filename_clean} -> {type_val}")
            except Exception as e:
                print(f"Error loading classifications: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"No classifications.csv found in cards or cards_modified")
        
        # Create URLs and metadata for cards
        card_data = []
        for card in cards:
            card_type = classifications.get(card, 'ENT')  # Default to ENT if not classified
            print(f"Card {card} -> type {card_type}")
            card_data.append({
                'url': f'/api/projects/{project_id}/card/{card}',
                'filename': card,
                'type': card_type
            })
        
        return jsonify({
            'cards': card_data,
            'total': len(card_data),
            'success': True
        })
        
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/api/projects/<project_id>/card/<filename>')
def serve_project_card(project_id, filename):
    """Serve a specific card image from project"""
    try:
        cards_path = project_manager.get_project_path(project_id, 'cards')
        if not cards_path or not cards_path.exists():
            return jsonify({'error': 'Cards folder not found', 'success': False}), 404
        
        return send_from_directory(cards_path, filename)
        
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 404


@app.route('/api/projects/<project_id>/card-modified/<filename>')
def serve_project_card_modified(project_id, filename):
    """Serve a specific modified card image from project"""
    try:
        cards_modified_path = project_manager.get_project_path(project_id, 'cards_modified')
        if not cards_modified_path or not cards_modified_path.exists():
            return jsonify({'error': 'Modified cards folder not found', 'success': False}), 404
        
        return send_from_directory(cards_modified_path, filename)
        
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 404


@app.route('/api/projects/<project_id>/tabular/load', methods=['POST'])
def load_project_tabular_data(project_id):
    """Load tabular data for a project - shows original image with bounding boxes"""
    try:
        # Verify project exists
        project_metadata = project_manager.get_project(project_id)
        if not project_metadata:
            return jsonify({'error': 'Project not found', 'success': False}), 404
        
        data = request.json
        img_num = int(data.get('img_num', 0))
        
        # Get project paths
        cards_path = project_manager.get_project_path(project_id, 'cards')
        images_path = project_manager.get_project_path(project_id, 'images')
        
        if not cards_path or not cards_path.exists():
            return jsonify({'error': 'No cards found in project', 'success': False}), 404
        
        if not images_path or not images_path.exists():
            return jsonify({'error': 'No images found in project', 'success': False}), 404
        
        # Load annotation CSVs
        mask_info_path = cards_path / 'mask_info.csv'
        mask_info_annots_path = cards_path / 'mask_info_annots.csv'
        
        if not mask_info_path.exists() or not mask_info_annots_path.exists():
            return jsonify({'error': 'Annotation CSV files not found', 'success': False}), 404
        
        # Read CSV files
        df_info = pd.read_csv(mask_info_path).fillna('')
        df_annots = pd.read_csv(mask_info_annots_path)
        
        # Add image_name and ID columns to annotations
        df_annots['image_name'] = df_annots['mask_file'].apply(
            lambda x: x.split('_mask_layer_')[0] if isinstance(x, str) and '_mask_layer_' in x else '')
        df_annots['ID'] = df_annots['mask_file'].apply(
            lambda x: x.split('layer_')[1].replace('.png', '') if isinstance(x, str) and 'layer_' in x else '0')
        
        # Get list of unique images
        unique_images = sorted(df_annots['image_name'].unique())
        
        if not unique_images:
            return jsonify({'error': 'No images found in annotations', 'success': False}), 404
        
        # Validate image number
        img_num = max(0, min(img_num, len(unique_images) - 1))
        current_image_name = unique_images[img_num]
        
        # Find original image file
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        original_image_path = None
        
        for ext in image_extensions:
            candidate = images_path / f"{current_image_name}{ext}"
            if candidate.exists():
                original_image_path = candidate
                break
        
        if not original_image_path:
            return jsonify({'error': f'Original image not found: {current_image_name}', 'success': False}), 404
        
        # Load and prepare image with annotations
        from PIL import Image
        import io

        with Image.open(original_image_path) as img:
            # Keep original size for bbox math
            original_size = img.size  # (width, height)

            # Create a reasonably sized preview but preserve aspect ratio (no rotation)
            img.thumbnail((1200, 1200))

            # Compute scale factors
            scale_x = img.size[0] / original_size[0]
            scale_y = img.size[1] / original_size[1]

            image_array = np.asarray(img, dtype=np.uint8)
        
        # Get annotations for this image
        image_annots = df_annots[df_annots['image_name'] == current_image_name]
        
        # Create scaled annotations list
        annotations = []
        for _, row in image_annots.iterrows():
            try:
                # Parse bbox string "(x1, y1, x2, y2)"
                bbox_str = str(row.get('bbox', '')).strip('()')
                coords = [int(x.strip()) for x in bbox_str.split(',')]

                # Scale coordinates (no rotation - keep original orientation)
                scaled_bbox = [
                    int(coords[0] * scale_x),
                    int(coords[1] * scale_y),
                    int(coords[2] * scale_x),
                    int(coords[3] * scale_y)
                ]

                annotations.append({
                    'bbox': scaled_bbox,
                    'label': str(row.get('ID', ''))
                })

            except Exception as e:
                print(f"Error processing annotation: {e}")
                continue
        
        # Convert image to base64
        buffer = io.BytesIO()
        Image.fromarray(image_array).save(buffer, format='PNG')
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        
    # Prepare table data from mask_info
        df_subset = df_info[df_info['file'] == current_image_name].copy()
        
        if not df_subset.empty:
            # Add ID column
            df_subset['ID'] = df_subset['mask_file'].apply(
                lambda x: x.split('layer_')[1] if isinstance(x, str) and 'layer_' in x else '0')
            
            # Drop internal columns
            drop_cols = [col for col in ['mask_file', 'file'] if col in df_subset.columns]
            if drop_cols:
                df_subset = df_subset.drop(columns=drop_cols)
            
            # Reorder with ID first
            columns_order = ['ID'] + [col for col in df_subset.columns if col != 'ID']
            df_subset = df_subset[columns_order]
            
            table_data = df_subset.to_dict('records')
            columns = list(df_subset.columns)
        else:
            # Create empty table structure
            table_data = []
            columns = ['ID', 'Notes']
        
        # Build image list with reviewed flags
        project_meta = project_metadata
        reviewed_list = project_meta.get('workflow_status', {}).get('reviewed_images', []) if project_meta else []
        image_list = [{'image_name': name, 'reviewed': (name in reviewed_list)} for name in unique_images]

        # Prepare full-resolution image URL for zoom (let frontend fetch it when user requests zoom)
        full_image_url = None
        for ext in image_extensions:
            candidate = images_path / f"{current_image_name}{ext}"
            if candidate.exists():
                full_image_url = f'/api/projects/{project_id}/image/{candidate.name}'
                break

        return jsonify({
            'image': f'data:image/png;base64,{img_base64}',
            'annotations': annotations,
            'table': table_data,
            'columns': columns,
            'current': img_num,
            'total': len(unique_images),
            'image_name': current_image_name,
            'image_list': image_list,
            'is_reviewed': (current_image_name in reviewed_list),
            'full_image_url': full_image_url,
            'success': True
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/api/projects/<project_id>/tabular/save', methods=['POST'])
def save_project_tabular_data(project_id):
    """Save tabular data for a project - updates mask_info.csv"""
    try:
        # Verify project exists
        project_metadata = project_manager.get_project(project_id)
        if not project_metadata:
            return jsonify({'error': 'Project not found', 'success': False}), 404
        
        data = request.json
        table_data = data.get('table')
        image_name = data.get('image_name')  # Current image being edited
        
        if not table_data:
            return jsonify({'error': 'Missing table data', 'success': False}), 400
        
        # Get cards folder path
        cards_path = project_manager.get_project_path(project_id, 'cards')
        
        if not cards_path or not cards_path.exists():
            return jsonify({'error': 'Cards folder not found', 'success': False}), 404
        
        # Convert to DataFrame
        df_new = pd.DataFrame(table_data)
        
        # Load existing mask_info.csv
        csv_path = cards_path / 'mask_info.csv'
        
        if csv_path.exists():
            try:
                df_existing = pd.read_csv(csv_path)
                
                # Remove old data for current image
                if image_name and 'file' in df_existing.columns:
                    df_existing = df_existing[df_existing['file'] != image_name]
                
                # Add file and mask_file columns to new data if not present
                if 'file' not in df_new.columns and image_name:
                    df_new['file'] = image_name
                
                if 'mask_file' not in df_new.columns and 'ID' in df_new.columns:
                    df_new['mask_file'] = df_new['ID'].apply(
                        lambda x: f"{image_name}_mask_layer_{x}" if image_name else f"mask_layer_{x}")
                
                # Combine old and new data
                df_combined = pd.concat([df_existing, df_new], ignore_index=True)
                
                # Save combined data
                df_combined.to_csv(csv_path, index=False)
                
            except Exception as e:
                print(f"Warning: Could not merge with existing CSV: {e}")
                # Just save new data
                df_new.to_csv(csv_path, index=False)
        else:
            # Add required columns if missing
            if 'file' not in df_new.columns and image_name:
                df_new['file'] = image_name
            
            if 'mask_file' not in df_new.columns and 'ID' in df_new.columns:
                df_new['mask_file'] = df_new['ID'].apply(
                    lambda x: f"{image_name}_mask_layer_{x}" if image_name else f"mask_layer_{x}")
            
            df_new.to_csv(csv_path, index=False)
        
        print(f"Saved tabular data to: {csv_path}")
        
        return jsonify({
            'message': 'Table saved successfully',
            'success': True
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/api/projects/<project_id>/tabular/ai-bibliographic', methods=['POST'])
def ai_extract_bibliographic(project_id):
    """Use Gemma 4 E2B-it to extract bibliographic info (tavola, figura, numero)
    from the original page image using bounding box coordinates."""
    try:
        import json as _json
        import re as _re

        project_metadata = project_manager.get_project(project_id)
        if not project_metadata:
            return jsonify({'error': 'Project not found', 'success': False}), 404

        data = request.json
        img_num = int(data.get('img_num', 0))

        cards_path = project_manager.get_project_path(project_id, 'cards')
        images_path = project_manager.get_project_path(project_id, 'images')

        mask_info_path = cards_path / 'mask_info.csv'
        mask_info_annots_path = cards_path / 'mask_info_annots.csv'

        if not mask_info_path.exists() or not mask_info_annots_path.exists():
            return jsonify({'error': 'Annotation CSV files not found', 'success': False}), 404

        df_info = pd.read_csv(mask_info_path).fillna('')
        df_annots = pd.read_csv(mask_info_annots_path)

        df_annots['image_name'] = df_annots['mask_file'].apply(
            lambda x: x.split('_mask_layer_')[0] if isinstance(x, str) and '_mask_layer_' in x else '')
        df_annots['ID'] = df_annots['mask_file'].apply(
            lambda x: x.split('layer_')[1].replace('.png', '') if isinstance(x, str) and 'layer_' in x else '0')

        unique_images = sorted(df_annots['image_name'].unique())
        if not unique_images:
            return jsonify({'error': 'No images found in annotations', 'success': False}), 404

        img_num = max(0, min(img_num, len(unique_images) - 1))
        current_image_name = unique_images[img_num]

        # Find original image file
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        original_image_path = None
        for ext in image_extensions:
            candidate = images_path / f"{current_image_name}{ext}"
            if candidate.exists():
                original_image_path = candidate
                break

        if not original_image_path:
            return jsonify({'error': f'Original image not found: {current_image_name}', 'success': False}), 404

        # Load image at high resolution for OCR (max 2400px on longest side)
        from PIL import Image as PILImage
        with PILImage.open(original_image_path) as img:
            orig_w, orig_h = img.size
            img_copy = img.copy()
            img_copy.thumbnail((2400, 2400), PILImage.LANCZOS)
            scale_x = img_copy.size[0] / orig_w
            scale_y = img_copy.size[1] / orig_h
            page_image = img_copy.convert('RGB')

        # Build bbox description for the prompt (scaled to OCR image size)
        image_annots = df_annots[df_annots['image_name'] == current_image_name]
        bbox_lines = []
        for _, row in image_annots.iterrows():
            try:
                bbox_str = str(row.get('bbox', '')).strip('()')
                coords = [int(x.strip()) for x in bbox_str.split(',')]
                scaled = [
                    int(coords[0] * scale_x), int(coords[1] * scale_y),
                    int(coords[2] * scale_x), int(coords[3] * scale_y)
                ]
                bbox_lines.append(f"- ID {row['ID']}: [{scaled[0]}, {scaled[1]}, {scaled[2]}, {scaled[3]}]")
            except Exception:
                continue

        if not bbox_lines:
            return jsonify({'error': 'No valid annotations found for this page', 'success': False}), 404

        id_list = [line.split(':')[0].replace('- ID ', '').strip() for line in bbox_lines]
        id_list_str = ', '.join(id_list)
        bbox_text = '\n'.join(bbox_lines)
        img_w, img_h = page_image.size
        prompt = (
            "This is a page from an archaeological publication about pottery. "
            f"The full image size is {img_w}x{img_h} pixels.\n"
            f"There are {len(bbox_lines)} pottery drawings with these EXACT IDs and pixel bounding boxes [x1, y1, x2, y2]:\n"
            f"{bbox_text}\n\n"
            f"IMPORTANT: Your JSON response MUST use EXACTLY these keys: {id_list_str}\n"
            "Do NOT invent sequential keys or rename IDs. Each key must match one of the IDs above.\n\n"
            "For EACH drawing, extract:\n"
            '- "page": the page number of the publication. '
            'Look at the EDGES and EXTREMITIES of the full image '
            '(top margin, bottom margin, corners) for a printed page number. '
            'This value is the SAME for all drawings on the page.\n'
            '- "plate": plate/table number (e.g. "Tav. III", "Pl. 12"). '
            'Usually at the top or bottom edge of the image, shared by all drawings.\n'
            '- "figure": figure number for the whole plate (e.g. "Fig. 3", "Abb. 5"). '
            'Usually at top or bottom of the image.\n'
            '- "number": the small reference number printed DIRECTLY NEXT TO or inside '
            'the specific drawing at the given bounding box coordinates. '
            'This is in a VARIABLE POSITION — examine carefully the area immediately '
            'surrounding each bounding box for a small printed number or label.\n\n'
            f"Respond ONLY with a valid JSON object using EXACTLY these keys: {id_list_str}\n"
            "Example (replace <IDx> with the real IDs from above):\n"
            '{\"<ID1>\": {\"page\": \"45\", \"plate\": \"Tav. III\", \"figure\": \"Fig. 5\", \"number\": \"1\"}, '
            '\"<ID2>\": {\"page\": \"45\", \"plate\": \"Tav. III\", \"figure\": \"Fig. 5\", \"number\": \"2a\"}}\n'
            "If a value is not found, use null."
        )

        # Load Gemma 4 E2B-it and run inference
        model, processor = load_gemma_model()

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": page_image},
                {"type": "text", "text": prompt}
            ]
        }]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        inputs = processor(text=text, images=[page_image], return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[-1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=1.0,
                top_p=0.95,
                top_k=64,
                do_sample=True
            )

        raw_response = processor.decode(outputs[0][input_len:], skip_special_tokens=True)
        print(f"[AI] Raw response: {raw_response[:300]}")

        # Extract JSON block from the response
        json_match = _re.search(r'\{.*\}', raw_response, _re.DOTALL)
        if not json_match:
            return jsonify({
                'error': f'Model did not return valid JSON. Response: {raw_response[:500]}',
                'success': False
            }), 500

        ai_result = _json.loads(json_match.group())

        # Add missing columns to df_info if needed
        for col in ['page', 'plate', 'figure', 'number']:
            if col not in df_info.columns:
                df_info[col] = ''

        # Write extracted values into df_info rows for current image
        # Use exact filename match to avoid partial-prefix collisions (e.g. ID "0" matching layer_01)
        for mask_id, values in ai_result.items():
            exact_mask_file = f"{current_image_name}_mask_layer_{mask_id}.png"
            row_mask = df_info['mask_file'] == exact_mask_file
            if not row_mask.any():
                # Try without extension in case stored differently
                exact_no_ext = f"{current_image_name}_mask_layer_{mask_id}"
                row_mask = df_info['mask_file'].apply(
                    lambda x: str(x).replace('.png', '') == exact_no_ext
                )
            if row_mask.any():
                for col in ['page', 'plate', 'figure', 'number']:
                    val = values.get(col)
                    if val is not None:
                        df_info.loc[row_mask, col] = str(val)
            else:
                print(f"[AI] Warning: no row found in df_info for mask_id={mask_id!r}, expected file={exact_mask_file!r}")

        df_info.to_csv(mask_info_path, index=False)

        # Return updated table for current image
        df_subset = df_info[df_info['file'] == current_image_name].copy()
        if not df_subset.empty:
            df_subset['ID'] = df_subset['mask_file'].apply(
                lambda x: x.split('layer_')[1] if isinstance(x, str) and 'layer_' in x else '0')
            drop_cols = [col for col in ['mask_file', 'file'] if col in df_subset.columns]
            if drop_cols:
                df_subset = df_subset.drop(columns=drop_cols)
            columns_order = ['ID'] + [col for col in df_subset.columns if col != 'ID']
            df_subset = df_subset[columns_order]
            table_data = df_subset.to_dict('records')
            columns = list(df_subset.columns)
        else:
            table_data = []
            columns = ['ID', 'page', 'plate', 'figure', 'number']

        return jsonify({
            'success': True,
            'table': table_data,
            'columns': columns,
            'ai_result': ai_result
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/api/projects/<project_id>/tabular/ai-bibliographic-batch', methods=['POST'])
def ai_extract_bibliographic_batch(project_id):
    """Run Gemma 4 E2B-it AI extraction on ALL images in the project (batch mode).
    Progress is streamed via the global operation_progress dict so the frontend
    can poll /api/progress."""
    try:
        import json as _json
        import re as _re

        project_metadata = project_manager.get_project(project_id)
        if not project_metadata:
            return jsonify({'error': 'Project not found', 'success': False}), 404

        cards_path = project_manager.get_project_path(project_id, 'cards')
        images_path = project_manager.get_project_path(project_id, 'images')

        mask_info_path = cards_path / 'mask_info.csv'
        mask_info_annots_path = cards_path / 'mask_info_annots.csv'

        if not mask_info_path.exists() or not mask_info_annots_path.exists():
            return jsonify({'error': 'Annotation CSV files not found', 'success': False}), 404

        df_info = pd.read_csv(mask_info_path).fillna('')
        df_annots = pd.read_csv(mask_info_annots_path)

        df_annots['image_name'] = df_annots['mask_file'].apply(
            lambda x: x.split('_mask_layer_')[0] if isinstance(x, str) and '_mask_layer_' in x else '')
        df_annots['ID'] = df_annots['mask_file'].apply(
            lambda x: x.split('layer_')[1].replace('.png', '') if isinstance(x, str) and 'layer_' in x else '0')

        unique_images = sorted(df_annots['image_name'].unique())
        if not unique_images:
            return jsonify({'error': 'No images found in annotations', 'success': False}), 404

        # Ensure English columns exist
        for col in ['page', 'plate', 'figure', 'number']:
            if col not in df_info.columns:
                df_info[col] = ''

        total = len(unique_images)
        update_operation_progress('ai_batch', 0, total, 'Loading AI model...')

        # Load model once before the loop
        model, processor = load_gemma_model()

        errors = []
        for idx, current_image_name in enumerate(unique_images):
            update_operation_progress('ai_batch', idx, total,
                                      f'Processing image {idx + 1}/{total}: {current_image_name}')

            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
            original_image_path = None
            for ext in image_extensions:
                candidate = images_path / f"{current_image_name}{ext}"
                if candidate.exists():
                    original_image_path = candidate
                    break

            if not original_image_path:
                errors.append(f'Image not found: {current_image_name}')
                continue

            from PIL import Image as PILImage
            with PILImage.open(original_image_path) as img:
                orig_w, orig_h = img.size
                img_copy = img.copy()
                img_copy.thumbnail((2400, 2400), PILImage.LANCZOS)
                scale_x = img_copy.size[0] / orig_w
                scale_y = img_copy.size[1] / orig_h
                page_image = img_copy.convert('RGB')

            image_annots = df_annots[df_annots['image_name'] == current_image_name]
            bbox_lines = []
            for _, row in image_annots.iterrows():
                try:
                    bbox_str = str(row.get('bbox', '')).strip('()')
                    coords = [int(x.strip()) for x in bbox_str.split(',')]
                    scaled = [
                        int(coords[0] * scale_x), int(coords[1] * scale_y),
                        int(coords[2] * scale_x), int(coords[3] * scale_y)
                    ]
                    bbox_lines.append(f"- ID {row['ID']}: [{scaled[0]}, {scaled[1]}, {scaled[2]}, {scaled[3]}]")
                except Exception:
                    continue

            if not bbox_lines:
                errors.append(f'No annotations for: {current_image_name}')
                continue

            id_list = [line.split(':')[0].replace('- ID ', '').strip() for line in bbox_lines]
            id_list_str = ', '.join(id_list)
            bbox_text = '\n'.join(bbox_lines)
            img_w, img_h = page_image.size
            prompt = (
                "This is a page from an archaeological publication about pottery. "
                f"The full image size is {img_w}x{img_h} pixels.\n"
                f"There are {len(bbox_lines)} pottery drawings with these EXACT IDs and pixel bounding boxes [x1, y1, x2, y2]:\n"
                f"{bbox_text}\n\n"
                f"IMPORTANT: Your JSON response MUST use EXACTLY these keys: {id_list_str}\n"
                "Do NOT invent sequential keys or rename IDs. Each key must match one of the IDs above.\n\n"
                "For EACH drawing, extract:\n"
                '- "page": the page number of the publication. '
                'Look at the EDGES and EXTREMITIES of the full image '
                '(top margin, bottom margin, corners) for a printed page number. '
                'This value is the SAME for all drawings on the page.\n'
                '- "plate": plate/table number (e.g. "Tav. III", "Pl. 12"). '
                'Usually at the top or bottom edge of the image, shared by all drawings.\n'
                '- "figure": figure number for the whole plate (e.g. "Fig. 3", "Abb. 5"). '
                'Usually at top or bottom of the image.\n'
                '- "number": the small reference number printed DIRECTLY NEXT TO or inside '
                'the specific drawing at the given bounding box coordinates. '
                'This is in a VARIABLE POSITION — examine carefully the area immediately '
                'surrounding each bounding box for a small printed number or label.\n\n'
                f"Respond ONLY with a valid JSON object using EXACTLY these keys: {id_list_str}\n"
                "Example (replace <IDx> with the real IDs from above):\n"
                '{\"<ID1>\": {\"page\": \"45\", \"plate\": \"Tav. III\", \"figure\": \"Fig. 5\", \"number\": \"1\"}, '
                '\"<ID2>\": {\"page\": \"45\", \"plate\": \"Tav. III\", \"figure\": \"Fig. 5\", \"number\": \"2a\"}}\n'
                "If a value is not found, use null."
            )

            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": page_image},
                    {"type": "text", "text": prompt}
                ]
            }]

            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            inputs = processor(text=text, images=[page_image], return_tensors="pt").to(model.device)
            input_len = inputs["input_ids"].shape[-1]

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=1.0,
                    top_p=0.95,
                    top_k=64,
                    do_sample=True
                )

            raw_response = processor.decode(outputs[0][input_len:], skip_special_tokens=True)
            print(f"[AI Batch] {current_image_name} response: {raw_response[:200]}")

            json_match = _re.search(r'\{.*\}', raw_response, _re.DOTALL)
            if not json_match:
                errors.append(f'No JSON from model for: {current_image_name}')
                continue

            try:
                ai_result = _json.loads(json_match.group())
            except _json.JSONDecodeError:
                errors.append(f'Invalid JSON for: {current_image_name}')
                continue

            for mask_id, values in ai_result.items():
                exact_mask_file = f"{current_image_name}_mask_layer_{mask_id}.png"
                row_mask = df_info['mask_file'] == exact_mask_file
                if not row_mask.any():
                    exact_no_ext = f"{current_image_name}_mask_layer_{mask_id}"
                    row_mask = df_info['mask_file'].apply(
                        lambda x: str(x).replace('.png', '') == exact_no_ext
                    )
                if row_mask.any():
                    for col in ['page', 'plate', 'figure', 'number']:
                        val = values.get(col)
                        if val is not None:
                            df_info.loc[row_mask, col] = str(val)

        df_info.to_csv(mask_info_path, index=False)
        clear_operation_progress()

        return jsonify({
            'success': True,
            'processed': total,
            'errors': errors
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        clear_operation_progress()
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/api/projects/<project_id>/tabular/export', methods=['POST'])
def export_project_tabular_csv(project_id):
    """Save combined tabular CSV (mask_info.csv) in the project folder"""
    try:
        # Verify project exists
        project_metadata = project_manager.get_project(project_id)
        if not project_metadata:
            return jsonify({'error': 'Project not found', 'success': False}), 404

        cards_path = project_manager.get_project_path(project_id, 'cards')
        if not cards_path or not cards_path.exists():
            return jsonify({'error': 'Cards folder not found', 'success': False}), 404

        # Get source CSV from temp location (if exists)
        temp_csv_path = cards_path / 'mask_info.csv'
        
        # Save to project root
        project_path = project_manager.get_project_path(project_id)
        export_csv_path = project_path / f"{project_id}_mask_info.csv"
        
        if temp_csv_path.exists():
            import shutil
            shutil.copy2(temp_csv_path, export_csv_path)
        else:
            return jsonify({'error': 'Combined CSV not found. Please save tabular data first.', 'success': False}), 404

        return jsonify({
            'message': f'CSV exported to {export_csv_path.name}',
            'path': str(export_csv_path),
            'success': True
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e), 'success': False}), 500


# ==================== EXPORT ====================

@app.route('/api/export', methods=['POST'])
def export_results():
    """Export final results"""
    try:
        data = request.json
        folder = data.get('folder')
        acronym = data.get('acronym')
        export_pdf = data.get('export_pdf', False)
        page_size = data.get('page_size', 'A4')
        scale_factor = float(data.get('scale_factor', 1.0))
        
        if not all([folder, acronym]):
            return jsonify({'error': 'Folder and acronym are required', 'success': False}), 400
        
        # Validate acronym
        if not acronym.replace('_', '').isalnum():
            return jsonify({'error': 'Acronym can only contain letters, numbers, and underscores', 'success': False}), 400
        
        result = export_processor.export_results(
            folder=folder,
            acronym=acronym,
            export_pdf=export_pdf,
            page_size=page_size,
            scale_factor=scale_factor
        )
        
        return jsonify({
            'message': result,
            'success': True
        })
        
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500


# ==================== STATIC FILES ====================

@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files"""
    return send_from_directory('static', filename)



@app.route('/api/projects/<project_id>/export', methods=['POST'])
def export_project_results(project_id):
    """Export final results for a project (with auto-merge if CSV exists)"""
    try:
        # Verify project exists
        project_metadata = project_manager.get_project(project_id)
        if not project_metadata:
            return jsonify({'error': 'Project not found', 'success': False}), 404
        
        data = request.json
        acronym = data.get('acronym')
        
        if not acronym:
            return jsonify({'error': 'Acronym is required', 'success': False}), 400
        
        # Validate acronym
        if not acronym.replace('_', '').isalnum():
            return jsonify({'error': 'Acronym can only contain letters, numbers, and underscores', 'success': False}), 400
        
        # Get project paths
        cards_path = project_manager.get_project_path(project_id, 'cards')
        cards_modified_path = project_manager.get_project_path(project_id, 'cards_modified')
        project_path = project_manager.get_project_path(project_id)
        
        if not cards_path or not cards_path.exists():
            return jsonify({'error': 'No cards folder found', 'success': False}), 404
        
        # Auto-merge: check if combined CSV exists in project root
        combined_csv_path = project_path / f"{project_id}_mask_info.csv"
        if combined_csv_path.exists():
            print(f"Found combined CSV, merging with classifications...")
            try:
                # Merge the CSVs
                import pandas as pd
                
                # Load combined CSV (tabular data)
                combined_df = pd.read_csv(combined_csv_path)
                print(f"Loaded combined CSV with {len(combined_df)} rows")
                print(f"Combined CSV columns: {list(combined_df.columns)}")
                
                # Load classifications if they exist
                classifications_path = cards_modified_path / 'classifications.csv' if cards_modified_path else None
                if classifications_path and classifications_path.exists():
                    classifications_df = pd.read_csv(classifications_path)
                    print(f"Loaded classifications CSV with {len(classifications_df)} rows")
                    print(f"Classifications CSV columns: {list(classifications_df.columns)}")
                    
                    # Ensure filename columns are compatible (remove .png if present in one but not other)
                    if 'filename' in combined_df.columns and 'filename' in classifications_df.columns:
                        # Normalize filenames - remove extension for matching
                        combined_df['filename_base'] = combined_df['filename'].str.replace('.png', '').str.replace('.jpg', '')
                        classifications_df['filename_base'] = classifications_df['filename'].str.replace('.png', '').str.replace('.jpg', '')
                        
                        # Merge on normalized filename
                        merged = pd.merge(
                            combined_df,
                            classifications_df,
                            on='filename_base',
                            how='left',
                            suffixes=('', '_class')
                        )
                        
                        # Keep original filename from combined
                        if 'filename_class' in merged.columns:
                            merged = merged.drop('filename_class', axis=1)
                        merged = merged.drop('filename_base', axis=1)
                        
                        # Save merged annotations
                        cards_modified_path.mkdir(exist_ok=True)
                        merged_path = cards_modified_path / 'merged_annotations.csv'
                        merged.to_csv(merged_path, index=False)
                        print(f"Auto-merged {len(merged)} annotations to {merged_path}")
                        print(f"Merged columns: {list(merged.columns)}")
                    else:
                        print("Warning: 'filename' column not found in one of the CSVs")
                        merged_path = cards_modified_path / 'merged_annotations.csv'
                        combined_df.to_csv(merged_path, index=False)
                else:
                    # No classifications, use combined CSV directly as merged
                    cards_modified_path.mkdir(exist_ok=True)
                    merged_path = cards_modified_path / 'merged_annotations.csv'
                    combined_df.to_csv(merged_path, index=False)
                    print(f"No classifications found, using combined CSV as merged")
                    
            except Exception as e:
                print(f"Warning: Auto-merge failed: {e}")
                import traceback
                traceback.print_exc()
                # Continue with export anyway
        
        # Determine export folder (prefer cards_modified if it has content)
        if cards_modified_path and cards_modified_path.exists() and any(cards_modified_path.iterdir()):
            export_folder = cards_modified_path
        else:
            export_folder = cards_path
        
        # Load merged annotations if available
        import pandas as pd
        merged_path = cards_modified_path / 'merged_annotations.csv' if cards_modified_path else None
        
        # Create final metadata with new IDs
        metadata_df = None
        if merged_path and merged_path.exists():
            metadata_df = pd.read_csv(merged_path)
            print(f"Loaded merged annotations: {len(metadata_df)} rows")
            print(f"Merged columns: {list(metadata_df.columns)}")
        elif combined_csv_path.exists():
            # Use combined CSV if no merged exists
            metadata_df = pd.read_csv(combined_csv_path)
            print(f"Loaded combined CSV: {len(metadata_df)} rows")
            print(f"Combined columns: {list(metadata_df.columns)}")
        
        # Load classifications to ensure we have type column
        classifications_df = None
        classifications_path = cards_modified_path / 'classifications.csv' if cards_modified_path else None
        if classifications_path and classifications_path.exists():
            classifications_df = pd.read_csv(classifications_path)
            print(f"Loaded classifications: {len(classifications_df)} rows")
            print(f"Classifications columns: {list(classifications_df.columns)}")
        
        # Create ZIP in temporary location
        import tempfile
        import zipfile
        
        temp_dir = tempfile.mkdtemp()
        zip_path = Path(temp_dir) / f"{acronym}.zip"
        
        try:
            # Get all card images sorted
            card_images = sorted([f for f in export_folder.iterdir() if f.suffix.lower() in ['.png', '.jpg', '.jpeg']])
            print(f"Found {len(card_images)} card images to export")
            
            # Prepare final metadata with new IDs
            final_metadata = []
            
            for idx, img_file in enumerate(card_images, 1):
                new_id_with_ext = f"{acronym}_{idx}{img_file.suffix}"  # Include extension
                
                # Initialize row with id
                row_data = {'id': new_id_with_ext}
                
                # Try to find matching row in metadata
                matched = False
                if metadata_df is not None:
                    # Try different column names for matching
                    for col in ['mask_file', 'filename', 'Filename', 'file']:
                        if col in metadata_df.columns:
                            # Normalize both sides for comparison (remove extensions)
                            img_base = img_file.stem  # filename without extension
                            
                            # Try exact match first
                            mask = metadata_df[col] == img_file.name
                            if not mask.any():
                                # Try without extension
                                mask = metadata_df[col].str.replace('.png', '').str.replace('.jpg', '').str.replace('.jpeg', '') == img_base
                            
                            if mask.any():
                                row = metadata_df[mask].iloc[0]
                                
                                # Copy all columns except unwanted ones
                                exclude_cols = ['mask_file', 'filename', 'Filename', 'filename_base', 'file', 'ID', 'id']
                                for metadata_col in metadata_df.columns:
                                    if metadata_col not in exclude_cols:
                                        row_data[metadata_col] = row[metadata_col]
                                
                                matched = True
                                print(f"Matched {img_file.name} via column '{col}'")
                                break
                
                # Ensure 'type' is present from classifications if available
                if classifications_df is not None and 'type' not in row_data:
                    # Try to match with classifications
                    img_base = img_file.stem
                    for col in ['filename', 'Filename']:
                        if col in classifications_df.columns:
                            mask = classifications_df[col].str.replace('.png', '').str.replace('.jpg', '').str.replace('.jpeg', '') == img_base
                            if mask.any():
                                class_row = classifications_df[mask].iloc[0]
                                if 'type' in class_row:
                                    row_data['type'] = class_row['type']
                                    print(f"Added type '{class_row['type']}' for {img_file.name}")
                                break
                
                final_metadata.append(row_data)
            
            # Create final metadata DataFrame
            final_df = pd.DataFrame(final_metadata)
            
            # Reorder columns: id first, then type (if present), then others alphabetically
            cols = ['id']
            if 'type' in final_df.columns:
                cols.append('type')
            # Add remaining columns alphabetically
            remaining = sorted([col for col in final_df.columns if col not in cols])
            cols.extend(remaining)
            final_df = final_df[cols]
            
            # Save metadata to temp file
            metadata_temp_path = Path(temp_dir) / f"{acronym}_metadata.csv"
            final_df.to_csv(metadata_temp_path, index=False)
            print(f"Created final metadata with {len(final_df)} rows and columns: {list(final_df.columns)}")
            
            # Create ZIP
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Add images with new names
                for idx, img_file in enumerate(card_images, 1):
                    new_name = f"{acronym}_{idx}{img_file.suffix}"
                    zipf.write(img_file, new_name)
                    print(f"Added {img_file.name} as {new_name}")
                
                # Add metadata
                zipf.write(metadata_temp_path, f"{acronym}_metadata.csv")
                print(f"Added metadata CSV")
        
            # Send the ZIP file
            return send_file(
                str(zip_path),
                as_attachment=True,
                download_name=f"{acronym}.zip",
                mimetype='application/zip'
            )
            
        finally:
            # Cleanup will happen after send_file completes
            pass
        
    except Exception as e:
        print(f"Error exporting project results: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e), 'success': False}), 500


# ==================== ERROR HANDLERS ====================

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Resource not found', 'success': False}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Internal server error', 'success': False}), 500


@app.route('/api/projects/<project_id>/thumbnail/<filename>')
def serve_project_thumbnail(project_id, filename):
    """Serve a thumbnail version of a project image"""
    try:
        from PIL import Image
        import io
        
        # Get images path
        images_path = project_manager.get_project_path(project_id, 'images')
        if not images_path or not images_path.exists():
            return jsonify({'error': 'Images folder not found', 'success': False}), 404
        
        image_path = images_path / filename
        if not image_path.exists():
            return jsonify({'error': 'Image not found', 'success': False}), 404
        
        # Open and create thumbnail
        with Image.open(image_path) as img:
            # Convert to RGB if necessary (for JPEG compatibility)
            if img.mode in ('RGBA', 'LA', 'P'):
                img = img.convert('RGB')
            
            # Create thumbnail (max 300px on longest side, maintain aspect ratio)
            img.thumbnail((300, 300), Image.Resampling.LANCZOS)
            
            # Save to memory buffer
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=85, optimize=True)
            buffer.seek(0)
            
            return send_file(buffer, mimetype='image/jpeg')
            
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500


if __name__ == '__main__':
    print("\n" + "="*80)
    print(" 🏺 PyPotteryLens Flask Application 🔍")
    print("="*80)
    print("\n🚀 Starting server...")
    print("📝 Browser will open at: http://localhost:5001")
    print("💡 Initialization will continue in background...")
    print("\n" + "="*80 + "\n")
    
    # Open browser immediately after Flask starts
    import webbrowser
    import threading
    
    def open_browser():
        import time
        time.sleep(1)  # Wait 1 second for Flask to start
        webbrowser.open('http://localhost:5001')
        print("🌐 Browser opened!")
    
    # Start browser in separate thread
    threading.Thread(target=open_browser, daemon=True).start()
    
    app.run(
        host='0.0.0.0',
        port=5001,
        debug=True,
        threaded=True,
        use_reloader=False  # Disable reloader to prevent double initialization
    )
