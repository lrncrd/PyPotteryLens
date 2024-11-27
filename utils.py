# utils.py

import numpy as np
import os
from pathlib import Path
from dataclasses import dataclass
import pandas as pd
from PIL import Image
import gradio as gr
import fitz
from ultralytics import YOLO
from skimage.filters import threshold_otsu, median
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square, disk
from scipy.ndimage import binary_dilation, binary_erosion
from typing import  Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass
import torch
import torchvision.transforms as transforms
from models import MultiHeadEfficientNet
import shutil
from reportlab.lib import pagesizes
from reportlab.pdfgen import canvas
from PIL import Image
import gc




@dataclass
class PDFConfig:
    """Configuration for PDF processing"""
    output_dir: Path

@dataclass
class ModelConfig:
    """Configuration for model processing"""
    models_dir: Path
    pred_output_dir: Path
    confidence: float = 0.5
    kernel_size: int = 2
    iterations: int = 10
    diagnostic: bool = False

@dataclass
class MaskExtractionConfig:
    """Configuration for mask extraction"""
    pdfimg_output_dir: Path  # Directory containing the original images
    pred_output_dir: Path    # Directory for predictions and output
    min_area_ratio: float = 0.001 # 0.005
    closing_kernel_size: int = 3
    output_suffix: str = "_card"
    mask_suffix: str = "_mask"

@dataclass
class AnnotationConfig:
    """Configuration for annotation processing"""
    pred_output_dir: Path

@dataclass
class TabularConfig:
    """Configuration for tabular processing"""
    pdfimg_output_dir: Path
    pred_output_dir: Path
    max_workers: int = 4  # For parallel processing
    cache_size: int = 32  # For LRU cache

class PDFProcessor:
    """Handles PDF to image conversion using PyMuPDF"""
    
    def __init__(self, config):
        self.config = config

    def process_pdf(self, pdf_path: str, split_pages: bool = False) -> str:
        """
        Convert PDF to images with optional page splitting
        
        Args:
            pdf_path: Path to PDF file
            split_pages: If True, splits each page into left and right halves
        """
        try:
            pdf_file_name = Path(pdf_path).stem
            output_folder = self.config.output_dir / pdf_file_name
            os.makedirs(output_folder, exist_ok=True)
            
            # Open PDF document
            doc = fitz.open(pdf_path)
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Get the pixel map with a good resolution
                pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
                
                # Convert to PIL Image
                img_data = pix.samples
                img = Image.frombytes("RGB", [pix.width, pix.height], img_data)
                
                if split_pages:
                    self._process_split_page(img, pdf_file_name, page_num, output_folder)
                else:
                    self._process_single_page(img, pdf_file_name, page_num, output_folder)
            
            doc.close()
            return f"PDF file {pdf_file_name} has been converted to JPG"
            
        except Exception as e:
            return f"Error processing PDF: {str(e)}"

    def _process_single_page(self, image: Image.Image, pdf_name: str, page_num: int, output_folder: Path):
        """Save a single page as one image"""
        output_image_name = f'{pdf_name}_page_{page_num}.jpg'
        image.save(output_folder / output_image_name, 'JPEG')

    def _process_split_page(self, image: Image.Image, pdf_name: str, page_num: int, output_folder: Path):
        """Split a page into left and right halves and save separately"""
        width, height = image.size
        mid_point = width // 2

        # Split into left and right pages
        left_page = image.crop((0, 0, mid_point, height))
        right_page = image.crop((mid_point, 0, width, height))

        # Save both pages with appropriate numbering
        left_page.save(output_folder / f'{pdf_name}_page_{page_num}a.jpg', 'JPEG')
        right_page.save(output_folder / f'{pdf_name}_page_{page_num}b.jpg', 'JPEG')

        
class ModelProcessor:
    """Handles model application and prediction"""
    
    def __init__(self, config: ModelConfig):
        self.config = config

    def apply_model(self,
                   folder: str,
                   model_name: str,
                   confidence: float,
                   diagnostic: bool,
                   kernel_size: float,
                   iterations: float) -> str:
        """Apply model to images in folder"""
        try:
            kernel_size = int(kernel_size)
            iterations = int(iterations)
            
            progress_tracker = gr.Progress()
            progress_tracker(0, desc="Starting model application")
            
            if not folder or not model_name:
                return "Please select both a folder and a model"
            
            # Load model from models directory
            model_path = self.config.models_dir / model_name
            if not model_path.exists():
                return f"Model not found: {model_name}"
            
            model = YOLO(model_path)
            
            # Setup output directory for masks
            output_folder = self.config.pred_output_dir / f"{folder}_mask"
            os.makedirs(output_folder, exist_ok=True)
            
            # Get images from pdf2img_outputs directory
            image_path = self.config.pred_output_dir.parent / "pdf2img_outputs" / folder
            if not image_path.exists():
                return f"Image folder not found: {folder}"
                
            images = os.listdir(image_path)
            
            if diagnostic:
                images = images[:25]
                
            for image_file in progress_tracker.tqdm(images):
                self._process_single_image(
                    image_file,
                    image_path,
                    model,
                    confidence,
                    kernel_size,
                    iterations,
                    output_folder
                )
            
            return f"Model applied successfully to {folder} with confidence={confidence}, kernel={kernel_size}, iterations={iterations}"
        except Exception as e:
            return f"Error applying model: {str(e)}"

    def _process_single_image(self,
                            image_file: str,
                            image_path: Path,
                            model: YOLO,
                            confidence: float,
                            kernel_size: int,
                            iterations: int,
                            output_folder: Path) -> None:
        """Process a single image with the model"""
        try:
            img = Image.open(image_path / image_file)
            results = model.predict(
                img,
                save_crop=False,
                conf=confidence,
                retina_masks=True
            )[0]
            
            if len(results) > 0:
                pred_masks = results.masks.data.cpu().numpy()
                save_mask(
                    img,
                    pred_masks,
                    image_file.split(".")[0],
                    output_folder,
                    kernel_size,
                    iterations,
                    export_masks=True
                )
        except Exception as e:
            print(f"Error processing image {image_file}: {str(e)}")


class MaskExtractor:
    """Handles mask extraction using black mask annotations"""
    
    def __init__(self, config: MaskExtractionConfig):
        self.config = config

    def _setup_directories(self, folder: str) -> tuple[Path, str, Path, Path]:
        """Setup and return required directories and image format"""
        img_folder = self.config.pdfimg_output_dir / folder
        img_format = os.listdir(img_folder)[0].split(".")[1]
        mask_folder = self.config.pred_output_dir / f"{folder}_mask"
        output_folder = self.config.pred_output_dir / f"{folder}_card"
        os.makedirs(output_folder, exist_ok=True)
        return img_folder, img_format, mask_folder, output_folder

    def _process_mask(self, mask_array: np.ndarray) -> np.ndarray:
        """Process mask array to get labeled regions"""
        thresh = threshold_otsu(mask_array)
        bw = closing(mask_array > thresh, square(3))
        cleared = clear_border(bw)
        return label(cleared.astype(int))
    
    def _create_region_mask(self, shape: tuple, region_bbox: tuple, region_mask: np.ndarray) -> np.ndarray:
        """Create full-size mask for a region"""
        minr, minc, maxr, maxc = region_bbox
        mask = np.zeros(shape)
        mask[minr:maxr, minc:maxc] = region_mask
        return mask

    def _expand_mask(self, mask: np.ndarray) -> np.ndarray:
        """Expand mask to match image dimensions"""
        mask_exp = np.expand_dims(mask * 255, axis=-1)
        return np.repeat(mask_exp, 3, axis=-1).astype(np.uint8)


    def _extract_region(self, 
                    region: 'RegionProperties', 
                    mask_array: np.ndarray, 
                    orig_array: np.ndarray, 
                    total_area: int) -> tuple[np.ndarray, tuple] | None:
        """
        Extract region using precise segmentation mask with PIL for image handling
        """
        if region.area < total_area * self.config.min_area_ratio:
            return None

        # Get bounding box coordinates
        minr, minc, maxr, maxc = region.bbox
        
        # Ensure dimensions match before processing
        mask_shape = mask_array.shape[:2]
        orig_shape = orig_array.shape[:2]
        
        if mask_shape != orig_shape:
            # Convert mask array to PIL Image for resizing
            mask_img = Image.fromarray(mask_array)
            # Resize mask to match original image dimensions
            mask_resized = mask_img.resize((orig_array.shape[1], orig_array.shape[0]), 
                                        resample=Image.Resampling.NEAREST)
            # Convert back to numpy array
            mask_array = np.array(mask_resized)
            
            # Recalculate bbox coordinates
            scale_y = orig_shape[0] / mask_shape[0]
            scale_x = orig_shape[1] / mask_shape[1]
            minr = int(minr * scale_y)
            maxr = int(maxr * scale_y)
            minc = int(minc * scale_x)
            maxc = int(maxc * scale_x)
            
            # Convert region mask to PIL Image and resize
            region_mask_img = Image.fromarray(region.image.astype(np.uint8) * 255)
            region_mask_resized = region_mask_img.resize((maxc - minc, maxr - minr), 
                                                        resample=Image.Resampling.NEAREST)
            region_mask = np.array(region_mask_resized) > 0
        else:
            region_mask = region.image

        # Create full-size mask
        full_mask = np.zeros_like(orig_array[:,:,0], dtype=bool)
        full_mask[minr:maxr, minc:maxc] = region_mask
        
        # Expand mask to match image dimensions
        mask_exp = np.expand_dims(full_mask, axis=-1).astype(np.uint8)
        mask_exp = np.repeat(mask_exp, 3, axis=-1)
        
        # Apply mask to original image
        masked_img = np.where(mask_exp == 0, 255, orig_array)
        
        # Crop to bounding box
        cropped = masked_img[minr:maxr, minc:maxc]
        
        return cropped, (minc, minr, maxc, maxr)

    def _save_metadata(self, 
                      metadata: list[tuple], 
                      annotations: list[tuple], 
                      output_folder: Path) -> None:
        """Save extraction metadata to CSV files"""
        if not metadata:
            return
            
        pd.DataFrame(metadata, columns=["file", "mask_file"]).to_csv(
            output_folder / "mask_info.csv", index=False
        )
        pd.DataFrame(annotations, columns=["bbox", "mask_file"]).to_csv(
            output_folder / "mask_info_annots.csv", index=False
        )

    def extract_masks(self, drop_folder_review: str) -> str:
        """Extract masks from images in folder"""
        try:
            progress_tracker = gr.Progress()
            progress_tracker(0, desc="Starting mask extraction")
            
            # Setup directories
            img_folder, img_format, mask_folder, output_folder = self._setup_directories(drop_folder_review)
            
            metadata = []
            annotations = []

            # Process each mask file
            for file in progress_tracker.tqdm(os.listdir(mask_folder)):
                base_filename = file.split(".")[0].replace("_mask_layer", "")
                
                # Load images
                mask_array = np.array(Image.open(mask_folder / file).convert("L"))
                orig_array = np.array(Image.open(img_folder / f"{base_filename}.{img_format}"))
                
                # Process mask and get labeled regions
                label_image = self._process_mask(mask_array)
                total_area = mask_array.size
                
                # Process each region
                for i, region in enumerate(regionprops(label_image)):
                    result = self._extract_region(region, mask_array, orig_array, total_area)
                    if result is None:
                        continue
                        
                    cropped, bbox = result
                    output_filename = f"{base_filename}_mask_layer_{i}.png"
                    
                    # Save cropped image
                    Image.fromarray(cropped).save(output_folder / output_filename)
                    
                    # Store metadata
                    metadata.append((base_filename, f"{base_filename}_mask_layer_{i}"))
                    annotations.append((bbox, output_filename))

            # Save metadata
            self._save_metadata(metadata, annotations, output_folder)
            
            if metadata:
                return f"Successfully extracted {len(metadata)} masks from '{drop_folder_review}'"
            return "No masks were extracted. Check if masks are properly drawn."

        except Exception as e:
            print(f"Error in mask extraction: {str(e)}")
            return f"Error extracting masks: {str(e)}"
    
class AnnotationProcessor:
    """Handles annotation processing"""
    
    def __init__(self, config: AnnotationConfig):
        self.config = config

    def save_annotation(self, folder: str, editor_data: Dict, current_image: str) -> bool:
        """Save annotation and return success status"""
        try:
            if not all([folder, editor_data, current_image]):
                return False
                    
            # Fast path for saving
            if 'layers' in editor_data and editor_data['layers']:
                layer = editor_data['layers'][0]
                if layer is not None:
                    # Setup paths
                    file_name = Path(current_image).stem + "_mask_layer"
                    saving_folder = folder + "_mask"
                    saving_path = self.config.pred_output_dir / saving_folder / f"{file_name}.png"
                    
                    # Ensure saving directory exists
                    os.makedirs(self.config.pred_output_dir / saving_folder, exist_ok=True)
                    
                    # Direct save
                    Image.fromarray(layer).save(saving_path)
                    
                    # Force Python garbage collection
                    del layer
                    gc.collect()
                    
                    return True
                    
            return False
            
        except Exception as e:
            print(f"Error saving annotation: {str(e)}")
            return False
        
    def file_selection(self, file_path: str) -> Dict:
        """Select file and prepare image data with performance optimizations"""
        try:
            if not file_path:
                return {"background": None, "layers": [], "composite": None}
                    
            file_path = Path(file_path)
            
            # Prepare mask path
            mask_dir_name = file_path.parent.name + "_mask"
            mask_name = file_path.stem + "_mask_layer.png"
            mask_path = self.config.pred_output_dir / mask_dir_name / mask_name

            # Load and resize original image
            with Image.open(file_path) as img:
                # Resize for preview while maintaining aspect ratio
                img.thumbnail((1200, 1200))
                img_background = np.asarray(img, dtype=np.uint8)
            
            # Check and load mask if exists
            layers = []
            if mask_path.exists():
                with Image.open(mask_path) as mask_img:
                    # Resize mask to match image dimensions
                    mask_img = mask_img.resize(img_background.shape[:2][::-1], Image.Resampling.NEAREST)
                    mask = np.asarray(mask_img, dtype=np.uint8)
                    layers = [mask]
            
            # Force cleanup
            gc.collect()
            
            return {
                "background": img_background,
                "layers": layers,
                "composite": img_background.copy()
            }
            
        except Exception as e:
            print(f"Error in file selection: {str(e)}")
            return {"background": None, "layers": [], "composite": None}

class ImageProcessor:
    """Handles image processing and display"""
    
    def __init__(self, pdfimg_output_dir: Path, pred_output_dir: Path):
        self.pdfimg_output_dir = Path(pdfimg_output_dir)
        self.pred_output_dir = Path(pred_output_dir)

    def return_images(self, folder: str) -> List[str]:
        """Return list of images in folder"""
        if not folder:
            return []
        try:
            return [
                str(self.pdfimg_output_dir / folder / img_name)
                for img_name in os.listdir(self.pdfimg_output_dir / folder)
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg'))
            ]
        except Exception as e:
            print(f"Error loading images: {str(e)}")
            return []
        

class TabularProcessor:
    """Handles tabular data viewing and editing for extracted masks"""
    
    def __init__(self, config: TabularConfig):
        self.pdfimg_output_dir = Path(config.pdfimg_output_dir).resolve()
        self.pred_output_dir = Path(config.pred_output_dir).resolve()
        self._current_file = None

    def get_results_folders(self) -> List[str]:
        """Get list of folders containing results with validation"""
        try:
            folder_list = [f for f in os.listdir(self.pred_output_dir) 
                          if f.endswith('_card') and not f.endswith('transformed_card')]
            
            # Validate each folder contains required files
            valid_folders = []
            for folder in folder_list:
                folder_path = self.pred_output_dir / folder
                if folder_path.is_dir():
                    mask_info = folder_path / "mask_info.csv"
                    mask_info_annots = folder_path / "mask_info_annots.csv"
                    if mask_info.exists() and mask_info_annots.exists():
                        valid_folders.append(folder)
                        print(f"Valid folder found: {folder}")
            
            return valid_folders
            
        except Exception as e:
            print(f"Error in get_results_folders: {str(e)}")
            return []

    def convert_bbox(self, bbox_str: str) -> tuple:
        """Convert bbox string to tuple with error handling"""
        try:
            # Remove parentheses and split
            bbox = bbox_str.strip('()').split(',')
            return tuple(int(float(coord)) for coord in bbox)
        except Exception as e:
            print(f"Error converting bbox {bbox_str}: {str(e)}")
            return (0, 0, 0, 0)

    def create_annotation_tuple(self, df: pd.DataFrame, image_name: str) -> List[tuple]:
        """Create list of annotation tuples with validation"""
        try:
            # Filter dataframe for current image
            df_selected = df[df['image_name'] == image_name].copy()
            
            if df_selected.empty:
                print(f"No annotations found for image {image_name}")
                return []
            
            # Create annotation tuples with error handling
            annotations = []
            for _, row in df_selected.iterrows():
                try:
                    bbox = self.convert_bbox(row['bbox'])
                    mask_id = str(row['ID']).strip()  # Ensure ID is string and clean
                    if all(isinstance(x, (int, float)) for x in bbox):
                        annotations.append((bbox, mask_id))
                except Exception as e:
                    print(f"Error processing annotation row: {str(e)}")
                    continue
                    
            return annotations
            
        except Exception as e:
            print(f"Error creating annotations: {str(e)}")
            return []

    def image_selection(self, folder: str, img_num: int) -> tuple:
        """Select and prepare image and associated data for display"""
        try:
            if not folder:
                return None, img_num, pd.DataFrame()

            # Setup paths
            context = folder.split("_card")[0]
            folder_mask_path = (self.pred_output_dir / f"{context}_mask").resolve()
            folder_img_path = (self.pdfimg_output_dir / context).resolve()
            csv_path = (self.pred_output_dir / folder).resolve()

            print(f"\nProcessing paths:")
            print(f"Mask folder: {folder_mask_path}")
            print(f"Image folder: {folder_img_path}")
            print(f"CSV folder: {csv_path}")

            # Validate paths existence
            if not all(p.exists() for p in [folder_mask_path, folder_img_path, csv_path]):
                print("Missing required folders")
                return None, img_num, pd.DataFrame()

            # Load CSV files
            try:
                mask_info_path = csv_path / "mask_info.csv"
                mask_info_annots_path = csv_path / "mask_info_annots.csv"

                print(f"\nReading CSV files:")
                print(f"mask_info.csv: {mask_info_path}")
                print(f"mask_info_annots.csv: {mask_info_annots_path}")

                if not mask_info_path.exists() or not mask_info_annots_path.exists():
                    print("Required CSV files not found")
                    return None, img_num, pd.DataFrame()

                # Read CSVs with explicit dtypes
                df = pd.read_csv(mask_info_path).fillna('')
                df_annots = pd.read_csv(mask_info_annots_path)
                
                # Clean and prepare annotation data
                df_annots['image_name'] = df_annots['mask_file'].apply(
                    lambda x: x.split('_mask')[0] if isinstance(x, str) else '')
                df_annots['ID'] = df_annots['mask_file'].apply(
                    lambda x: x.split('layer_')[1] if isinstance(x, str) else '')

            except Exception as e:
                print(f"Error reading CSV files: {str(e)}")
                return None, img_num, pd.DataFrame()

            # Get valid images with corresponding masks
            images = []
            for f in os.listdir(folder_img_path):
                if not f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                    
                mask_file = f.split(".")[0] + "_mask_layer.png"
                if os.path.exists(folder_mask_path / mask_file):
                    images.append(f)

            if not images:
                print("No valid images found")
                return None, img_num, pd.DataFrame()

            # Get current image
            img_num = max(0, min(img_num, len(images) - 1))
            current_image = images[img_num]
            image_base_name = current_image.split(".")[0]
            
            print(f"\nProcessing image: {image_base_name}")
            
            # Store current file
            self._current_file = image_base_name
            
            # Prepare display data
            df_subset = df[df["file"] == image_base_name].copy()
            
            if df_subset.empty:
                print(f"No data found for image {image_base_name}")
                return None, img_num, pd.DataFrame()

            df_subset["ID"] = df_subset["mask_file"].apply(lambda x: x.split('layer_')[1] if isinstance(x, str) else '')
            
            # Clean up columns
            drop_cols = [col for col in ["mask_file", "file"] if col in df_subset.columns]
            if drop_cols:
                df_subset.drop(columns=drop_cols, inplace=True)
            
            # Reorder columns
            columns_order = ["ID"] + [col for col in df_subset.columns if col != "ID"]
            df_display = df_subset[columns_order]

            # Create image with annotations
            try:
                img_path = folder_img_path / current_image
                if not img_path.exists():
                    print(f"Image file not found: {img_path}")
                    return None, img_num, df_display

                image = Image.open(img_path)
                annotations = self.create_annotation_tuple(df_annots, image_base_name)
                
                print(f"Created {len(annotations)} annotations")
                
                return (
                    gr.AnnotatedImage(value=[image, annotations]), 
                    img_num, 
                    df_display
                )
                
            except Exception as e:
                print(f"Error creating annotated image: {str(e)}")
                return None, img_num, df_display

        except Exception as e:
            print(f"Error in image selection: {str(e)}")
            return None, img_num, pd.DataFrame()

    def save_table(self, table: pd.DataFrame, folder: str) -> None:
        """Save table with robust error handling"""
        try:
            if self._current_file is None:
                print("No file currently selected")
                return

            csv_path = self.pred_output_dir / folder / "mask_info.csv"
            if not csv_path.exists():
                print(f"CSV file not found: {csv_path}")
                return

            # Read existing table
            existing_table = pd.read_csv(csv_path)
            
            # Add back file information
            table = table.copy()
            table['file'] = self._current_file
            table["mask_file"] = table["file"] + "_mask_layer_" + table["ID"]
            
            if "ID" in table.columns:
                table.drop(columns=["ID"], inplace=True)
            
            # Update existing table
            existing_table.set_index("mask_file", inplace=True)
            table.set_index("mask_file", inplace=True)

            # Update values
            for col in table.columns:
                if col not in existing_table.columns:
                    existing_table[col] = pd.NA
                existing_table.loc[table.index, col] = table[col]

            # Clean up and save
            existing_table.reset_index(inplace=True)
            existing_table = existing_table.loc[:, ~existing_table.columns.str.contains('Header')]
            existing_table.to_csv(csv_path, index=False)
            
            print(f"Table saved successfully to {csv_path}")
            
        except Exception as e:
            print(f"Error saving table: {str(e)}")
            
def save_mask(img: Image.Image,
             masks_array: np.ndarray,
             img_name: str = "",
             output_dir: str = ".",
             kernel_size: int = 5,
             num_iterations: int = 10,
             export_masks: bool = False) -> None:
    """Save mask layers for detected objects"""
    output_path = Path(output_dir)
    
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)

    dilated_masks = _process_masks(masks_array, kernel, num_iterations)

    combined_mask = _combine_masks(dilated_masks, kernel)
    
    if export_masks:
        _export_mask(combined_mask, output_path, img_name)

def _process_masks(masks_array: np.ndarray,
                  kernel: np.ndarray,
                  num_iterations: int) -> List[np.ndarray]:
    """Process individual masks"""
    return [
        binary_dilation(mask.copy(), iterations=num_iterations, structure=kernel)
        for mask in masks_array
    ]

def _combine_masks(masks: List[np.ndarray], kernel: np.ndarray) -> np.ndarray:
    """Combine multiple masks into one"""
    combined = np.sum(masks, axis=0)
    ###
    combined = binary_erosion(combined, iterations=2, structure=kernel)
    ###
    return median(combined, footprint=disk(5))

def _export_mask(mask: np.ndarray,
                output_path: Path,
                img_name: str) -> None:
    """Export processed mask"""
    mask_repeated = np.repeat(np.expand_dims(mask * 128, 2), 4, axis=2)
    mask_rgba = Image.fromarray(mask_repeated.astype(np.uint8), mode="RGBA")
    mask_rgba.save(output_path / f"{img_name}_mask_layer.png")


# Update in utils_new.py

@dataclass
class SecondStepConfig:
    """Configuration for second step processing"""
    pred_output_dir: Path
    model_path: Path = Path("models/model_classifier.pth")
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    processed_suffix: str = "_processed"

class SecondStepProcessor:
    def __init__(self, config: SecondStepConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.model = self._load_model()
        self.transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229])
        ])
        # Add flip options with defaults
        self.auto_flip_vertical = True
        self.auto_flip_horizontal = True

    def _load_model(self) -> Optional[torch.nn.Module]:
        """Load the trained model"""
        try:
            print(f"Attempting to load model from {self.config.model_path}")
            
            if not self.config.model_path.exists():
                print(f"Model file not found at {self.config.model_path}")
                return None
            
            # Create model instance
            model = MultiHeadEfficientNet()
            
            # Load state dict with device mapping
            checkpoint = torch.load(self.config.model_path, map_location=self.device)
            
            # Handle both checkpoint dict and state dict formats
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                #if 'epoch' in checkpoint:
                #    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
            else:
                model.load_state_dict(checkpoint)
            
            # Move to device and set to eval mode
            model = model.to(self.device)
            model.eval()
            
            print("Model loaded successfully")
            return model
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return None


    def get_transformed_folder_path(self, folder: str) -> Path:
        """Get path to the transformed folder"""
        base_folder = folder.replace("_card", "")
        return self.config.pred_output_dir / f"{base_folder}_transformed_card"

    def get_original_path(self, folder: str, filename: str) -> Path:
        """Get path to original image"""
        return self.config.pred_output_dir / folder / filename

    def get_transformed_path(self, folder: str, filename: str) -> Path:
        """Get path to transformed image"""
        transformed_folder = self.get_transformed_folder_path(folder)
        return transformed_folder / filename

    def set_flip_options(self, flip_vertical: bool, flip_horizontal: bool):
        """Update flip options for model processing"""
        self.auto_flip_vertical = flip_vertical
        self.auto_flip_horizontal = flip_horizontal
        print(f"Updated flip options - vertical: {flip_vertical}, horizontal: {flip_horizontal}")

    def manual_flip(self, folder: str, filename: str, flip_type: str) -> Optional[Image.Image]:
        """Manually flip an image vertically or horizontally"""
        try:
            # Get paths
            transformed_folder = self.get_transformed_folder_path(folder)
            transformed_path = transformed_folder / filename
            
            # Load image from transformed folder if it exists, otherwise from original
            if transformed_path.exists():
                image = Image.open(transformed_path).convert('L')
            else:
                image_path = self.get_original_path(folder, filename)
                image = Image.open(image_path).convert('L')
            
            # Apply the requested flip
            if flip_type == "vertical":
                transformed = image.transpose(Image.FLIP_TOP_BOTTOM)  # For vertical flip (top-down)
            elif flip_type == "horizontal":
                transformed = image.transpose(Image.FLIP_LEFT_RIGHT)  # For horizontal flip (left-right)
            else:
                print(f"Unknown flip type: {flip_type}")
                return None
            
            # Ensure transformed folder exists
            os.makedirs(transformed_folder, exist_ok=True)
            
            # Save the transformed image
            transformed.save(transformed_path)
            
            return transformed
        
           
        except Exception as e:
            print(f"Error in manual flip: {str(e)}")
            return None
        
    def _update_flip_status(self, folder: str, filename: str, flip_type: str):
        """Update the status in results CSV after a manual flip"""
        try:
            results = self.load_results(folder)
            if results.empty:
                return
            
            mask = results['filename'] == filename
            if not any(mask):
                return
            
            # Update position or rotation based on flip type
            if flip_type == "vertical":
                current_position = results.loc[mask, 'position'].iloc[0]
                new_position = "TOP" if current_position == "BOTTOM" else "BOTTOM"
                results.loc[mask, 'position'] = new_position
            elif flip_type == "horizontal":
                current_rotation = results.loc[mask, 'rotation'].iloc[0]
                new_rotation = "RIGHT" if current_rotation == "LEFT" else "LEFT"
                results.loc[mask, 'rotation'] = new_rotation
            
            # Save updated results
            transformed_folder = self.get_transformed_folder_path(folder)
            save_path = transformed_folder / 'classifications.csv'
            results.to_csv(save_path, index=False)
            
        except Exception as e:
            print(f"Error updating flip status: {str(e)}")

    def process_folder(self, folder: str) -> pd.DataFrame:
        """Process all mask images in the folder"""
        try:
            results = []
            source_folder = self.config.pred_output_dir / folder
            transformed_folder = self.get_transformed_folder_path(folder)
            
            # Create transformed folder
            transformed_folder.mkdir(exist_ok=True)
            
            if not source_folder.exists():
                print(f"Source folder not found: {source_folder}")
                return pd.DataFrame()
            
            image_files = [f for f in os.listdir(source_folder) if f.endswith('.png')]
            print(f"Found {len(image_files)} masks to analyze")
            
            for file in image_files:
                try:
                    image_path = source_folder / file
                    print(f"Processing {file}...")
                    
                    type_pred, pos_pred, rot_pred, transformed_image = self.process_image(str(image_path))
                    
                    if all((type_pred, pos_pred, rot_pred)):
                        # Save transformed image
                        transformed_path = transformed_folder / file
                        if transformed_image:
                            transformed_image.save(transformed_path)
                            print(f"Saved transformed image to {transformed_path}")
                        
                        results.append({
                            'filename': file,
                            'type': type_pred,
                            'position': pos_pred,
                            'rotation': rot_pred
                        })
                        print(f"Successfully processed {file}: Type={type_pred}, Pos={pos_pred}, Rot={rot_pred}")
                    
                except Exception as e:
                    print(f"Error processing individual image {file}: {str(e)}")
                    continue
            
            # Create and save results DataFrame
            results_df = pd.DataFrame(results)
            if not results_df.empty:
                save_path = transformed_folder / 'classifications.csv'
                results_df.to_csv(save_path, index=False)
                print(f"Saved results for {len(results_df)} images")
                
            return results_df
            
        except Exception as e:
            print(f"Error in process_folder: {str(e)}")
            return pd.DataFrame()

    def process_image(self, image_path: str) -> tuple[str, str, str, Image.Image]:
        """Process a single image with respect to flip options"""
        try:
            if self.model is None:
                print("Model not loaded")
                return None, None, None, None
                
            image = Image.open(image_path).convert('L')
            image_tensor = self.transforms(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                pred_type, pred_position, pred_rotation = self.model(image_tensor)
                
            type_label = "ENT" if pred_type.item() < 0.5 else "FRAG"
            position_label = "BOTTOM" if pred_position.item() < 0.5 else "TOP"
            rotation_label = "LEFT" if pred_rotation.item() < 0.5 else "RIGHT"
            
            # Transform image according to enabled flip options and predictions
            transformed = self._transform_image(
                image, 
                position_label if self.auto_flip_vertical else None,
                rotation_label if self.auto_flip_horizontal else None
            )
            
            return type_label, position_label, rotation_label, transformed
            
        except Exception as e:
            print(f"Error processing image {image_path}: {str(e)}")
            return None, None, None, None

    def _transform_image(self, image: Image.Image, position: Optional[str], rotation: Optional[str]) -> Image.Image:
        """Transform image based on position and rotation if enabled"""
        transformed = image.copy()
        
        if position == "BOTTOM":
            transformed = transformed.transpose(Image.FLIP_TOP_BOTTOM)
        if rotation == "LEFT":
            transformed = transformed.transpose(Image.FLIP_LEFT_RIGHT)
            
        return transformed

    def load_results(self, folder: str) -> pd.DataFrame:
        """Load results from the transformed folder"""
        transformed_folder = self.get_transformed_folder_path(folder)
        results_path = transformed_folder / 'classifications.csv'
        
        if results_path.exists():
            return pd.read_csv(results_path)
        return pd.DataFrame()

    def update_result(self, folder: str, filename: str, updates: dict):
        """Update result and transform image if needed"""
        try:
            results = self.load_results(folder)
            if results.empty:
                print("No results found to update")
                return
            
            # Find the row to update
            mask = results['filename'] == filename
            if not any(mask):
                print(f"No entry found for {filename}")
                return
                
            # Update the values
            for key, value in updates.items():
                if key in results.columns:
                    results.loc[mask, key] = value
                    
                    # If position or rotation changed, transform the image
                    if key in ['position', 'rotation']:
                        try:
                            # Load and transform image
                            image_path = self.get_original_path(folder, filename)
                            image = Image.open(image_path).convert('L')
                            
                            row = results[mask].iloc[0]
                            transformed = self._transform_image(
                                image,
                                row['position'],
                                row['rotation']
                            )
                            
                            # Save transformed image
                            transformed_path = self.get_transformed_path(folder, filename)
                            transformed.save(transformed_path)
                            print(f"Saved updated transformed image to {transformed_path}")
                            
                        except Exception as e:
                            print(f"Error updating transformed image: {str(e)}")
            
            # Save updated results
            transformed_folder = self.get_transformed_folder_path(folder)
            save_path = transformed_folder / 'classifications.csv'
            results.to_csv(save_path, index=False)
            print(f"Saved updated results to {save_path}")
            
        except Exception as e:
            print(f"Error updating results: {str(e)}")
            raise e
        

def download_model(url: str = 'https://huggingface.co/lrncrd/PyPotteryLens/resolve/main/BasicModelv8_v01.pt', 
                  dest_path: str = 'models_vision/BasicModelv8_v01.pt') -> bool:
    """Download model file from url to specified path"""
    try:
        import os
        import requests
        from pathlib import Path

        dest_path = Path(dest_path)
        os.makedirs(dest_path.parent, exist_ok=True)
        
        if dest_path.exists():
            print('[✓] Model already exists in models_vision directory')
            return True
            
        print(f'[*] Downloading model from {url}...')
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024
        downloaded = 0
        
        with open(dest_path, 'wb') as f:
            for data in response.iter_content(block_size):
                downloaded += len(data)
                f.write(data)
                if total_size > 0:
                    percent = int((downloaded / total_size) * 100)
                    print(f'\r[*] Download progress: {percent}% ({downloaded}/{total_size} bytes)', end='')
                    
        print('\n[✓] Model downloaded successfully')
        return True
        
    except Exception as e:
        print(f'\n[!] Error downloading model: {str(e)}')
        return False
    

@dataclass
class ExportConfig:
    """Configuration for final export processing"""
    pred_output_dir: Path
    export_pdf: bool = False
    pdf_page_size: str = 'A4'
    scale_factor: float = 1.0

class ExportProcessor:
    """Handles final export processing with PDF export capability"""
    
    def __init__(self, config: ExportConfig):
        self.config = config
        self.pdf_exporter = None  # Will be initialized when needed

    def export_results(self, folder: str, acronym: str, export_pdf: bool = False, 
                        page_size: str = 'A4', scale_factor: float = 1.0) -> str:
            """
            Export processed images and metadata with optional PDF export
            """
            try:
                # Setup paths
                base_folder = folder.split("_card")[0]
                source_folder = self.config.pred_output_dir / f"{base_folder}_transformed_card"
                export_folder = self.config.pred_output_dir / f"{acronym}"
                
                if not source_folder.exists():
                    return f"Transformed folder not found. Please process images first."

                merged_annotations_path = source_folder / "merged_annotations.csv"
                    
                if not merged_annotations_path.exists():
                    return "Merged annotations file not found. Please merge annotations first."

                try:
                    # Load merged_annotations
                    metadata = pd.read_csv(merged_annotations_path)
                    
                    # Create the export folder
                    os.makedirs(export_folder, exist_ok=True)

                    # Create new sequential IDs and track exported image paths
                    image_data = []  # List to store (path, new_id) tuples
                    metadata['new_id'] = [f"{acronym}_{i+1}" for i in range(len(metadata))]
                    metadata.set_index('new_id', inplace=True)
                    
                    # Copy transformed images with new names
                    copied_count = 0
                    for idx, row in metadata.iterrows():
                        try:
                            source_image = source_folder / f"{row['filename']}.png"
                            if source_image.exists():
                                dest_image = export_folder / f"{idx}.png"
                                shutil.copy2(source_image, dest_image)
                                image_data.append((str(dest_image), idx))  # Store path and new_id
                                copied_count += 1
                        except Exception as e:
                            print(f"Error copying image {idx}: {str(e)}")
                            continue

                    # Clean up and save metadata
                    metadata = metadata.drop('filename', axis=1, errors='ignore')
                    metadata.to_csv(export_folder / f"{acronym}_metadata.csv")
                    
                    # Generate PDF if requested
                    pdf_message = ""
                    if export_pdf and image_data:
                        pdf_path = export_folder / f"{acronym}_catalog.pdf"
                        pdf_exporter = PDFExporter(
                            page_size=page_size,
                            scale_factor=scale_factor
                        )
                        if pdf_exporter.generate_pdf(str(pdf_path), image_data):
                            pdf_message = f" PDF catalog generated at {pdf_path}."
                        else:
                            pdf_message = " Warning: PDF generation failed."

                    if copied_count == 0:
                        return "Warning: No images were exported."
                        
                    return (f"Export complete: {copied_count} images exported to {export_folder} "
                        f"with prefix '{acronym}_'.{pdf_message}")
                    
                except Exception as e:
                    print(f"Detailed error: {str(e)}")
                    return f"Error processing data: {str(e)}"
                
            except Exception as e:
                print(f"Export error: {str(e)}")
                return f"Error during export: {str(e)}"
        


class PDFExporter:
    """Handles PDF generation with optimized image arrangement"""
    
    PAGE_SIZES = {
        'A4': pagesizes.A4,
        'A3': pagesizes.A3,
        'A5': pagesizes.A5,
        'LETTER': pagesizes.LETTER,
        'LEGAL': pagesizes.LEGAL
    }
    
    def __init__(self, page_size: str = 'A4', margin: int = 50, scale_factor: float = 1.0):
        self.page_size = self.PAGE_SIZES.get(page_size, pagesizes.A4)
        self.margin = margin
        self.scale_factor = scale_factor
        self.width = self.page_size[0] - (2 * margin)
        self.height = self.page_size[1] - (2 * margin)

    def generate_pdf(self, output_path: str, image_data: List[tuple]) -> bool:
        """Generate PDF with optimized image layout and labels"""
        try:
            # Create PDF canvas
            c = canvas.Canvas(output_path, pagesize=self.page_size)
            
            # Set up font for labels
            c.setFont("Helvetica", 10)
            
            # Get optimized layout
            optimizer = LayoutOptimizer(
                page_width=self.page_size[0],
                page_height=self.page_size[1],
                margin=self.margin
            )
            
            pages = optimizer.pack_images(image_data, self.scale_factor)
            
            # Generate each page
            for page in pages:
                for img_info in page:
                    # Draw the image
                    c.drawImage(
                        img_info['path'],
                        img_info['x'],
                        img_info['y'],
                        width=img_info['width'],
                        height=img_info['height'],
                        preserveAspectRatio=True
                    )
                    
                    # Center and draw the label
                    c.setFillColorRGB(0, 0, 0)
                    
                    # Get the width of the text to center it
                    text_width = c.stringWidth(img_info['new_id'], "Helvetica", 10)
                    
                    # Calculate center position
                    center_x = img_info['x'] + (img_info['width'] / 2) - (text_width / 2)
                    
                    # Draw centered text
                    c.drawString(
                        center_x,
                        img_info['label_y'],
                        img_info['new_id']
                    )
                
                c.showPage()
            
            c.save()
            return True
            
        except Exception as e:
            print(f"Error generating PDF: {str(e)}")
            return False

        

class LayoutOptimizer:
    """Optimizes image layout for PDF pages"""
    
    def __init__(self, page_width: float, page_height: float, margin: float = 50):
        self.page_width = page_width - (2 * margin)
        self.page_height = page_height - (2 * margin)
        self.margin = margin
        # Add spacing between images
        self.horizontal_spacing = 20  # Space between images horizontally
        self.vertical_spacing = 40    # Space between images vertically (includes room for label)
        self.label_height = 20        # Height reserved for the label

    def optimize_row_layout(self, images: List[Dict], start_y: float) -> Tuple[List[Dict], float]:
        """Optimize layout for a single row of images with spacing"""
        if not images:
            return [], 0
            
        # Calculate total width including spacing between images
        total_width = sum(img['width'] for img in images) + (len(images) - 1) * self.horizontal_spacing
        total_height = max(img['height'] for img in images) + self.label_height
        
        # Scale factor to fit row width if needed
        scale = min(1.0, self.page_width / total_width)
        
        # Position images in row
        x = self.margin
        layout = []
        
        for img in images:
            scaled_width = img['width'] * scale
            scaled_height = img['height'] * scale
            
            layout.append({
                'path': img['path'],
                'new_id': img['new_id'],  # Add new_id to layout
                'x': x,
                'y': start_y,
                'width': scaled_width,
                'height': scaled_height,
                'label_y': start_y + scaled_height + 5  # Position label below image
            })
            
            x += scaled_width + self.horizontal_spacing
            
        return layout, total_height + self.vertical_spacing

    def pack_images(self, images: List[tuple], scale_factor: float = 1.0) -> List[List[Dict]]:
        """Pack images into pages using an optimized layout"""
        # Get image dimensions and apply scale factor
        image_info = []
        for img_path, new_id in images:  # Now expecting tuples of (path, new_id)
            with Image.open(img_path) as img:
                w, h = img.size
                image_info.append({
                    'path': img_path,
                    'new_id': new_id,  # Store new_id
                    'width': w * scale_factor,
                    'height': h * scale_factor,
                    'aspect_ratio': w / h
                })
        
        # Sort images by height for better packing
        image_info.sort(key=lambda x: x['height'], reverse=True)
        
        pages = []
        current_page = []
        y_position = self.margin
        current_row = []
        row_width = 0
        
        for img in image_info:
            # Check if adding this image exceeds page width
            if row_width + img['width'] + self.horizontal_spacing > self.page_width:
                # Layout current row
                row_layout, row_height = self.optimize_row_layout(current_row, y_position)
                current_page.extend(row_layout)
                
                # Move to next row
                y_position += row_height + self.vertical_spacing
                
                # Check if we need a new page
                if y_position + img['height'] + self.label_height > self.page_height:
                    pages.append(current_page)
                    current_page = []
                    y_position = self.margin
                
                # Start new row with current image
                current_row = [img]
                row_width = img['width']
            else:
                # Add image to current row
                current_row.append(img)
                row_width += img['width'] + self.horizontal_spacing
        
        # Handle remaining images
        if current_row:
            row_layout, _ = self.optimize_row_layout(current_row, y_position)
            current_page.extend(row_layout)
        
        if current_page:
            pages.append(current_page)
        
        return pages