import os
import numpy as np
from pathlib import Path
import shutil
from PIL import Image
import cv2
from typing import Tuple, List, Dict
import yaml
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import random

class MaskToYOLOSegConverter:
    """Converts PyPotteryLens mask outputs to YOLO segmentation format"""
    
    def __init__(self, config: Dict):
        self.masks_dir = Path(config['masks_dir'])
        self.output_dir = Path(config['output_dir'])
        self.original_images_dir = Path(config['original_images_dir'])
        self.class_name = config.get('class_name', 'pottery')
        self.train_ratio = config.get('train_ratio', 0.8)
        self.diagnostic_dir = Path(config['output_dir']) / 'diagnostics'
        self.min_points = config.get('min_points', 8)  # Minimum points for polygon approximation
        self.epsilon_factor = config.get('epsilon_factor', 0.005)  # Controls polygon approximation accuracy
        
        # Create YOLO dataset structure
        self.dataset_dir = self.output_dir / 'dataset'
        self.images_dir = self.dataset_dir / 'images'
        self.labels_dir = self.dataset_dir / 'labels'
        
        # Create directories
        for split in ['train', 'val']:
            (self.images_dir / split).mkdir(parents=True, exist_ok=True)
            (self.labels_dir / split).mkdir(parents=True, exist_ok=True)
        
        self.diagnostic_dir.mkdir(parents=True, exist_ok=True)

    def process_mask(self, mask_path: Path) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Process a single mask and extract YOLO segmentation format polygons
        Returns the mask and list of normalized polygon coordinates
        """
        # Read mask
        mask = np.array(Image.open(mask_path).convert('L'))
        height, width = mask.shape
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Process each contour into a simplified polygon
        yolo_polygons = []
        for contour in contours:
            # Calculate epsilon for polygon approximation based on contour perimeter
            epsilon = self.epsilon_factor * cv2.arcLength(contour, True)
            # Approximate polygon
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Only include if we have enough points
            if len(approx) >= self.min_points:
                # Normalize coordinates
                polygon = approx.squeeze().astype(float)
                polygon[:, 0] /= width  # normalize x
                polygon[:, 1] /= height  # normalize y
                yolo_polygons.append(polygon)
        
        return mask, yolo_polygons

    def create_diagnostic_plots(self, original_path: Path, mask_path: Path, 
                              polygons: List[np.ndarray], output_path: Path) -> None:
        """Create diagnostic visualization comparing original, mask, and YOLO polygons"""
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot original image
        original_img = np.array(Image.open(original_path))
        height, width = original_img.shape[:2]
        ax1.imshow(original_img)
        ax1.set_title('Original Image')
        
        # Plot mask
        mask = np.array(Image.open(mask_path).convert('L'))
        ax2.imshow(mask, cmap='gray')
        ax2.set_title('Mask')
        
        # Plot original with YOLO polygons
        ax3.imshow(original_img)
        
        for polygon in polygons:
            # Denormalize coordinates for plotting
            plot_poly = polygon.copy()
            plot_poly[:, 0] *= width
            plot_poly[:, 1] *= height
            
            poly = Polygon(plot_poly, fill=False, edgecolor='r', linewidth=2)
            ax3.add_patch(poly)
        
        ax3.set_title('YOLO Polygons')
        
        # Remove axes for cleaner look
        for ax in [ax1, ax2, ax3]:
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    def create_summary_plots(self, split_data: Dict[str, List[Path]]) -> None:
        """Create summary statistics and visualizations"""
        fig = plt.figure(figsize=(15, 10))
        
        # 1. Distribution of number of polygons per image
        polygons_per_image = {}
        total_polygons = 0
        points_per_polygon = []
        
        for split, paths in split_data.items():
            polygons_per_image[split] = []
            for mask_path in paths:
                _, polygons = self.process_mask(mask_path)
                polygons_per_image[split].append(len(polygons))
                total_polygons += len(polygons)
                
                # Collect points per polygon
                for polygon in polygons:
                    points_per_polygon.append(len(polygon))
        
        # Plot distributions
        plt.subplot(2, 2, 1)
        plt.hist(polygons_per_image['train'], alpha=0.5, label='Train', bins=20)
        plt.hist(polygons_per_image['val'], alpha=0.5, label='Val', bins=20)
        plt.xlabel('Polygons per Image')
        plt.ylabel('Count')
        plt.title('Distribution of Polygons per Image')
        plt.legend()
        
        # 2. Points per polygon distribution
        plt.subplot(2, 2, 2)
        plt.hist(points_per_polygon, bins=30)
        plt.xlabel('Points per Polygon')
        plt.ylabel('Count')
        plt.title('Distribution of Polygon Complexity')
        
        # 3. Summary statistics
        plt.subplot(2, 2, 3)
        plt.axis('off')
        summary_text = (
            f'Dataset Summary:\n\n'
            f'Total Images: {len(split_data["train"]) + len(split_data["val"])}\n'
            f'Training Images: {len(split_data["train"])}\n'
            f'Validation Images: {len(split_data["val"])}\n'
            f'Total Polygons: {total_polygons}\n'
            f'Avg Polygons per Image: {total_polygons/(len(split_data["train"]) + len(split_data["val"])):.2f}\n'
            f'Avg Points per Polygon: {np.mean(points_per_polygon):.2f}\n'
            f'Min Points per Polygon: {min(points_per_polygon)}\n'
            f'Max Points per Polygon: {max(points_per_polygon)}'
        )
        plt.text(0.1, 0.9, summary_text, fontsize=10, va='top')
        
        # Save summary plot
        plt.tight_layout()
        plt.savefig(self.diagnostic_dir / 'dataset_summary.png')
        plt.close()

    def create_yolo_segmentation(self, polygons: List[np.ndarray], output_path: Path) -> None:
        """Create YOLO format segmentation file"""
        with open(output_path, 'w') as f:
            for polygon in polygons:
                # Class index is 0 (single class)
                # YOLO seg format: class_idx x1 y1 x2 y2 ... xn yn
                coords_str = ' '.join(f'{coord:.6f}' for coord in polygon.ravel())
                line = f"0 {coords_str}\n"
                f.write(line)

    def create_dataset_yaml(self) -> None:
        """Create YAML file for YOLO dataset configuration"""
        yaml_content = {
            'path': str(self.dataset_dir.absolute()),
            'train': str(self.images_dir / 'train'),
            'val': str(self.images_dir / 'val'),
            'names': {0: self.class_name},
            'task': 'segment'  # Specify segmentation task
        }
        
        with open(self.dataset_dir / 'dataset.yaml', 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=False)

    def convert(self) -> None:
        """Convert all masks to YOLO segmentation format with diagnostic visualizations"""
        # Get all mask files
        mask_files = list(self.masks_dir.glob('*_mask_layer.png'))
        
        if not mask_files:
            raise ValueError(f"No mask files found in {self.masks_dir}")
            
        # Split into train/val
        train_masks, val_masks = train_test_split(
            mask_files, 
            train_size=self.train_ratio,
            random_state=42
        )
        
        split_data = {
            'train': train_masks,
            'val': val_masks
        }
        
        # Create summary plots
        self.create_summary_plots(split_data)
        
        # Process splits
        for split_name, split_masks in split_data.items():
            # Randomly select some images for diagnostic visualization
            diagnostic_samples = random.sample(
                split_masks, 
                min(5, len(split_masks))
            )
            
            for mask_path in split_masks:
                try:
                    # Get original image name
                    original_name = mask_path.stem.replace('_mask_layer', '')
                    original_path = self.original_images_dir / f"{original_name}.jpg"
                    
                    if not original_path.exists():
                        print(f"Warning: Original image not found for {mask_path}")
                        continue
                    
                    # Process mask
                    mask, yolo_polygons = self.process_mask(mask_path)
                    
                    if not yolo_polygons:
                        print(f"Warning: No valid polygons found in {mask_path}")
                        continue
                    
                    # Create diagnostic visualization for sampled images
                    if mask_path in diagnostic_samples:
                        self.create_diagnostic_plots(
                            original_path,
                            mask_path,
                            yolo_polygons,
                            self.diagnostic_dir / f"{original_name}_diagnostic.png"
                        )
                    
                    # Copy original image
                    shutil.copy2(
                        original_path,
                        self.images_dir / split_name / f"{original_name}.jpg"
                    )
                    
                    # Create YOLO segmentation annotation
                    self.create_yolo_segmentation(
                        yolo_polygons,
                        self.labels_dir / split_name / f"{original_name}.txt"
                    )
                    
                except Exception as e:
                    print(f"Error processing {mask_path}: {str(e)}")
                    continue
        
        # Create dataset.yaml
        self.create_dataset_yaml()
        
        print(f"""
        Dataset created successfully:
        - Total masks processed: {len(mask_files)}
        - Training images: {len(train_masks)}
        - Validation images: {len(val_masks)}
        - Dataset configuration: {self.dataset_dir}/dataset.yaml
        - Diagnostic visualizations: {self.diagnostic_dir}
        """)

if __name__ == "__main__":
    # Configuration
    config = {
        'masks_dir': 'path_to_masks',  # Directory containing *_mask_layer.png files
        'original_images_dir': 'path_to_original_images',  # Directory containing original images (PDF2images output)
        'output_dir': 'output_dir',  # Where to create the YOLO dataset
        'class_name': 'Pot',  # Class name for YOLO
        'train_ratio': 0.8,  # Train/val split ratio
        'min_points': 8,  # Minimum points for polygon approximation
        'epsilon_factor': 0.005  # Controls polygon approximation accuracy
    }
    
    # Create and run converter
    converter = MaskToYOLOSegConverter(config)
    ### create output directory with dataset name
    converter.output_dir.mkdir(parents=True, exist_ok=True)

    converter.convert()

    
