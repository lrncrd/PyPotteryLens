import os
from pathlib import Path
import yaml
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import shutil
from typing import Dict, Optional

class YOLOFineTuner:
    """Handles YOLO model fine-tuning with experiment tracking"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.experiment_dir = Path(config['experiment_dir'])
        self.model_name = config['model_name']
        self.dataset_yaml = Path(config['dataset_yaml'])
        
        # Training parameters
        self.epochs = config.get('epochs', 100)
        self.batch_size = config.get('batch_size', 8)
        self.image_size = config.get('image_size', 640)
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
        # Learning rate parameters
        self.initial_lr = config.get('initial_lr', 0.001)
        self.final_lr = config.get('final_lr', 0.0001)
        self.warmup_epochs = config.get('warmup_epochs', 3)
        
        # Create experiment directory structure
        self.setup_experiment_dir()
        
        # Save configuration
        self.save_config()

    def setup_experiment_dir(self):
        """Setup experiment directory structure"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = f"{self.model_name}_{timestamp}"
        self.experiment_path = self.experiment_dir / self.experiment_name
        
        # Create directories
        for subdir in ['models', 'plots', 'logs']:
            (self.experiment_path / subdir).mkdir(parents=True, exist_ok=True)
            
        # Copy dataset YAML to experiment directory
        if self.dataset_yaml.exists():
            shutil.copy2(self.dataset_yaml, self.experiment_path)

    def save_config(self):
        """Save experiment configuration"""
        config_path = self.experiment_path / 'config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)

    def create_callbacks(self):
        """Create training callbacks"""
        class TrainingCallback:
            def __init__(self, trainer):
                self.trainer = trainer

            def on_train_start(self):
                """Called when training starts"""
                print(f"\nStarting training experiment: {trainer.experiment_name}")
                print(f"Device: {trainer.device}")
                
            def on_train_epoch_end(self):
                """Called at end of each training epoch"""
                # Access metrics through trainer.trainer (YOLO trainer instance)
                metrics = self.trainer.trainer.metrics
                print(f"\nEpoch {self.trainer.trainer.epoch}:")
                print(f"mAP50: {metrics['metrics/mAP50(B)']:.4f}")
                print(f"mAP50-95: {metrics['metrics/mAP50-95(B)']:.4f}")
                print(f"Precision: {metrics['metrics/precision(B)']:.4f}")
                print(f"Recall: {metrics['metrics/recall(B)']:.4f}")
                
            def on_train_end(self):
                """Called when training ends"""
                print("\nTraining completed!")
                # Save final plots
                if hasattr(self.trainer.trainer, 'plot_metrics'):
                    self.trainer.trainer.plot_metrics()
                
        return TrainingCallback(self)

    def plot_training_history(self, metrics: Dict):
        """Plot training history"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot mAP
        ax1.plot(metrics['metrics/mAP50(B)'], label='mAP50')
        ax1.plot(metrics['metrics/mAP50-95(B)'], label='mAP50-95')
        ax1.set_title('mAP Progress')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('mAP')
        ax1.legend()
        
        # Plot loss
        ax2.plot(metrics['train/box_loss'], label='Box Loss')
        ax2.plot(metrics['train/seg_loss'], label='Seg Loss')
        ax2.plot(metrics['train/cls_loss'], label='Class Loss')
        ax2.set_title('Training Losses')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        
        # Plot precision/recall
        ax3.plot(metrics['metrics/precision(B)'], label='Precision')
        ax3.plot(metrics['metrics/recall(B)'], label='Recall')
        ax3.set_title('Precision & Recall')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Value')
        ax3.legend()
        
        # Plot learning rate
        if 'lr/pg0' in metrics:
            ax4.plot(metrics['lr/pg0'], label='Learning Rate')
            ax4.set_title('Learning Rate Schedule')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Learning Rate')
            ax4.legend()
        
        plt.tight_layout()
        plt.savefig(self.experiment_path / 'plots' / 'training_history.png')
        plt.close()

    def fine_tune(self):
        """Execute fine-tuning process"""
        try:
            # Load model
            if self.model_name.endswith('.pt'):
                # Load custom pretrained model
                model = YOLO(self.model_name)
            else:
                # Load from YOLO pretrained models
                model = YOLO(f"yolov8{self.model_name}-seg.pt")
            
            # Training arguments
            args = {
                'data': str(self.dataset_yaml),
                'epochs': self.epochs,
                'imgsz': self.image_size,
                'batch': self.batch_size,
                'device': self.device,
                'project': str(self.experiment_path / 'models'),
                'name': 'train',
                'exist_ok': True,
                'pretrained': True,
                'optimizer': 'AdamW',
                'lr0': self.initial_lr,
                'lrf': self.final_lr,
                'warmup_epochs': self.warmup_epochs,
                'val': True,
                'save': True,
                'save_period': -1,  # Save only best and last
                'patience': 50,  # Early stopping patience
                'plots': True
            }
            
            # Create callback
            callback = self.create_callbacks()
            
            # Train
            results = model.train(**args, callbacks=[callback])
            
            # Plot and save training history
            if hasattr(results, 'keys'):
                self.plot_training_history(results)
            
            # Save final model
            final_model_path = self.experiment_path / 'models' / 'train' / 'weights' / 'best.pt'
            if final_model_path.exists():
                shutil.copy2(
                    final_model_path,
                    self.experiment_path / 'models' / f'{self.experiment_name}_best.pt'
                )
            
            return True, "Training completed successfully!"
            
        except Exception as e:
            return False, f"Error during training: {str(e)}"

    def validate_model(self, model_path: Optional[Path] = None):
        """Validate the trained model"""
        try:
            if model_path is None:
                model_path = self.experiment_path / 'models' / f'{self.experiment_name}_best.pt'
            
            if not model_path.exists():
                return False, "Model file not found"
            
            # Load model
            model = YOLO(str(model_path))
            
            # Validate
            metrics = model.val(
                data=str(self.dataset_yaml),
                imgsz=self.image_size,
                batch=self.batch_size,
                device=self.device,
                plots=True
            )
            
            # Save validation results
            validation_results = {
                'mAP50': metrics.box.map50,
                'mAP50-95': metrics.box.map,
                'precision': metrics.box.precision,
                'recall': metrics.box.recall
            }
            
            results_path = self.experiment_path / 'logs' / 'validation_results.yaml'
            with open(results_path, 'w') as f:
                yaml.dump(validation_results, f, default_flow_style=False)
            
            return True, validation_results
            
        except Exception as e:
            return False, f"Error during validation: {str(e)}"

if __name__ == "__main__":
    # Example configuration
    config = {
        'experiment_dir': 'yolo_experiments',  # Directory to store experiments
        'model_name': 'n',  # 'n', 's', 'm', 'l', 'x' or path to .pt file
        'dataset_yaml': 'path/to/dataset.yaml',  # Path to dataset YAML
        'epochs': 100,
        'batch_size': 8,
        'image_size': 640,
        'initial_lr': 0.001,
        'final_lr': 0.0001,
        'warmup_epochs': 3
    }
    
    # Create and run fine-tuning
    trainer = YOLOFineTuner(config)
    success, message = trainer.fine_tune()
    
    if success:
        print("\nStarting validation...")
        success, val_results = trainer.validate_model()
        if success:
            print("\nValidation Results:")
            for metric, value in val_results.items():
                print(f"{metric}: {value:.4f}")
        else:
            print(f"\nValidation failed: {val_results}")
    else:
        print(f"\nTraining failed: {message}")
