# Standard library imports
from pathlib import Path
import time

# Third-party imports
import numpy as np
import pandas as pd
from PIL import Image
import timm
from tqdm import tqdm

# PyTorch imports
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

# Sklearn imports
from sklearn.model_selection import train_test_split


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
from pathlib import Path
from datetime import datetime

class TrainingVisualizer:
    def __init__(self, save_dir: str = 'training_plots'):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        
        # Set style
        #plt.style.use('seaborn')
        self.task_names = ['Type', 'Position', 'Rotation']
        self.class_names = [
            ['ENT', 'FRAG'],
            ['BOTTOM', 'TOP'],
            ['LEFT', 'RIGHT']
        ]
    
    def plot_training_history(self, history):
        """Plot training and validation metrics with correct epoch numbering"""
        fig = plt.figure(figsize=(15, 10))
        
        # Get number of epochs
        num_epochs = len(history['train_loss'])
        epochs = list(range(1, num_epochs + 1))  # Start from 1
        
        # Plot Loss
        plt.subplot(2, 2, 1)
        plt.plot(epochs, history['train_loss'], label='Train Loss')
        plt.plot(epochs, history['val_loss'], label='Val Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.5)

        ### make y ticks integers
        #plt.yticks(np.arange(0, num_epochs + 1, step=1))
        
        # Plot Learning Rate
        plt.subplot(2, 2, 2)
        steps = range(len(history['learning_rates']))
        plt.plot(steps, history['learning_rates'])
        plt.title('Learning Rate Schedule')
        plt.xlabel('Step')
        plt.ylabel('Learning Rate')
        plt.grid(True, alpha=0.5)

        #plt.yticks(np.arange(0, num_epochs + 1, step=1))
        
        # Plot Training Accuracy
        plt.subplot(2, 2, 3)
        for i, task in enumerate(self.task_names):
            acc = [epoch_acc[i] for epoch_acc in history['train_acc']]
            plt.plot(epochs, acc, label=task)
        plt.title('Training Accuracy by Task')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.5)

        #plt.yticks(np.arange(0, num_epochs + 1, step=1))
        
        # Plot Validation Accuracy
        plt.subplot(2, 2, 4)
        for i, task in enumerate(self.task_names):
            acc = [epoch_acc[i] for epoch_acc in history['val_acc']]
            plt.plot(epochs, acc, label=task)
        plt.title('Validation Accuracy by Task')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.5)

        #plt.yticks(np.arange(0, num_epochs + 1, step=1))
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_confusion_matrices(self, model, test_loader, device):
        """Plot confusion matrices for all tasks"""
        model.eval()
        predictions = {task: [] for task in self.task_names}
        targets = {task: [] for task in self.task_names}
        
        with torch.no_grad():
            for images, target_tuple in test_loader:
                images = images.to(device)
                outputs = model(images)
                
                # Get predictions
                for i, (output, target) in enumerate(zip(outputs, target_tuple)):
                    task = self.task_names[i]
                    pred = (torch.sigmoid(output) > 0.5).cpu().numpy()
                    predictions[task].extend(pred.flatten())
                    targets[task].extend(target.cpu().numpy().flatten())
        
        # Plot confusion matrices
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for i, task in enumerate(self.task_names):
            cm = confusion_matrix(targets[task], predictions[task])
            sns.heatmap(cm, annot=True, fmt='d', ax=axes[i],
                       xticklabels=self.class_names[i],
                       yticklabels=self.class_names[i],
                       cmap='Blues')
            axes[i].set_title(f'{task} Confusion Matrix')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Generate and save classification reports
        reports = {}
        for i, task in enumerate(self.task_names):
            report = classification_report(
                targets[task], 
                predictions[task],
                target_names=self.class_names[i],
                output_dict=True
            )
            reports[task] = report
        
        return reports
    
    def plot_prediction_distribution(self, model, test_loader, device):
        """Plot distribution of prediction probabilities"""
        model.eval()
        probabilities = {task: [] for task in self.task_names}
        correct_probs = {task: [] for task in self.task_names}
        incorrect_probs = {task: [] for task in self.task_names}
        
        with torch.no_grad():
            for images, target_tuple in test_loader:
                images = images.to(device)
                outputs = model(images)
                
                for i, (output, target) in enumerate(zip(outputs, target_tuple)):
                    task = self.task_names[i]
                    probs = torch.sigmoid(output).cpu().numpy().flatten()
                    targets = target.cpu().numpy().flatten()
                    preds = (probs > 0.5).astype(int)
                    
                    probabilities[task].extend(probs)
                    
                    # Separate correct and incorrect predictions
                    correct_mask = preds == targets
                    correct_probs[task].extend(probs[correct_mask])
                    incorrect_probs[task].extend(probs[~correct_mask])
        
        # Plot probability distributions
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for i, task in enumerate(self.task_names):
            # Plot correct predictions
            if correct_probs[task]:
                sns.kdeplot(data=correct_probs[task], ax=axes[i], 
                           color='green', label='Correct')
            
            # Plot incorrect predictions
            if incorrect_probs[task]:
                sns.kdeplot(data=incorrect_probs[task], ax=axes[i], 
                           color='red', label='Incorrect')
            
            axes[i].set_title(f'{task} Prediction Distribution')
            axes[i].set_xlabel('Prediction Probability')
            axes[i].set_ylabel('Density')
            axes[i].axvline(x=0.5, color='black', linestyle='--', alpha=0.5)
            axes[i].legend()
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'prediction_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()

# Rest of your code remains the same starting with PotteryDataset class
class PotteryDataset(Dataset):
    def __init__(self, image_paths, type_labels, position_labels, rotation_labels):
        self.image_paths = image_paths
        self.type_labels = type_labels
        self.position_labels = position_labels
        self.rotation_labels = rotation_labels
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229])
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load and transform image
        image = Image.open(self.image_paths[idx]).convert('L')
        image = self.transform(image)
        
        # Convert labels to tensors and reshape to match model output
        type_label = torch.tensor(self.type_labels[idx], dtype=torch.float32).unsqueeze(0)
        position_label = torch.tensor(self.position_labels[idx], dtype=torch.float32).unsqueeze(0)
        rotation_label = torch.tensor(self.rotation_labels[idx], dtype=torch.float32).unsqueeze(0)
        
        return image, (type_label, position_label, rotation_label)

def prepare_data(data_dir, metadata_file, test_size=0.2, val_size=0.2):
    """Prepare train, validation, and test datasets"""
    # Read metadata
    df = pd.read_csv(metadata_file)
    
    # Create label mappings
    type_map = {'ENT': 0, 'FRAG': 1}
    position_map = {'BOTTOM': 0, 'TOP': 1}
    rotation_map = {'LEFT': 0, 'RIGHT': 1}
    
    # Convert labels
    df['type_encoded'] = df['type'].map(type_map)
    df['position_encoded'] = df['position'].map(position_map)
    df['rotation_encoded'] = df['rotation'].map(rotation_map)
    
    # Create full image paths
    data_dir = Path(data_dir)
    df['full_path'] = df['filename'].apply(lambda x: str(data_dir / x))
    
    # Split data
    train_val_df, test_df = train_test_split(
        df, test_size=test_size, stratify=df['type'], random_state=42
    )
    
    train_df, val_df = train_test_split(
        train_val_df, test_size=val_size, stratify=train_val_df['type'], random_state=42
    )
    
    # Create datasets
    train_dataset = PotteryDataset(
        train_df['full_path'].values,
        train_df['type_encoded'].values,
        train_df['position_encoded'].values,
        train_df['rotation_encoded'].values
    )
    
    val_dataset = PotteryDataset(
        val_df['full_path'].values,
        val_df['type_encoded'].values,
        val_df['position_encoded'].values,
        val_df['rotation_encoded'].values
    )
    
    test_dataset = PotteryDataset(
        test_df['full_path'].values,
        test_df['type_encoded'].values,
        test_df['position_encoded'].values,
        test_df['rotation_encoded'].values
    )
    
    return train_dataset, val_dataset, test_dataset



class MultiHeadEfficientNet(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super().__init__()
        
        self.backbone = timm.create_model(
            'efficientnetv2_rw_s',
            pretrained=True,
            in_chans=1,
            num_classes=0
        )
        
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 224, 224)
            backbone_features = self.backbone(dummy).shape[1]
        
        self.head1 = self._create_head(backbone_features, dropout_rate)
        self.head2 = self._create_head(backbone_features, dropout_rate)
        self.head3 = self._create_head(backbone_features, dropout_rate)

    def _create_head(self, in_features, dropout_rate):
        return nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(dropout_rate/2),
            nn.Linear(256, 1)  # Remove Sigmoid here
        )

    def forward(self, x):
        features = self.backbone(x)
        return (
            self.head1(features),
            self.head2(features),
            self.head3(features)
        )

def train_model(model, train_loader, val_loader, num_epochs=5, device='cuda', save_dir=None):
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    
    steps_per_epoch = len(train_loader)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=1e-3,
        epochs=num_epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.3,
        div_factor=25,
        final_div_factor=1000,
    )
    
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler()
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'learning_rates': []
    }
    
    # Create checkpoint path
    if save_dir:
        checkpoint_path = save_dir / 'best_model.pth'
    else:
        checkpoint_path = Path('best_model.pth')
    
    print(f"\nStarting training for {num_epochs} epochs...")
    print(f"Training batches per epoch: {len(train_loader)}")
    print(f"Validation batches per epoch: {len(val_loader)}")
    print(f"Checkpoints will be saved to: {checkpoint_path}\n")
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        model.train()
        train_loss = 0
        train_correct = [0, 0, 0]
        train_total = 0
        
        # Create progress bar
        pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                   desc=f'Epoch {epoch+1}/{num_epochs} [Train]',
                   bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}',
                   ncols=100)
        
        for batch_idx, (images, targets) in pbar:
            images = images.to(device)
            targets = tuple(t.to(device) for t in targets)
            
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = sum(criterion(output, target) 
                          for output, target in zip(outputs, targets))
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            train_loss += loss.item()
            train_total += images.size(0)
            
            # Calculate accuracies
            for i, (output, target) in enumerate(zip(outputs, targets)):
                preds = (torch.sigmoid(output) > 0.5).float()
                train_correct[i] += (preds == target).sum().item()
            
            # Update progress bar
            current_lr = scheduler.get_last_lr()[0]
            avg_loss = train_loss / (batch_idx + 1)
            accuracies = [correct / train_total for correct in train_correct]
            
            pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'lr': f'{current_lr:.2e}',
                'type_acc': f'{accuracies[0]:.4f}',
                'pos_acc': f'{accuracies[1]:.4f}',
                'rot_acc': f'{accuracies[2]:.4f}'
            })
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = [0, 0, 0]
        val_total = 0
        
        # Validation progress bar
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]',
                       bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}',
                       ncols=100)
        
        with torch.no_grad():
            for images, targets in val_pbar:
                images = images.to(device)
                targets = tuple(t.to(device) for t in targets)
                
                outputs = model(images)
                loss = sum(criterion(output, target) 
                          for output, target in zip(outputs, targets))
                
                val_loss += loss.item()
                val_total += images.size(0)
                
                for i, (output, target) in enumerate(zip(outputs, targets)):
                    preds = (torch.sigmoid(output) > 0.5).float()
                    val_correct[i] += (preds == target).sum().item()
                
                # Update validation progress bar
                avg_val_loss = val_loss / val_total
                val_accuracies = [correct / val_total for correct in val_correct]
                
                val_pbar.set_postfix({
                    'loss': f'{avg_val_loss:.4f}',
                    'type_acc': f'{val_accuracies[0]:.4f}',
                    'pos_acc': f'{val_accuracies[1]:.4f}',
                    'rot_acc': f'{val_accuracies[2]:.4f}'
                })
        
        # Calculate epoch metrics
        train_loss = train_loss / len(train_loader)
        val_loss = val_loss / len(val_loader)
        train_acc = [correct / train_total for correct in train_correct]
        val_acc = [correct / val_total for correct in val_correct]
        epoch_time = time.time() - epoch_start_time
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['learning_rates'].append(scheduler.get_last_lr()[0])
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{num_epochs} Summary ({epoch_time:.1f}s):")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        for i, task in enumerate(['Type', 'Position', 'Rotation']):
            print(f"{task:>8} - Train Acc: {train_acc[i]:.4f} | Val Acc: {val_acc[i]:.4f}")
        
        # Save checkpoint if best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'history': history,
            }, checkpoint_path)
            print(f"\nNew best model saved! (Val Loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"\nNo improvement for {patience_counter} epochs.")
        
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
        
        print("-" * 80 + "\n")
    
    return model, history

def main():
    # Set device and enable deterministic training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"Using device: {device}")
    
    # Prepare datasets
    train_dataset, val_dataset, test_dataset = prepare_data(
        data_dir=r'PATH_TO_IMAGES',
        metadata_file=r'PATH_TO_METADATA'
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize model
    model = MultiHeadEfficientNet().to(device)
    
    # Initialize visualizer
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    visualizer = TrainingVisualizer(save_dir=f'training_results_{timestamp}')

    # Set training parameters
    num_epochs = 10
    
    # Train model
    print("\nStarting training...")
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        device=device
    )
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'history': history,
        'final': True
    }, f'training_results_{timestamp}/final_model.pth')
    
    print("\nGenerating diagnostic plots...")
    
    # Plot training history
    visualizer.plot_training_history(history)
    print("Training history plots saved")
    
    # Generate confusion matrices and get classification reports
    print("Generating confusion matrices...")
    reports = visualizer.plot_confusion_matrices(model, test_loader, device)
    print("Confusion matrices saved")
    
    # Plot prediction distributions
    print("Generating prediction distribution plots...")
    visualizer.plot_prediction_distribution(model, test_loader, device)
    print("Prediction distributions saved")
    
    # Save classification reports
    output_file = Path(f'training_results_{timestamp}') / 'classification_reports.txt'
    
    print("\nSaving classification reports...")
    with open(output_file, 'w') as f:
        f.write(f"Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        for task, report in reports.items():
            f.write(f"\n{task} Classification Report:\n")
            f.write("-" * 50 + "\n")
            for class_name, metrics in report.items():
                if isinstance(metrics, dict):
                    f.write(f"{class_name}:\n")
                    for metric_name, value in metrics.items():
                        f.write(f"  {metric_name}: {value:.4f}\n")
            f.write("\n")
    
    print(f"\nTraining completed! All results saved in 'training_results_{timestamp}' directory")
    print("\nResults include:")
    print("- Training history plots")
    print("- Confusion matrices")
    print("- Prediction distribution plots")
    print("- Classification reports")
    print("- Final model weights")

if __name__ == "__main__":
    main()