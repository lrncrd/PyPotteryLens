import timm
import torch
import torch.nn as nn
import torchvision.models as models
from typing import Tuple



class MultiHeadEfficientNet(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super().__init__()
        
        # Load pretrained backbone
        self.backbone = timm.create_model(
            'efficientnetv2_rw_s',
            pretrained=False, #True
            in_chans=1,
            num_classes=0
        )
        
        # Get backbone output features
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 224, 224)
            backbone_features = self.backbone(dummy).shape[1]
        
        # Create classification heads without the extra BatchNorm
        self.head1 = self._create_head(backbone_features, dropout_rate)  # Type
        self.head2 = self._create_head(backbone_features, dropout_rate)  # Position
        self.head3 = self._create_head(backbone_features, dropout_rate)  # Rotation

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
            nn.Linear(256, 1)  # Removed the extra BatchNorm1d layer
        )

    def forward(self, x):
        features = self.backbone(x)
        return (
            self.head1(features),
            self.head2(features),
            self.head3(features)
        )