import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class EfficientNetBaseline(nn.Module):
    """
    Phase 5: Baseline EfficientNet for robust image classification / feature extraction.
    Outputs logits for the 4 classes (No fire, Smoke, Fire, Fire + Smoke).
    """
    def __init__(self, num_classes: int = 4):
        super(EfficientNetBaseline, self).__init__()
        # Use pretrained weights for transfer learning
        self.backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(in_features, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
