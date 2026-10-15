import torch
import torch.nn as nn

class BaselineCNN(nn.Module):
    """
    Baseline 3-stage 2D CNN classifier.
    Designed to process 224x224 RGB images and output logits for 4 classes:
    0: No fire, 1: Smoke, 2: Fire, 3: Fire + Smoke
    """
    def __init__(self, num_classes: int = 4):
        super(BaselineCNN, self).__init__()
        
        # Stage 1
        self.stage1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # 224x224 -> 112x112
        )
        
        # Stage 2
        self.stage2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # 112x112 -> 56x56
        )
        
        # Stage 3
        self.stage3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # 56x56 -> 28x28
        )
        
        # Classifier Head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        logits = self.classifier(x)
        return logits
