import os
import sys
import glob
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision import transforms

# Ensure python path includes root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from models.baseline_cnn.model import BaselineCNN
from evaluation.metrics import calculate_classification_metrics, save_metrics

NUM_CLASSES = 4
CLASSES = ["Fire", "Smoke", "Clouds/Fog", "Dust/Clear"]
EPOCHS = 3
BATCH_SIZE = 2

class RealCameraDataset(Dataset):
    """Loads actual camera surveillance images and labels from datasets/camera_surveillance."""
    def __init__(self, images_dir, labels_dir, transform=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.samples = []
        
        image_files = glob.glob(os.path.join(images_dir, "*.png")) + glob.glob(os.path.join(images_dir, "*.jpg"))
        for img_path in image_files:
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            lbl_path = os.path.join(labels_dir, f"{base_name}.txt")
            label = 1  # Default to Smoke class if no label file
            if os.path.exists(lbl_path):
                with open(lbl_path, "r") as f:
                    first_line = f.readline().strip()
                    if first_line:
                        parts = first_line.split()
                        label = int(parts[0])
            self.samples.append((img_path, min(label, NUM_CLASSES - 1)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)


def train_and_evaluate():
    print(f"==================================================")
    print(f"--- Training Baseline CNN on Real Camera Data ---")
    print(f"==================================================")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_img_dir = os.path.abspath("datasets/camera_surveillance/train/images")
    train_lbl_dir = os.path.abspath("datasets/camera_surveillance/train/labels")
    val_img_dir = os.path.abspath("datasets/camera_surveillance/val/images")
    val_lbl_dir = os.path.abspath("datasets/camera_surveillance/val/labels")
    
    train_dataset = RealCameraDataset(train_img_dir, train_lbl_dir, transform=transform)
    val_dataset = RealCameraDataset(val_img_dir, val_lbl_dir, transform=transform)
    
    print(f"Loaded real train samples: {len(train_dataset)}, val samples: {len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = BaselineCNN(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training Loop
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            
        epoch_loss = running_loss / len(train_dataset)
        print(f"Epoch [{epoch+1}/{EPOCHS}] - Real Training Loss: {epoch_loss:.4f}")
        
    # Evaluation
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(labels.numpy().tolist())
            
    metrics = calculate_classification_metrics(all_targets, all_preds, classes=CLASSES)
    metrics["model"] = "baseline_cnn"
    metrics["dataset"] = "HPWREN Real Camera Surveillance Dataset"
    metrics["real_dataset"] = True
    
    save_metrics(metrics, "output/evaluation/baseline_cnn_metrics.json")
    print("\n--- Baseline CNN Evaluation on Real Data Complete ---")
    print(json.dumps(metrics, indent=2))
    return metrics

if __name__ == "__main__":
    train_and_evaluate()
