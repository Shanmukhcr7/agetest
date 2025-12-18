# Age Prediction from Images (Folder-based Dataset)
# Optimized for macOS Apple Silicon (M1/M2/M3/M4) using MPS

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

# --------------------------------------------------
# Device Configuration (Apple Metal GPU)
# --------------------------------------------------
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

# --------------------------------------------------
# Dataset Structure Expected
# --------------------------------------------------
# age_prediction/test/
# ├── 001/
# │    ├── 7148.jpg
# ├── 002/
# ├── 003/
# ...
# └── 100/
# Folder name = age label

class AgeDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform

        for age_folder in sorted(os.listdir(root_dir)):
            age_path = os.path.join(root_dir, age_folder)

            if not os.path.isdir(age_path):
                continue

            try:
                age = int(age_folder)  # '001' → 1
            except ValueError:
                continue

            for img_name in os.listdir(age_path):
                if img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append(
                        (os.path.join(age_path, img_name), age)
                    )

        print(f"Total images found: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, age = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(age, dtype=torch.float32)

# --------------------------------------------------
# Image Transforms
# --------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# --------------------------------------------------
# Dataset & DataLoader
# --------------------------------------------------
DATASET_PATH = "/Users/eeshanrohith/Desktop/Age X/age_prediction_up/age_prediction/test"

train_dataset = AgeDataset(
    root_dir=DATASET_PATH,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=8,   # safe for 8GB RAM
    shuffle=True,
    num_workers=0
)

# --------------------------------------------------
# Model: ResNet18 (Regression)
# --------------------------------------------------
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 1)
model = model.to(device)

# --------------------------------------------------
# Loss & Optimizer
# --------------------------------------------------
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# --------------------------------------------------
# Training Loop
# --------------------------------------------------
epochs = 20

for epoch in range(epochs):
    model.train()
    total_loss = 0.0

    for images, ages in train_loader:
        images = images.to(device)
        ages = ages.to(device).unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, ages)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}")

# --------------------------------------------------
# Save Model
# --------------------------------------------------
MODEL_PATH = "age_model.pth"
torch.save(model.state_dict(), MODEL_PATH)
print(f"Model saved as {MODEL_PATH}")

# --------------------------------------------------
# Prediction Function
# --------------------------------------------------
def predict_age(image_path):
    model.eval()
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        age = model(image).item()

    return round(age, 1)

# Example usage (uncomment to test)
# print(predict_age("/Users/eeshanrohith/Desktop/Age X/age_prediction_up/age_prediction/test/001/7148.jpg"))