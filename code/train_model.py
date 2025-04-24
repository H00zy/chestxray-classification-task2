import os
import pandas as pd
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

# === Paths ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # no double dirname anymore
BASE_DIR = os.path.dirname(BASE_DIR)  # go one level up from /code/
LABELS_FILE = os.path.join(BASE_DIR, "data", "processed", "train_labels.csv")
IMAGES_DIR = os.path.join(BASE_DIR, "data", "images", "images")
MODEL_DIR = os.path.join(BASE_DIR, "Models")
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pth")

# Ensure model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# === Config ===
NUM_EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
IMAGE_SIZE = 224
PATIENCE = 3

TARGET_CLASSES = ["Atelectasis", "Effusion", "Infiltration", "No Finding", "Other Disease"]
class_to_idx = {cls: idx for idx, cls in enumerate(TARGET_CLASSES)}

# === Dataset ===
class ChestXrayDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(IMAGES_DIR, row["Image Index"])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = class_to_idx[row["label"]]
        return image, label

# === Transforms ===
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

# === Load & Split Data ===
df = pd.read_csv(LABELS_FILE)
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)

train_dataset = ChestXrayDataset(train_df, transform=train_transform)
val_dataset = ChestXrayDataset(val_df, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# === Compute class weights ===
labels = [class_to_idx[label] for label in train_df["label"]]
class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(labels), y=labels)
class_weights = torch.tensor(class_weights, dtype=torch.float)

# === Model Setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, len(TARGET_CLASSES))
model = model.to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# === Training ===
print("🚀 Training started...")
best_loss = float("inf")
patience_counter = 0

for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)

    # === Validation ===
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    avg_val_loss = val_loss / len(val_loader)

    print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")

    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        patience_counter = 0
        torch.save(model.state_dict(), MODEL_PATH)
        print("✅ Best model saved.")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print("⏹️ Early stopping triggered.")
            break

print("✅ Training complete.")
