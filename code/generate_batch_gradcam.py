# generate_batch_gradcam.py

import os
import numpy as np
import torch
from torchvision import models, transforms
from torch.nn import functional as F
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import cv2

# === Paths ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "Models", "best_model.pth")
LABELS_FILE = os.path.join(BASE_DIR, "data", "processed", "test_labels.csv")
IMAGES_DIR = os.path.join(BASE_DIR, "data", "images", "images")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "gradcam_samples")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Setup ===
TARGET_CLASSES = ["Atelectasis", "Effusion", "Infiltration", "No Finding", "Other Disease"]
class_to_idx = {cls: idx for idx, cls in enumerate(TARGET_CLASSES)}
idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, len(TARGET_CLASSES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device).eval()

# === Hooks ===
gradients = None
activations = None

def backward_hook(module, grad_input, grad_output):
    global gradients
    gradients = grad_output[0]

def forward_hook(module, input, output):
    global activations
    activations = output

model.layer4[-1].register_forward_hook(forward_hook)
model.layer4[-1].register_backward_hook(backward_hook)

# === Transforms ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# === Load test labels ===
df = pd.read_csv(LABELS_FILE)
df = df[df["Image Index"].isin(os.listdir(IMAGES_DIR))].reset_index(drop=True)

# === Track correctly & incorrectly predicted samples ===
correct, incorrect = [], []

for _, row in df.iterrows():
    image_path = os.path.join(IMAGES_DIR, row["Image Index"])
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Forward pass
    output = model(input_tensor)
    pred_idx = output.argmax(dim=1).item()
    true_idx = class_to_idx[row["label"]]

    if pred_idx == true_idx:
        correct.append((image_path, true_idx, pred_idx))
    else:
        incorrect.append((image_path, true_idx, pred_idx))

    if len(correct) >= 5 and len(incorrect) >= 5:
        break

selected = correct[:5] + incorrect[:5]

# === Generate and save Grad-CAM overlays ===
for img_path, true_idx, pred_idx in selected:
    image = Image.open(img_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    model.zero_grad()
    output = model(input_tensor)
    class_loss = output[0, pred_idx]
    class_loss.backward()

    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    for i in range(activations.shape[1]):
        activations[:, i, :, :] *= pooled_gradients[i]
    heatmap = torch.mean(activations, dim=1).squeeze()
    heatmap = np.maximum(heatmap.detach().cpu().numpy(), 0)
    heatmap /= np.max(heatmap)

    heatmap = cv2.resize(heatmap, (image.size[0], image.size[1]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(np.array(image), 0.6, heatmap, 0.4, 0)

    label_true = idx_to_class[true_idx]
    label_pred = idx_to_class[pred_idx]
    status = "correct" if true_idx == pred_idx else "wrong"
    filename = f"{status}_{label_true}_Pred{label_pred}_{os.path.basename(img_path)}"
    cv2.imwrite(os.path.join(OUTPUT_DIR, filename), overlay[:, :, ::-1])  # RGB -> BGR

print(f"✅ Grad-CAM samples saved to: {OUTPUT_DIR}")
