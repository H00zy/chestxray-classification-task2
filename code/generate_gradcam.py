import os
import numpy as np
import torch
from torchvision import models, transforms
from torch.nn import functional as F
from PIL import Image
import matplotlib.pyplot as plt
import cv2

# === Paths ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pth")
IMAGE_PATH = os.path.join(BASE_DIR, "data", "images", "images", "00020666_003.png")  # <-- CHANGE this if needed
OUTPUT_PATH = os.path.join(BASE_DIR, "outputs", "gradcam_overlay.png")

# === Load model ===
model = models.resnet50(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 5)
model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
model.eval()

# === Hook to capture gradients ===
gradients = None
activations = None

def backward_hook(module, grad_input, grad_output):
    global gradients
    gradients = grad_output[0]

def forward_hook(module, input, output):
    global activations
    activations = output

# Register hooks
final_conv = model.layer4[-1]
final_conv.register_forward_hook(forward_hook)
final_conv.register_backward_hook(backward_hook)

# === Image transformation ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

image = Image.open(IMAGE_PATH).convert("RGB")
input_tensor = transform(image).unsqueeze(0)

# === Forward pass ===
output = model(input_tensor)
pred_class = output.argmax(dim=1).item()

# === Backward pass ===
model.zero_grad()
class_loss = output[0, pred_class]
class_loss.backward()

# === Generate Grad-CAM ===
pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
for i in range(activations.shape[1]):
    activations[:, i, :, :] *= pooled_gradients[i]

heatmap = torch.mean(activations, dim=1).squeeze()
heatmap = np.maximum(heatmap.detach().numpy(), 0)
heatmap /= np.max(heatmap)

# === Overlay heatmap on image ===
heatmap = cv2.resize(heatmap, (image.size[0], image.size[1]))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

image_np = np.array(image)
overlay = cv2.addWeighted(image_np, 0.6, heatmap, 0.4, 0)

# === Save the result ===
cv2.imwrite(OUTPUT_PATH, overlay[:, :, ::-1])  # Convert RGB to BGR for OpenCV
print("✅ Grad-CAM saved to:", OUTPUT_PATH)
