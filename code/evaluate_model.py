import os
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.preprocessing import label_binarize

# === Paths ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABELS_FILE = os.path.join(BASE_DIR, "data", "processed", "test_labels.csv")
IMAGES_DIR = os.path.join(BASE_DIR, "data", "images", "images")
MODEL_PATH = os.path.join(BASE_DIR, "Models", "best_model.pth")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Config ===
IMAGE_SIZE = 224
BATCH_SIZE = 32
TARGET_CLASSES = ["Atelectasis", "Effusion", "Infiltration", "No Finding", "Other Disease"]
class_to_idx = {cls: idx for idx, cls in enumerate(TARGET_CLASSES)}
idx_to_class = {v: k for k, v in class_to_idx.items()}

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
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

# === Load Data ===
df = pd.read_csv(LABELS_FILE)
dataset = ChestXrayDataset(df, transform=transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# === Model ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, len(TARGET_CLASSES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

# === Inference ===
all_preds, all_labels, all_scores = [], [], []

with torch.no_grad():
    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
        all_scores.extend(outputs.cpu().numpy())

# === Metrics ===
report = classification_report(all_labels, all_preds, target_names=TARGET_CLASSES, digits=4)
print("\n📊 Classification Report:\n", report)
with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "w") as f:
    f.write(report)

# === Confusion Matrix ===
conf_mat = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues",
            xticklabels=TARGET_CLASSES, yticklabels=TARGET_CLASSES)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
plt.close()

# === ROC Curves ===
y_true_bin = label_binarize(all_labels, classes=list(range(len(TARGET_CLASSES))))
fpr, tpr, roc_auc = {}, {}, {}
all_scores = np.array(all_scores)

for i in range(len(TARGET_CLASSES)):
    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], all_scores[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(10, 8))
for i in range(len(TARGET_CLASSES)):
    plt.plot(fpr[i], tpr[i], label=f'{TARGET_CLASSES[i]} (AUC = {roc_auc[i]:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.title('ROC Curve by Class')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "roc_curves.png"))
plt.close()
