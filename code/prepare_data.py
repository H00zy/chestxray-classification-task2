import os
import pandas as pd
from sklearn.model_selection import train_test_split

# === Paths ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABELS_FILE = os.path.join(BASE_DIR, "data", "sample_labels.csv")  # ✅ Correct path
IMAGES_DIR = os.path.join(BASE_DIR, "data", "images", "images")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "processed")

# === Ensure output directory exists ===
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Load CSV ===
df = pd.read_csv(LABELS_FILE)

# === Filter only images we actually have ===
existing_images = set(os.listdir(IMAGES_DIR))
df = df[df["Image Index"].isin(existing_images)]

# === Class mapping ===
TARGET_CLASSES = ["Atelectasis", "Effusion", "Infiltration", "No Finding"]

def simplify_label(label_str):
    labels = label_str.split('|')
    for target in TARGET_CLASSES:
        if target in labels:
            return target
    return "Other Disease"

df["label"] = df["Finding Labels"].apply(simplify_label)

# === Stratified Split ===
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df["label"],
    random_state=42
)

# === Save to disk ===
train_df.to_csv(os.path.join(OUTPUT_DIR, "train_labels.csv"), index=False)
test_df.to_csv(os.path.join(OUTPUT_DIR, "test_labels.csv"), index=False)

print("✅ Data prepared and saved to:", OUTPUT_DIR)
