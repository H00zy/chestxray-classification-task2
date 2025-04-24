# 🩻 Chest X-ray Classification (NIH Sample Dataset)

This repository contains a deep learning workflow for thoracic disease classification using NIH Chest X-ray images. It was developed for a technical AI assessment and includes end-to-end preprocessing, model development, Grad-CAM visualization, evaluation, and reporting.

## 📌 Objective

- Train a ResNet50-based classifier on labeled chest X-ray samples  
- Apply data augmentation and class rebalancing strategies  
- Visualize model decisions using Grad-CAM and batch Grad-CAM  
- Evaluate using classification metrics and ROC curves

## 📁 Project Structure

chestxray-classification-task2/
├── code/
│   ├── prepare_data.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   ├── generate_gradcam.py
│   ├── generate_batch_gradcam.py
│   └── gradcam_visualization.py
├── models/ (ResNet50 trained weights)
├── outputs/
│   ├── classification_report.txt
│   ├── confusion_matrix.png
│   ├── roc_curves.png
│   ├── gradcam_overlay.png
│   └── gradcam_samples/
├── ChestXray_Classification_Report.docx
├── ChestXray_Classification_Slides.pptx
└── requirements.txt

## ✅ Deliverables

- ✅ Stratified sampling & processed dataset  
- ✅ ResNet50 model with class-weighted loss  
- ✅ Evaluation metrics & heatmap visualizations  
- ✅ Grad-CAM overlays and batch visualization  
- ✅ Slide deck and technical report for Task 2 results

## 🛠️ Installation

```bash
pip install -r requirements.txt
```

## 👨‍💻 Author

Hazzaa Alghamdi  
Digital healthcare innovator | AI-driven decision maker | Python-powered insights  
🔗 github.com/H00zy
