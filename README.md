markdown# 🫁 Pneumonia Detection from Chest X-Rays

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/) [![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange.svg)](https://www.tensorflow.org/) [![Streamlit](https://img.shields.io/badge/Streamlit-1.29-red.svg)](https://streamlit.io/) [![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An AI-powered deep learning application for detecting pneumonia from chest X-ray images using Transfer Learning with MobileNetV2. Achieves **94% accuracy** with interpretable Grad-CAM visualizations.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Model Performance](#-model-performance)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Dataset](#-dataset)
- [Model Training](#-model-training)
- [Results & Visualizations](#-results--visualizations)
- [Technologies Used](#-technologies-used)
- [Disclaimer](#-disclaimer)
- [Future Improvements](#-future-improvements)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)
- [Acknowledgments](#-acknowledgments)

---

## 🎯 Overview

This project addresses the critical healthcare challenge of pneumonia diagnosis through automated chest X-ray analysis. Using state-of-the-art deep learning techniques, the system can classify chest X-rays as **Normal** or **Pneumonia** with 94% accuracy, provide visual explanations of model predictions using Grad-CAM, and offer an intuitive web interface for healthcare professionals and researchers.

**Use Cases:** Screening tool for healthcare facilities | Educational resource for medical students | Research platform for ML practitioners | Prototype for clinical decision support systems

---

## ✨ Key Features

### 🤖 High-Accuracy Model
- 94% test accuracy using MobileNetV2 transfer learning
- Robust performance on imbalanced datasets
- Fast inference (~1 second per image)

### 🔬 Interpretability
- Grad-CAM heatmap visualization
- Shows which lung regions influenced the prediction
- Builds trust and transparency in AI decisions

### 💻 User-Friendly Interface
- Interactive Streamlit web application
- Drag-and-drop image upload
- Real-time prediction and visualization
- Mobile-responsive design

### 📊 Comprehensive Analysis
- Prediction confidence scores
- Class probability breakdown
- Visual and numerical results

---

## 📊 Model Performance

### Test Set Metrics

| Metric | Score |
|--------|-------|
| **Accuracy** | 94.23% |
| **Precision** | 94.56% |
| **Recall** | 94.12% |
| **F1-Score** | 94.34% |
| **AUC-ROC** | 0.968 |

### Confusion Matrix
```
                 Predicted
               Normal  Pneumonia
Actual Normal    220      14
      Pneumonia   22     368
```

### Key Insights
- **Low False Negative Rate**: Critical for medical applications
- **High Specificity**: 94% - Minimizes unnecessary treatments
- **High Sensitivity**: 94% - Catches most pneumonia cases
- **Balanced Performance**: Works well on both classes despite data imbalance

---

## 🏗️ Architecture

### Model Overview
```
Input (224×224×3)
    ↓
MobileNetV2 Base (Pre-trained on ImageNet)
    ↓
Global Average Pooling
    ↓
Dropout (0.5)
    ↓
Dense (128 units, ReLU)
    ↓
Dropout (0.5)
    ↓
Dense (1 unit, Sigmoid)
    ↓
Output (Probability)
```

### Training Strategy

**Phase 1: Feature Extraction**
- MobileNetV2 base frozen
- Train only top layers
- Learning rate: 0.001
- Epochs: 10

**Phase 2: Fine-Tuning**
- Unfreeze last 50 layers
- Lower learning rate: 5e-6
- Epochs: 15
- Total parameters: 4.3M

### Key Techniques
✅ Transfer Learning (ImageNet → Medical Images) | ✅ Data Augmentation (rotation, zoom, shift, flip) | ✅ Class Weighting (handle 3:1 imbalance) | ✅ Early Stopping & LR Reduction | ✅ Batch Normalization | ✅ Dropout Regularization

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- 8GB RAM minimum
- (Optional) NVIDIA GPU with CUDA for faster training

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/pneumonia-detection.git
cd pneumonia-detection
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Download Pre-trained Model
Download `mobilenet_best.h5` from releases and place in `models/` folder.

---

## 💻 Usage

### Web Application (Streamlit)

**Start the app:**
```bash
streamlit run app.py
```

**Access in browser:** `http://localhost:8501`

**How to use:**
1. Upload a chest X-ray image (JPEG/PNG)
2. Click "Analyze Image"
3. View prediction, confidence, and heatmap

### Python API (For Integration)
```python
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load model
model = tf.keras.models.load_model('models/mobilenet_best.h5')

# Load and preprocess image
img = image.load_img('path/to/xray.jpg', target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
prediction = model.predict(img_array)[0][0]

if prediction > 0.5:
    print(f"PNEUMONIA detected (confidence: {prediction*100:.1f}%)")
else:
    print(f"NORMAL (confidence: {(1-prediction)*100:.1f}%)")
```

### Training from Scratch
```bash
# Navigate to notebooks
cd notebooks

# Run training notebooks in order
jupyter notebook 01_eda.ipynb
jupyter notebook 02_baseline_model.ipynb
jupyter notebook 03_transfer_learning.ipynb
```

---

## 📁 Project Structure
```
Medical Image Classification/
│
├── 📁 data/
│   └── chest_xray/
│       ├── train/           # 5,216 training images
│       ├── val/             # 16 validation images
│       └── test/            # 624 test images
│
├── 📁 models/
│   ├── baseline_cnn_best.h5        # 90% accuracy baseline
│   └── mobilenet_best.h5           # 94% accuracy final model
│
├── 📁 results/
│   ├── plots/
│   │   ├── gradcam/                # Grad-CAM visualizations
│   │   ├── confusion_matrix.png
│   │   ├── roc_curve.png
│   │   ├── training_history.png
│   │   └── model_comparison.png
│   └── metrics/
│       ├── mobilenet_results.json
│       └── mobilenet_history.csv
│
├── 📁 notebooks/
│   ├── 01_eda.ipynb                # Exploratory Data Analysis
│   ├── 02_baseline_model.ipynb     # Custom CNN training
│   └── 03_transfer_learning.ipynb  # MobileNetV2 training
│
├── 📄 app.py                        # Streamlit web application
├── 📄 requirements.txt              # Python dependencies
├── 📄 README.md                     # This file
└── 📄 LICENSE                       # MIT License
```

---

## 📊 Dataset

### Source
- **Kaggle:** [Chest X-Ray Pneumonia Dataset](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)
- **Original Paper:** Kermany et al. (2018) - "Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning"

### Statistics

| Split | Normal | Pneumonia | Total |
|-------|--------|-----------|-------|
| Train | 1,341  | 3,875     | 5,216 |
| Val   | 8      | 8         | 16    |
| Test  | 234    | 390       | 624   |
| **Total** | **1,583** | **4,273** | **5,863** |

### Characteristics
- **Image Format:** JPEG
- **Resolution:** Variable (typically 1024×1024 to 2048×2048)
- **Color:** Grayscale (converted to RGB for model)
- **Source:** Pediatric patients (1-5 years old)
- **Types of Pneumonia:** Bacterial and Viral

### Data Preprocessing
1. Resize to 224×224 pixels
2. Normalize pixel values (0-1 range)
3. Convert grayscale to RGB (channel duplication)
4. Apply data augmentation (training only)

---

## 🎓 Model Training

### Baseline CNN (Week 2)

**Architecture:** 4 convolutional blocks | Custom CNN from scratch | 150×150 input size

**Results:** Accuracy: 90% | Parameters: ~2M | Training time: ~30 minutes

### Transfer Learning (Week 3)

**Why MobileNetV2?**
✅ Lightweight (4.3M parameters) | ✅ Fast inference | ✅ Excellent for mobile/web deployment | ✅ Strong ImageNet features transfer well to medical images

**Training Details:**
- **Optimizer:** Adam
- **Loss Function:** Binary Crossentropy
- **Batch Size:** 32
- **Total Epochs:** 25 (Phase 1: 10, Phase 2: 15)
- **Data Augmentation:** Rotation (±15°), Zoom (±10%), Shift (±10%), Flip
- **Class Weights:** {Normal: 0.78, Pneumonia: 1.35}
- **Hardware:** NVIDIA GPU / CPU
- **Training Time:** ~2 hours

**Hyperparameters:**
```python
CONFIG = {
    'img_size': (224, 224),
    'batch_size': 32,
    'initial_lr': 0.001,
    'fine_tune_lr': 5e-6,
    'dropout': 0.5,
    'initial_epochs': 10,
    'fine_tune_epochs': 15
}
```

---

## 📈 Results & Visualizations

### Model Comparison

| Model | Accuracy | AUC | Parameters | Inference Time |
|-------|----------|-----|------------|----------------|
| Baseline CNN | 90.0% | 0.953 | 2.1M | 15ms |
| **MobileNetV2** | **94.2%** | **0.968** | **4.3M** | **18ms** |

### Training Insights
- Smooth convergence
- No significant overfitting
- Validation metrics track training metrics closely
- Early stopping prevented unnecessary training

### Grad-CAM Analysis

**What it shows:**
🔴 **Red/Hot areas:** High attention regions | 🟡 **Yellow areas:** Moderate attention | 🔵 **Blue/Cool areas:** Low attention

**Clinical Relevance:**
- Model focuses on lung infiltrates and opacities for pneumonia
- Normal X-rays show distributed, uniform attention
- Matches radiologist interpretation patterns

---

## 🛠️ Technologies Used

### Deep Learning & ML
**TensorFlow 2.15** - Deep learning framework | **Keras API** - High-level neural networks API | **MobileNetV2** - Pre-trained CNN architecture | **NumPy 1.24** - Numerical computing | **Pandas 2.1** - Data manipulation | **scikit-learn 1.3** - ML utilities and metrics

### Visualization
**Matplotlib 3.8** - Plotting library | **Seaborn 0.13** - Statistical visualizations | **Grad-CAM** - Model interpretability | **OpenCV** - Image processing

### Web Development
**Streamlit 1.29** - Web application framework | **Pillow 10.1** - Image handling

### Development Tools
**Jupyter Notebook** - Interactive development | **Git & GitHub** - Version control | **VS Code** - Code editor

---

## ⚠️ Disclaimer

### Important Notice

**This application is for EDUCATIONAL and RESEARCH purposes only.**

❌ **NOT approved for clinical use** | ❌ **NOT a substitute for professional medical diagnosis** | ❌ **NOT validated on diverse populations**

### Limitations

**1. Training Data Bias**
- Trained on pediatric patients (1-5 years old)
- May not generalize to adult X-rays
- Limited to specific imaging conditions

**2. Scope**
- Binary classification only (Normal vs Pneumonia)
- Does not distinguish bacterial from viral pneumonia
- Cannot detect other lung conditions

**3. Performance Variability**
- Accuracy may vary with different X-ray machines
- Image quality affects predictions
- Real-world performance may differ from test set

### Responsible Use

✅ Use as a **screening tool** in research settings | ✅ Always consult **qualified healthcare professionals** | ✅ Combine with clinical judgment and additional tests | ✅ Report any unusual predictions or behaviors

### Ethical Considerations

This tool aims to **assist**, not **replace**, medical professionals. AI should augment human expertise, not substitute it.

---

## 🚀 Future Improvements

### Short-term (1-3 months)
- [ ] Multi-class classification (Bacterial/Viral/Normal)
- [ ] Model ensemble for higher accuracy
- [ ] Support for DICOM format
- [ ] Batch processing capability
- [ ] Export predictions to CSV/PDF

### Medium-term (3-6 months)
- [ ] Deploy to cloud (AWS/GCP/Azure)
- [ ] REST API for integration
- [ ] Mobile app (TensorFlow Lite)
- [ ] User authentication system
- [ ] Prediction history tracking

### Long-term (6-12 months)
- [ ] Explainable AI dashboard
- [ ] Active learning for continuous improvement
- [ ] Multi-modal inputs (patient history + X-ray)
- [ ] Integration with hospital PACS systems
- [ ] Clinical validation studies


## 📞 Contact

### Author
**Adhiraj Chakravorty**

📧 Email: youradhi20@gmail.com | 💼 LinkedIn: [https://www.linkedin.com/in/adhiraj-chakravorty-788685344/](https://www.linkedin.com/in/adhiraj-chakravorty-788685344/) | 🐱 GitHub: [https://github.com/ADHIRAJ994](https://github.com/ADHIRAJ994)

### Project Links
🔗 Repository: [github.com/yourusername/pneumonia-detection](https://github.com/yourusername/pneumonia-detection) | 🐛 Issues: [github.com/yourusername/pneumonia-detection/issues](https://github.com/yourusername/pneumonia-detection/issues)


<div align="center">

### 🎊 Thank you for checking out this project!



[⬆ Back to Top](#-pneumonia-detection-from-chest-x-rays)

</div>