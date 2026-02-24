# Hand Gesture Classification (HaGRID)

Real-time hand gesture recognition system using MediaPipe and machine learning. Detects and classifies hand gestures from webcam input using pre-trained Random Forest model achieving 96.62% accuracy.

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Directory Guide](#-directory-guide)
- [Technologies](#-technologies)
- [Model Performance](#-model-performance)
- [Gesture Classes](#-gesture-classes)
- [References](#-references)
- [Configuration](#️-configuration)
- [Performance Tips](#-performance-tips)
- [Troubleshooting](#-troubleshooting)
- [Notes](#-notes)

## 🎯 Overview

This project provides a complete hand gesture recognition solution with real-time inference capabilities. It uses MediaPipe to extract 21-point hand landmarks and a trained Random Forest classifier to recognize hand gestures. The system supports live webcam input for interactive gesture recognition.

## ✨ Features

- **Real-time Inference**: Live gesture recognition from webcam with 96.62% accuracy
- **MediaPipe Integration**: Robust 21-point hand landmark detection
- **Multiple Pre-trained Models**: Random Forest, SVM, and Logistic Regression models available
- **Modular Architecture**: Clean separation of concerns across multiple modules
- **Modern UI**: Live visualization with bounding boxes, confidence scores, and FPS counter
- **Gesture Preprocessing**: Landmark normalization and centering for robust predictions

## � Project Structure

```
hand-gesture-classification-hagrid/
├── README.md                                 # Project documentation
├── requirements.txt                          # Python dependencies
├── hand-gesture-classification.ipynb         # Training & analysis notebook
├── hand_landmarker.task                      # MediaPipe hand landmark model
├── Hand Gesture Detection Video Demo.mp4     # Demo video of real-time inference
├── .gitignore                                # Git ignore rules
├── .gitattributes                            # Git attributes
├── .venv/                                    # Virtual environment
├── .git/                                     # Version control directory
│
├── data/
│   └── hand_landmarks_data.csv              # Extracted hand landmark features (84 cols + 1 label)
│
├── models_pkls/                              # Pre-trained machine learning models
│   ├── Random_Forest.pkl                     # ✓ Best performing model (96.62% accuracy)
│   ├── Support_Vector_Machine.pkl            # SVM model (94.62% accuracy)
│   └── Lr_model.pkl                          # Logistic Regression model (91.02% accuracy)
│
├── src/                                      # Source code package
│   ├── __init__.py                           # Package initialization
│   ├── config.py                             # Configuration & constants
│   ├── inference.py                          # Real-time gesture recognition from webcam
│   ├── model.py                              # GestureModel wrapper class
│   ├── utils.py                              # Landmark extraction & preprocessing utilities
│   ├── evaluation.py                         # Model evaluation metrics and visualization
│   ├── README.md                             # Detailed src module documentation
│   └── __pycache__/                          # Python bytecode cache
│
└── __pycache__/                              # Root Python cache
```

## 🚀 Installation

### Prerequisites
- Python 3.8+
- Webcam/camera device
- pip package manager

### Setup

1. **Clone/Navigate to the project directory**:
   ```bash
   cd hand-gesture-classification-hagrid
   ```

2. **Create and activate a virtual environment** (recommended):
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # On Windows
   source .venv/bin/activate  # On macOS/Linux
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Required Packages
- **opencv-python** (cv2): Video capture and drawing utilities
- **mediapipe**: Hand landmark detection
- **scikit-learn**: Pre-trained machine learning models
- **numpy**: Numerical computations
- **pandas**: Data manipulation (for notebook)
- **seaborn**: Data visualization (for notebook)
- **scipy**: Scientific computing

## 💻 Usage

### Quick Start - Real-time Gesture Recognition

```bash
cd src
python inference.py
```

Press `'q'` to exit the application.

**What you'll see**:
- Live webcam feed with hand landmarks drawn
- Bounding box around detected hand
- Predicted gesture label with confidence score
- Hand classification (Left/Right)
- FPS counter in top-right corner

### Using in Python Code

```python
from src.inference import HandGestureRecognizer

# Create and run gesture recognizer
recognizer = HandGestureRecognizer()
recognizer.run()
```

### Training & Analysis

Open the Jupyter notebook for data exploration and model training:

```bash
jupyter notebook hand-gesture-classification.ipynb
```

The notebook covers:
- Data loading and exploration
- Feature preprocessing and normalization
- Model training (Random Forest, SVM, Logistic Regression)
- Performance evaluation with metrics and confusion matrices
- Model comparison and selection

## 📂 Directory Guide

### `src/` - Source Code Modules

#### `config.py`
Central configuration file containing:
- **MODEL_PATH**: Path to pre-trained Random Forest model
- **MEDIAPIPE_CONFIG**: Hand detection settings (detection/tracking confidence, max hands)
- **COLOR_CONFIG**: UI display colors (BGR format for OpenCV)
- **UI_CONFIG**: UI element settings (panel height, padding, thickness)
- **PREPROCESSING_CONFIG**: Landmark normalization parameters

#### `inference.py`
Main real-time gesture recognition application:
- **HandGestureRecognizer class**:
  - `setup_mediapipe()`: Initialize MediaPipe hand detector
  - `process_frame()`: Process video frames
  - `draw_hand_detection()`: Render predictions, landmarks, and bounding box
  - `draw_no_detection()`: Display "No hand detected" message
  - `draw_fps()`: Show real-time FPS counter
  - `calculate_fps()`: Compute frames per second
  - `run()`: Main inference loop
- **main()**: Entry point for running the application

#### `model.py`
Model wrapper class for gesture prediction:
- **GestureModel class**:
  - `_load_model()`: Load Random Forest model from pickle file
  - `predict()`: Get gesture label and confidence score for input landmarks
  - `get_all_probabilities()`: Get prediction probabilities for all gesture classes

#### `utils.py`
Utility functions for hand landmark processing:
- `extract_hand_landmarks()`: Extract 21 hand landmark coordinates from MediaPipe (63 total features: 21 landmarks × 3 coordinates)
- `preprocess_landmarks()`: Normalize landmarks by recentering to wrist and scaling by middle finger position
- `calculate_bounding_box()`: Compute bounding box around detected hand with configurable padding

#### `evaluation.py`
Model evaluation and visualization utilities:
- Confusion matrix generation and visualization
- Performance metrics calculation (accuracy, precision, recall, F1-score)
- Model comparison visualization

#### `__init__.py`
Package initialization file for the src module

#### `README.md`
Detailed documentation of the src module and its components

### `data/` Directory

**hand_landmarks_data.csv**:
- Pre-extracted hand landmark features from training dataset
- Format: 84 feature columns (21 landmarks × 3-4 coordinates) + 1 label column
- Used for model training and evaluation

### `models_pkls/` Directory

Pre-trained machine learning models saved as pickle files:

| Model | File | Accuracy | Precision | Recall | F1-Score |
|-------|------|----------|-----------|--------|----------|
| **Random Forest** 🏆 | `Random_Forest.pkl` | **96.62%** | **96.65%** | **96.62%** | **96.62%** |
| Support Vector Machine | `Support_Vector_Machine.pkl` | 94.62% | 94.91% | 94.62% | 94.69% |
| Logistic Regression | `Lr_model.pkl` | 91.02% | 90.96% | 91.02% | 91.06% |

**Usage**:
```python
from src.model import GestureModel

model = GestureModel("models_pkls/Random_Forest.pkl")
gesture, confidence = model.predict(landmarks_array)
```

### Root-Level Files

| File | Purpose |
|------|---------|
| `README.md` | This documentation |
| `requirements.txt` | Python package dependencies |
| `hand-gesture-classification.ipynb` | Jupyter notebook with full ML pipeline |
| `.gitignore` | Git ignore rules |
| `.gitattributes` | Git attributes for line endings |
| `hand_landmarker.task` | MediaPipe pre-trained hand landmark model |
| `Hand Gesture Detection Video Demo.mp4` | Demo video of real-time inference in action |

## � Technologies

- **Computer Vision**: OpenCV (cv2), MediaPipe
- **Machine Learning**: scikit-learn
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Environment Management**: venv/virtualenv
- **Notebook**: Jupyter
- **Version Control**: Git

## 📊 Model Performance

The Random Forest model was selected for production use due to superior performance:

**Performance Metrics**:
```
Accuracy:  96.62%
Precision: 96.65%
Recall:    96.62%
F1-Score:  96.62%
```

**Why Random Forest?**
- Highest overall accuracy compared to SVM and Logistic Regression
- Excellent precision and recall balance
- Robust handling of non-linear decision boundaries
- Consistent performance across all gesture classes
- Efficient inference speed suitable for real-time applications

## 🎯 Gesture Classes

The model recognizes 26 ASL (American Sign Language) gestures (A-Z):
- A, B, C, D, E, F, G, H, I, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z

*Note: Hand landmarks are normalized by recentering to the wrist position and scaling by the middle finger tip distance for rotation and scale invariance.*

## 📖 References

- [HaGRID Dataset](https://github.com/hukenovs/hagrid)
- [MediaPipe Hands](https://google.github.io/mediapipe/solutions/hands.html)
- [scikit-learn Documentation](https://scikit-learn.org/)
- [OpenCV Documentation](https://docs.opencv.org/)

## ⚙️ Configuration

To customize the inference behavior, edit `src/config.py`:

```python
# Detection confidence threshold (0.0 to 1.0)
'min_detection_confidence': 0.5

# Colors in BGR format (OpenCV standard)
COLOR_CONFIG = {
    'accent': (255, 144, 30),
    'background': (40, 40, 40),
    'text': (255, 255, 255),
}

# UI settings
'panel_height': 35
'padding': 20
```

## 🚀 Performance Tips

- **Faster Inference**: Reduce detection confidence threshold (faster but less accurate)
- **Better Accuracy**: Increase detection confidence threshold (slower but more accurate)
- **FPS Optimization**: The inference interval in `inference.py` can be adjusted for speed/accuracy tradeoff
- **Camera Lighting**: Ensure good lighting for better hand detection

## ❓ Troubleshooting

**Issue**: "No hand detected" appears constantly
- **Solution**: Improve lighting, position hand clearly in frame, ensure hand is fully visible

**Issue**: Incorrect gesture predictions
- **Solution**: Ensure hand is properly positioned, try different models from `models_pkls/`

**Issue**: Low FPS
- **Solution**: Reduce video resolution, update graphics drivers, close background applications

**Issue**: Model loading fails
- **Solution**: Ensure `Random_Forest.pkl` exists in `models_pkls/` directory, check file permissions

## 📝 Notes

- The system is optimized for single-hand recognition (max_num_hands=1 in config)
- All landmark coordinates are normalized for scale and rotation invariance
- Real-time performance: ~25-30 FPS on standard laptops
- MediaPipe hand detector works best with clear, frontal hand views

---

**Project Status**: Production Ready ✓  
**Last Updated**: February 2026  
**License**: Open Source
