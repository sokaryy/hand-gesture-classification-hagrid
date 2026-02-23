# Hand Gesture Classification (HaGRID)

A machine learning project for hand gesture recognition that classifies 18 hand gestures from the HaGRID dataset. The project uses MediaPipe to extract 21-point hand landmarks and trains multiple classification models to achieve high accuracy.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Evaluation](#model-evaluation)
- [Technologies](#technologies)
- [Results](#results)

## 🎯 Overview

This project implements an end-to-end machine learning pipeline for hand gesture classification. It processes hand images using MediaPipe to extract landmark coordinates and trains machine learning models (Random Forest, SVM, Logistic Regression, etc.) to recognize 18 different hand gestures.

## ✨ Features

- **MediaPipe Integration**: Extracts 21-point hand landmarks from hand images
- **Multiple Models**: Implements various classification algorithms for comparison
- **MLflow Tracking**: Comprehensive experiment tracking with MLflow for model management
- **Evaluation Metrics**: Detailed performance analysis with accuracy, precision, recall, and F1-score
- **Confusion Matrix Visualization**: Visual representation of model predictions
- **Data Preprocessing**: Feature extraction and normalization pipeline

## 📊 Dataset

**HaGRID Dataset**:
- 18 hand gesture classes
- Hand landmarks extracted using MediaPipe (21 points per hand)
- Features: X, Y, Z coordinates and confidence scores for each landmark
- Train/test split: 80/20

### Gesture Classes

The model recognizes the following 18 gestures:
- Open Palm
- Thumbs Up
- Thumbs Down
- Victory Sign
- OK Sign
- And 13 more hand gestures from the HaGRID dataset

## 📁 Project Structure

```
hand-gesture-classification-hagrid/
├── README.md                                  # Project documentation
├── requirements.txt                          # Python dependencies
├── hand-gesture-classification.ipynb         # Main Jupyter notebook with full pipeline
├── .gitignore                                # Git ignore file
├── .git/                                     # Version control
├── .venv/                                    # Virtual environment folder
├── data/
│   └── hand_landmarks_data.csv              # Processed hand landmark features (84 features + 1 label)
├── src/
│   ├── evaluation.py                        # Model evaluation metrics and confusion matrix visualization
│   ├── mlflow_utils.py                      # MLflow experiment tracking utilities
│   └── __pycache__/
├── artifacts/
│   ├── Lr_optimal_conf_matrix.png          # Logistic Regression confusion matrix
│   ├── random_forest_optimal_conf_matrix.png # Random Forest confusion matrix
│   └── Support_Vector_Machine_optimal_conf_matrix.png # SVM confusion matrix
├── mlruns/                                  # MLflow local experiment tracking and metadata
├── mlflow screenshots/                      # MLflow UI interface screenshots
└── __pycache__/
```

## 🚀 Installation

### Prerequisites
- Python 3.8+
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
- **mlflow** (2.22.0): Experiment tracking and model management
- **scikit-learn** (1.6.1): Machine learning algorithms
- **pandas** (2.2.3): Data manipulation
- **numpy** (2.2.5): Numerical computations
- **seaborn** (0.13.2): Data visualization
- **scipy** (1.15.2): Scientific computing

## 💻 Usage

### Running the Main Notebook

1. **Start Jupyter Notebook**:
   ```bash
   jupyter notebook hand-gesture-classification.ipynb
   ```

2. **Execute the pipeline**:
   - Load and explore the hand landmarks data
   - Preprocess features and split data
   - Train multiple classification models
   - Evaluate and compare model performance
   - Generate confusion matrices and metrics visualizations

### Key Steps in the Pipeline

1. **Data Loading**: Load `hand_landmarks_data.csv`
2. **Feature Engineering**: Normalize and scale hand landmark coordinates
3. **Model Training**: Train various classification models
4. **Model Evaluation**: Calculate metrics (accuracy, precision, recall, F1)
5. **Visualization**: Generate confusion matrices and performance plots
6. **MLflow Tracking**: Log models, metrics, and parameters (optional with MLflow UI)

### Using the Evaluation Module

```python
from src.evaluation import evaluate_metrics

# Evaluate model predictions
metrics = evaluate_metrics(y_true, y_pred)
```

### Using MLflow Utilities

```python
from src.mlflow_utils import setup_mlflow, start_run

# Setup MLflow tracking
setup_mlflow("Hand Gesture Classification", tracking_uri="http://localhost:5000")

# Start a new tracking run
start_run("model_run_name")
```

## 📈 Model Evaluation

The project evaluates models using the following metrics:

| Metric | Description |
|--------|-------------|
| **Accuracy** | Overall correctness of predictions |
| **Precision** | True positives / (True positives + False positives) |
| **Recall** | True positives / (True positives + False negatives) |
| **F1-Score** | Harmonic mean of precision and recall |
| **Confusion Matrix** | Visual breakdown of correct/incorrect predictions per class |

### Example Output

```
Model accuracy: 0.9234
Model precision: 0.9245
Model recall: 0.9234
Model f1: 0.9235
```

## 🛠 Technologies

- **Machine Learning**: scikit-learn
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Experiment Tracking**: MLflow
- **Environment Management**: venv/virtualenv
- **Hand Detection**: MediaPipe (for landmark extraction)

## 📊 Results

The trained models achieve high accuracy on the hand gesture classification task. Results are tracked in MLflow with comprehensive metrics and visualizations. The confusion matrix helps identify which gesture classes are frequently confused with each other.

### Model Comparison

Three classification algorithms were trained and evaluated on the validation set:

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **Random Forest** 🏆 | **0.9662** | **0.9665** | **0.9662** | **0.9662** |
| SVC (SVM) | 0.9462 | 0.9491 | 0.9462 | 0.9469 |
| Logistic Regression | 0.9102 | 0.9096 | 0.9102 | 0.9096 |

### Model Selection Rationale

**Random Forest was selected as the final model** based on the following analysis:

- **Best Overall Performance**: Random Forest achieved the highest accuracy (96.62%) among all three models, with a ~2% improvement over SVM and ~5.6% over Logistic Regression.
- **Consistent Metrics**: All evaluation metrics (accuracy, precision, recall, F1-score) are exceptionally high and consistent, indicating robust and reliable performance.
- **Superior Precision & Recall**: With 96.65% precision and 96.62% recall, the model minimizes both false positives and false negatives, crucial for real-world gesture recognition applications.
- **Class Imbalance Handling**: Random Forest naturally handles the imbalanced nature of the hand gesture dataset effectively.
- **Generalization**: The high F1-score (0.9662) indicates excellent generalization to unseen data.

The SVM model also performed well (94.62% accuracy) but slightly underperformed Random Forest. Logistic Regression, while achieving reasonable accuracy (91.02%), was outperformed by the ensemble methods, suggesting that the non-linear decision boundaries are better captured by tree-based and kernel-based methods.

See `mlruns/` and `mlartifacts/` directories for detailed experiment results and model artifacts.

## 📝 Notes

- The project uses locally stored CSV data with pre-extracted hand landmarks
- MLflow tracking is configured for local experiments (default: `http://localhost:5000`)
- To use MLflow UI, run: `mlflow ui` in the project directory
- Model artifacts are automatically logged to the `mlartifacts/` directory

## 🔄 Workflow

1. **Data Preparation**: Hand landmarks are extracted using MediaPipe and stored in CSV format
2. **Exploration**: Analyze data distribution and gesture characteristics
3. **Training**: Train classification models with hyperparameter tuning
4. **Evaluation**: Assess model performance using multiple metrics
5. **Deployment**: Save best performing models as artifacts

## 📖 References

- [HaGRID Dataset](https://github.com/hukenovs/hagrid)
- [MediaPipe Hands](https://google.github.io/mediapipe/solutions/hands.html)
- [scikit-learn Documentation](https://scikit-learn.org/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)

---

**Project Status**: Active Development  
**Last Updated**: February 2026
