# Hand Gesture Recognition Module

This module provides a complete real-time hand gesture recognition system using MediaPipe and a trained Random Forest model.

## File Structure

### `config.py`
Contains all configuration constants and settings:
- **MODEL_PATH**: Path to the Random Forest pickle model
- **MEDIAPIPE_CONFIG**: Hand detection parameters
- **COLOR_CONFIG**: UI display colors (BGR format)
- **UI_CONFIG**: UI element settings (panel height, padding, etc.)
- **PREPROCESSING_CONFIG**: Landmark preprocessing parameters

### `utils.py`
Utility functions for hand landmark processing:
- `extract_hand_landmarks()`: Extract 21 hand landmark coordinates from MediaPipe
- `preprocess_landmarks()`: Normalize landmarks by recentering and scaling
- `calculate_bounding_box()`: Compute bounding box around detected hand

### `model.py`
Model wrapper class `GestureModel`:
- `__init__()`: Load the Random Forest model from pickle file
- `predict()`: Get gesture prediction and confidence score
- `get_all_probabilities()`: Get probabilities for all gesture classes

### `inference.py`
Main execution script with `HandGestureRecognizer` class:
- Real-time webcam processing
- Hand detection and gesture classification
- Live visualization with landmarks and predictions
- FPS counter and UI rendering

## Usage

### Basic Usage
```bash
python inference.py
```

### In Python Code
```python
from inference import HandGestureRecognizer

recognizer = HandGestureRecognizer()
recognizer.run()
```

### Custom Configuration
Edit `config.py` to adjust:
- Detection confidence thresholds
- Display colors
- Preprocessing behavior

## Controls
- **'q'**: Exit the application
- Press Ctrl+C to terminate if stuck

## Features
- Real-time hand landmark detection
- Gesture classification with confidence scores
- Hand identification (Left/Right)
- FPS monitoring
- Modern UI with bounding boxes and panels
- Smooth frame processing with error handling

## Requirements
- OpenCV (`cv2`)
- MediaPipe
- NumPy
- scikit-learn (for model)

## Model Information
The system uses a pre-trained Random Forest model (`models_pkls/Random_Forest.pkl`) trained on hand gesture data. The model expects preprocessed landmark features as input.

## Gesture Labels
The model predicts 26 ASL (American Sign Language) gestures: A-Z

## Error Handling
- Automatically detects if webcam is unavailable
- Displays "No hand detected" when hands are not visible
- Graceful cleanup on exit
