"""Utility functions for hand landmark processing."""

import numpy as np
from config import PREPROCESSING_CONFIG


def extract_hand_landmarks(hand_landmarks):
    """
    Extract landmark coordinates from MediaPipe hand detection.
    
    Args:
        hand_landmarks: MediaPipe hand landmarks object
        
    Returns:
        np.ndarray: Flattened array of landmark coordinates (21 landmarks × 3 coords)
    """
    landmarks = []
    for landmark in hand_landmarks.landmark:
        landmarks.extend([landmark.x, landmark.y, landmark.z])
    
    return np.array(landmarks).flatten()


def preprocess_landmarks(landmarks_array):
    """
    Preprocess hand landmarks by recentering and normalizing.
    
    Args:
        landmarks_array: Raw landmark coordinates array
        
    Returns:
        np.ndarray: Preprocessed landmark array
    """
    processed_array = landmarks_array.copy()
    
    # Recenter hand to wrist (x0, y0)
    if PREPROCESSING_CONFIG['recenter_to_wrist']:
        x0 = processed_array[0]
        y0 = processed_array[1]
        processed_array[0::3] -= x0
        processed_array[1::3] -= y0
    
    # Normalize to middle finger tip (x12, y12)
    if PREPROCESSING_CONFIG['normalize_to_middle_finger']:
        x12 = processed_array[36]  # 12*3 --> x12
        y12 = processed_array[37]  # 12*3+1 --> y12
        epsilon = PREPROCESSING_CONFIG['epsilon']
        processed_array[0::3] /= (x12 + epsilon)
        processed_array[1::3] /= (y12 + epsilon)
    
    return processed_array


def calculate_bounding_box(hand_landmarks, frame_width, frame_height, padding=20):
    """
    Calculate bounding box coordinates for hand.
    
    Args:
        hand_landmarks: MediaPipe hand landmarks object
        frame_width: Video frame width in pixels
        frame_height: Video frame height in pixels
        padding: Padding around the hand in pixels
        
    Returns:
        tuple: (x_min, y_min, x_max, y_max) bounding box coordinates
    """
    x_coords = [lm.x for lm in hand_landmarks.landmark]
    y_coords = [lm.y for lm in hand_landmarks.landmark]
    
    # Convert normalized coordinates (0.0 to 1.0) to pixel coordinates
    x_min = int(min(x_coords) * frame_width)
    x_max = int(max(x_coords) * frame_width)
    y_min = int(min(y_coords) * frame_height)
    y_max = int(max(y_coords) * frame_height)
    
    # Add padding
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(frame_width, x_max + padding)
    y_max = min(frame_height, y_max + padding)
    
    return x_min, y_min, x_max, y_max
