"""Configuration and constants for hand gesture recognition."""

from pathlib import Path
import os

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
MODEL_PATH = os.path.join(PROJECT_ROOT, 'models_pkls', 'Random_Forest.pkl')

# MediaPipe Hand Detection Settings
MEDIAPIPE_CONFIG = {
    'static_image_mode': False,
    'max_num_hands': 1,
    'min_detection_confidence': 0.5,
    'min_tracking_confidence': 0.5
}

# Display Colors (BGR format for OpenCV)
COLOR_CONFIG = {
    'accent': (255, 144, 30),      # Sleek Blue
    'background': (40, 40, 40),    # Dark Gray
    'text': (255, 255, 255),       # White
    'alert': (0, 0, 255),          # Red for no detection
    'success': (0, 255, 0)         # Green
}

# UI Settings
UI_CONFIG = {
    'panel_height': 35,
    'padding': 20,
    'font': 'FONT_HERSHEY_DUPLEX',
    'thickness': 2
}

# Preprocessing Settings
PREPROCESSING_CONFIG = {
    'recenter_to_wrist': True,
    'normalize_to_middle_finger': True,
    'epsilon': 1e-6
}
