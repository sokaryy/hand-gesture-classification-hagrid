"""Model loading and prediction utilities."""

import pickle
import numpy as np
from collections import deque, Counter
from config import MODEL_PATH


class GestureModel:
    """Wrapper class for loading and using the gesture recognition model."""
    
    def __init__(self, model_path=MODEL_PATH):
        """
        Initialize and load the Random Forest model.
        
        Args:
            model_path: Path to the pickled model file
        """
        self.model = self._load_model(model_path)
        self.prediction_history = deque(maxlen=3)
    
    @staticmethod
    def _load_model(model_path):
        """
        Load the Random Forest model from pickle file.
        
        Args:
            model_path: Path to the model pickle file
            
        Returns:
            Loaded model object
            
        Raises:
            FileNotFoundError: If model file doesn't exist
        """
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            print(f"✓ Model loaded successfully from {model_path}")
            return model
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file not found at {model_path}")
        except Exception as e:
            raise RuntimeError(f"Error loading model: {str(e)}")
    
    def predict(self, landmarks_array):
        """
        Predict gesture from landmark array.
        
        Args:
            landmarks_array: Preprocessed landmark coordinates
            
        Returns:
            tuple: (gesture_label, confidence_score)
        """
        probabilities = self.model.predict_proba([landmarks_array])[0]
        best_class_index = np.argmax(probabilities)

        current_confidence = probabilities[best_class_index]
        raw_prediction = self.model.classes_[best_class_index]
        self.prediction_history.append(raw_prediction)
        final_prediction = Counter(self.prediction_history).most_common(1)[0][0] # get mode of the queue (most common value in last 3 predictions)
        return final_prediction, current_confidence
    
    def get_all_probabilities(self, landmarks_array):
        """
        Get prediction probabilities for all gesture classes.
        
        Args:
            landmarks_array: Preprocessed landmark coordinates
            
        Returns:
            dict: Mapping of gesture labels to probabilities
        """
        probabilities = self.model.predict_proba([landmarks_array])[0]
        return {label: prob for label, prob in zip(self.model.classes_, probabilities)}
