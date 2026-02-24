"""Main inference script for real-time hand gesture recognition."""

import cv2
import mediapipe as mp
import warnings
import time

import numpy as np

from config import MEDIAPIPE_CONFIG, COLOR_CONFIG, UI_CONFIG
from model import GestureModel
from utils import extract_hand_landmarks, preprocess_landmarks, calculate_bounding_box

# Suppress sklearn warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


class HandGestureRecognizer:
    """Real-time hand gesture recognition system using webcam and MediaPipe."""
    
    def __init__(self):
        """Initialize the gesture recognizer with MediaPipe and model."""
        self.setup_mediapipe()
        self.model = GestureModel()
        self.pTime = 0
        self.frame_counter = 0
        self.inference_interval = 3 # Run the ML model every 3 frames
    
    def setup_mediapipe(self):
        """Initialize MediaPipe hand detector."""
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(**MEDIAPIPE_CONFIG)
        print("✓ MediaPipe initialized")
    
    def process_frame(self, frame):
        """
        Process a single video frame.
        
        Args:
            frame: Input video frame (BGR)
            
        Returns:
            tuple: (processed_frame, detection_results)
        """
        # Flip frame for selfie view
        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape
        
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect hand landmarks
        results = self.hands.process(rgb_frame)
        
        return frame, results, h, w
    
    def draw_hand_detection(self, frame, hand_landmarks, x_min, y_min, x_max, y_max, 
                           gesture_label, confidence, handedness):
        """
        Draw hand landmarks, bounding box, and predictions on frame.
        
        Args:
            frame: Video frame to draw on
            hand_landmarks: MediaPipe hand landmarks
            x_min, y_min, x_max, y_max: Bounding box coordinates
            gesture_label: Predicted gesture label
            confidence: Confidence score
            handedness: Hand classification (left/right)
        """
        # Draw hand landmarks
        self.mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            self.mp_hands.HAND_CONNECTIONS,
            self.mp_drawing.DrawingSpec(
                color=COLOR_CONFIG['text'], 
                thickness=2, 
                circle_radius=2
            ),
            self.mp_drawing.DrawingSpec(
                color=COLOR_CONFIG['accent'], 
                thickness=2
            )
        )
        
        # Draw bounding box
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), 
                     COLOR_CONFIG['accent'], UI_CONFIG['thickness'])
        
        # Top panel with gesture and confidence
        panel_height = UI_CONFIG['panel_height']
        cv2.rectangle(frame, (x_min, y_min - panel_height), (x_max, y_min), 
                     COLOR_CONFIG['accent'], cv2.FILLED)
        
        text = f"{gesture_label} ({confidence:.2f})"
        cv2.putText(
            frame, text, (x_min + 5, y_min - 10),
            cv2.FONT_HERSHEY_DUPLEX, 0.6, COLOR_CONFIG['text'], 1, cv2.LINE_AA
        )
        
        # Bottom panel with handedness
        cv2.rectangle(frame, (x_min, y_max), (x_max, y_max + 25), 
                     COLOR_CONFIG['background'], cv2.FILLED)
        
        hand_text = f"{handedness.classification[0].label} Hand"
        cv2.putText(
            frame, hand_text, (x_min + 5, y_max + 18),
            cv2.FONT_HERSHEY_DUPLEX, 0.5, COLOR_CONFIG['text'], 1, cv2.LINE_AA
        )
    
    def draw_no_detection(self, frame):
        """Draw 'No hand detected' message on frame."""
        cv2.rectangle(frame, (10, 10), (250, 50), 
                     COLOR_CONFIG['background'], cv2.FILLED)
        cv2.putText(
            frame, "No hand detected", (20, 35),
            cv2.FONT_HERSHEY_DUPLEX, 0.6, COLOR_CONFIG['alert'], 1, cv2.LINE_AA
        )
    
    def draw_fps(self, frame, fps):
        """Draw FPS counter on top-right corner."""
        h, w = frame.shape[:2]
        cv2.putText(
            frame, f"FPS: {int(fps)}", (w - 120, 40),
            cv2.FONT_HERSHEY_DUPLEX, 0.7, COLOR_CONFIG['success'], 1, cv2.LINE_AA
        )
    
    def calculate_fps(self, current_time):
        """
        Calculate FPS and update timing.
        
        Args:
            current_time: Current time value
            
        Returns:
            float: Current FPS
        """
        fps = 1 / (current_time - self.pTime) if (current_time - self.pTime) > 0 else 0
        self.pTime = current_time
        return fps
    
    def run(self):
        """Main inference loop for real-time gesture recognition."""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("✗ Error: Cannot access webcam")
            return
        
        print("✓ Webcam accessed successfully")
        print("Starting gesture recognition. Press 'q' to exit.")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                frame, results, h, w = self.process_frame(frame)
                
                # Calculate current FPS
                cTime = time.time()
                fps = self.calculate_fps(cTime)
                
                # Process detections
                if results.multi_hand_landmarks and results.multi_handedness:
                    for hand_landmarks, handedness in zip(
                        results.multi_hand_landmarks, 
                        results.multi_handedness
                    ):
                        if self.frame_counter % self.inference_interval == 0:
                        # Extract and preprocess landmarks
                            landmarks_array = extract_hand_landmarks(hand_landmarks)
                            landmarks_array = preprocess_landmarks(landmarks_array)
                            
                            # Make prediction
                            gesture_label, confidence = self.model.predict(landmarks_array)
                        
                        # Calculate bounding box
                        bbox = calculate_bounding_box(
                            hand_landmarks, w, h, 
                            padding=UI_CONFIG['padding']
                        )
                        x_min, y_min, x_max, y_max = bbox
                        
                        # Draw results
                        self.draw_hand_detection(
                            frame, hand_landmarks, x_min, y_min, x_max, y_max,
                            gesture_label, confidence, handedness
                        )
                        self.frame_counter += 1
                else:
                    self.draw_no_detection(frame)
                
                # Draw FPS
                self.draw_fps(frame, fps)
                
                # Display frame
                cv2.imshow('Hand Gesture Recognition', frame)
                
                # Exit on 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        finally:
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
            self.hands.close()
            print("\n✓ Application closed")


def main():
    """Entry point for the gesture recognition application."""
    recognizer = HandGestureRecognizer()
    recognizer.run()


if __name__ == "__main__":
    main()
