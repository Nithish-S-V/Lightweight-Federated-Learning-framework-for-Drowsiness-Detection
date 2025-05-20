import cv2
import numpy as np
import tensorflow as tf
from models.mobilenet_capsnet import MobileNetCapsNet
from utils.data_processor import DataProcessor

class DrowsinessDetector:
    def __init__(self, model_weights_path=None):
        self.model = MobileNetCapsNet()
        self.model.compile_model()
        if model_weights_path:
            self.model.model.load_weights(model_weights_path)
        
        # Initialize face detector
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Initialize variables for drowsiness detection
        self.drowsy_frames = 0
        self.alert_threshold = 10  # Number of consecutive drowsy frames to trigger alert
        
    def preprocess_frame(self, frame):
        """Preprocess frame for model input"""
        # Convert to RGB (model expects RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        if len(faces) == 0:
            return None
            
        # Process the largest face
        (x, y, w, h) = max(faces, key=lambda face: face[2] * face[3])
        
        # Add margin to face detection
        margin = int(0.2 * w)
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(frame.shape[1] - x, w + 2 * margin)
        h = min(frame.shape[0] - y, h + 2 * margin)
        
        # Crop and preprocess face region
        face_img = frame_rgb[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (224, 224))
        face_img = face_img / 127.5 - 1.0
        
        return {
            'processed_image': face_img,
            'face_coords': (x, y, w, h)
        }
        
    def detect(self, frame):
        """Detect drowsiness in a frame"""
        # Preprocess frame
        result = self.preprocess_frame(frame)
        if result is None:
            return frame, "No face detected", "alert"
            
        face_img = result['processed_image']
        (x, y, w, h) = result['face_coords']
        
        # Make prediction
        prediction = self.model.model.predict(
            np.expand_dims(face_img, axis=0),
            verbose=0
        )[0]
        
        # Get drowsiness probability (index 1 is drowsy class)
        drowsy_prob = prediction[1]
        
        # Update drowsy frames counter
        if drowsy_prob > 0.5:
            self.drowsy_frames += 1
        else:
            self.drowsy_frames = 0
            
        # Determine status and color
        if self.drowsy_frames >= self.alert_threshold:
            status = "DROWSY!"
            color = (0, 0, 255)  # Red
            state = "drowsy"
        else:
            status = "Alert"
            color = (0, 255, 0)  # Green
            state = "alert"
            
        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        
        # Add text with drowsiness probability
        text = f"{status} ({drowsy_prob:.2f})"
        cv2.putText(frame, text, (x, y-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                   
        return frame, text, state
        
    def run_webcam(self):
        """Run real-time detection using webcam"""
        cap = cv2.VideoCapture(0)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Detect drowsiness
            frame, text, state = self.detect(frame)
            
            # Display the frame
            cv2.imshow('Drowsiness Detection', frame)
            
            # Break loop on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()
