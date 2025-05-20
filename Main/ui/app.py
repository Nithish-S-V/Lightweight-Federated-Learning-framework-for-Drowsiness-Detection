import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QPushButton, QLabel, QFileDialog)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from inference.realtime_detector import DrowsinessDetector

class DrowsinessDetectionUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Drowsiness Detection System")
        self.detector = DrowsinessDetector()
        
        # Initialize camera
        self.camera = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        
        self.init_ui()
        
    def init_ui(self):
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Create display area
        self.display_label = QLabel()
        self.display_label.setAlignment(Qt.AlignCenter)
        self.display_label.setMinimumSize(640, 480)
        layout.addWidget(self.display_label)
        
        # Create status label
        self.status_label = QLabel("Status: Not Running")
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)
        
        # Create buttons
        button_layout = QHBoxLayout()
        
        self.start_button = QPushButton("Start Webcam")
        self.start_button.clicked.connect(self.toggle_camera)
        button_layout.addWidget(self.start_button)
        
        self.image_button = QPushButton("Load Image")
        self.image_button.clicked.connect(self.load_image)
        button_layout.addWidget(self.image_button)
        
        layout.addLayout(button_layout)
        
        # Set window size
        self.setMinimumSize(800, 600)
        
    def toggle_camera(self):
        if self.timer.isActive():
            self.timer.stop()
            self.camera.release()
            self.camera = None
            self.start_button.setText("Start Webcam")
            self.status_label.setText("Status: Stopped")
        else:
            self.camera = cv2.VideoCapture(0)
            if self.camera.isOpened():
                self.timer.start(30)  # Update every 30ms
                self.start_button.setText("Stop Webcam")
                self.status_label.setText("Status: Running")
            else:
                self.status_label.setText("Status: Error - Could not open camera")
                
    def update_frame(self):
        if self.camera is None:
            return
            
        ret, frame = self.camera.read()
        if ret:
            # Process frame
            frame, text, state = self.detector.detect(frame)
            
            # Update status
            self.status_label.setText(f"Status: {text}")
            
            # Convert frame to Qt format
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            
            # Scale image to fit label while maintaining aspect ratio
            scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
                self.display_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            
            self.display_label.setPixmap(scaled_pixmap)
            
    def load_image(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg)"
        )
        
        if file_name:
            # Read and process image
            frame = cv2.imread(file_name)
            if frame is not None:
                frame, text, state = self.detector.detect(frame)
                
                # Update status
                self.status_label.setText(f"Status: {text}")
                
                # Convert and display image
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                
                scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
                    self.display_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
                
                self.display_label.setPixmap(scaled_pixmap)
            else:
                self.status_label.setText("Status: Error loading image")
                
    def closeEvent(self, event):
        if self.camera is not None:
            self.camera.release()
        event.accept()

def main():
    app = QApplication(sys.argv)
    window = DrowsinessDetectionUI()
    window.show()
    sys.exit(app.exec_())
