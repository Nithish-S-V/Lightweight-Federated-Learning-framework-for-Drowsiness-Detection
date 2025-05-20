import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import cv2
import numpy as np
import tensorflow as tf
from standard_fl.models.mobilenet_capsnet import MobileNetCapsNet
from standard_fl.encryption.paillier_manager import PaillierManager
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class EdgeNode:
    def __init__(self, node_id, base_dir):
        """Initialize edge node
        
        Args:
            node_id (str): Client ID (e.g., 'client_1', 'client_2')
            base_dir (str): Base directory containing round-specific data folders
        """
        self.node_id = node_id
        self.base_dir = base_dir
        self.model = MobileNetCapsNet()
        self.model.compile_model()
        self.paillier = PaillierManager()
        self.current_round = 1
        
    def setup_data_generators(self, round_num, batch_size=32):
        """Setup data generators for specific round
        
        Args:
            round_num (int): Current round number
            batch_size (int): Batch size for training
        """
        round_dir = os.path.join(self.base_dir, f"round_{round_num}")
        if not os.path.exists(round_dir):
            raise ValueError(f"Round directory not found: {round_dir}")
            
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            validation_split=0.2
        )
        
        train_generator = train_datagen.flow_from_directory(
            round_dir,
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode='binary',
            subset='training'
        )
        
        valid_generator = train_datagen.flow_from_directory(
            round_dir,
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode='binary',
            subset='validation'
        )
        
        return train_generator, valid_generator
        
    def train_round(self, round_num, epochs=10, batch_size=32):
        """Train model on data for specific round
        
        Args:
            round_num (int): Round number to train on
            epochs (int): Number of epochs to train
            batch_size (int): Batch size for training
            
        Returns:
            dict: Training history and metrics
        """
        print(f"\nTraining {self.node_id} - Round {round_num}")
        
        # Setup data generators for this round
        train_generator, valid_generator = self.setup_data_generators(round_num, batch_size)
        
        # Calculate class weights
        n_drowsy = len(os.listdir(os.path.join(self.base_dir, f"round_{round_num}", 'drowsy')))
        n_non_drowsy = len(os.listdir(os.path.join(self.base_dir, f"round_{round_num}", 'non_drowsy')))
        total_samples = n_drowsy + n_non_drowsy
        
        weight_for_0 = (1 / n_non_drowsy) * (total_samples / 2.0)
        weight_for_1 = (1 / n_drowsy) * (total_samples / 2.0)
        class_weights = {0: weight_for_0, 1: weight_for_1}
        
        print(f"Training on {train_generator.samples} samples")
        print(f"Validating on {valid_generator.samples} samples")
        
        # Train the model
        history = self.model.model.fit(
            train_generator,
            validation_data=valid_generator,
            epochs=epochs,
            class_weight=class_weights,
            verbose=1
        )
        
        # Evaluate
        metrics = self.evaluate_round(round_num)
        
        return {
            'history': history.history,
            'metrics': metrics
        }
        
    def evaluate_round(self, round_num):
        """Evaluate model on validation data for specific round"""
        _, valid_generator = self.setup_data_generators(round_num)
        
        scores = self.model.model.evaluate(valid_generator)
        metrics = dict(zip(self.model.model.metrics_names, scores))
        
        print(f"\nEvaluation results for {self.node_id} - Round {round_num}:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
            
        return metrics
        
    def get_encrypted_parameters(self):
        """Get encrypted model parameters"""
        params = self.model.get_classification_layer_parameters()
        return self.paillier.encrypt_parameters(params)
        
    def update_model(self, encrypted_global_params):
        """Update model with new global parameters"""
        decrypted_params = self.paillier.decrypt_parameters(encrypted_global_params)
        self.model.set_classification_layer_parameters(decrypted_params)
        
    def detect_drowsiness(self, frame):
        """Real-time drowsiness detection"""
        frame = cv2.resize(frame, (224, 224))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame / 255.0
        frame = np.expand_dims(frame, axis=0)
        
        predictions = self.model.model.predict(frame)
        return predictions[0]
        
    def get_dataset_size(self):
        """Get the total number of training samples for the current round"""
        round_dir = os.path.join(self.base_dir, f"round_{self.current_round}")
        n_drowsy = len(os.listdir(os.path.join(round_dir, 'drowsy')))
        n_non_drowsy = len(os.listdir(os.path.join(round_dir, 'non_drowsy')))
        return n_drowsy + n_non_drowsy
