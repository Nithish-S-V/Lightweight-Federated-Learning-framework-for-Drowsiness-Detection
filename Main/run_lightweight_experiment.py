import os
import sys
import json
import numpy as np
import tensorflow as tf
from lightweight_fl.models.student_model import LightweightCNN
from lightweight_fl.encryption.lightweight_encryption import LightweightEncryption

class LightweightServer:
    def __init__(self):
        self.model = LightweightCNN()
        self.encryption = LightweightEncryption()
        self.received_weights = []
        self.current_round = 0
        self.total_rounds = 5
        
    def aggregate_weights(self):
        """Average the encrypted weights from clients"""
        averaged_weights = []
        num_clients = len(self.received_weights)
        
        for weights_list_tuple in zip(*self.received_weights):
            # Decrypt weights before averaging
            decrypted_weights = []
            for weights in zip(*weights_list_tuple):
                decrypted = [self.encryption.decrypt(w) for w in weights]
                decrypted_weights.append(np.array(decrypted))
            
            # Calculate average
            averaged = np.mean(decrypted_weights, axis=0)
            averaged_weights.append(averaged)
            
        return averaged_weights
        
    def save_model(self):
        """Save the current global model"""
        save_dir = 'checkpoints/lightweight'
        os.makedirs(save_dir, exist_ok=True)
        self.model.save_weights(
            f'{save_dir}/global_model_round_{self.current_round}.h5'
        )

class LightweightClient:
    def __init__(self, client_id, data_path):
        self.client_id = client_id
        self.model = LightweightCNN()
        self.base_data_path = data_path
        self.current_round = 0
        self.encryption = LightweightEncryption()
        
    def load_round_data(self, round_num):
        """Load training data for specific round"""
        round_path = os.path.join(
            self.base_data_path, 
            f'round_{round_num + 1}'
        )
        if not os.path.exists(round_path):
            raise ValueError(
                f"Data for round {round_num + 1} not found at {round_path}"
            )
        
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            validation_split=0.2
        )
        
        train_generator = train_datagen.flow_from_directory(
            round_path,
            target_size=(224, 224),
            batch_size=32,
            class_mode='binary'
        )
        
        return train_generator
        
    def train_local_model(self, train_generator):
        """Train local model for one epoch"""
        self.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        history = self.model.fit(
            train_generator,
            epochs=1,
            verbose=1
        )
        return history

def run_federated_learning():
    """Run federated learning with lightweight model"""
    # Initialize server and clients
    server = LightweightServer()
    clients = [
        LightweightClient(0, "data/non_iid_data/client_0"),
        LightweightClient(1, "data/non_iid_data/client_1")
    ]
    
    # Training loop
    for round_num in range(server.total_rounds):
        print(f"\nStarting round {round_num + 1}")
        
        # Train each client
        for client in clients:
            print(f"\nTraining Client {client.client_id}")
            
            # Get current global weights
            client.model.set_weights(server.model.get_weights())
            
            # Load round data and train
            train_generator = client.load_round_data(round_num)
            history = client.train_local_model(train_generator)
            
            # Get and encrypt weights
            weights = client.model.get_weights()
            encrypted_weights = []
            for layer_weights in weights:
                shape = layer_weights.shape
                flat_weights = layer_weights.flatten()
                encrypted = [client.encryption.encrypt(float(w)) for w in flat_weights]
                encrypted_weights.append(np.array(encrypted).reshape(shape).tolist())
            
            # Submit weights to server
            server.received_weights.append(encrypted_weights)
            
        # Server aggregates weights
        new_weights = server.aggregate_weights()
        server.model.set_weights(new_weights)
        
        # Save model and clear received weights
        server.save_model()
        server.received_weights = []
        server.current_round += 1
        
        print(f"Round {round_num + 1} completed")

if __name__ == "__main__":
    print("Starting lightweight federated learning...")
    run_federated_learning()
