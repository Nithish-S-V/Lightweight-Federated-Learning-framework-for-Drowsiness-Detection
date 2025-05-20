import os
import numpy as np
import requests
import tensorflow as tf
from models.student_model import LightweightCNN
from encryption.lightweight_encryption import LightweightEncryption

class LightweightClient:
    def __init__(self, client_id, server_url, data_path):
        self.client_id = client_id
        self.server_url = server_url
        self.model = LightweightCNN()
        self.base_data_path = data_path
        self.current_round = 0
        
        # Setup encryption
        self.encryption = LightweightEncryption()
        self.setup_encryption()
        
        # Optimizer with reduced memory usage
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07
        )
        
    def setup_encryption(self):
        """Get encryption key from server"""
        response = requests.get(f'{self.server_url}/get_encryption_key')
        key = response.json()['key'].encode('utf-8')
        self.encryption.set_key(key)
        
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
        
        # Memory efficient data loading
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
            batch_size=16,  # Smaller batch size for Raspberry Pi
            class_mode='binary'
        )
        
        return train_generator
        
    def get_global_model(self):
        """Get current global model from server"""
        response = requests.get(f'{self.server_url}/get_model')
        data = response.json()
        
        # Update local model with received weights
        weights = [np.array(w) for w in data['weights']]
        self.model.set_weights(weights)
        self.current_round = data['round']
        
        return data['round']
        
    def train_local_model(self, train_generator):
        """Train local model with memory-efficient approach"""
        steps_per_epoch = len(train_generator)
        
        for step in range(steps_per_epoch):
            # Get batch
            x_batch, y_batch = next(train_generator)
            
            with tf.GradientTape() as tape:
                # Forward pass
                y_pred = self.model(x_batch, training=True)
                # Calculate loss
                loss = tf.keras.losses.binary_crossentropy(y_batch, y_pred)
                loss = tf.reduce_mean(loss)
            
            # Calculate gradients
            grads = tape.gradient(loss, self.model.trainable_variables)
            # Apply gradients
            self.optimizer.apply_gradients(
                zip(grads, self.model.trainable_variables)
            )
            
            if step % 10 == 0:
                print(f"Step {step}/{steps_per_epoch}, Loss: {loss:.4f}")
        
    def submit_weights(self):
        """Send encrypted weights to server"""
        weights = self.model.get_weights()
        encrypted_weights = self.encryption.encrypt_weights(weights)
        
        response = requests.post(
            f'{self.server_url}/submit_weights',
            json={
                'weights': encrypted_weights,
                'client_id': self.client_id
            }
        )
        return response.json()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--client_id', type=int, required=True)
    parser.add_argument('--server_host', type=str, default='localhost')
    parser.add_argument('--server_port', type=int, default=5000)
    parser.add_argument('--data_path', type=str, required=True)
    args = parser.parse_args()
    
    server_url = f'http://{args.server_host}:{args.server_port}'
    client = LightweightClient(args.client_id, server_url, args.data_path)
    
    while True:
        # Get current round from server
        round_num = client.get_global_model()
        
        if round_num >= 5:  # Maximum 5 rounds
            print("Training completed!")
            break
            
        print(f"\nStarting training round {round_num + 1}")
        
        # Load round-specific data
        try:
            train_generator = client.load_round_data(round_num)
        except ValueError as e:
            print(f"Error: {e}")
            break
            
        # Train local model
        client.train_local_model(train_generator)
        
        # Submit weights to server
        response = client.submit_weights()
        print(f"Server response: {response['message']}")
        
        if response['status'] == 'success':
            print(f"Round {response['round']} completed")

if __name__ == '__main__':
    main()
