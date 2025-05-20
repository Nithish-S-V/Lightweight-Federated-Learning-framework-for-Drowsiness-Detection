import os
import numpy as np
import requests
import tensorflow as tf
from models.mobilenet_capsnet import MobileNetCapsNet
from encryption.paillier_encryption import PaillierEncryption

class FederatedClient:
    def __init__(self, client_id, server_url, data_path):
        self.client_id = client_id
        self.server_url = server_url
        self.model = MobileNetCapsNet()
        self.base_data_path = data_path
        self.current_round = 0
        
        # Get public key from server
        response = requests.get(f'{self.server_url}/get_public_key')
        self.public_key = response.json()['public_key']
        self.encryption = PaillierEncryption()
        self.encryption.set_public_key(self.public_key)
        
    def load_round_data(self, round_num):
        """Load training data for specific round"""
        round_path = os.path.join(self.base_data_path, f'round_{round_num + 1}')
        if not os.path.exists(round_path):
            raise ValueError(f"Data for round {round_num + 1} not found at {round_path}")
            
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
        
    def get_global_model(self):
        """Get current global model from server"""
        response = requests.get(f'{self.server_url}/get_model')
        data = response.json(),
        
        # Update local model with received weights
        weights = [np.array(w) for w in data['weights']]
        self.model.set_weights(weights)
        self.current_round = data['round']
        
        return data['round']
        
    def train_local_model(self, train_generator):
        """Train local model for one epoch"""
        history = self.model.fit(
            train_generator,
            epochs=1,
            verbose=1
        )
        return history
        
    def submit_weights(self):
        """Send encrypted local model weights to server"""
        weights = self.model.get_weights()
        
        # Encrypt weights
        encrypted_weights = []
        for layer_weights in weights:
            # Flatten weights, encrypt, and reshape back
            shape = layer_weights.shape
            flat_weights = layer_weights.flatten()
            encrypted = [self.encryption.encrypt(float(w)) for w in flat_weights]
            encrypted_weights.append(np.array(encrypted).reshape(shape).tolist())
        
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
    client = FederatedClient(args.client_id, server_url, args.data_path)
    
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
        history = client.train_local_model(train_generator)
        
        # Submit weights to server
        response = client.submit_weights()
        print(f"Server response: {response['message']}")
        
        if response['status'] == 'success':
            print(f"Round {response['round']} completed")

if __name__ == '__main__':
    main()
