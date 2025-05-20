import os
import flask
import numpy as np
from flask import Flask, request, jsonify
from models.mobilenet_capsnet import MobileNetCapsNet
import tensorflow as tf
from encryption.paillier_encryption import PaillierEncryption

app = Flask(__name__)

class FederatedServer:
    def __init__(self):
        self.global_model = MobileNetCapsNet()
        # Initialize with pre-trained weights if available
        if os.path.exists('checkpoints/initial_model.h5'):
            self.global_model.load_weights('checkpoints/initial_model.h5')
            
        self.received_weights = []
        self.client_status = {}  # Track which clients submitted weights
        self.current_round = 0
        self.total_rounds = 5
        self.num_clients = 2
        
        # Initialize encryption
        self.encryption = PaillierEncryption()
        self.public_key = self.encryption.get_public_key()
        
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
        os.makedirs('checkpoints', exist_ok=True)
        self.global_model.save_weights(f'checkpoints/global_model_round_{self.current_round}.h5')

server = FederatedServer()

@app.route('/get_public_key', methods=['GET'])
def get_public_key():
    """Send public key to clients"""
    return jsonify({
        'public_key': server.public_key
    })

@app.route('/get_model', methods=['GET'])
def get_model():
    """Send current global model weights to clients"""
    weights = server.global_model.get_weights()
    return jsonify({
        'weights': [w.tolist() for w in weights],
        'round': server.current_round
    })

@app.route('/submit_weights', methods=['POST'])
def submit_weights():
    """Receive encrypted weights from clients and aggregate if all clients submitted"""
    data = request.get_json()
    client_id = data['client_id']
    encrypted_weights = data['weights']
    
    # Convert encrypted weights back to numpy arrays
    client_weights = [np.array(w) for w in encrypted_weights]
    
    # Store weights and update client status
    if client_id not in server.client_status or server.client_status[client_id] != server.current_round:
        server.received_weights.append(client_weights)
        server.client_status[client_id] = server.current_round
    
    # If all clients submitted weights for current round
    if len([c for c, r in server.client_status.items() if r == server.current_round]) == server.num_clients:
        # Aggregate weights
        new_weights = server.aggregate_weights()
        server.global_model.set_weights(new_weights)
        
        # Save model
        server.save_model()
        
        # Clear received weights and increment round
        server.received_weights = []
        server.current_round += 1
        
        return jsonify({
            'status': 'success',
            'message': f'Round {server.current_round} completed',
            'round': server.current_round
        })
    
    return jsonify({
        'status': 'waiting',
        'message': f'Received encrypted weights from client {client_id}. Waiting for other clients.',
        'round': server.current_round
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
