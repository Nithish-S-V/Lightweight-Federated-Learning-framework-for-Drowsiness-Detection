import os
import flask
import numpy as np
from flask import Flask, request, jsonify
from models.student_model import LightweightCNN
from models.knowledge_transfer import KnowledgeDistillation
from encryption.lightweight_encryption import LightweightEncryption
import tensorflow as tf

app = Flask(__name__)

class LightweightServer:
    def __init__(self):
        # Initialize models
        self.global_model = LightweightCNN()
        self.teacher_model = None  # Load teacher model if available
        self.knowledge_transfer = KnowledgeDistillation()
        
        # Load teacher model if available
        if os.path.exists('D:/Major Project/Main/models/initial_model.h5'):
            self.teacher_model = tf.keras.models.load_model('D:/Major Project/Main/models/initial_model.h5')
        
        # Initialize encryption
        self.encryption = LightweightEncryption()
        self.key = self.encryption.generate_key()
        
        # Training state
        self.received_weights = []
        self.client_status = {}
        self.current_round = 0
        self.total_rounds = 5
        self.num_clients = 2
        
    def aggregate_weights(self):
        """Average the received weights from clients"""
        averaged_weights = []
        weights_list = [
            self.encryption.decrypt_weights(w) 
            for w in self.received_weights
        ]
        
        for weights_list_tuple in zip(*weights_list):
            averaged_weights.append(
                np.array([np.array(weights).mean(axis=0) 
                         for weights in zip(*weights_list_tuple)])
            )
        
        return averaged_weights
        
    def save_model(self):
        """Save the current global model"""
        os.makedirs('checkpoints', exist_ok=True)
        self.global_model.save_weights(
            f'checkpoints/lightweight_model_round_{self.current_round}.h5'
        )

server = LightweightServer()

@app.route('/get_encryption_key', methods=['GET'])
def get_encryption_key():
    """Send encryption key to clients"""
    return jsonify({
        'key': server.key.decode('utf-8')
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
    """Receive encrypted weights from clients"""
    data = request.get_json()
    client_id = data['client_id']
    encrypted_weights = data['weights']
    
    # Store weights and update client status
    if (client_id not in server.client_status or 
        server.client_status[client_id] != server.current_round):
        server.received_weights.append(encrypted_weights)
        server.client_status[client_id] = server.current_round
    
    # If all clients submitted weights for current round
    if len([c for c, r in server.client_status.items() 
            if r == server.current_round]) == server.num_clients:
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
        'message': f'Received weights from client {client_id}. Waiting for others.',
        'round': server.current_round
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
