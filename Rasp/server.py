import os
import flask
import numpy as np
from flask import Flask, request, jsonify
from student_model import LightweightCNN
from knowledge_transfer import KnowledgeDistillation
from lightweight_encryption import LightweightEncryption
import tensorflow as tf
app = Flask(__name__)



#<---------------------------------------------------------------->
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras.saving import register_keras_serializable
@register_keras_serializable(package="Custom")
class CapsuleLayer(Layer):
    def __init__(self, num_capsule, dim_capsule, routings=3, **kwargs):
        super(CapsuleLayer, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        
    def build(self, input_shape):
        self.input_num_capsule = input_shape[1]
        self.input_dim_capsule = input_shape[2]
        
        self.W = self.add_weight(
            shape=[self.num_capsule, self.input_num_capsule,
                   self.dim_capsule, self.input_dim_capsule],
            initializer='glorot_uniform',
            name='W')
            
        self.built = True
        
    def call(self, inputs):
        # inputs.shape = [None, input_num_capsule, input_dim_capsule]
        # Expand dims to [None, input_num_capsule, 1, 1, input_dim_capsule]
        inputs_expand = K.expand_dims(K.expand_dims(inputs, 2), 2)
        
        # Replicate num_capsule dimension to prepare being multiplied by W
        # inputs_tiled.shape = [None, input_num_capsule, num_capsule, 1, input_dim_capsule]
        inputs_tiled = K.tile(inputs_expand, [1, 1, self.num_capsule, 1, 1])
        
        # Compute `inputs * W` by scanning inputs_tiled on dimension 0
        # W.shape = [num_capsule, input_num_capsule, dim_capsule, input_dim_capsule]
        # x.shape = [num_capsule, input_num_capsule, dim_capsule]
        inputs_hat = tf.scan(lambda x, y: K.batch_dot(y, self.W, [3, 3]),
                           elems=inputs_tiled,
                           initializer=K.zeros([self.input_num_capsule,
                                              self.num_capsule,
                                              1,
                                              self.dim_capsule]))
                                              
        # Routing algorithm
        b = tf.zeros(shape=[K.shape(inputs_hat)[0],
                           self.input_num_capsule,
                           self.num_capsule, 1])
                           
        for i in range(self.routings):
            c = tf.nn.softmax(b, axis=2)
            outputs = squash(K.sum(c * inputs_hat, axis=1, keepdims=True))
            
            if i < self.routings - 1:
                b += K.sum(inputs_hat * outputs, axis=-1, keepdims=True)
                
        return K.reshape(outputs, [-1, self.num_capsule, self.dim_capsule])
        
    def compute_output_shape(self, input_shape):
        return (None, self.num_capsule, self.dim_capsule)

def squash(vectors, axis=-1):
    """Squashing function for capsule values"""
    s_squared_norm = K.sum(K.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / K.sqrt(s_squared_norm + K.epsilon())
    return scale * vectors

#<---------------------------------------------------------------->



class LightweightServer:
    def __init__(self):
        # Initialize models
        self.global_model = LightweightCNN()
        self.teacher_model = None  # Load teacher model if available
        self.knowledge_transfer = KnowledgeDistillation()
        
        # Load teacher model if available
        if os.path.exists('D:/Major Project/Main/models/initial_model.h5'):
            self.teacher_model = tf.keras.models.load_model('D:/Major Project/Main/models/initial_model.h5', custom_objects={"CapsuleLayer": CapsuleLayer})
        
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
