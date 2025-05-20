import socket
import struct
import zlib
import tempfile
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import tenseal as ts
from sklearn.metrics import classification_report
from tensorflow.keras.saving import register_keras_serializable
from tensorflow.keras import layers, initializers, backend as K
from sklearn.metrics import confusion_matrix, accuracy_score
import base64
import hashlib
from cryptography.fernet import Fernet

def setup_ckks_context():
    """Setup CKKS encryption context for homomorphic encryption."""
    context = ts.context(
        scheme=ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60]
    )
    context.global_scale = 2 ** 40
    context.generate_galois_keys()
    return context

def encrypt_weights(model, context):
    """Encrypt model weights using CKKS encryption."""
    encrypted_weights = []
    for layer in model.get_weights():
        flattened = layer.flatten()
        encrypted_weights.append(ts.ckks_vector(context, flattened))
    return encrypted_weights
import numpy as np

import numpy as np

def decrypt_weights(encrypted_weights, context, model):
    """Decrypt encrypted weights and reshape them to match the model's structure."""
    decrypted_weights = []
    model_shapes = [w.shape for w in model.get_weights()]  # Get original weight shapes

    for i, (enc_vec, shape) in enumerate(zip(encrypted_weights, model_shapes)):
        decrypted_layer = np.array(enc_vec.decrypt())  # Decrypt to NumPy array
        decrypted_layer = decrypted_layer.reshape(shape)  # Reshape to original shape
        decrypted_weights.append(decrypted_layer)

    print(f"Number of decrypted weights: {len(decrypted_weights)}")
    
    return decrypted_weights



# Custom Capsule Network Components
@register_keras_serializable(package="Custom")
class Length(layers.Layer):
    def call(self, inputs, **kwargs):
        return K.sqrt(K.sum(K.square(inputs), -1) + K.epsilon())
    
    def compute_output_shape(self, input_shape):
        return input_shape[:-1]
    
    def get_config(self):
        return super(Length, self).get_config()

@tf.keras.saving.register_keras_serializable(package="Custom")
class CapsuleLayer(layers.Layer):
    def __init__(self, num_capsule, dim_capsule, routings=3, **kwargs):
        super().__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings

    def build(self, input_shape):
        self.input_num_capsule = input_shape[1]
        self.input_dim_capsule = input_shape[2]
        
        self.W = self.add_weight(
            shape=[1, self.input_num_capsule, self.num_capsule, self.dim_capsule, self.input_dim_capsule],
            initializer=initializers.glorot_uniform(),
            name='W'
        )
        self.built = True

    def call(self, inputs):
        inputs_expand = K.expand_dims(K.expand_dims(inputs, 2), 2)
        W_tiled = K.tile(self.W, [K.shape(inputs)[0], 1, 1, 1, 1])
        inputs_hat = tf.squeeze(tf.matmul(W_tiled, inputs_expand, transpose_b=True), axis=-1)
        b = tf.zeros(shape=[K.shape(inputs)[0], self.input_num_capsule, self.num_capsule])

        for i in range(self.routings):
            c = tf.nn.softmax(b, axis=2)
            c_expand = K.expand_dims(c, -1)
            outputs = self.squash(tf.reduce_sum(inputs_hat * c_expand, axis=1))
            if i < self.routings - 1:
                b += tf.reduce_sum(inputs_hat * K.expand_dims(c, -1), axis=-1)
        
        return outputs
    def get_config(self):
        config = super().get_config()
        config.update({
            "num_capsule": self.num_capsule,
            "dim_capsule": self.dim_capsule,
            "routings": self.routings
        })
        return config
    def squash(self, vectors, axis=-1):
        s_squared_norm = K.sum(K.square(vectors), axis, keepdims=True)
        scale = s_squared_norm / (1 + s_squared_norm) / K.sqrt(s_squared_norm + K.epsilon())
        return scale * vectors

@tf.keras.saving.register_keras_serializable(package="Custom", name="margin_loss")
def margin_loss(y_true, y_pred):
    y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=2)
    L = y_true * tf.square(tf.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * tf.square(tf.maximum(0., y_pred - 0.1))
    return tf.reduce_mean(tf.reduce_sum(L, axis=1))

class MobileNetCapsNet:
    def __init__(self, input_shape=(224, 224, 3)):
        self.input_shape = input_shape
        self.model = self._build_model()
    
    def _build_model(self):
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=self.input_shape,
            include_top=False,
            weights='imagenet'
        )
        base_model.trainable = False
        
        x = base_model.output
        x = layers.Conv2D(256, 3, activation='relu')(x)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Reshape((-1, 256))(x)
        
        x = CapsuleLayer(num_capsule=8, dim_capsule=16, routings=3)(x)
        x = CapsuleLayer(num_capsule=2, dim_capsule=32, routings=3)(x)
        outputs = Length()(x)
        
        return tf.keras.Model(inputs=base_model.input, outputs=outputs)
    
    def compile_model(self, learning_rate=0.001):
        """Compile the model with appropriate loss and optimizer"""
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        self.model.compile(
            optimizer=optimizer,
            loss=self.margin_loss,
            metrics=['accuracy']
        )
        
    @staticmethod
    def margin_loss(y_true, y_pred):
        """Margin loss for capsule network"""
        # Convert y_true to one-hot if it isn't already
        if len(K.int_shape(y_true)) == 1:
            y_true = tf.one_hot(tf.cast(y_true, 'int32'), 2)
            
        L = y_true * tf.square(tf.maximum(0., 0.9 - y_pred)) + \
            0.5 * (1 - y_true) * tf.square(tf.maximum(0., y_pred - 0.1))
        return tf.reduce_mean(tf.reduce_sum(L, axis=1))

# Load pre-trained global model
def load_model_from_file():
    return tf.keras.models.load_model("D:/Major Project/Rasp/Data/drowsiness_model.keras",
                                      custom_objects={'CapsuleLayer': CapsuleLayer,
                                                      'Length': Length,
                                                      'margin_loss': margin_loss})


def aggregate_encrypted_weights(encrypted_client_weights):
    """Perform FedAvg aggregation directly on encrypted weights."""
    num_clients = len(encrypted_client_weights)
    aggregated_weights = []
    
    for layer_idx in range(len(encrypted_client_weights[0])):
        layer_sum = encrypted_client_weights[0][layer_idx].copy()
        for i in range(1, num_clients):
            layer_sum += encrypted_client_weights[i][layer_idx]
        aggregated_layer = layer_sum * (1 / num_clients)
        aggregated_weights.append(aggregated_layer)
    
    return aggregated_weights
def generate_aes_key():
    """Generate AES key for model encryption."""
    custom_key = "secretkey"
    hashed_key = hashlib.sha256(custom_key.encode()).digest()
    encoded_key = base64.urlsafe_b64encode(hashed_key)
    return Fernet(encoded_key)
# Send the global model to the client
def send_model(client, model, cipher):
    """Encrypt and send the full model to the client."""
    with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as tmp:
        model.save(tmp.name)
        tmp.seek(0)
        model_data = tmp.read()
    compressed_data = zlib.compress(model_data)
    encrypted_data = cipher.encrypt(compressed_data)
    data_length = struct.pack('>I', len(encrypted_data))
    client.sendall(data_length)
    client.sendall(encrypted_data)
    print("Encrypted model sent to client.")

# def receive_encrypted_weights(client, context):
#     """Receive encrypted weights from client."""
#     data_length = struct.unpack('>I', client.recv(4))[0]
#     received_data = client.recv(data_length)
#     return ts.ckks_vector_from(context, received_data)
# def receive_encrypted_weights(client):
#     """Receive encrypted weights from client."""
#     data_length = struct.unpack('>I', client.recv(4))[0]
#     received_data = client.recv(data_length)
#     return ts.deserialize(received_data)
# Evaluate the global model
def evaluate_model(model, dataset_dir):
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)
    test_gen = datagen.flow_from_directory(
                    dataset_dir,
                    target_size=(224, 224),
                    batch_size=32,
                    class_mode='binary',
                    shuffle=False
                )
    predictions = np.argmax(model.predict(test_gen), axis=1)
    true_labels = test_gen.classes
    test_accuracy = accuracy_score(true_labels, predictions)
    print(f"Test Accuracy (global model): {test_accuracy:.2%}")
    print("\nClassification Report after updating global model:")
    print(classification_report(true_labels, predictions, target_names=list(test_gen.class_indices.keys())))
def receive_encrypted_weights(client, context):
    """Receive and correctly deserialize encrypted weights from the client."""

    # Receive the length of the incoming data
    data_length = struct.unpack(">I", client.recv(4))[0]
    print(f"Expected data size: {data_length} bytes")

    # Receive all data
    received_data = b""
    while len(received_data) < data_length:
        packet = client.recv(data_length - len(received_data))
        if not packet:
            raise ConnectionError("Connection lost while receiving data")
        received_data += packet

    print(f"Actual received data size: {len(received_data)} bytes")

    # Decompress the received data
    decompressed_data = zlib.decompress(received_data)
    print(f"Decompressed data size: {len(decompressed_data)} bytes")

    # Deserialize weights
    encrypted_weights = []
    offset = 0

    while offset < len(decompressed_data):
        try:
            # Read the length of the next serialized vector
            vec_length = struct.unpack(">I", decompressed_data[offset:offset+4])[0]
            offset += 4

            # Extract and deserialize the vector
            vec_data = decompressed_data[offset:offset+vec_length]
            vec = ts.ckks_vector_from(context, vec_data)
            encrypted_weights.append(vec)
            offset += vec_length

        except Exception as e:
            print(f"Error during deserialization at offset {offset}: {e}")
            break

    print(f"Number of encrypted weights received: {len(encrypted_weights)}")

    return encrypted_weights


# Server socket for federated learning
def server_socket():
    context = setup_ckks_context()
    cipher = generate_aes_key()
    global_model = load_model_from_file()
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(('0.0.0.0', 5000))  # Bind to all available IPs on port 5000
    server.listen(10)  # Listen for connections

    num_clients = 1  # Number of clients participating in federated learning
    client_sockets = []

    # Accept connections from all clients once
    print("Waiting for clients to connect...")
    for i in range(num_clients):
        client, addr = server.accept()
        print(f"Client {i + 1} connected from {addr}.")
        client_sockets.append(client)

    for round_num in range(1, 6):  # Perform 5 rounds
        print(f"\n==== Round {round_num} ====")
        encrypted_client_weights = []

        # Step 1: Send the global model to all clients
        for client in client_sockets:
            send_model(client, global_model, cipher)
        # Step 2: Receive updated weights from all clients
        for client in client_sockets:
            encrypted_weights = receive_encrypted_weights(client, context)
            encrypted_client_weights.append(encrypted_weights)
        aggregated_weights = aggregate_encrypted_weights(encrypted_client_weights)
        # Decrypt weights
        decrypted_weights = decrypt_weights(aggregated_weights, context, global_model)

        # Set weights properly
        global_model.set_weights(decrypted_weights)  # ✅ Now it should work


        print(f"Global model updated after round {round_num}")

        # Step 4: Evaluate the updated global model
        evaluate_model(global_model, "D:/Major Project/Rasp/Data/test")

    # Step 5: Close all client connections after all rounds
    for client in client_sockets:
        client.close()
        print("Client connection closed.")

    server.close()
    print("Server shutdown.")

if __name__ == "__main__":
    server_socket()