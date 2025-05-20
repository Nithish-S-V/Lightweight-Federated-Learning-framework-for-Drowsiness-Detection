import socket
import struct
import zlib
import tempfile
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report
from tensorflow.keras.models import load_model, clone_model
import threading
from tensorflow.keras.saving import register_keras_serializable
from tensorflow.keras import layers, initializers, backend as K
from sklearn.metrics import confusion_matrix, accuracy_score
from tensorflow.keras.models import load_model
from sklearn.cluster import KMeans
from cryptography.fernet import Fernet
import base64
import hashlib
from pathlib import Path
from datetime import datetime

# Generate a shared secret key for encryption (must be the same on server and client)
# KEY = Fernet.generate_key()
# print("Generated Key:", KEY)
# cipher_suite = Fernet(KEY)


# @tf.keras.saving.register_keras_serializable(package="Custom", name="binary_crossentropy_loss")
def binary_crossentropy_loss(y_true, y_pred):
    return tf.keras.losses.binary_crossentropy(y_true, y_pred)

# Your custom key (e.g., a name or phrase)
custom_key = "secretkey"
# Hash the custom key using SHA-256 to get a 32-byte value
hashed_key = hashlib.sha256(custom_key.encode()).digest()
# Encode the hashed key as a URL-safe base64 string
encoded_key = base64.urlsafe_b64encode(hashed_key)
# Use the encoded key
KEY = encoded_key
cipher_suite = Fernet(KEY)


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
    
INPUT_SHAPE = (224, 224, 3)
BATCH_SIZE = 32
#DATA_DIR = Path("D:/Major Project/Rasp/Data")
TEST_DIR = r"D:\Major Project\Final Proper\FederatedData_balanced\test"
#AGGREGATION_METHOD = "FedAvg"  # Change to: FedTrimmedAvg, FedMA, FedCDA, FedPA, FedBN

### Aggregation Methods ###

def fedavg_aggregate(client_weights, client_data_sizes):
    total_samples = sum(client_data_sizes)
    num_layers = len(client_weights[0])
    aggregated_weights = [np.zeros_like(client_weights[0][i]) for i in range(num_layers)]
    for i in range(len(client_weights)):
        weight_factor = client_data_sizes[i] / total_samples
        for layer in range(num_layers):
            aggregated_weights[layer] += weight_factor * np.array(client_weights[i][layer])
    return aggregated_weights

def fedma_aggregate(client_weights):
    num_layers = len(client_weights[0])
    return [np.mean([client_weights[i][j] for i in range(len(client_weights))], axis=0) for j in range(num_layers)]

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def compute_model_similarity(client1_weights, client2_weights):
    flat1 = np.concatenate([layer.flatten() for layer in client1_weights])
    flat2 = np.concatenate([layer.flatten() for layer in client2_weights])
    return cosine_similarity(flat1, flat2)

def cluster_clients(client_weights, num_clusters=2):
    flattened_weights = [np.concatenate([layer.flatten() for layer in weights]) for weights in client_weights]
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    return kmeans.fit_predict(flattened_weights)

def fedCDA(client_weights, past_model_weights, round_index, mode="divergence", num_clusters=2):
    """
    FedCDA Aggregation:
    - mode="divergence": Uses cosine similarity to adapt weight blending between 2 clients
    - mode="cluster": Uses KMeans to cluster clients and aggregate within clusters
    """
    num_layers = len(client_weights[0])
    adaptive_lambda = 1 / (1 + round_index)

    if mode == "divergence":
        if len(client_weights) != 2:
            raise ValueError("Divergence mode only supports 2 clients")
        # Compute cosine similarity
        similarity = compute_model_similarity(client_weights[0], client_weights[1])
        w1 = 0.5 + 0.5 * similarity   # similarity = 1 → w1 = 1
        w2 = 1 - w1

        aggregated_weights = [
            w1 * np.array(client_weights[0][i]) + w2 * np.array(client_weights[1][i])
            for i in range(num_layers)
        ]

    elif mode == "cluster":
        cluster_labels = cluster_clients(client_weights, num_clusters)
        cluster_aggregates = []

        for cluster in range(num_clusters):
            cluster_indices = [i for i, label in enumerate(cluster_labels) if label == cluster]
            if not cluster_indices:
                continue
            cluster_weights = [client_weights[i] for i in cluster_indices]
            cluster_agg = [
                np.mean([cw[layer] for cw in cluster_weights], axis=0)
                for layer in range(num_layers)
            ]
            cluster_aggregates.append(cluster_agg)

        aggregated_weights = [
            np.mean([cluster[layer] for cluster in cluster_aggregates], axis=0)
            for layer in range(num_layers)
        ]

    else:
        raise ValueError("Unsupported mode. Use 'divergence' or 'cluster'.")

    # Temporal smoothing with past model weights
    return [
        (1 - adaptive_lambda) * aggregated_weights[i] + adaptive_lambda * np.array(past_model_weights[i])
        for i in range(num_layers)
    ]

def fedpa_aggregate(client_weights, global_weights):
    lambda_factor = 0.5
    num_layers = len(global_weights)
    aggregated_weights = []
    for j in range(num_layers):
        client_layer_aggregated = np.mean([np.array(client_weights[i][j]) for i in range(len(client_weights))], axis=0)
        updated_layer = (lambda_factor * np.array(global_weights[j])) + ((1 - lambda_factor) * client_layer_aggregated)
        aggregated_weights.append(updated_layer)
    return aggregated_weights

def fedbn_aggregate(client_weights):
    aggregated_weights = []
    for layer_weights in zip(*client_weights):
        if len(layer_weights[0].shape) == 1:  # Likely batch norm stats
            aggregated_weights.append(layer_weights[0])  # Keep first client's BN layer (local)
        else:
            aggregated_weights.append(np.mean(layer_weights, axis=0))
    return aggregated_weights

def fedtrimmedavg_aggregate(client_weights, beta=0.2):
    num_clients = len(client_weights)
    k = int(beta * num_clients)
    k = max(0, min(k, (num_clients - 1) // 2))
    aggregated_weights = []
    for layer_idx in range(len(client_weights[0])):
        layer_stack = np.array([client[layer_idx] for client in client_weights])
        original_shape = layer_stack.shape[1:]
        flattened = layer_stack.reshape(num_clients, -1)
        sorted_flat = np.sort(flattened, axis=0)
        trimmed = sorted_flat[k:num_clients-k] if k > 0 else sorted_flat
        avg_flat = np.mean(trimmed, axis=0)
        aggregated_weights.append(avg_flat.reshape(original_shape))
    return aggregated_weights

# Model loading and communication


# Define the aggregation methods
AGGREGATION_METHODS = ["FedAvg", "FedTrimmedAvg", "FedMA", "FedCDA", "FedPA", "FedBN"]#["FedCDA"]

# Load pre-trained global model

def load_model_from_file():
    return load_model(r"D:\Major Project\Final Proper\Models\drowsiness_model_With_balanced_lesser.keras",
                      custom_objects={"CapsuleLayer": CapsuleLayer, "Length": Length, "margin_loss": margin_loss})

# Send the global model to the client
def send_global_model(client, global_model, round_num):
    with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as tmp:
        global_model.save(tmp.name)
        tmp.seek(0)
        model_data = tmp.read()

    compressed_data = zlib.compress(model_data)
    encrypted_data = cipher_suite.encrypt(compressed_data)
    data_length = struct.pack('>I', len(encrypted_data))

    client.sendall(data_length)
    chunk_size = 1024 * 1024
    with tqdm(total=len(encrypted_data), unit="B", unit_scale=True, desc="Sending model") as pbar:
        for i in range(0, len(encrypted_data), chunk_size):
            chunk = encrypted_data[i:i + chunk_size]
            client.sendall(chunk)
            pbar.update(len(chunk))
    print(f"Global model for Round {round_num} sent to client.")
    # client.sendall(struct.pack('>I', round_num))

# Receive weights from the client
def receive_client_weights(client):
    data_length = struct.unpack('>I', client.recv(4))[0]
    received_data = b""
    while len(received_data) < data_length:
        packet = client.recv(1024 * 1024)
        if not packet:
            break
        received_data += packet

    decrypted_data = cipher_suite.decrypt(received_data)
    decompressed_data = zlib.decompress(decrypted_data)
    with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as tmp:
        tmp.write(decompressed_data)
        tmp.flush()
        model = load_model(tmp.name, custom_objects={"CapsuleLayer": CapsuleLayer, "Length": Length, "margin_loss": margin_loss})
        return model.get_weights()

# Evaluate model
INPUT_SHAPE = (224, 224, 3)
BATCH_SIZE = 32

def evaluate_model(model, dataset_dir, method_name, round_num):
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
    test_gen = test_datagen.flow_from_directory(
        dataset_dir,
        target_size=INPUT_SHAPE[:2],
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        color_mode='rgb',
        shuffle=False
    )
    y_pred = np.argmax(model.predict(test_gen), axis=1)
    y_true = test_gen.classes

    accuracy = np.mean(y_true == y_pred)
    with open(f"aggregation_{method_name}_results.txt", "a") as f:
        f.write(f"Round {round_num} - Accuracy: {accuracy:.4f}\n")
        f.write(classification_report(y_true, y_pred, target_names=['Eyeclose', 'Neutral', 'Yawn'], zero_division=0))
        f.write("\n\n")
    print(f"{method_name} - Round {round_num} Accuracy: {accuracy:.4f}")

# Aggregation dispatcher
def aggregate_weights_dispatch(method, client_weights, global_weights=None, round_index=1, past_model_weights=None):
    if method == "FedAvg":
        return fedavg_aggregate(client_weights, [1] * len(client_weights))
    elif method == "FedTrimmedAvg":
        return fedtrimmedavg_aggregate(client_weights)
    elif method == "FedMA":
        return fedma_aggregate(client_weights)
    elif method == "FedCDA":
        return fedCDA(client_weights, past_model_weights, round_index)
    elif method == "FedPA":
        return fedpa_aggregate(client_weights, global_weights)
    elif method == "FedBN":
        return fedbn_aggregate(client_weights)
    else:
        raise ValueError(f"Unknown aggregation method: {method}")
def evaluate_model_old(model, dataset_dir):
# Define test image generator
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

    # Load test images
    test_gen = test_datagen.flow_from_directory(
        dataset_dir,
        target_size=INPUT_SHAPE[:2],
        batch_size=BATCH_SIZE,
        class_mode='categorical',  
        color_mode='rgb',
        shuffle=False
    )
    # Get predictions (probabilities)
    # Get predictions (probabilities)
    y_pred = np.argmax(model.predict(test_gen), axis=1)
    y_true = test_gen.classes

    print("\nTest Metrics:")
    print(f"Accuracy: {np.mean(y_true == y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Eyeclose','Neutral', 'Yawn'], zero_division=0))
# Server socket for federated learning
def server_socket():
    original_model = load_model_from_file()
    evaluate_model_old(original_model, TEST_DIR)
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(('0.0.0.0', 5000))
    server.listen(10)

    num_clients = 2
    client_sockets = []
    print("Waiting for clients to connect...")
    for i in range(num_clients):
        client, addr = server.accept()
        print(f"Client {i + 1} connected from {addr}.")
        client_sockets.append(client)

    for method in AGGREGATION_METHODS:
        print(f"\n=== Starting aggregation: {method} ===")
        global_model = clone_model(original_model)
        global_model.set_weights(original_model.get_weights())
        past_weights = global_model.get_weights()

        for round_num in range(1, 4):
            print(f"\n[{method}] Round {round_num}")
            client_weights = []

            for client in client_sockets:
                send_global_model(client, global_model, round_num)

            for i, client in enumerate(client_sockets):
                print(f"Receiving weights from client {i + 1} for Round {round_num}...")
                weights = receive_client_weights(client)
                client_weights.append(weights)

            aggregated_weights = aggregate_weights_dispatch(
                method, client_weights,
                global_weights=global_model.get_weights(),
                round_index=round_num,
                past_model_weights=past_weights
            )
            global_model.set_weights(aggregated_weights)
            past_weights = aggregated_weights

            evaluate_model(global_model, r"D:\Major Project\Final Proper\FederatedData_balanced\test", method, round_num)

    for client in client_sockets:
        client.close()
        print("Client connection closed.")

    server.close()
    print("Server shutdown.")

if __name__ == "__main__":
    server_socket()
