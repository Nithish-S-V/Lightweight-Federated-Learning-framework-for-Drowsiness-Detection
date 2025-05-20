import socket
import struct
import zlib
import tempfile
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report
from tensorflow.keras.models import load_model, clone_model
from copy import deepcopy
import pandas as pd
import matplotlib.pyplot as plt
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
DATA_DIR = Path("D:/Major Project/Rasp/Data")
TEST_DIR = r"D:\\Major Project\\Final Proper\\3class_test"
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

def cluster_clients(client_weights, num_clusters=3):
    flattened_weights = [np.concatenate([layer.flatten() for layer in weights]) for weights in client_weights]
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    return kmeans.fit_predict(flattened_weights)

def fedcda_aggregate(client_weights, past_model_weights, round_index, num_clusters=3):
    cluster_labels = cluster_clients(client_weights, num_clusters)
    num_layers = len(client_weights[0])
    aggregated_weights = [np.zeros_like(client_weights[0][i]) for i in range(num_layers)]
    for cluster in range(num_clusters):
        cluster_indices = [i for i, label in enumerate(cluster_labels) if label == cluster]
        if len(cluster_indices) == 0:
            continue
        cluster_weights = [client_weights[i] for i in cluster_indices]
        for layer in range(num_layers):
            aggregated_weights[layer] = np.mean(
                [np.array(cluster_weights[i][layer]) for i in range(len(cluster_weights))], axis=0)
    adaptive_lambda = 1 / (1 + round_index)
    return [
        (1 - adaptive_lambda) * aggregated_weights[layer] + adaptive_lambda * np.array(past_model_weights[layer])
        for layer in range(num_layers)
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
AGGREGATION_METHODS = ["FedAvg", "FedTrimmedAvg", "FedMA", "FedCDA", "FedPA", "FedBN"]

# Load pre-trained global model

def load_model_from_file():
    return load_model(r"D:\Major Project\Final Proper\Models\drowsiness_model_final_WithClientData_lesser.keras",
                      custom_objects={"CapsuleLayer": CapsuleLayer, "Length": Length, "margin_loss": margin_loss})

# Send the global model to the client
def send_global_model(client, global_model, method_name):
    # === Save and Encrypt Model ===
    with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as tmp:
        global_model.save(tmp.name)
        tmp.seek(0)
        model_data = tmp.read()

    compressed_data = zlib.compress(model_data)
    encrypted_data = cipher_suite.encrypt(compressed_data)
    data_length = struct.pack('>I', len(encrypted_data))

    # === Send Model ===
    client.sendall(data_length)
    chunk_size = 1024 * 1024
    with tqdm(total=len(encrypted_data), unit="B", unit_scale=True, desc="Sending model") as pbar:
        for i in range(0, len(encrypted_data), chunk_size):
            chunk = encrypted_data[i:i + chunk_size]
            client.sendall(chunk)
            pbar.update(len(chunk))
    print(f"✅ Global model sent to client.")

    # === Send Aggregation Method Name ===
    method_bytes = method_name.encode()
    client.sendall(struct.pack('>I', len(method_bytes)))  # send length
    client.sendall(method_bytes)                          # send method name
    print(f"🧠 Sent aggregation method: {method_name}")
    #client.sendall(struct.pack('>I', round_num))

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
        return fedcda_aggregate(client_weights, past_model_weights, round_index)
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
def server_socket():
    aggregation_methods = {
        "FedAvg": fedavg_aggregate,
        "FedTrimmedAvg": fedtrimmedavg_aggregate,
        "FedMA": fedma_aggregate,
        "FedCDA": lambda w, past_weights, round_num: fedcda_aggregate(w, past_weights, round_num, num_clusters=2),
        "FedPA": lambda w, global_weights: fedpa_aggregate(w, global_weights),
        "FedBN": fedbn_aggregate
    }

    results = {method: [] for method in aggregation_methods}
    num_clients = 2
    num_rounds = 3

    print("Waiting for clients to connect...")
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(('0.0.0.0', 5000))
    server.listen(10)
    client_sockets = [server.accept()[0] for _ in range(num_clients)]
    print("All clients connected.")

    base_model = load_model_from_file()
    evaluate_model_old(base_model, TEST_DIR)

    experiment_id = 0
    for method_name, aggregate_fn in aggregation_methods.items():
        print(f"\n======== Testing Aggregation: {method_name} ========")
        global_model = tf.keras.models.clone_model(base_model)
        global_model.set_weights(base_model.get_weights())
        past_model_weights = deepcopy(global_model.get_weights())

        for round_num in range(1, num_rounds + 1):
            print(f"\n---- Round {round_num} ({method_name}) ----")
            experiment_id += 1
            client_weights = []

            for client in client_sockets:
                send_global_model(client, global_model,method_name)

            for i, client in enumerate(client_sockets):
                print(f"Receiving weights from client {i + 1}...")
                weights = receive_client_weights(client)
                client_weights.append(weights)

            global_weights = global_model.get_weights()

            if method_name == "FedAvg":
                aggregated_weights = aggregate_fn(client_weights, [1] * len(client_weights))
            elif method_name == "FedCDA":
                aggregated_weights = aggregate_fn(client_weights, past_model_weights, round_num, num_clusters=2)
            elif method_name == "FedPA":
                aggregated_weights = aggregate_fn(client_weights, global_weights)
            elif method_name == "FedTrimmedAvg":
                aggregated_weights = aggregate_fn(client_weights)
            elif method_name == "FedMA":
                aggregated_weights = aggregate_fn(client_weights)
            elif method_name == "FedBN":
                aggregated_weights = aggregate_fn(client_weights)
            else:
                raise ValueError(f"Unknown aggregation method: {method}")

            global_model.set_weights(aggregated_weights)
            past_model_weights = deepcopy(aggregated_weights)

            print(f"Evaluating {method_name} after Round {round_num}")
            test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
            test_gen = test_datagen.flow_from_directory(TEST_DIR, target_size=INPUT_SHAPE[:2],
                                                        batch_size=BATCH_SIZE, class_mode='categorical',
                                                        color_mode='rgb', shuffle=False)
            y_pred = np.argmax(global_model.predict(test_gen), axis=1)
            y_true = test_gen.classes
            acc = np.mean(y_true == y_pred)
            results[method_name].append((round_num, acc))

            print(f"[{method_name}] Accuracy after Round {round_num}: {acc:.4f}")
            print(classification_report(y_true, y_pred, target_names=['Eyeclose', 'Neutral', 'Yawn'], zero_division=0))

    all_data = []
    for method, rounds in results.items():
        for round_num, acc in rounds:
            all_data.append({'Aggregation': method, 'Round': round_num, 'Accuracy': acc})

    df = pd.DataFrame(all_data)
    df.to_csv('aggregation_summary.csv', index=False)
    print("✅ Saved accuracy summary to aggregation_summary.csv")

    plt.figure(figsize=(10, 6))
    for method, rounds in results.items():
        x = [r for r, _ in rounds]
        y = [a for _, a in rounds]
        plt.plot(x, y, marker='o', label=method)

    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.title("Federated Aggregation Comparison")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("aggregation_accuracy_plot.png")
    print("✅ Saved plot to aggregation_accuracy_plot.png")
    plt.show()

    for client in client_sockets:
        client.close()
    server.close()
    print("Server shutdown.")

if __name__ == "__main__":
    server_socket()