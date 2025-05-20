import socket
import struct
import zlib
import tempfile
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report
from tensorflow.keras.saving import register_keras_serializable
from tensorflow.keras import layers, initializers, backend as K
from sklearn.metrics import confusion_matrix, accuracy_score
from cryptography.fernet import Fernet
import base64
import hashlib
import time
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

from tensorflow.keras.models import load_model
# Load pre-trained global model
def load_model_from_file():
    return load_model(r"D:\Major Project\Rasp\Final_Results_MEtrcs\drowsiness_model_teacher_very_less_complex.keras",
                      custom_objects={"CapsuleLayer": CapsuleLayer, "Length": Length, "margin_loss": margin_loss})

# def aggregate_weights(client_weights, beta=0.2):
#     """Implements Federated Trimmed Averaging (FedTrimmedAvg) aggregation"""
#     if not client_weights:
#         return []
    
#     num_clients = len(client_weights)
#     k = int(beta * num_clients)
#     k = max(0, min(k, (num_clients - 1) // 2))  # Ensure valid trim size
    
#     aggregated_weights = []
    
#     # Process each layer independently
#     for layer_idx in range(len(client_weights[0])):
#         # Stack all clients' layer weights into array
#         layer_stack = np.array([client[layer_idx] for client in client_weights])
#         original_shape = layer_stack.shape[1:]
        
#         # Flatten for parameter-wise processing
#         flattened = layer_stack.reshape(num_clients, -1)
        
#         # Sort and trim parameters
#         sorted_flat = np.sort(flattened, axis=0)
#         trimmed = sorted_flat[k:num_clients-k] if k > 0 else sorted_flat
        
#         # Compute mean of remaining values
#         avg_flat = np.mean(trimmed, axis=0)
        
#         # Restore original layer shape
#         aggregated_weights.append(avg_flat.reshape(original_shape))
    
#     return aggregated_weights


# Send the global model to the client
def send_global_model(client, global_model, round_num):
    with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as tmp:
        global_model.save(tmp.name)
        tmp.seek(0)
        model_data = tmp.read()

    compressed_data = zlib.compress(model_data)
    encrypted_data = cipher_suite.encrypt(compressed_data)  # Encrypt the data
    data_length = struct.pack('>I', len(encrypted_data))

    client.sendall(data_length)  # Send encrypted data size
    chunk_size = 1024 * 1024  # 1 MB per chunk
    with tqdm(total=len(encrypted_data), unit="B", unit_scale=True, desc="Sending model") as pbar:
        for i in range(0, len(encrypted_data), chunk_size):
            chunk = encrypted_data[i:i + chunk_size]
            client.sendall(chunk)
            pbar.update(len(chunk))
    print(f"Global model for Round {round_num} sent to client.")
    client.sendall(struct.pack('>I', round_num))  # Send round number

# Receive weights from the client
def receive_client_weights(client):
    data_length = struct.unpack('>I', client.recv(4))[0]  # Get size of the encrypted data
    received_data = b""
    while len(received_data) < data_length:
        packet = client.recv(1024 * 1024)
        if not packet:
            break
        received_data += packet

    decrypted_data = cipher_suite.decrypt(received_data)  # Decrypt the data
    decompressed_data = zlib.decompress(decrypted_data)
    with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as tmp:
        tmp.write(decompressed_data)
        tmp.flush()
        model =load_model(tmp.name,custom_objects={'CapsuleLayer': CapsuleLayer,
                                                       'Length': Length,
                                                       'margin_loss': margin_loss})
        return model.get_weights()
from pathlib import Path
INPUT_SHAPE = (224, 224, 3)
BATCH_SIZE = 32
DATA_DIR = Path("D:/Major Project/Rasp/Data")
TEST_DIR = r"D:\Major Project\Rasp\old\test"
# Evaluate the global model
import numpy as np
from sklearn.cluster import KMeans

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-8)

def compute_model_similarity(client1_weights, client2_weights):
    flat1 = np.concatenate([layer.flatten() for layer in client1_weights])
    flat2 = np.concatenate([layer.flatten() for layer in client2_weights])
    return cosine_similarity(flat1, flat2)

def cluster_clients(client_weights, num_clusters=2):
    flattened_weights = [np.concatenate([layer.flatten() for layer in weights]) for weights in client_weights]
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    return kmeans.fit_predict(flattened_weights)

def aggregate_weights(client_weights, past_model_weights, round_index, mode="divergence", num_clusters=2):
    """
    FedCDA Aggregation:
    - mode="divergence": Uses cosine similarity to adapt weight blending between 2 clients
    - mode="cluster": Uses KMeans to cluster clients and aggregate within clusters
    """
    if not client_weights:
        raise ValueError("No client weights provided.")

    num_layers = len(client_weights[0])
    adaptive_lambda = 1 / (1 + round_index)

    if mode == "divergence":
        if len(client_weights) != 2:
            raise ValueError("Divergence mode only supports 2 clients.")
        similarity = compute_model_similarity(client_weights[0], client_weights[1])
        w1 = 0.5 + 0.5 * similarity
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

    return [
        (1 - adaptive_lambda) * aggregated_weights[i] + adaptive_lambda * np.array(past_model_weights[i])
        for i in range(num_layers)
    ]
def evaluate_model(model, dataset_dir):
# Define test image generator
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

    # Load test images
    test_gen = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=INPUT_SHAPE[:2],
        batch_size=BATCH_SIZE,
        class_mode='categorical',  # Ensure binary classification mode
        shuffle=False
    )
    # Get predictions (probabilities)
    y_pred_prob = model.predict(test_gen)

    # Convert probabilities to binary labels (0 or 1)
    y_pred = (y_pred_prob >= 0.5).astype(int).flatten()
    y_true = test_gen.classes  # True labels
    loss, accuracy = model.evaluate(test_gen)
    print(f"Test Accuracy: {accuracy:.4f}")
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Not Drowsy', 'Drowsy']))
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_score, accuracy_score, confusion_matrix,
    f1_score, recall_score, classification_report
)
def evaluate_model_start(model, dataset_dir, show_confusion_matrix=False):
    # Data preparation
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)
    test_gen = datagen.flow_from_directory(
        dataset_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        shuffle=False
    )

    # Predictions and metrics
    start_time = time.time()
    predictions = np.argmax(model.predict(test_gen), axis=1)
    end_time = time.time()
    inference_latency = end_time - start_time

    true_labels = test_gen.classes
    class_names = list(test_gen.class_indices.keys())

    # Calculate metrics
    precision = precision_score(true_labels, predictions, average='weighted')
    test_accuracy = accuracy_score(true_labels, predictions)
    confusion_mat = confusion_matrix(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average='weighted')
    recall = recall_score(true_labels, predictions, average='weighted')

    # Print metrics
    print(f"\nTest Accuracy : {test_accuracy:.2%}")
    print(f"Precision: {precision:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Inference Latency: {inference_latency:.4f} seconds")

    # Classification report
    print("\nClassification Report:")
    print(classification_report(true_labels, predictions, target_names=class_names))

    # Confusion matrix (optionally plot)
    print("\nConfusion Matrix:")
    print(confusion_mat)

    if show_confusion_matrix:
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_mat, annot=True, fmt='g', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.ylabel('Actual', fontsize=13)
        plt.title('Confusion Matrix', fontsize=17, pad=20)
        plt.xlabel('Prediction', fontsize=13)
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)
        plt.show()

    return confusion_mat, class_names
# Server socket for federated learning
def server_socket():
    global_model = load_model_from_file()
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(('0.0.0.0', 5000))  # Bind to all available IPs on port 5000
    server.listen(10)  # Listen for connections #use distiller.student when training
    past_model_weights = global_model.get_weights()
    num_clients = 2  # Number of clients participating in federated learning
    client_sockets = []
    evaluate_model_start(global_model, r"D:\Major Project\Rasp\old\test")
    # Accept connections from all clients once
    print("Waiting for clients to connect...")
    for i in range(num_clients):
        client, addr = server.accept()
        print(f"Client {i + 1} connected from {addr}.")
        client_sockets.append(client)

    for round_num in range(1, 6):  # Perform 5 rounds
        print(f"\n==== Round {round_num} ====")
        client_weights = []

        # Step 1: Send the global model to all clients
        for client in client_sockets:
            send_global_model(client, global_model, round_num)

        # Step 2: Receive updated weights from all clients
        for i, client in enumerate(client_sockets):
            print(f"Receiving weights from client {i + 1} for Round {round_num}...")
            weights = receive_client_weights(client)
            client_weights.append(weights)

        # Step 3: Aggregate weights and update the global model
        aggregated_weights = aggregate_weights(
            client_weights,
            past_model_weights=past_model_weights,
            round_index=round_num,
            mode="cluster"  # or "divergence" if you're doing 2-client adaptive weighting
        )
        global_model.set_weights(aggregated_weights)
        past_model_weights = global_model.get_weights()
        print(f"Global model updated after Round {round_num}.")
        # Step 4: Evaluate the updated global model
        evaluate_model(global_model, r"D:\Major Project\Rasp\old\test")

    # Step 5: Close all client connections after all rounds
    for client in client_sockets:
        client.close()
        print("Client connection closed.")

    server.close()
    print("Server shutdown.")

if __name__ == "__main__":
    server_socket()