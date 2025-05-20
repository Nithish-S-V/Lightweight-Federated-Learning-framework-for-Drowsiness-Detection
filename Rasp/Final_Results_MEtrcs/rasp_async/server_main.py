# === Standard Library ===
import base64
import hashlib
import os
import shutil
import socket
import struct
import tempfile
import threading
import zlib
from datetime import datetime
from pathlib import Path
import socket
import threading
import time
# === Cryptography ===
from cryptography.fernet import Fernet
# === Progress Bar / Utilities ===
from tqdm import tqdm
# === Data Handling & Metrics ===
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.model_selection import train_test_split
# === TensorFlow / Keras ===
import tensorflow as tf
from tensorflow.keras import backend as K, initializers, layers
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.models import clone_model, load_model
from tensorflow.keras.saving import register_keras_serializable
# If using standalone Keras (not just tf.keras)
from keras import Model, ops
from keras.optimizers import Adam
import time
import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_score, accuracy_score, confusion_matrix,
    f1_score, recall_score, classification_report
)

#======================================================================
#Encryption
#======================================================================

# Your custom key (e.g., a name or phrase)
custom_key = "secretkey"
# Hash the custom key using SHA-256 to get a 32-byte value
hashed_key = hashlib.sha256(custom_key.encode()).digest()
# Encode the hashed key as a URL-safe base64 string
encoded_key = base64.urlsafe_b64encode(hashed_key)
# Use the encoded key
KEY = encoded_key
cipher_suite = Fernet(KEY)

#======================================================================
# Custom Capsule Network Components
#======================================================================

@register_keras_serializable(package="Custom")
class Length(layers.Layer):
    def call(self, inputs, **kwargs):
        return K.sqrt(K.sum(K.square(inputs), -1) + K.epsilon())
    
    def compute_output_shape(self, input_shape):
        return input_shape[:-1]
    
    def get_config(self):
        return super(Length, self).get_config()

def squash(vectors, axis=-1):
    s_squared_norm = K.sum(K.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / K.sqrt(s_squared_norm + K.epsilon())
    return scale * vectors

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
    L = y_true * tf.square(tf.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * tf.square(tf.maximum(0., y_pred - 0.1))

    return tf.reduce_mean(tf.reduce_sum(L, axis=1))

# @tf.keras.saving.register_keras_serializable(package="Custom", name="binary_crossentropy_loss")
def binary_crossentropy_loss(y_true, y_pred):
    return tf.keras.losses.binary_crossentropy(y_true, y_pred)

#======================================================================
# Student model architecture
#======================================================================

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, regularizers

class MobileNetStudent:
    def __init__(self, input_shape=(224, 224, 3), num_classes=3, learning_rate=3e-4):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.model = self._build_model()
        self._compile_model()

    def _build_model(self):
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=self.input_shape,
            include_top=False,
            weights="imagenet",  # Use pre-trained weights
            alpha=0.35  # Reduced alpha to make it smaller
        )
        base_model.trainable = False  # Freeze the base model initially

        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.6)(x)  # Increased dropout
        x = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(0.001))(x)  # Reduced units
        outputs = layers.Dense(self.num_classes, activation="softmax")(x)

        return Model(inputs=base_model.input, outputs=outputs)

    def _compile_model(self):
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss="categorical_crossentropy",
            metrics=["CategoricalAccuracy"]
        )

    def get_callbacks(self):
        lr_scheduler = keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, verbose=1
        )
        early_stopping = keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        )
        model_checkpoint = keras.callbacks.ModelCheckpoint(
            "best_model.h5", save_best_only=True, monitor="val_loss", mode="min"
        )
        return [lr_scheduler, early_stopping]

#======================================================================
# Load model fn
#======================================================================

def load_model_from_file():
    return load_model(r"D:\Major Project\Rasp\Final_Results_MEtrcs\rasp_async\teacher.keras",
                      custom_objects={"CapsuleLayer": CapsuleLayer, "Length": Length, "margin_loss": margin_loss})

#======================================================================
# Aggregation fedavg
#======================================================================

def fedavg_aggregate(client_weights, client_data_sizes):
    total_samples = sum(client_data_sizes)
    num_layers = len(client_weights[0])
    aggregated_weights = [np.zeros_like(client_weights[0][i]) for i in range(num_layers)]
    for i in range(len(client_weights)):
        weight_factor = client_data_sizes[i] / total_samples
        for layer in range(num_layers):
            aggregated_weights[layer] += weight_factor * np.array(client_weights[i][layer])
    return aggregated_weights



#======================================================================
# Send and Receive functions
#======================================================================

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
    
#======================================================================
# Evaluate functions
#======================================================================

def evaluate_model(model, dataset_dir):
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)
    data_flow = datagen.flow_from_directory(
        dataset_dir,
        target_size=(224,224),
        batch_size=32,
        class_mode='binary',
        shuffle=False,
    )
    predictions = np.argmax(model.predict(data_flow), axis=1)
    true_labels = data_flow.classes
    test_accuracy = accuracy_score(true_labels, predictions)
    print(f"Test Accuracy : {test_accuracy:.2%}")
    print("\nClassification Report after updating global model:")
    print(classification_report(true_labels, predictions, target_names=list(data_flow.class_indices.keys())))
def evaluate_model_start(model, dataset_dir, show_confusion_matrix=False):
    # Data preparation
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)
    test_gen = datagen.flow_from_directory(
        dataset_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode="categorical",
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

#======================================================================
# Flops and model size fn
#======================================================================

# import os
# import tensorflow as tf
# from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

# def get_model_size_mb(model_path: str) -> float:
#     """Get the size of a saved model file in MB."""
#     if not os.path.exists(model_path):
#         raise FileNotFoundError(f"Model file '{model_path}' not found!")
    
#     return os.path.getsize(model_path) / (1024 ** 2)  # Convert bytes to MB

# def save_temp_model(model, model_path: str) -> None:
#     """Save a model to a temporary file."""
#     model.save(model_path)  # Save the model

# def calculate_compression_ratio(teacher_size: float, student_size: float) -> float:
#     """Calculate compression ratio between teacher and student models."""
#     return teacher_size / student_size if student_size > 0 else float('inf')

# import tensorflow as tf
# from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

# def get_flops(model):
#     """
#     Compute FLOPs for a given model.
#     """
#     concrete_func = tf.function(model).get_concrete_function(
#         tf.TensorSpec([1] + list(model.input_shape[1:]), model.input.dtype)
#     )
    
#     frozen_func = convert_variables_to_constants_v2(concrete_func)
#     graph = frozen_func.graph

#     flops = tf.compat.v1.profiler.profile(
#         graph,
#         options=tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
#     )

    # return flops.total_float_ops if flops is not None else 0

#======================================================================
# Distillation fn
#======================================================================

import tensorflow as tf
from tensorflow import keras
import os

class ServerDistiller(keras.Model):
    def __init__(self, teacher, student, temp=3.0, alpha=0.1, grad_clip=1.0):
        super().__init__()
        self.teacher = teacher
        self.student = student
        self.temp = temp
        self.alpha = alpha
        self.grad_clip = grad_clip
        self.teacher.trainable = False  # Freeze the teacher model

        # Metrics
        self.total_loss_metric = keras.metrics.Mean(name="total_loss")
        self.student_loss_metric = keras.metrics.Mean(name="student_loss")
        self.distill_loss_metric = keras.metrics.Mean(name="distill_loss")
        self.acc_metric = keras.metrics.CategoricalAccuracy(name="accuracy")

    def compile(self, optimizer, **kwargs):
        """Properly compiles the model with loss validation bypass."""
        kwargs.pop('loss', None)  # Prevent Keras validation issues
        super().compile(optimizer=optimizer, loss=self._dummy_loss, **kwargs)

        # Define actual loss functions
        self.student_loss_fn = keras.losses.CategoricalCrossentropy(from_logits=False)  
        self.distill_loss_fn = keras.losses.KLDivergence()

    def _dummy_loss(self, y_true, y_pred):
        """Dummy loss function to bypass Keras validation checks."""
        return 0.0

    def train_step(self, data):
        """Custom training step for knowledge distillation."""
        x, y = data

        with tf.GradientTape() as tape:
            teacher_logits = self.teacher(x, training=False)  # Teacher inference mode
            student_probs = self.student(x, training=True)  # Student training mode

            # Compute student loss (cross-entropy with true labels)
            student_loss = self.student_loss_fn(y, student_probs)

            # Compute distillation loss (teacher-student KL divergence)
            teacher_probs = tf.nn.softmax(teacher_logits / self.temp, axis=1)
            distill_loss = (self.temp ** 2) * self.distill_loss_fn(teacher_probs, student_probs)  # Scale KL divergence

            # Total loss: weighted sum of both
            total_loss = self.alpha * student_loss + (1 - self.alpha) * distill_loss

        # Compute gradients & apply clipping
        gradients = tape.gradient(total_loss, self.student.trainable_variables)
        gradients = [tf.clip_by_norm(g, self.grad_clip) for g in gradients]
        self.optimizer.apply_gradients(zip(gradients, self.student.trainable_variables))

        # Update metrics
        self.total_loss_metric.update_state(total_loss)
        self.student_loss_metric.update_state(student_loss)
        self.distill_loss_metric.update_state(distill_loss)
        self.acc_metric.update_state(y, student_probs)

        return {m.name: m.result() for m in self.metrics}

    @property
    def metrics(self):
        """Returns list of tracked metrics."""
        return [
            self.total_loss_metric,
            self.student_loss_metric,
            self.distill_loss_metric,
            self.acc_metric
        ]

    def call(self, inputs, training=False):
        """Forward pass using student model."""
        return self.student(inputs, training=training)

    # def save_student_model(self):
    #     """Saves the distilled student model to disk."""
    #     save_path = r"C:\Jeeva\college\sem 8\Major project\drowsy\models\drowsiness_model_student_Distilled_1_epoch.keras"
        
    #     # Ensure the directory exists before saving
    #     os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
    #     self.student.save(save_path)
    #     print(f"✅ Distilled student model saved at: {save_path}")

#======================================================================
# Configuration
#======================================================================

INPUT_SHAPE = (224, 224, 3)
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001
DATA_DIR = Path("train")
TEST_SIZE = 0.2

#======================================================================
# Data generators
#======================================================================

# train_dir1 = r"D:\Major Project\final datset\Initial_unbalance_data\train"
test_dir1 = r"D:\Major Project\Rasp\old\test"

# Define ImageDataGenerators for training, validation, and testing
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0 / 255,
    # rotation_range=20,
    # width_shift_range=0.2,
    # height_shift_range=0.2,
    # horizontal_flip=True,
    validation_split=0.2  # Split for validation
)

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)

# Create training, validation, and test generators
# train_gen = train_datagen.flow_from_directory(
#     train_dir1,
#     target_size=INPUT_SHAPE[:2],
#     batch_size=BATCH_SIZE,
#     class_mode="categorical",
#     subset="training",
#     shuffle=False
# )

# val_gen = train_datagen.flow_from_directory(
#     train_dir1,
#     target_size=INPUT_SHAPE[:2],
#     batch_size=BATCH_SIZE,
#     class_mode="categorical",
#     subset="validation"
# )

test_gen = test_datagen.flow_from_directory(
    test_dir1,
    target_size=INPUT_SHAPE[:2],
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False  # Ensures label order consistency
)

#======================================================================
# Server socket
#======================================================================

# 🛠️ Declare as real globals
# global_model = None
# past_model_weights = None
lock = threading.Lock()

def server_socket():

    teacher_model = load_model_from_file()
    print("Evaluation of teacher")
    # evaluate_model_start(teacher_model,test_dir1)
    student = MobileNetStudent()
    # model_student = student.model
    # model_student.compile(
    #     optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
    #     loss=CategoricalCrossentropy(from_logits=False),
    #     metrics=[CategoricalAccuracy()]) # Load once at the start
    # history2 = model_student.fit(
    #         train_gen,
    #         validation_data=val_gen,
    #         epochs=5,
    #         verbose=0,
    #         callbacks = student.get_callbacks()
    #     )
    # model_student = tf.keras.models.load_model(r"C:\Jeeva\college\sem 8\Major project\drowsy\models\drowsiness_model_student.keras")
    # print("Evaluation of Pre-distilled student")
    # evaluate_model_start(model_student,test_dir1)
    # distiller = ServerDistiller(teacher_model, model_student, temp=3.0, alpha=0.1)
    # Compile with reasonable learning rate and explicit metrics
    # distiller.compile(
    #     optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    #     metrics=[distiller.acc_metric]
    # )
    # Train model
    # distiller.fit(train_gen, validation_data=val_gen, epochs=5,verbose=0)
    distiller = tf.keras.models.load_model(r"D:\Major Project\Rasp\Final_Results_MEtrcs\rasp_async\distilled.keras")
    print("Evaluation of distilled student")
    # evaluate_model_start(distiller, test_dir1) #use distiller.student when training
    
    # # Save temporary teacher model & calculate size
    # teacher_path = "temp_teacher.keras"
    # save_temp_model(teacher_model, teacher_path)
    # teacher_size = get_model_size_mb(teacher_path)

    # # Save pre-distillation student model & calculate size
    # student_pre_path = "temp_student_pre.keras"
    # save_temp_model(model_student, student_pre_path)
    # student_pre_size = get_model_size_mb(student_pre_path)

    # # Save post-distillation student model & calculate size
    # student_post_path = "temp_student_post.keras"
    # save_temp_model(distiller, student_post_path) #use distiller.student when training
    # student_post_size = get_model_size_mb(student_post_path)

    # # Compute FLOPs for each model
    # teacher_flops = get_flops(teacher_model)
    # student_pre_flops = get_flops(model_student)
    # student_post_flops = get_flops(distiller) #use distiller.student when training

    # Print model sizes and FLOPs
    # print("\n===== Model Size & FLOPs Comparison =====")
    # print(f"Teacher Model        : {teacher_size:.2f} MB | FLOPs: {teacher_flops / 1e9:.3f} GFLOPs")
    # print(f"Student (Pre-Distill): {student_pre_size:.2f} MB | FLOPs: {student_pre_flops / 1e9:.3f} GFLOPs")
    # print(f"Student (Post-Distill): {student_post_size:.2f} MB | FLOPs: {student_post_flops / 1e9:.3f} GFLOPs")

    # # Calculate compression ratios
    # student_pre_compression = calculate_compression_ratio(teacher_size, student_pre_size)
    # student_post_compression = calculate_compression_ratio(teacher_size, student_post_size)

    # Print compression ratios
    # print("\n===== Compression Ratios =====")
    # print(f"Pre-Distillation Student Compression: {student_pre_compression:.2f}x")
    # print(f"Post-Distillation Student Compression: {student_post_compression:.2f}x")
    
    global global_model, past_model_weights, weightarray

    global_model = distiller  # use distiller.student when training
    past_model_weights = global_model.get_weights()
    weightarray = {}  # Store client_id : weights

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(('0.0.0.0', 5000))
    server.listen(10)

    client_sockets = []

    num_clients = 2  # Number of expected clients
    
    # --- received_clients is initialized here ---
    received_clients = {round_num: {} for round_num in range(1, 6)}

    # --- Now, receive received_clients as an argument ---
    def handle_client(client_socket, client_id, received_clients):
        global global_model, past_model_weights
        
        for round_num in range(1, 6):
            try:
                print(f"[Client {client_id}] Sending global model for Round {round_num}...")
                send_global_model(client_socket, global_model, round_num)

                print(f"[Client {client_id}] Waiting for updated weights...")
                client_weights = receive_client_weights(client_socket)

                with lock:
                    print(f"[Client {client_id}] Received weights.")

                    received_clients[round_num][client_id] = client_weights

                    if len(received_clients[round_num]) == num_clients:
                        print(f"[Server] Both clients sent in round {round_num}. Performing aggregation...")

                        temp_weights = list(received_clients[round_num].values())
                        aggregated_weights = fedavg_aggregate(temp_weights, [1] * num_clients)

                        global_model.set_weights(aggregated_weights)
                        past_model_weights = global_model.get_weights()

                        evaluate_model(global_model, r"d:\Major Project\Rasp\old\test")

                        received_clients[round_num] = {}

                    elif len(received_clients[round_num]) == 1:
                        print(f"[Server] Only client {client_id} sent. Updating model without aggregation.")

                        global_model.set_weights(client_weights)
                        past_model_weights = global_model.get_weights()

                        evaluate_model(global_model, r"d:\Major Project\Rasp\old\test")

            except Exception as e:
                print(f"[Client {client_id}] Error: {e}")
                break

        client_socket.close()
        print(f"[Client {client_id}] Connection closed.")

    print("Waiting for clients to connect...")
    for i in range(num_clients):
        client_socket, addr = server.accept()
        print(f"Client {i + 1} connected from {addr}.")
        client_sockets.append(client_socket)

    threads = []
    for i, client_socket in enumerate(client_sockets):
        client_thread = threading.Thread(target=handle_client, args=(client_socket, i + 1, received_clients))
        client_thread.start()
        threads.append(client_thread)

    for t in threads:
        t.join()

    server.close()
    print("Server shutdown.")

if __name__ == "__main__":
    server_socket()