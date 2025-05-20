import socket
import struct
import zlib
import tempfile
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
import numpy as np
from tqdm import tqdm

from tensorflow.keras.layers import Layer
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.saving import register_keras_serializable
from tensorflow.keras import backend as K


@register_keras_serializable(package="Custom")
class CapsuleLayer(tf.keras.layers.Layer):
    def __init__(self, num_capsule, dim_capsule, num_routing, **kwargs):
        super(CapsuleLayer, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.num_routing = num_routing

    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=(input_shape[-1], self.num_capsule * self.dim_capsule),
            initializer="glorot_uniform",
            trainable=True,
            name="capsule_kernel"
        )
        super(CapsuleLayer, self).build(input_shape)

    def call(self, inputs):
        inputs_hat = tf.linalg.matmul(inputs, self.kernel)
        inputs_hat = tf.reshape(inputs_hat, (-1, self.num_capsule, self.dim_capsule))

        b = tf.zeros(shape=(tf.shape(inputs)[0], self.num_capsule, 1))
        for _ in range(self.num_routing):
            c = tf.nn.softmax(b, axis=1)
            outputs = self.squash(tf.reduce_sum(c * inputs_hat, axis=1, keepdims=True))
            b += tf.linalg.matmul(inputs_hat, outputs, transpose_b=True)

        return tf.squeeze(outputs, axis=1)

    def squash(self, vectors, axis=-1):
        s_squared_norm = tf.reduce_sum(tf.square(vectors), axis, keepdims=True)
        scale = s_squared_norm / (1 + s_squared_norm) / tf.sqrt(s_squared_norm + tf.keras.backend.epsilon())
        return scale * vectors


# Load images for training and evaluation
# Load images for training and evaluation
def load_image_data(base_dir, set_num):
    dataset_dir = f"{base_dir}/Set {set_num}"  # Directory for the current set
    datagen = ImageDataGenerator(rescale=1.0 / 255)
    
    # Debugging check to verify the directory
    print(f"Loading data from: {dataset_dir}")
    
    return datagen.flow_from_directory(
        dataset_dir, 
        target_size=(128, 128), 
        batch_size=16, 
        class_mode='categorical', 
        shuffle=True
    )

from sklearn.metrics import confusion_matrix, accuracy_score

# Evaluate the model
def evaluate_model(model, dataset_dir,set_num):
    # Load the test dataset using ImageDataGenerator
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)
    data_flow = datagen.flow_from_directory(
        dataset_dir,
        target_size=(128, 128),
        batch_size=16,
        class_mode='categorical',
        shuffle=False,
    )

    # Predict on the test data
    predictions = np.argmax(model.predict(data_flow), axis=1)  # Convert probabilities to class labels
    true_labels = data_flow.classes  # True labels from the dataset

    # Compute test accuracy
    test_accuracy = accuracy_score(true_labels, predictions)
    print(f"Test Accuracy (Himachal Pradesh Set {set_num}): {test_accuracy:.2%}")

    # Compute confusion matrix
    conf_matrix = confusion_matrix(true_labels, predictions)
    print("\nConfusion Matrix:")
    print(conf_matrix)

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(true_labels, predictions, target_names=list(data_flow.class_indices.keys())))


# Client function
def client_socket(base_dir):
    # Step 1: Establish connection with the server
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(('localhost', 5000))  # Replace with server's IP
    print("Connected to server.")

    for set_num in range(1, 6):  # Automate rounds for all 5 sets
        print(f"\n==== Round {set_num} ====")

        # Step 2: Receive the model size
        data_length = struct.unpack('>I', client.recv(4))[0]
        print(f"Receiving model ({data_length} bytes)...")

        # Step 3: Receive the compressed model
        received_data = b""
        with tqdm(total=data_length, unit="B", unit_scale=True, desc="Receiving model") as pbar:
            while len(received_data) < data_length:
                packet = client.recv(1024 * 1024)  # Receive in 1 MB chunks
                if not packet:
                    break
                received_data += packet
                pbar.update(len(packet))

        # Step 4: Decompress the model
        decompressed_data = zlib.decompress(received_data)

        # Step 5: Load the received global model
        with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as tmp:
            tmp.write(decompressed_data)
            tmp.flush()
            model = tf.keras.models.load_model(tmp.name, custom_objects={"CapsuleLayer": CapsuleLayer})
        print(f"Model for Round {set_num} loaded successfully.")

        # Step 6: Train the model locally on the client's dataset
        data_flow = load_image_data(base_dir, set_num)
        print(f"Training locally on Set {set_num}...")
        model.fit(data_flow, epochs=20, verbose=1)

        # Evaluate the locally trained model
        set_dir = f"{base_dir}/Set {set_num}"
        evaluate_model(model, set_dir, set_num)  # Pass the directory for evaluation


        # Step 7: Send updated weights back to the server
        with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as tmp:
            model.save(tmp.name)
            tmp.seek(0)
            updated_weights = tmp.read()

        compressed_data = zlib.compress(updated_weights)
        compressed_size = len(compressed_data)

        print(f"Sending updated weights ({compressed_size} bytes) to the server...")
        client.sendall(struct.pack('>I', compressed_size))  # Send the size of the compressed data
        with tqdm(total=compressed_size, unit="B", unit_scale=True, desc="Uploading weights") as pbar:
            client.sendall(compressed_data)
            pbar.update(compressed_size)

        print(f"Training and weight update for Round {set_num} completed.\n")

    # Step 8: Close the connection after all rounds
    client.close()
    print("Connection to server closed.")


if __name__ == "__main__":
    client_socket('D:/Major Project/Fl dataset/Himachal Pradesh')  # For Jammu
