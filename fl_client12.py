import socket
import struct
import zlib
import tempfile
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
import numpy as np
from tqdm import tqdm


# Custom Capsule Layer for loading the model
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
def load_image_data(base_dir, set_num):
    dataset_dir = f"{base_dir}/Set {set_num}"
    datagen = ImageDataGenerator(rescale=1.0 / 255)
    return datagen.flow_from_directory(
        dataset_dir, target_size=(128, 128), batch_size=16, class_mode='categorical', shuffle=True
    )


# Evaluate the model
def evaluate_model(model, data_flow):
    predictions = np.argmax(model.predict(data_flow), axis=1)
    true_labels = data_flow.classes
    print("\nClassification Report:")
    print(classification_report(true_labels, predictions, target_names=data_flow.class_indices.keys()))


# Client function
def client_socket(base_dir):
    for set_num in range(1, 6):  # Automate rounds for all 5 sets
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.settimeout(30)  # Set a timeout for robustness
        client.connect(('localhost', 5000))  # Replace with server's IP
        print(f"Connected to server for training on Set {set_num}...")

        # Step 1: Receive the model size
        data_length = struct.unpack('>I', client.recv(4))[0]
        print(f"Receiving model ({data_length} bytes)...")

        # Step 2: Receive the compressed model
        received_data = b""
        with tqdm(total=data_length, unit="B", unit_scale=True, desc="Receiving model") as pbar:
            while len(received_data) < data_length:
                packet = client.recv(1024 * 1024)
                if not packet:
                    break
                received_data += packet
                pbar.update(len(packet))

        # Step 3: Decompress the model
        decompressed_data = zlib.decompress(received_data)

        # Step 4: Load the global model
        with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as tmp:
            tmp.write(decompressed_data)
            tmp.flush()
            model = tf.keras.models.load_model(tmp.name, custom_objects={"CapsuleLayer": CapsuleLayer})

        print(f"Model for Round {set_num} loaded successfully.")

        # Step 5: Train locally
        data_flow = load_image_data(base_dir, set_num)
        print(f"Training locally on Set {set_num}...")
        model.fit(data_flow, epochs=2, verbose=1)

        # Evaluate the locally trained model
        evaluate_model(model, data_flow)

        # Step 6: Send updated weights to the server
        with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as tmp:
            model.save(tmp.name)
            tmp.seek(0)
            updated_weights = tmp.read()

        compressed_data = zlib.compress(updated_weights)
        compressed_size = len(compressed_data)

        print(f"Sending updated weights ({compressed_size} bytes) to the server...")
        client.sendall(struct.pack('>I', compressed_size))  # Send size
        with tqdm(total=compressed_size, unit="B", unit_scale=True, desc="Uploading weights") as pbar:
            client.sendall(compressed_data)
            pbar.update(compressed_size)

        print(f"Training on Set {set_num} completed.\n")
        client.close()


if __name__ == "__main__":
    client_socket('D:/Major Project/Fl dataset/Jammu and Kashmir')  # For Jammu
