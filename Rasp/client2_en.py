import socket
import struct
import zlib
import tempfile
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, accuracy_score
from tensorflow.keras.saving import register_keras_serializable
from cryptography.fernet import Fernet
import base64
import hashlib

# # Use the same key as the server
# KEY = b'ySWKD3Esbml8jDJWZZiXwkSMDKyv96MHUvGLnaKhVMo='  # Replace with the actual key shared with the server
# cipher_suite = Fernet(KEY)

# Your custom key (e.g., a name or phrase)
custom_key = "secretkey"
# Hash the custom key using SHA-256 to get a 32-byte value
hashed_key = hashlib.sha256(custom_key.encode()).digest()
# Encode the hashed key as a URL-safe base64 string
encoded_key = base64.urlsafe_b64encode(hashed_key)
# Use the encoded key
KEY = encoded_key
cipher_suite = Fernet(KEY)

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

def load_image_data(base_dir, set_num):
    dataset_dir = f"{base_dir}/Set {set_num}"
    datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)  # 20% for validation
    
    train_flow = datagen.flow_from_directory(
        dataset_dir,
        target_size=(128, 128),
        batch_size=16,
        class_mode='categorical',
        subset='training',  # Specify training subset
        shuffle=True
    )

    val_flow = datagen.flow_from_directory(
        dataset_dir,
        target_size=(128, 128),
        batch_size=16,
        class_mode='categorical',
        subset='validation',  # Specify validation subset
        shuffle=True
    )
    
    return train_flow, val_flow

# Evaluate the global model
def evaluate_model(model, dataset_dir):
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)
    data_flow = datagen.flow_from_directory(
        dataset_dir,
        target_size=(128, 128),
        batch_size=16,
        class_mode='categorical',
        shuffle=False,
    )
    predictions = np.argmax(model.predict(data_flow), axis=1)
    true_labels = data_flow.classes
    test_accuracy = accuracy_score(true_labels, predictions)
    print(f"Test Accuracy (Jammu Set): {test_accuracy:.2%}")
    print("\nClassification Report after updating global model:")
    print(classification_report(true_labels, predictions, target_names=list(data_flow.class_indices.keys())))

# Client function
def client_socket(base_dir):
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(('192.168.77.22', 5000))  # Replace with server's IP
    print("Connected to server.")

    for set_num in range(1, 6):  # Automate rounds for all 5 sets
        print(f"\n==== Round {set_num} ====")

        # Step 2: Receive the encrypted model size
        data_length = struct.unpack('>I', client.recv(4))[0]
        print(f"Receiving encrypted model ({data_length} bytes)...")

        # Step 3: Receive the encrypted model
        received_data = b""
        with tqdm(total=data_length, unit="B", unit_scale=True, desc="Receiving model") as pbar:
            while len(received_data) < data_length:
                packet = client.recv(1024 * 1024)  # Receive in 1 MB chunks
                if not packet:
                    break
                received_data += packet
                pbar.update(len(packet))

        # Step 4: Decrypt the model
        decrypted_data = cipher_suite.decrypt(received_data)  # Decrypt the data
        decompressed_data = zlib.decompress(decrypted_data)

        # Step 5: Load the received global model
        with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as tmp:
            tmp.write(decompressed_data)
            tmp.flush()
            model = tf.keras.models.load_model(tmp.name, custom_objects={"CapsuleLayer": CapsuleLayer})
        print(f"Model for Round {set_num} loaded successfully.")

        train_flow, val_flow = load_image_data(base_dir, set_num)
        print(f"Training locally on Set {set_num}...")
        model.fit(train_flow, epochs=1, validation_data=val_flow, verbose=0)

        # Evaluate the locally trained model
        evaluate_model(model, "D:/Major Project/Fl dataset/Test")

        # Step 7: Send updated weights back to the server
        with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as tmp:
            model.save(tmp.name)
            tmp.seek(0)
            updated_weights = tmp.read()

        compressed_data = zlib.compress(updated_weights)
        encrypted_data = cipher_suite.encrypt(compressed_data)  # Encrypt the data
        compressed_size = len(encrypted_data)

        print(f"Sending updated weights ({compressed_size} bytes) to the server...")
        client.sendall(struct.pack('>I', compressed_size))  # Send the size of the encrypted data
        with tqdm(total=compressed_size, unit="B", unit_scale=True, desc="Uploading weights") as pbar:
            client.sendall(encrypted_data)
            pbar.update(compressed_size)

        print(f"Training and weight update for Round {set_num} completed.\n")

    # Step 8: Close the connection after all rounds
    client.close()
    print("Connection to server closed.")

if __name__ == "__main__":
    client_socket('D:/Major Project/Fl dataset/Jammu and Kashmir')