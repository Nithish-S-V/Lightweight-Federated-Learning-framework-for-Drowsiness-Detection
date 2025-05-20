import socket
import struct
import zlib
import tempfile
import tensorflow as tf
import numpy as np
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report
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
class CapsuleLayer(Layer):
    def __init__(self, num_capsule, dim_capsule, num_routing, **kwargs):
        super(CapsuleLayer, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.num_routing = num_routing

    def build(self, input_shape):
        # Define weights for the capsule layer
        self.kernel = self.add_weight(
            shape=(input_shape[-1], self.num_capsule * self.dim_capsule),
            initializer="glorot_uniform",
            trainable=True,
            name="capsule_kernel",
        )
        super(CapsuleLayer, self).build(input_shape)

    def call(self, inputs):
        # Implement the forward pass for the Capsule Layer
        inputs_hat = tf.linalg.matmul(inputs, self.kernel)
        inputs_hat = tf.reshape(inputs_hat, (-1, self.num_capsule, self.dim_capsule))

        b = tf.zeros(shape=(tf.shape(inputs)[0], self.num_capsule, 1))
        for i in range(self.num_routing):
            c = tf.nn.softmax(b, axis=1)
            outputs = self.squash(tf.reduce_sum(c * inputs_hat, axis=1, keepdims=True))
            b += tf.linalg.matmul(inputs_hat, outputs, transpose_b=True)

        return tf.squeeze(outputs, axis=1)

    def squash(self, vectors, axis=-1):
        s_squared_norm = tf.reduce_sum(tf.square(vectors), axis, keepdims=True)
        scale = s_squared_norm / (1 + s_squared_norm) / tf.sqrt(s_squared_norm + K.epsilon())
        return scale * vectors

#--------------------------------------------------------------

# Load pre-trained global model
def load_model_from_file():
    return tf.keras.models.load_model('D:/Major Project/capsule_mobilenet_best_model4.keras', custom_objects={"CapsuleLayer": CapsuleLayer})


# Aggregate weights
def aggregate_weights(client_weights):
    aggregated_weights = []
    num_clients = len(client_weights)
    for layer in client_weights[0]:
        aggregated_weights.append(np.zeros_like(layer))
    for weights in client_weights:
        for i, layer in enumerate(weights):
            aggregated_weights[i] += layer
    aggregated_weights = [layer / num_clients for layer in aggregated_weights]
    return aggregated_weights


# Send the global model to the client
def send_global_model(client, global_model, round_num):
    with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as tmp:
        global_model.save(tmp.name)
        tmp.seek(0)
        model_data = tmp.read()

    compressed_data = zlib.compress(model_data)
    data_length = struct.pack('>I', len(compressed_data))

    client.sendall(data_length)  # Send model size
    chunk_size = 1024 * 1024  # 1 MB per chunk
    with tqdm(total=len(compressed_data), unit="B", unit_scale=True, desc="Sending model") as pbar:
        for i in range(0, len(compressed_data), chunk_size):
            chunk = compressed_data[i:i + chunk_size]
            client.sendall(chunk)
            pbar.update(len(chunk))
    print(f"Global model for Round {round_num} sent to client.")
    client.sendall(struct.pack('>I', round_num))  # Send round number


# Receive weights from the client
def receive_client_weights(client):
    data_length = struct.unpack('>I', client.recv(4))[0]  # Get size of the data
    received_data = b""
    while len(received_data) < data_length:
        packet = client.recv(1024 * 1024)
        if not packet:
            break
        received_data += packet

    decompressed_data = zlib.decompress(received_data)
    with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as tmp:
        tmp.write(decompressed_data)
        tmp.flush()
        model = tf.keras.models.load_model(tmp.name, custom_objects={"CapsuleLayer": CapsuleLayer})
        return model.get_weights()


# Evaluate the global model
from sklearn.metrics import confusion_matrix, accuracy_score

# Evaluate the global model
def evaluate_model(model, dataset_dir):
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
    print(f"Updated Global Model Test Accuracy: {test_accuracy:.2%}")

    # Compute confusion matrix
    conf_matrix = confusion_matrix(true_labels, predictions)
    print("\nConfusion Matrix:")
    print(conf_matrix)

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(true_labels, predictions, target_names=list(data_flow.class_indices.keys())))

# Server socket for federated learning
def server_socket():
    global_model = load_model_from_file()
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(('0.0.0.0', 5000))  # Bind to all available IPs on port 5000
    server.listen(10)  # Listen for connections

    num_clients = 2  # Number of clients participating in federated learning
    client_sockets = []

    # Accept connections from all clients once
    print("Waiting for clients to connect...")
    for i in range(num_clients):
        client, addr = server.accept()
        print(f"Client {i + 1} connected from {addr}.")
        client_sockets.append(client)

    for round_num in range(1, 6):  # Perform 5 rounds
        print(f"\n==== Round {round_num} ====")
        client_weights = []

        # Step 1: Send the appropriate global model to all clients
        for client in client_sockets:
            send_global_model(client, global_model, round_num)
            if round_num == 1:
                print(f"Initial global model sent to client for Round {round_num}.")
            else:
                print(f"Updated global model sent to client for Round {round_num}.")

        # Step 2: Receive updated weights from all clients
        for i, client in enumerate(client_sockets):
            print(f"Receiving weights from client {i + 1} for Round {round_num}...")
            weights = receive_client_weights(client)
            client_weights.append(weights)

        # Step 3: Aggregate weights and update the global model
        aggregated_weights = aggregate_weights(client_weights)
        global_model.set_weights(aggregated_weights)
        print(f"Global model updated after Round {round_num}.")

        # Step 4: Evaluate the updated global model
        evaluate_model(global_model, "D:/Major Project/Fl dataset/Test")  # Replace with test dataset path

    # Step 5: Close all client connections after all rounds
    for client in client_sockets:
        client.close()
        print("Client connection closed.")

    server.close()
    print("Server shutdown.")



if __name__ == "__main__":
    server_socket()
