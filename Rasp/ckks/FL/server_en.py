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
        poly_modulus_degree=16384,
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



def decrypt_weights(encrypted_weights, context, model):
    """Decrypt encrypted weights and reshape them to match the model's structure."""
    model_shapes = [w.shape for w in model.get_weights()]  # Get original weight shapes

    decrypted_weights = [
        np.array(enc_vec.decrypt()).reshape(shape)  
        for enc_vec, shape in zip(encrypted_weights, model_shapes)
    ]

    print(f"Successfully decrypted {len(decrypted_weights)} weight tensors.")
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



# def aggregate_encrypted_weights(encrypted_client_weights):
#     """Perform encrypted FedAvg aggregation with CKKS while preserving precision."""
#     num_clients = len(encrypted_client_weights)
#     aggregated_weights = []

#     for layer_idx in range(len(encrypted_client_weights[0])):
#         layer_sum = encrypted_client_weights[0][layer_idx].copy()

#         # ✅ Sum encrypted weights across all clients
#         for i in range(1, num_clients):
#             layer_sum += encrypted_client_weights[i][layer_idx]

#         # ✅ Use `layer_sum.size()` instead of `len(layer_sum)`
#         scaling_factor = [1 / num_clients] * layer_sum.size()
#         scale_vector = ts.ckks_vector(layer_sum.context(), scaling_factor)
#         aggregated_layer = layer_sum * scale_vector  # Safe division

#         aggregated_weights.append(aggregated_layer)

#     return aggregated_weights
import tenseal as ts

def aggregate_encrypted_weights(encrypted_client_weights):
    """Perform encrypted FedAvg aggregation with CKKS while preserving precision."""
    
    # ✅ Step 1: Ensure input is valid
    if not encrypted_client_weights or not all(encrypted_client_weights):
        print("❌ Error: Empty or invalid encrypted weights received!")
        return None

    num_clients = len(encrypted_client_weights)
    print(f"🔹 Number of clients contributing weights: {num_clients}")

    num_layers = len(encrypted_client_weights[0])
    aggregated_weights = []

    # ✅ Step 2: Verify all weights are CKKS vectors
    for client_idx, weights in enumerate(encrypted_client_weights):
        for layer_idx, vec in enumerate(weights):
            if not isinstance(vec, ts.CKKSVector):
                print(f"❌ Error: Invalid vector at client {client_idx}, layer {layer_idx} - Type: {type(vec)}")
                return None

    # ✅ Step 3: Aggregate encrypted weights layer-wise
    for layer_idx in range(num_layers):
        try:
            layer_sum = encrypted_client_weights[0][layer_idx].copy()
        except Exception as e:
            print(f"❌ Error copying layer {layer_idx}: {e}")
            return None

        # ✅ Step 4: Sum encrypted weights across all clients
        for client_idx in range(1, num_clients):
            try:
                layer_sum += encrypted_client_weights[client_idx][layer_idx]
            except Exception as e:
                print(f"❌ Error adding weights from client {client_idx}, layer {layer_idx}: {e}")
                return None

        # ✅ Step 5: Verify size before scaling
        try:
            layer_size = layer_sum.size()
            scaling_factor = [1 / num_clients] * layer_size
            scale_vector = ts.ckks_vector(layer_sum.context(), scaling_factor)
            aggregated_layer = layer_sum * scale_vector  # Safe division
        except Exception as e:
            print(f"❌ Error scaling layer {layer_idx}: {e}")
            return None

        print(f"✅ Successfully aggregated layer {layer_idx}")

        aggregated_weights.append(aggregated_layer)

    print(f"✅ Aggregation complete. Total layers aggregated: {len(aggregated_weights)}")
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

# def receive_encrypted_weights(client, context):
#     """Receive and correctly deserialize encrypted weights from the client."""
    
#     # ✅ Receive the total data length (first 4 bytes)
#     header = client.recv(4)
#     if not header:
#         raise ConnectionError("Failed to receive data length header")
#     data_length = struct.unpack(">I", header)[0]
#     print(f"Expected data size: {data_length} bytes")

#     # ✅ Ensure full data is received
#     received_data = bytearray()
#     while len(received_data) < data_length:
#         packet = client.recv(min(4096, data_length - len(received_data)))  # 4KB chunks
#         if not packet:
#             raise ConnectionError("Connection lost while receiving data")
#         received_data.extend(packet)

#     print(f"Actual received data size: {len(received_data)} bytes")

#     # ✅ Decompress the received data
#     try:
#         decompressed_data = zlib.decompress(received_data)
#     except zlib.error as e:
#         raise ValueError(f"Decompression failed: {e}")

#     print(f"Decompressed data size: {len(decompressed_data)} bytes")

#     # ✅ Deserialize weights
#     encrypted_weights = []
#     offset = 0

#     while offset < len(decompressed_data):
#         try:
#             # Read the next vector's length (first 4 bytes)
#             vec_length = struct.unpack(">I", decompressed_data[offset:offset+4])[0]
#             offset += 4

#             # Extract and deserialize the vector
#             vec_data = decompressed_data[offset:offset+vec_length]
#             vec = ts.ckks_vector_from(context, vec_data)
#             encrypted_weights.append(vec)
#             offset += vec_length

#         except Exception as e:
#             print(f"Error during deserialization at offset {offset}: {e}")
#             break

#     print(f"Number of encrypted weights received: {len(encrypted_weights)}")

#     return encrypted_weights

# def receive_encrypted_weights(client, context):
#     """Receive and correctly deserialize encrypted weights from the client."""
    
#     # ✅ Step 1: Receive the total data length (first 4 bytes)
#     header = client.recv(4)
#     if not header or len(header) != 4:
#         raise ConnectionError("Failed to receive data length header")
    
#     data_length = struct.unpack(">I", header)[0]
#     print(f"🔹 Expected data size: {data_length} bytes")

#     # ✅ Step 2: Ensure full data is received
#     received_data = bytearray()
#     while len(received_data) < data_length:
#         packet = client.recv(min(4096, data_length - len(received_data)))  # 4KB chunks
#         if not packet:
#             raise ConnectionError("Connection lost while receiving data")
#         received_data.extend(packet)

#     print(f"✅ Received data size: {len(received_data)} bytes")
    
#     if len(received_data) != data_length:
#         print(f"❌ Warning: Data size mismatch! Expected {data_length}, but received {len(received_data)} bytes.")

#     # ✅ Step 3: Print first & last 50 bytes to check for corruption
#     print(f"🔍 Raw received data (first 50 bytes): {received_data[:50]}")
#     print(f"🔍 Raw received data (last 50 bytes): {received_data[-50:]}")

#     # ✅ Step 4: Decompress the received data
#     try:
#         decompressed_data = zlib.decompress(received_data)
#     except zlib.error as e:
#         raise ValueError(f"❌ Decompression failed: {e}")

#     print(f"✅ Decompressed data size: {len(decompressed_data)} bytes")
#     print(f"🔍 Decompressed data (first 50 bytes): {decompressed_data[:50]}")
#     print(f"🔍 Decompressed data (last 50 bytes): {decompressed_data[-50:]}")

#     # ✅ Step 5: Deserialize weights
#     offset = 0
#     serialized_weights = []

#     while offset < len(decompressed_data):
#         if offset + 4 > len(decompressed_data):  # Prevent out-of-bounds read
#             print("❌ Incomplete data: missing vector length header")
#             break

#         vec_length = struct.unpack(">I", decompressed_data[offset:offset+4])[0]
#         offset += 4

#         if offset + vec_length > len(decompressed_data):
#             print(f"❌ Invalid vector length: expected {vec_length} bytes, but only {len(decompressed_data) - offset} bytes remain")
#             break

#         serialized_weights.append(decompressed_data[offset:offset+vec_length])
#         offset += vec_length

#     print(f"✅ Extracted {len(serialized_weights)} serialized weight vectors.")

#     # ✅ Step 6: Debug serialized weights before deserialization
#     for idx, data in enumerate(serialized_weights):
#         if not isinstance(data, bytes):
#             print(f"❌ Invalid data type at index {idx}: {type(data)}")
#         if len(data) == 0:
#             print(f"❌ Empty vector at index {idx}")

#     # ✅ Step 7: Deserialize using list comprehension (Handle exceptions)
#     try:
#         encrypted_weights = [ts.ckks_vector_from(context, data) for data in serialized_weights]
#     except Exception as e:
#         print(f"❌ Error during deserialization: {e}")
#         return None

#     print(f"✅ Number of encrypted weights received: {len(encrypted_weights)}")

#     return encrypted_weights if encrypted_weights else None
import struct
import zlib
import hashlib
import socket

def receive_encrypted_weights(client, context):
    """Efficiently receive and deserialize encrypted weights from the client with integrity check."""

    # ✅ Step 1: Receive header (4-byte length + 32-byte checksum)
    header = bytearray(36)  # Preallocate buffer for efficiency
    bytes_received = client.recv_into(header, 36)
    
    if bytes_received != 36:
        raise ConnectionError("❌ Failed to receive header (Data size + Checksum)")

    # ✅ Step 2: Extract total size and checksum
    total_size = struct.unpack(">I", header[:4])[0]
    expected_checksum = header[4:36]  # 32 bytes
    print(f"🔹 Expected data size: {total_size} bytes")
    print(f"🔹 Expected checksum: {expected_checksum.hex()}")

    # ✅ Step 3: Receive compressed data efficiently
    received_data = bytearray(total_size)
    view = memoryview(received_data)  # Avoid unnecessary copies
    total_received = 0

    while total_received < total_size:
        chunk_size = min(65536, total_size - total_received)  # Use 64KB chunks
        bytes_received = client.recv_into(view[total_received:], chunk_size)
        if not bytes_received:
            raise ConnectionError("❌ Connection lost while receiving data")
        total_received += bytes_received

    print(f"✅ Received data size: {len(received_data)} bytes")

    # ✅ Step 4: Verify checksum for data integrity
    actual_checksum = hashlib.sha256(received_data).digest()
    if actual_checksum != expected_checksum:
        raise ValueError(f"❌ Checksum mismatch! Data may be corrupted.")

    print(f"✅ Checksum verified successfully!")

    # ✅ Step 5: Decompress the data
    try:
        decompressed_data = zlib.decompress(received_data)
    except zlib.error as e:
        raise ValueError(f"❌ Decompression failed: {e}")

    print(f"✅ Decompressed data size: {len(decompressed_data)} bytes")

    # ✅ Step 6: Deserialize encrypted weights
    offset = 0
    serialized_weights = []
    data_len = len(decompressed_data)

    while offset < data_len:
        if offset + 4 > data_len:
            raise ValueError("❌ Incomplete data: Missing vector length header")

        vec_length = struct.unpack(">I", decompressed_data[offset:offset+4])[0]
        offset += 4

        if offset + vec_length > data_len:
            raise ValueError(f"❌ Invalid vector length: Expected {vec_length} bytes, but only {data_len - offset} bytes remain")

        serialized_weights.append(decompressed_data[offset:offset+vec_length])
        offset += vec_length

    print(f"✅ Extracted {len(serialized_weights)} serialized weight vectors.")

    # ✅ Step 7: Deserialize into encrypted weight objects
    try:
        encrypted_weights = [ts.ckks_vector_from(context, data) for data in serialized_weights]
    except Exception as e:
        raise ValueError(f"❌ Error during deserialization: {e}")

    print(f"✅ Successfully received {len(encrypted_weights)} encrypted weights")

    return encrypted_weights if encrypted_weights else None





def print_model_weights(model):
    """Print model layer names, weight names, shapes, and values."""
    for layer in model.layers:
        print(f"Layer Name: {layer.name}")
        for weight in layer.weights:
            print(f"  Weight Name: {weight.name}")
            print(f"  Weight Shape: {weight.shape}")
            print(f"  Weight Values: \n{weight.numpy()}\n")
# Server socket for federated learning
def server_socket():
    context = setup_ckks_context()
    cipher = generate_aes_key()
    global_model = load_model_from_file()
    evaluate_model(global_model, "D:/Major Project/Rasp/Data/test")
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
        encrypted_client_weights = []

        # Step 1: Send the global model to all clients
        for client in client_sockets:
            send_model(client, global_model, cipher)
        # Step 2: Receive updated weights from all clients
        for client in client_sockets:
            encrypted_weights = receive_encrypted_weights(client, context)
            print(type(encrypted_weights))
            encrypted_client_weights.append(encrypted_weights)
        print(type(encrypted_client_weights))
        aggregated_weights = aggregate_encrypted_weights(encrypted_client_weights)
        # Decrypt weights
        decrypted_weights = decrypt_weights(aggregated_weights, context, global_model)

        # Set weights properly
        global_model.set_weights(decrypted_weights)  # ✅ Now it should work
        # print_model_weights(global_model)


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