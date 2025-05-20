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
from tensorflow.keras.losses import KLDivergence


# Generate a shared secret key for encryption (must be the same on server and client)
# KEY = Fernet.generate_key()
# print("Generated Key:", KEY)
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
        L = y_true * tf.square(tf.maximum(0., 0.9 - y_pred)) + \
            0.5 * (1 - y_true) * tf.square(tf.maximum(0., y_pred - 0.1))
        return tf.reduce_mean(tf.reduce_sum(L, axis=1))

# Distilation

class DistillationManager:
    def __init__(self, teacher_model):
        self.teacher = teacher_model
        self.student = self.create_distill_ready_student()
        self.temperature = 4.0  # From Keras example best practices
    
    def create_distill_ready_student(self):
        """Create student with dual outputs for logits and probabilities"""
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(224, 224, 3),
            include_top=False,
            alpha=0.35,
            weights=None
        )
        
        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)
        
        # Logits output
        true_out = layers.Dense(2, name='true_logits')(x)
        
        # Soft probabilities with temperature scaling
        soft_out = layers.Lambda(lambda x: x/self.temperature, name='temp_scaling')(true_out)
        soft_out = layers.Activation('softmax', name='soft_probs')(soft_out)
        
        return tf.keras.Model(inputs=base_model.input, outputs=[true_out, soft_out])

    def generate_soft_predictions(self, temp=4.0):
        """Generate temperature-scaled teacher predictions"""
        self.temperature = temp
        logits = self.teacher.predict(self.proxy_data)
        return tf.nn.softmax(logits / self.temperature).numpy()



# Load pre-trained global model
def load_model_from_file():
    return tf.keras.models.load_model("D:/Major Project/Rasp/Data/drowsiness_model.keras",
                                      custom_objects={'CapsuleLayer': CapsuleLayer,
                                                      'Length': Length,
                                                      'margin_loss': margin_loss})

def aggregate_weights(client_weights, beta=0.2):
    """Implements Federated Trimmed Averaging (FedTrimmedAvg) aggregation"""
    if not client_weights:
        return []
    
    num_clients = len(client_weights)
    k = int(beta * num_clients)
    k = max(0, min(k, (num_clients - 1) // 2))  # Ensure valid trim size
    
    aggregated_weights = []
    
    # Process each layer independently
    for layer_idx in range(len(client_weights[0])):
        # Stack all clients' layer weights into array
        layer_stack = np.array([client[layer_idx] for client in client_weights])
        original_shape = layer_stack.shape[1:]
        
        # Flatten for parameter-wise processing
        flattened = layer_stack.reshape(num_clients, -1)
        
        # Sort and trim parameters
        sorted_flat = np.sort(flattened, axis=0)
        trimmed = sorted_flat[k:num_clients-k] if k > 0 else sorted_flat
        
        # Compute mean of remaining values
        avg_flat = np.mean(trimmed, axis=0)
        
        # Restore original layer shape
        aggregated_weights.append(avg_flat.reshape(original_shape))
    
    return aggregated_weights


# Send the global model to the client
def send_global_model(client, model, soft_labels, round_num):
    # Send compressed and encrypted soft labels
    client.sendall(struct.pack('>II', soft_labels.shape[0], soft_labels.shape[1]))
    
    # Existing compression/encryption code
    compressed_logits = zlib.compress(soft_labels.tobytes())
    encrypted_logits = cipher_suite.encrypt(compressed_logits)
    client.sendall(encrypted_logits)

    # Send compressed and encrypted model
    with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as tmp:
        model.save(tmp.name)
        tmp.seek(0)
        model_data = tmp.read()

    compressed_data = zlib.compress(model_data)
    encrypted_data = cipher_suite.encrypt(compressed_data)
    data_length = struct.pack('>I', len(encrypted_data))

    # Send model header and chunked data
    client.sendall(data_length)
    chunk_size = 1024 * 1024  # 1 MB chunks
    with tqdm(total=len(encrypted_data), unit="B", unit_scale=True, desc="Sending model") as pbar:
        for i in range(0, len(encrypted_data), chunk_size):
            chunk = encrypted_data[i:i + chunk_size]
            client.sendall(chunk)
            pbar.update(len(chunk))

    # Finalize transmission
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
        model = tf.keras.models.load_model(tmp.name, custom_objects={'CapsuleLayer': CapsuleLayer,
                                                      'Length': Length,
                                                      'margin_loss': margin_loss})
        return model.get_weights()

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

# Server socket for federated learning
def server_socket():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(('0.0.0.0', 5000))  # Bind to all available IPs on port 5000
    server.listen(10)  # Listen for connections

    num_clients = 2  # Number of clients participating in federated learning
    client_sockets = []
    teacher = load_model_from_file()
    distill_manager = DistillationManager(teacher)
    soft_labels = distill_manager.generate_soft_labels() 
    print("Soft labels shape:", soft_labels.shape)# Uncomment this line
    compressed_labels = distill_manager.compress_logits(soft_labels)
    print("Soft labels shape:", compressed_labels.shape)
    global_model = distill_manager.student  # Initialize with student model

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
            send_global_model(client, global_model, compressed_labels, round_num)  

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
        evaluate_model(global_model, "D:/Major Project/Rasp/Data/test")

    # Step 5: Close all client connections after all rounds
    for client in client_sockets:
        client.close()
        print("Client connection closed.")

    server.close()
    print("Server shutdown.")

if __name__ == "__main__":
    server_socket()