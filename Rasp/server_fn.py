import socket
import threading
import numpy as np
import tensorflow as tf
from pathlib import Path
from cryptography.fernet import Fernet
import hashlib
import base64
import pickle
import zlib
import tempfile
from tensorflow.keras import layers, initializers, backend as K
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tensorflow.keras.saving import register_keras_serializable

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

# Federated Server Implementation
class FederatedServer:
    def __init__(self, model_path, host='0.0.0.0', port=5000, trim_percent=0.2):
        self.host = host
        self.port = port
        self.clients = {}
        self.lock = threading.Lock()
        self.trim_percent = trim_percent
        self.global_model = tf.keras.models.load_model(
            model_path,
            custom_objects={
                'CapsuleLayer': CapsuleLayer,
                'Length': Length,
                'margin_loss': margin_loss
            }
        )
        # Initialize encryption
        self._initialize_encryption()
        self.server_thread = threading.Thread(target=self._run_server)
        self.server_thread.start()

    def _initialize_encryption(self, custom_key="federated-secret-key"):
        # PBKDF2 key derivation with 100k iterations
        salt = b'secure_salt_123'
        kdf = hashlib.pbkdf2_hmac(
            'sha256',
            custom_key.encode(),
            salt,
            100000,  # High iteration count
            dklen=32
        )
        self.fernet_key = base64.urlsafe_b64encode(kdf)
        self.cipher = Fernet(self.fernet_key)


    def _run_server(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((self.host, self.port))
            s.listen(5)
            print(f"Server listening on {self.host}:{self.port}")
            
            while True:
                conn, addr = s.accept()
                threading.Thread(target=self._handle_client, args=(conn, addr)).start()

    # server_fn.py (Corrected _handle_client)
    def _handle_client(self, conn, addr):
        try:
            with conn:
                # 1. Key Exchange First
                conn.sendall(self.fernet_key)
                
                # 2. Send Initial Model
                self._send_model(conn, 0)
                
                # 3. FL Rounds
                while True:
                    # 4. Receive Client Update
                    encrypted_update = self._receive_all(conn)
                    client_weights = pickle.loads(zlib.decompress(self.cipher.decrypt(encrypted_update)))
                    
                    with self.lock:
                        self.clients[addr] = client_weights
                    
                    # 5. Aggregate & Send Updated Model
                    self.federated_averaging()
                    self._send_model(conn, self.current_round)

        
        except Exception as e:
            print(f"Client {addr} error: {str(e)}")
        finally:
            with self.lock:
                if addr in self.clients:
                    del self.clients[addr]

    def _send_model(self, conn, round_num):
        try:
            with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as tmp:
                self.global_model.save(tmp.name)
                tmp.seek(0)
                model_data = tmp.read()
            
            compressed = zlib.compress(model_data)
            encrypted = self.cipher.encrypt(compressed)
            
            # Send data size
            conn.sendall(len(encrypted).to_bytes(4, 'big'))
            
            # Chunked transfer
            chunk_size = 1024 * 1024  # 1MB
            for i in range(0, len(encrypted), chunk_size):
                conn.sendall(encrypted[i:i+chunk_size])
            
            # Send round number
            conn.sendall(round_num.to_bytes(4, 'big'))
        
        finally:
            Path(tmp.name).unlink(missing_ok=True)

    def federated_averaging(self):
        with self.lock:
            if len(self.clients) < 2:
                return

            # Get all client updates and current global weights
            client_weights = list(self.clients.values())
            global_weights = self.global_model.get_weights()

            # 1. Calculate update magnitudes using L2 norm
            update_magnitudes = [
                np.mean([np.linalg.norm(cw - gw) 
                    for cw, gw in zip(weights, global_weights)])
                for weights in client_weights
            ]

            # 2. Sort clients by update magnitude
            sorted_indices = np.argsort(update_magnitudes)
            sorted_weights = [client_weights[i] for i in sorted_indices]

            # 3. Trim 20% of clients (10% from each end)
            trim_n = int(len(client_weights) * self.trim_percent / 2)
            valid_weights = sorted_weights[trim_n:-trim_n] if trim_n > 0 else sorted_weights

            # 4. Layer-wise IQR filtering and aggregation
            aggregated_weights = []
            for layer_idx in range(len(valid_weights[0])):
                layer_values = [client[layer_idx] for client in valid_weights]
                
                # Calculate IQR for the layer
                q25, q75 = np.percentile(layer_values, [25, 75], axis=0)
                iqr = q75 - q25
                
                # Define valid range (1.5*IQR from quartiles)
                lower_bound = q25 - 1.5 * iqr
                upper_bound = q75 + 1.5 * iqr
                
                # Filter outliers
                mask = np.all((layer_values >= lower_bound) & 
                            (layer_values <= upper_bound), axis=1)
                filtered = np.compress(mask, layer_values, axis=0)
                
                # Mean aggregation of valid updates
                aggregated_weights.append(np.mean(filtered, axis=0))

            # Update global model with aggregated weights
            self.global_model.set_weights(aggregated_weights)


    def run(self, rounds=5, test_dir=None):
        print(f"Starting federated learning ({rounds} rounds)")
        for round_num in range(1, rounds+1):
            self.current_round = round_num
            print(f"\n=== Round {round_num}/{rounds} ===")
            
            # Wait for clients
            while len(self.clients) < 2: pass
            
            # Aggregate updates
            self.federated_averaging()
            
            # Evaluate
            if test_dir:
                datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
                test_gen = datagen.flow_from_directory(
                    test_dir,
                    target_size=(224, 224),
                    batch_size=32,
                    class_mode='binary',
                    shuffle=False
                )
                
                y_pred = np.argmax(self.global_model.predict(test_gen), axis=1)
                y_true = test_gen.classes
                
                print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
                print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    # Helper methods remain the same
    def _receive_data_size(self, conn):
        size_data = self._receive_all(conn, 4)
        return int.from_bytes(size_data, byteorder='big') if size_data else 0

    def _receive_all(self, conn, length):
        data = b''
        while len(data) < length:
            packet = conn.recv(length - len(data))
            if not packet: break
            data += packet
        return data

    def process_client_update(self, encrypted_data):
        decrypted = self.cipher.decrypt(encrypted_data)
        return pickle.loads(zlib.decompress(decrypted))

    def evaluate_global_model(self, test_dir):
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
        test_gen = datagen.flow_from_directory(
            test_dir,
            target_size=(224, 224),
            batch_size=32,
            class_mode='binary',
            shuffle=False
        )
        
        y_pred = np.argmax(self.global_model.predict(test_gen), axis=1)
        y_true = test_gen.classes
        
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'confusion_matrix': confusion_matrix(y_true, y_pred)
        }

    def save_final_model(self, save_path="final_federated_model.keras"):
        self.global_model.save(save_path)
        print(f"Final model saved to {save_path}")

# if __name__ == "__main__":
#     server = FederatedServer(
#         initial_model_path="D:/Major Project/Rasp/Data/drowsiness_model.keras",
#         host='0.0.0.0',
#         port=5000,
#         trim_percent=0.2
#     )
    
#     try:
#         server.run_federated_learning(
#             num_rounds=5,
#             test_dir="D:/Major Project/Rasp/Data/test"
#         )
#     finally:
#         server.save_final_model()
if __name__ == "__main__":
    server = FederatedServer(
        model_path="D:/Major Project/Rasp/Data/drowsiness_model.keras",
        host='0.0.0.0',
        port=5000,
        trim_percent=0.2
    )
    server.run(rounds=5, test_dir="D:/Major Project/Rasp/Data/test")
