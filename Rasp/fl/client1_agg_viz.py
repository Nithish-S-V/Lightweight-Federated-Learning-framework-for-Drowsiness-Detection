import socket
import struct
import zlib
import tempfile
import tensorflow as tf
from sklearn.metrics import classification_report
import numpy as np
from tqdm import tqdm
from tensorflow.keras import layers, initializers, backend as K # type: ignore
from tensorflow.keras.saving import register_keras_serializable
from cryptography.fernet import Fernet
from sklearn.metrics import accuracy_score, f1_score, recall_score, classification_report, confusion_matrix
import base64
import hashlib
import time
import os

# Use the same key as the server
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


time_list=[]
INPUT_SHAPE = (224, 224, 3)
BATCH_SIZE = 16
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

@tf.keras.saving.register_keras_serializable(package="Custom", name="margin_loss")
def margin_loss(y_true, y_pred):
    # Remove explicit one-hot conversion if labels are already categorical
    if len(y_true.shape) == 2:  # Already one-hot encoded
        y_true_ = y_true
    else:  # Convert sparse labels to one-hot
        y_true_ = tf.one_hot(tf.cast(y_true, tf.int32), depth=2)
    
    L = y_true_ * tf.square(tf.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true_) * tf.square(tf.maximum(0., y_pred - 0.1))
    return tf.reduce_mean(tf.reduce_sum(L, axis=1))


@register_keras_serializable(package="Custom")
class CapsuleLayer(layers.Layer):
    def __init__(self, num_capsule, dim_capsule, routings=3, **kwargs):
        super(CapsuleLayer, self).__init__(**kwargs)
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
            outputs = squash(tf.reduce_sum(inputs_hat * c_expand, axis=1))
            
            if i < self.routings - 1:
                b += tf.reduce_sum(inputs_hat * K.expand_dims(c, -1), axis=-1)
        
        return outputs

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
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss=margin_loss,
            metrics=['accuracy'],
            experimental_run_tf_function=False
        )


def load_image_data(base_dir, set_num):
    dataset_dir = f"{base_dir}/set{set_num}"
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                rescale=1./255,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True,
                validation_split=0.2,
                
            )
    
    train_flow = datagen.flow_from_directory(
        dataset_dir,
        target_size=INPUT_SHAPE[:2],
        batch_size=BATCH_SIZE,
        class_mode='categorical',  # Change from 'binary' to 'categorical'
        color_mode='rgb',
        subset='training',
        shuffle=True 
    )

    val_flow = datagen.flow_from_directory(
        dataset_dir,
        target_size=INPUT_SHAPE[:2],
        batch_size=BATCH_SIZE,
        class_mode='categorical',  # Change from 'binary' to 'categorical'
        color_mode='rgb',
        subset='validation'
    )
    
    return train_flow, val_flow
# @tf.keras.saving.register_keras_serializable(package="Custom", name="binary_crossentropy_loss")
def binary_crossentropy_loss(y_true, y_pred):
    return tf.keras.losses.binary_crossentropy(y_true, y_pred)
# Evaluate the global model
from pathlib import Path
INPUT_SHAPE = (224, 224, 3)
BATCH_SIZE = 32
DATA_DIR = Path("D:/Major Project/Rasp/Data")
TEST_DIR = r"D:\Major Project\Final Proper\FederatedData_balanced\test"
# List of aggregation methods
AGGREGATION_METHODS = ["FedAvg", "FedTrimmedAvg", "FedMA", "FedCDA", "FedPA", "FedBN"]#[ "FedCDA"]#

# Track total number of evaluations so far
total_set_counter = 0  # This should be managed outside the function if calling repeatedly

def evaluate_model(model, dataset_dir, set_num, start_time, end_time, model_size_kb):
    global total_set_counter

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

    # Predict
    y_pred = np.argmax(model.predict(test_gen), axis=1)
    y_true = test_gen.classes
    acc = np.mean(y_true == y_pred)
    report = classification_report(
        y_true, y_pred,
        target_names=['Eyeclose', 'Neutral', 'Yawn'],
        zero_division=0
    )

    # Determine method index using total_set_counter
    method_index = total_set_counter // 3
    method_name = AGGREGATION_METHODS[method_index] if method_index < len(AGGREGATION_METHODS) else "UnknownMethod"

    # File to write (one per aggregation method)
    file_name = f"Client_One_Evaluation_results_{method_name}.txt"

    # Print to console
    print(f"Method: {method_name} | Set: {set_num}")
    print(f"Accuracy: {acc:.4f}")
    print("Classification Report:")
    print(report)

    # Append results to the file
    with open(file_name, "a") as f:
        f.write(f"\n--- {method_name} | Set {set_num} ---\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\n" + "="*60 + "\n")

    total_set_counter += 1


from tensorflow.keras.models import load_model

def client_socket(base_dir):
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        client.connect(('localhost', 5000))  # Replace with server's IP
        print("Connected to server.")

        while True:
            for set_num in range(1, 4):  # Automate rounds for all 3 sets
                print(f"\n==== Round {set_num} ====")

                try:
                    # Step 2: Receive the encrypted model size
                    raw_size = client.recv(4)
                    if not raw_size:
                        print("Server disconnected during size receive.")
                        return
                    data_length = struct.unpack('>I', raw_size)[0]
                    print(f"Receiving encrypted model ({data_length} bytes)...")

                    # Step 3: Receive the encrypted model
                    received_data = b""
                    with tqdm(total=data_length, unit="B", unit_scale=True, desc="Receiving model") as pbar:
                        while len(received_data) < data_length:
                            packet = client.recv(min(1024 * 1024, data_length - len(received_data)))
                            if not packet:
                                print("Server disconnected during data receive.")
                                return
                            received_data += packet
                            pbar.update(len(packet))

                    # Step 4: Decrypt and decompress the model
                    decrypted_data = cipher_suite.decrypt(received_data)
                    decompressed_data = zlib.decompress(decrypted_data)

                    # Step 5: Load model
                    with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as tmp:
                        tmp.write(decompressed_data)
                        tmp.flush()
                        model = load_model(tmp.name, custom_objects={
                            "CapsuleLayer": CapsuleLayer,
                            "Length": Length,
                            "margin_loss": margin_loss
                        })
                        model.save(tmp.name)
                        model_size_kb = os.path.getsize(tmp.name) / 1024

                    print(f"Model for Round {set_num} loaded successfully.")

                    train_flow, val_flow = load_image_data(base_dir, set_num)
                    print(f"Training locally on Set {set_num}...")
                    start_time = time.time()
                    model.fit(train_flow, epochs=5, validation_data=val_flow, verbose=0)
                    end_time = time.time()

                    evaluate_model(model, r"D:\Major Project\Final Proper\FederatedData_balanced\test",set_num, start_time, end_time, model_size_kb)

                    # Step 7: Send updated weights
                    with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as tmp:
                        model.save(tmp.name)
                        tmp.seek(0)
                        updated_weights = tmp.read()

                    compressed_data = zlib.compress(updated_weights)
                    encrypted_data = cipher_suite.encrypt(compressed_data)
                    compressed_size = len(encrypted_data)

                    print(f"Sending updated weights ({compressed_size} bytes) to the server...")
                    client.sendall(struct.pack('>I', compressed_size))
                    with tqdm(total=compressed_size, unit="B", unit_scale=True, desc="Uploading weights") as pbar:
                        client.sendall(encrypted_data)
                        pbar.update(compressed_size)

                    print(f"Training and weight update for Round {set_num} completed.\n")

                except (socket.error, struct.error, Exception) as e:
                    print(f"Error during round {set_num}: {e}")
                    return  # Exit the function on error

    except Exception as e:
        print(f"Could not connect to server: {e}")
    finally:
        client.close()
        print("Connection to server closed.")

if __name__ == "__main__":
    client_socket(r"D:\Major Project\Final Proper\FederatedData_balanced\client1")
