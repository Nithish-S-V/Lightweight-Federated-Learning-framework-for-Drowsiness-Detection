import socket
import struct
import zlib
import tempfile
import tensorflow as tf
from sklearn.metrics import classification_report
import numpy as np
from tqdm import tqdm
import tenseal as ts
from tensorflow.keras import layers, initializers, backend as K # type: ignore
from tensorflow.keras.saving import register_keras_serializable
from sklearn.metrics import accuracy_score, f1_score, recall_score, classification_report, confusion_matrix
import base64
import hashlib
import time
import os
from cryptography.fernet import Fernet

time_list=[]
def setup_ckks_context():
    """Setup CKKS encryption context for homomorphic encryption."""
    context = ts.context(
        scheme=ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60]
    )
    context.global_scale = 2 ** 40
    context.generate_galois_keys()
    return context

import tenseal as ts
import numpy as np

def encrypt_weights(model, context):
    """Encrypt model weights using CKKS encryption."""
    encrypted_weights = []

    # Extract model weights
    model_weights = model.get_weights()
    print(f"Extracted {len(model_weights)} weight tensors from the model")

    if len(model_weights) == 0:
        raise ValueError("Error: Model has no weights to encrypt!")

    for i, layer in enumerate(model_weights):
        flattened = layer.flatten()
        
        # Ensure the layer has data
        if flattened.size == 0:
            print(f"Warning: Layer {i} is empty!")

        encrypted_weights.append(ts.ckks_vector(context, flattened))

    print(f"Successfully encrypted {len(encrypted_weights)} weight tensors")
    
    if len(encrypted_weights) == 0:
        raise ValueError("Error: No weights were encrypted!")

    return encrypted_weights




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
    dataset_dir = f"{base_dir}/round_{set_num}"
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            validation_split=0.2
        )  # 20% for validation
    
    train_flow = datagen.flow_from_directory(
        dataset_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='training',  # Specify training subset
        shuffle=True
    )

    val_flow = datagen.flow_from_directory(
        dataset_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='validation',  # Specify validation subset
        shuffle=True
    )
    
    return train_flow, val_flow

# Evaluate the global model
def evaluate_model(model, dataset_dir,start_time,end_time,model_size_kb):
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)
    data_flow = datagen.flow_from_directory(
        dataset_dir,
        target_size=(224,224),
        batch_size=32,
        class_mode='categorical',
        shuffle=False,
    )
    predictions = np.argmax(model.predict(data_flow), axis=1)
    true_labels = data_flow.classes
    test_accuracy = accuracy_score(true_labels, predictions)
    confusion_mat = confusion_matrix(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average='weighted')
    recall = recall_score(true_labels, predictions, average='weighted')
    # Model size and number of parameters
    num_params = model.count_params()
    training_time = end_time - start_time
    time_list.append(training_time)
    print(f"Test Accuracy :    {test_accuracy:.2%}")
    print(f"F1 Score:          {f1:.4f}")
    print(f"Recall:            {recall:.4f}")
    print(f"Training Time:     {training_time:.2f} seconds")
    print(f"Number of Params:  {num_params:,}")
    print(f"Model Size:        {model_size_kb:.2f} KB")
    print("\nConfusion Matrix:")
    print(confusion_mat)
    print("\nClassification Report after updating global model:")
    print(classification_report(true_labels, predictions, target_names=list(data_flow.class_indices.keys())))

import struct
import tenseal as ts
import zlib

def send_encrypted_weights(client, encrypted_weights):
    """Serialize and send encrypted weights to the server."""

    # Ensure encrypted weights exist
    if not encrypted_weights:
        raise ValueError("Error: No encrypted weights to send!")

    serialized_weights = [vec.serialize() for vec in encrypted_weights]
    print(f"Number of encrypted weights being sent: {len(serialized_weights)}")

    # Convert list of serialized weights into a single bytes object with separators
    encrypted_bytes = b"".join([struct.pack(">I", len(w)) + w for w in serialized_weights])

    # Compress the data
    compressed_data = zlib.compress(encrypted_bytes)

    # Send data length first
    client.sendall(struct.pack(">I", len(compressed_data)))

    # Send actual compressed encrypted data
    with tqdm(total=len(compressed_data), unit="B", unit_scale=True, desc="Uploading weights") as pbar:
            client.sendall(compressed_data)
            pbar.update(len(compressed_data))

    print(f"Encrypted data sent! Size: {len(compressed_data)} bytes")

    # Send actual compressed encrypted data
    

    
# Client function
def client_socket(base_dir):
    context = setup_ckks_context()
    # Symmetric Key for Model Transmission (Fernet Encryption)
    custom_key = "secretkey"
    hashed_key = hashlib.sha256(custom_key.encode()).digest()
    encoded_key = base64.urlsafe_b64encode(hashed_key)
    cipher_suite = Fernet(encoded_key)

    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(('localhost', 5000))  # Replace with server's IP
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

        decrypted_data = cipher_suite.decrypt(received_data)
        decompressed_data = zlib.decompress(decrypted_data)

        # Step 5: Load the received global model
        with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as tmp:
            tmp.write(decompressed_data)
            tmp.flush()
            model = tf.keras.models.load_model(tmp.name, custom_objects={'CapsuleLayer': CapsuleLayer,
                                                                         'Length': Length,
                                                                         'margin_loss': margin_loss
                                                                         })
            model.save(tmp.name)
            model_size_kb = os.path.getsize(tmp.name) / 1024
            
        print(f"Model for Round {set_num} loaded successfully.")

        train_flow, val_flow = load_image_data(base_dir, set_num)
        print(f"Training locally on Set {set_num}...")
        start_time = time.time()
        model.fit(train_flow, epochs=1, validation_data=val_flow, verbose = 0)
        end_time = time.time()

        # Evaluate the locally trained model
        evaluate_model(model, "D:/Major Project/Rasp/Data/test",start_time,end_time,model_size_kb)
        print(f"Model has {len(model.get_weights())} weight tensors")
        encrypted_data = encrypt_weights(model, context)
        send_encrypted_weights(client,encrypted_data)
        # # encrypted_data = encrypted_weights.serialize()
        # encrypted_bytes = b"".join(encrypted_data)  # Convert list to single bytes object
        # compressed_data = zlib.compress(encrypted_bytes)
        # compressed_size = len(compressed_data)
        # # Step 7: Send updated weights back to the server
        # print(f"Sending encrypted weights ({compressed_size} bytes) to the server...")
        # client.sendall(struct.pack('>I', compressed_size))  # Send the size of the encrypted data
        # with tqdm(total=compressed_size, unit="B", unit_scale=True, desc="Uploading weights") as pbar:
        #     client.sendall(compressed_data)
        #     pbar.update(compressed_size)

        print(f"Training and weight update for Round {set_num} completed.\n")
        time_avg = sum(time_list) / len(time_list)
        print(f"Average Time Taken for All Rounds{time_avg}.\n")

    # Step 8: Close the connection after all rounds
    client.close()
    print("Connection to server closed.")

if __name__ == "__main__":
    client_socket("D:/Major Project/Rasp/Data/client1")