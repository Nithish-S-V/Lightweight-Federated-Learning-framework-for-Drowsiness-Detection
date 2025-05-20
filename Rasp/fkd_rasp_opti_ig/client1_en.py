import socket
import struct
import zlib
import tempfile
import tensorflow as tf
from sklearn.metrics import classification_report
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, accuracy_score
from tensorflow.keras import layers, initializers, backend as K # type: ignore
from tensorflow.keras.saving import register_keras_serializable
from cryptography.fernet import Fernet
import base64
import hashlib

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
    """
    Capsule Network Margin Loss with Shape Compatibility Fix
    - Expects y_true in categorical format (one-hot encoded)
    - Processes batch dimensions correctly
    - Maintains capsule network properties
    """
    # Convert labels to float32 if needed
    y_true = tf.cast(y_true, y_pred.dtype)
    
    # Ensure matching ranks by expanding dimensions if needed
    if len(y_true.shape) == len(y_pred.shape) - 1:
        y_true = tf.expand_dims(y_true, -1)
    
    # Calculate margin components
    positive_loss = y_true * tf.square(tf.maximum(0.9 - y_pred, 0.))
    negative_loss = 0.5 * (1 - y_true) * tf.square(tf.maximum(y_pred - 0.1, 0.))
    
    # Sum over capsules and average over batch
    return tf.reduce_mean(tf.reduce_sum(positive_loss + negative_loss, axis=-1))



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
def evaluate_model(model, dataset_dir):
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
    print(f"Test Accuracy : {test_accuracy:.2%}")
    print("\nClassification Report after updating global model:")
    print(classification_report(true_labels, predictions, target_names=list(data_flow.class_indices.keys())))

# Client function
def client_socket(base_dir):
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
        decrypted_data = cipher_suite.decrypt(received_data)  # Decrypt the data
        decompressed_data = zlib.decompress(decrypted_data)

        # Step 5: Load the received global model
        with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as tmp:
            tmp.write(decompressed_data)
            tmp.flush()
            model = tf.keras.models.load_model(tmp.name)
        print(f"Model for Round {set_num} loaded successfully.")


        train_flow, val_flow = load_image_data(base_dir, set_num)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),loss=margin_loss,metrics=['accuracy'])
        print(f"Training locally on Set {set_num}...")
        model.fit(train_flow, epochs=1, validation_data=val_flow, verbose = 1)

        # Evaluate the locally trained model
        evaluate_model(model, "D:/Major Project/Rasp/Data/test")

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
    client_socket("D:/Major Project/Rasp/Data/client1")