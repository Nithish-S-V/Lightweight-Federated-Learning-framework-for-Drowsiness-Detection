import socket
import struct
import zlib
import tempfile
import tensorflow as tf
from sklearn.metrics import classification_report
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, accuracy_score
from tensorflow.keras.saving import register_keras_serializable
from tensorflow.keras import layers, initializers, backend as K # type: ignore
from tensorflow.keras.losses import KLDivergence


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
    L = y_true * tf.square(tf.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * tf.square(tf.maximum(0., y_pred - 0.1))
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

def receive_model_and_labels(client):
    rows, cols = struct.unpack('>II', client.recv(8))
    # Receive soft labels size and data
    logits_size = struct.unpack('>I', client.recv(4))[0]
    received_logits = b""
    with tqdm(total=logits_size, unit="B", unit_scale=True, desc="Receiving soft labels") as pbar:
        while len(received_logits) < logits_size:
            packet = client.recv(min(1024 * 1024, logits_size - len(received_logits)))
            if not packet:
                break
            received_logits += packet
            pbar.update(len(packet))
    
    # Process soft labels
    decompressed_logits = zlib.decompress(cipher_suite.decrypt(received_logits))
    
    # Reshape to original dimensions
    soft_labels = np.frombuffer(decompressed_logits, dtype=np.float16).astype(np.float32).reshape(rows, cols)
    
    # Receive model size and data
    model_size = struct.unpack('>I', client.recv(4))[0]
    print(f"Receiving encrypted model ({model_size} bytes)...")
    
    received_model = b""
    with tqdm(total=model_size, unit="B", unit_scale=True, desc="Receiving model") as pbar:
        while len(received_model) < model_size:
            packet = client.recv(1024 * 1024)
            if not packet:
                break
            received_model += packet
            pbar.update(len(packet))
    
    # Process model data
    decrypted_model = cipher_suite.decrypt(received_model)
    decompressed_model = zlib.decompress(decrypted_model)
    return decompressed_model, soft_labels

def client_socket(base_dir):
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(('localhost', 5000))
    print("Connected to server.")

    for set_num in range(1, 6):
        print(f"\n==== Round {set_num} ====")

        # Receive both model and soft labels
        decompressed_model_data, soft_labels = receive_model_and_labels(client)

        # Load the received global model
        with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as tmp:
            tmp.write(decompressed_model_data)
            tmp.flush()
            model = tf.keras.models.load_model(tmp.name)
        print(f"Model and soft labels for Round {set_num} loaded successfully.")

        train_flow, val_flow = load_image_data(base_dir, set_num)
        
        # Create distillation-aware data generator
        class DistillDataGenerator(tf.keras.utils.Sequence):
            def __init__(self, base_flow, soft_labels, batch_size=32):
                super().__init__()  # Critical initialization
                self.base_flow = base_flow
                self.soft_labels = soft_labels
                self.batch_size = batch_size

                
            def __len__(self):
                return len(self.base_flow)
            
            def __getitem__(self, idx):
                x, y = self.base_flow[idx]
                start = idx * self.batch_size
                end = start + self.batch_size
                soft_batch = self.soft_labels[start:end]
                
                assert soft_batch.shape == (x.shape[0], 2), \
                    f"Soft label shape mismatch: {soft_batch.shape} vs ({x.shape[0]}, 2)"
                    
                return x, {'true_labels': y, 'soft_labels': soft_batch}


        class DistillValidationDataGenerator(tf.keras.utils.Sequence):
            def __init__(self, base_flow):
                super().__init__()
                self.base_flow = base_flow

            def __len__(self):
                return len(self.base_flow)

            def __getitem__(self, idx):
                x, y = self.base_flow[idx]
                assert y.shape == (32, 2), f"Invalid validation shape: {y.shape}"
                return x, {'true_labels': y, 'soft_labels': y}  # Dual labels       

        # Configure distillation training
        print(f"Training with knowledge distillation on Set {set_num}...")
        model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                      loss={'true_labels': margin_loss,
                            'soft_labels': KLDivergence()},
                      loss_weights=[0.7, 0.3],
                      metrics={'true_labels': 'accuracy'})

        
        # Configure distillation training
        print(f"Training with knowledge distillation on Set {set_num}...")
        # Add validation generator class


        # Modify model.fit() call
        model.fit(
            DistillDataGenerator(train_flow, soft_labels),
            epochs=5,
            validation_data=DistillValidationDataGenerator(val_flow),  # Updated
            verbose=1
        )

        # Evaluate the distilled model
        evaluate_model(model, "D:/Major Project/Rasp/Data/test")

        # Send updated weights back to server
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

        print(f"Distillation training and weight update for Round {set_num} completed.\n")

    client.close()
    print("Connection to server closed.")

@tf.keras.saving.register_keras_serializable(package="Custom", name="distill_loss")
def distillation_loss(y_true, y_pred):
    print("True labels shape:", y_true['true_labels'].shape)
    print("Soft labels shape:", y_true['soft_labels'].shape)
    print("Pred true shape:", y_pred['true_labels'].shape)
    print("Pred soft shape:", y_pred['soft_labels'].shape)
    # y_pred contains both outputs ('true_labels' and 'soft_labels')
    margin_loss = margin_loss(y_true['true_labels'], y_pred['true_labels'])
    kl_loss = KLDivergence()(y_true['soft_labels'], y_pred['soft_labels'])
    return 0.7 * margin_loss + 0.3 * kl_loss




if __name__ == "__main__":
    client_socket("D:/Major Project/Rasp/Data/client2")