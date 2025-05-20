import numpy as np
import tensorflow as tf
from pathlib import Path
from cryptography.fernet import Fernet
import base64
import os
import pickle
import zlib
import socket
import time
from tensorflow.keras import layers, initializers, backend as K # type: ignore
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tensorflow.keras.saving import register_keras_serializable # type: ignore
from tqdm import tqdm
from tqdm.keras import TqdmCallback

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
    y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=2)
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
            metrics=['accuracy']
        )

class EdgeNode:
    def __init__(self, client_id, data_dir):
        self.client_id = client_id
        self.data_dir = Path(data_dir)
        self.model = MobileNetCapsNet()
        self.model.compile_model()
        self.current_round = 1
        self.cipher = None

    def _calculate_metrics(self, generator):
        with tqdm(total=len(generator), desc="Evaluating", unit="sample") as eval_bar:
            y_pred = []
            y_true = []
            for x, y in generator:
                y_pred.extend(np.argmax(self.model.model.predict(x), axis=1))
                y_true.extend(y)
                eval_bar.update(len(x))
        
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'confusion_matrix': confusion_matrix(y_true, y_pred),
            'classification_report': classification_report(
                y_true, y_pred, 
                target_names=['Not Drowsy', 'Drowsy'],
                output_dict=True
            ),
            'class_distribution': np.bincount(y_true),
            'error_rate': 1 - accuracy_score(y_true, y_pred)
        }

    def train_round(self, round_num, epochs=10, batch_size=32):
        print(f"\n=== Client {self.client_id} - Round {round_num} ===")
        
        round_dir = self.data_dir / f"round_{round_num}"
        
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            validation_split=0.2
        )
        
        train_gen = train_datagen.flow_from_directory(
            round_dir,
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode='binary',
            subset='training'
        )
        
        val_gen = train_datagen.flow_from_directory(
            round_dir,
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode='binary',
            subset='validation'
        )
        
        with tqdm(total=epochs, desc=f"Round {round_num} Epochs") as epoch_bar:
            history = self.model.model.fit(
                train_gen,
                validation_data=val_gen,
                epochs=epochs,
                verbose=0,
                callbacks=[
                    TqdmCallback(
                        desc="Batch Progress", 
                        metrics_format="{name}: {value:0.4f}",
                        batch_size=batch_size
                    )
                ]
            )
            epoch_bar.update(epochs)
        
        with tqdm(total=len(val_gen), desc="Validation") as val_bar:
            train_metrics = self._calculate_metrics(train_gen)
            val_metrics = self._calculate_metrics(val_gen)
            val_bar.update(len(val_gen))
        
        print(f"\nClient {self.client_id} - Training Metrics:")
        print(f"Accuracy: {train_metrics['accuracy']:.4f}")
        print(f"Error Rate: {train_metrics['error_rate']:.4f}")
        
        print(f"\nClient {self.client_id} - Validation Metrics:")
        print(f"Accuracy: {val_metrics['accuracy']:.4f}")
        print("Confusion Matrix:\n", val_metrics['confusion_matrix'])
        
        return {
            'history': history.history,
            'metrics': {'train': train_metrics, 'val': val_metrics},
            'encrypted_weights': self.get_encrypted_parameters()
        }

    def get_encrypted_parameters(self):
        weights = self.model.model.get_weights()
        serialized = pickle.dumps(weights)
        compressed = zlib.compress(serialized)
        return self.cipher.encrypt(compressed)

class FederatedClient:
    def __init__(self, client_id, data_dir, server_host, server_port):
        self.client_id = client_id
        self.data_dir = data_dir
        self.server_host = server_host
        self.server_port = server_port
        self.edge_node = EdgeNode(client_id, data_dir)

    def connect(self):
        max_retries = 3
        retry_count = 0
        
        with tqdm(desc="Connection Attempt", total=max_retries, leave=False) as conn_bar:
            while retry_count < max_retries:
                try:
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                        s.connect((self.server_host, self.server_port))
                        key_data = b''
                        while len(key_data) < 44:
                            chunk = s.recv(44 - len(key_data))
                            if not chunk: break
                            key_data += chunk
                        self.edge_node.cipher = Fernet(key_data)
                        
                        with tqdm(desc="Training Rounds", unit="round") as round_bar:
                            while True:
                                self._receive_model_update(s)
                                results = self.edge_node.train_round(self.edge_node.current_round)
                                self._send_encrypted_update(s, results['encrypted_weights'])
                                self.edge_node.current_round += 1
                                round_bar.update(1)
                    break
                except (ConnectionResetError, TimeoutError) as e:
                    retry_count += 1
                    conn_bar.update(1)
                    time.sleep(5)

    def _receive_model_update(self, conn):
        data_size = int.from_bytes(conn.recv(4), byteorder='big')
        encrypted_data = conn.recv(data_size)
        
        decrypted = self.edge_node.cipher.decrypt(encrypted_data)
        decompressed = zlib.decompress(decrypted)
        weights = pickle.loads(decompressed)
        
        self.edge_node.model.model.set_weights(weights)

    def _send_encrypted_update(self, conn, encrypted_data):
        conn.sendall(len(encrypted_data).to_bytes(4, byteorder='big'))
        conn.sendall(encrypted_data)

if __name__ == "__main__":
    CLIENT_ID = "client_2"
    DATA_DIR = r"Data/client2"
    SERVER_IP = "localhost"
    SERVER_PORT = 5000
    
    client = FederatedClient(
        client_id=CLIENT_ID,
        data_dir=DATA_DIR,
        server_host=SERVER_IP,
        server_port=SERVER_PORT
    )
    client.connect()
