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
import time
import psutil

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




class MobileNetCapsNet:
    def __init__(self, input_shape=(224, 224, 3)):
        self.input_shape = input_shape
        self.model = self._build_model()

    def _build_model(self):
        # Scaled-down MobileNetV2
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=self.input_shape,
            include_top=False,
            weights='imagenet',
            alpha=0.35  # Reduced from teacher's 1.0
        )
        
        # Consolidated capsule architecture
        x = base_model.output
        x = layers.Conv2D(128, 3)(x)  # Reduced from 256
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Reshape((-1, 128))(x)
        x = CapsuleLayer(num_capsule=4, dim_capsule=12, routings=2)(x)  # Single layer
        outputs = Length()(x)
        
        return tf.keras.Model(inputs=base_model.input, outputs=outputs)

    def quantize_model(self):
        """Post-training quantization for TF 2.18"""
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Custom quantization for capsule layers
        def representative_dataset():
            for _ in range(100):
                yield [np.random.rand(1, *self.input_shape).astype(np.float32)]
                
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
        
        return converter.convert()

    def compile_model(self, learning_rate=0.001):
        """Compile with native TF optimization"""
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=learning_rate,
                decay_steps=1000,
                decay_rate=0.96
            )
        )
        
        self.model.compile(
            optimizer=optimizer,
            loss=self.margin_loss,
            metrics=['accuracy']
        )
        self.tb_callback = tf.keras.callbacks.TensorBoard(
            log_dir='logs/quantization',
            profile_batch=(10,20)
        )

    @staticmethod
    def margin_loss(y_true, y_pred):
        # Maintain original margin loss implementation
        y_true = tf.one_hot(tf.cast(tf.squeeze(y_true), tf.int32), depth=2)
        L = y_true * tf.square(tf.maximum(0., 0.9 - y_pred)) + \
            0.5 * (1 - y_true) * tf.square(tf.maximum(0., y_pred - 0.1))
        return tf.reduce_mean(tf.reduce_sum(L, axis=1))

def compile_model(model, learning_rate=0.001):
        """Compile with native TF optimization"""
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=learning_rate,
                decay_steps=1000,
                decay_rate=0.96
            )
        )
        
        model.compile(
            optimizer=optimizer,
            loss=margin_loss,
            metrics=['accuracy']
        )
        tb_callback = tf.keras.callbacks.TensorBoard(
            log_dir='logs/quantization',
            profile_batch=(10,20)
        )
class ServerDistiller(tf.keras.Model):
    def __init__(self, teacher, student):
        super().__init__()
        self.teacher = teacher
        self.student = student
        self.acc_metric = tf.keras.metrics.CategoricalAccuracy(name="accuracy")
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

    def compile(self, optimizer, temperature=2.0, alpha=0.1):
        super().compile(optimizer=optimizer)
        self.temperature = temperature
        self.alpha = alpha
        self.kl_loss = tf.keras.losses.KLDivergence()

    def train_step(self, data):
        # Maintain full-precision calculations for distillation
        x, y = data
        teacher_pred = self.teacher(x, training=False)
        
        with tf.GradientTape() as tape:
            student_pred = self.student(x, training=True)
            
            # Temperature-scaled distillation
            teacher_probs = tf.nn.softmax(teacher_pred/self.temperature)
            student_probs = tf.nn.softmax(student_pred/self.temperature)
            
            distillation_loss = self.kl_loss(teacher_probs, student_probs)
            student_loss = self.student.compiled_loss(y, student_pred)
            total_loss = self.alpha*student_loss + (1-self.alpha)*distillation_loss

        # Update metrics and apply gradients
        y_labels = tf.argmax(y, axis=1)
        pred_labels = tf.argmax(student_pred, axis=1)
        self.acc_metric.update_state(y_labels, pred_labels)
        
        gradients = tape.gradient(total_loss, self.student.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.student.trainable_weights))
        
        self.loss_tracker.update_state(total_loss)
        return {
            "loss": self.loss_tracker.result(),
            "student_loss": student_loss,
            "distillation_loss": distillation_loss,
            "accuracy": self.acc_metric.result()
        }
    def thermal_profile(self, data, steps):
        temps = []
        for batch in data.take(steps):
            start_temp = psutil.sensors_temperatures()['coretemp'][0].current
            self.predict(batch[0])  # Inference only
            end_temp = psutil.sensors_temperatures()['coretemp'][0].current
            temps.append(end_temp - start_temp)
        
        print(f"Thermal Impact Δ: {np.mean(temps):.1f}°C/batch")
    def quantized_predict(self, data):
        """Quantized inference for resource-constrained devices"""
        quant_model = self.student.quantize_model()
        interpreter = tf.lite.Interpreter(model_content=quant_model)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        interpreter.set_tensor(input_details[0]['index'], data)
        interpreter.invoke()
        return interpreter.get_tensor(output_details[0]['index'])

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
def send_global_model(client, global_model, round_num):
    with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as tmp:
        global_model.save(tmp.name)
        tmp.seek(0)
        model_data = tmp.read()

    compressed_data = zlib.compress(model_data)
    encrypted_data = cipher_suite.encrypt(compressed_data)  # Encrypt the data
    data_length = struct.pack('>I', len(encrypted_data))

    client.sendall(data_length)  # Send encrypted data size
    chunk_size = 1024 * 1024  # 1 MB per chunk
    with tqdm(total=len(encrypted_data), unit="B", unit_scale=True, desc="Sending model") as pbar:
        for i in range(0, len(encrypted_data), chunk_size):
            chunk = encrypted_data[i:i + chunk_size]
            client.sendall(chunk)
            pbar.update(len(chunk))
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
                    class_mode='categorical',
                    shuffle=False
                )
    predictions = np.argmax(model.predict(test_gen), axis=1)
    true_labels = test_gen.classes
    test_accuracy = accuracy_score(true_labels, predictions)
    print(f"Test Accuracy (global model): {test_accuracy:.2%}")
    print("\nClassification Report after updating global model:")
    print(classification_report(true_labels, predictions, target_names=list(test_gen.class_indices.keys())))
    # Inference speed test
    start_time = time.perf_counter()
    predictions = np.argmax(model.predict(test_gen), axis=1)
    latency = (time.perf_counter() - start_time) * 1000 / len(predictions)
    
    # Thermal monitoring
    temps = psutil.sensors_temperatures()['coretemp']
    max_temp = max(t.current for t in temps)
    
    # Memory usage (requires psutil)
    process = psutil.Process()
    mem_usage = process.memory_info().rss / 1024**2  # MB
    
    print(f"""
    Quantization Results:
    - Model Accuracy: {test_accuracy:.2%}
    - Avg Inference Latency: {latency:.2f}ms
    - Peak CPU Temp: {max_temp}°C
    - Memory Usage: {mem_usage:.1f}MB
    """)


def benchmark_quantized_model(tflite_model_path):
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()
    
    # Warmup
    input_details = interpreter.get_input_details()
    interpreter.set_tensor(input_details[0]['index'], np.random.rand(1,224,224,3).astype(np.float32))
    interpreter.invoke()
    
    # Latency test
    times = []
    for _ in range(100):
        start = time.perf_counter()
        interpreter.invoke()
        times.append((time.perf_counter() - start)*1000)
    
    print(f"Quantized Model Performance:")
    print(f"- Average latency: {np.mean(times):.2f}ms (±{np.std(times):.2f})")
    print(f"- 95th percentile: {np.percentile(times, 95):.2f}ms")


def get_server_validation_data(train_dir):
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )
    
    val_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    # Create dataset without loading all data into memory
    dataset = tf.data.Dataset.from_generator(
        lambda: val_gen,
        output_signature=(
            tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 2), dtype=tf.float32)
        )
    )
    
    steps = val_gen.samples // val_gen.batch_size
    if val_gen.samples % val_gen.batch_size != 0:
        steps += 1
        
    return dataset, steps


        
# Server socket for federated learning
def server_socket():
    teacher_model = load_model_from_file()
    student_model = tf.keras.models.clone_model(teacher_model)

    for layer in student_model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            # Only modify Conv2D layers, not DepthwiseConv2D
            layer.filters = max(1, int(layer.filters * 0.35))
        elif isinstance(layer, CapsuleLayer):
            # Adjust capsule dimensions
            layer.num_capsule = 4
            layer.dim_capsule = 12
            
    compile_model(student_model)
    # Before student_model.set_weights()
    print("\nTeacher Model Summary:")
    teacher_model.summary()

    print("\nStudent Model Summary:")
    student_model.summary()

    # Compare layer counts
    print(f"\nTeacher weights: {len(teacher_model.get_weights())}")
    print(f"Student weights: {len(student_model.get_weights())}")

    student_model.set_weights(teacher_model.get_weights())
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

        # Step 1: Send the global model to all clients
        for client in client_sockets:
            send_global_model(client, student_model, round_num)

        # Collect client updates
        client_weights = [receive_client_weights(c) for c in client_sockets]
        
        # Server-side aggregation and distillation
       # Server-side aggregation
        aggregated_weights = aggregate_weights(client_weights)
        student_model.set_weights(aggregated_weights)

        # Final round quantization
        if round_num == 5:
            tflite_model = student_model.quantize_model()
            with open('deployed.tflite', 'wb') as f:
                f.write(tflite_model)
            
            # Compare models
            evaluate_model(student_model, "D:/Major Project/Rasp/Data/test")
            evaluate_model(tf.lite.Interpreter('deployed.tflite'), "D:/Major Project/Rasp/Data/test")
            benchmark_quantized_model('deployed.tflite')
        else:
            # Standard distillation
            distiller = ServerDistiller(teacher=teacher_model, student=student_model)
            distiller.compile(optimizer=tf.keras.optimizers.Adam(0.001))
            val_data, steps = get_server_validation_data("D:/Major Project/Rasp/Data/train")
            distiller.fit(val_data, steps_per_epoch=steps, epochs=2)
            teacher_model.set_weights(student_model.get_weights())

        evaluate_model(student_model, "D:/Major Project/Rasp/Data/test")

    # Step 5: Close all client connections after all rounds
    for client in client_sockets:
        client.close()
        print("Client connection closed.")

    server.close()
    print("Server shutdown.")

if __name__ == "__main__":
    server_socket()