import socket
import struct
import zlib
import tempfile
import tensorflow as tf
import numpy as np
import numpy
from tqdm import tqdm
from sklearn.metrics import classification_report
from tensorflow.keras.saving import register_keras_serializable
from tensorflow.keras import layers, initializers, backend as K
from sklearn.metrics import accuracy_score, f1_score, recall_score, classification_report, confusion_matrix
from cryptography.fernet import Fernet
import base64
import hashlib
import time
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
time_list =[]

import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph

def get_flops(model):
    # Convert Keras model to a concrete function
    concrete_function = tf.function(model).get_concrete_function(
        tf.TensorSpec([1] + list(model.input_shape[1:]), model.inputs[0].dtype)
    )
    frozen_func, _ = convert_variables_to_constants_v2_as_graph(concrete_function)

    # Calculate FLOPs using TensorFlow's profiler
    run_meta = tf.compat.v1.RunMetadata()
    opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
    flops = tf.compat.v1.profiler.profile(
        graph=frozen_func.graph,
        run_meta=run_meta,
        cmd='op',
        options=opts
    )
    return flops.total_float_ops  # Returns total FLOPs


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
        outputs = tf.keras.layers.Reshape((2,))(outputs)
        
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
        # Dynamic shape handling for unknown ranks
        rank = tf.rank(y_pred)
        y_pred = tf.cond(
            tf.equal(rank, 1),
            lambda: tf.expand_dims(y_pred, -1),
            lambda: y_pred
        )
        
        # Convert labels to one-hot encoding
        y_true = tf.one_hot(tf.cast(tf.squeeze(y_true), tf.int32), depth=2)
        
        # Margin loss calculation
        L = y_true * tf.square(tf.maximum(0., 0.9 - y_pred)) + \
            0.5 * (1 - y_true) * tf.square(tf.maximum(0., y_pred - 0.1))
        
        return tf.reduce_mean(tf.reduce_sum(L, axis=1))


class ServerDistiller(tf.keras.Model):
    def __init__(self, teacher, student, temp=3, alpha=0.1):
        super().__init__()
        self.teacher = teacher
        self.student = student
        self.temp = temp
        self.alpha = alpha
        self.teacher.trainable = False
        
        # Initialize metrics
        self.loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.acc_metric = tf.keras.metrics.CategoricalAccuracy(name="acc")

    def compile(self, optimizer, grad_clip=1.0, **kwargs):
        # Include loss in super().compile() call
        super().compile(
            loss=margin_loss,  # Add this line
            optimizer=optimizer,
            **kwargs
        )
        self.grad_clip = grad_clip
        self.kl_loss = tf.keras.losses.KLDivergence()

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            teacher_pred = self.teacher(x, training=False)
            student_pred = self.student(x, training=True)
            
            student_loss = margin_loss(y, student_pred)
            distillation_loss = self.kl_loss(
                tf.nn.softmax(teacher_pred/self.temp, axis=1),
                tf.nn.softmax(student_pred/self.temp, axis=1)
            )
            total_loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss
            
        gradients = tape.gradient(total_loss, self.student.trainable_variables)
        gradients = [tf.clip_by_norm(g, self.grad_clip) for g in gradients]
        self.optimizer.apply_gradients(zip(gradients, self.student.trainable_variables))
        
        # Update metrics
        self.loss_tracker.update_state(total_loss)
        self.acc_metric.update_state(y, student_pred)
        
        return {
            "loss": self.loss_tracker.result(),
            "accuracy": self.acc_metric.result()
        }

    @property
    def metrics(self):
        return [self.loss_tracker, self.acc_metric]


    def call(self, inputs, training=False):
        return self.student(inputs, training=training)

    



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
    weights = [w.numpy() if isinstance(w, tf.Tensor) else w 
          for w in global_model.get_weights()]

    global_model.set_weights(weights)
    with tempfile.NamedTemporaryFile(suffix=".keras") as tmp:
        global_model.save(tmp.name, save_format='keras')
        tmp.seek(0)
        model_data = tmp.read()
    # with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as tmp:
    #     global_model.save(tmp.name)
    #     tmp.seek(0)
    #     model_data = tmp.read()

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
def evaluate_model(model, dataset_dir,start_time,end_time):
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
    print("\nConfusion Matrix:")
    print(confusion_mat)
    print("\nClassification Report after updating global model:")
    print(classification_report(true_labels, predictions, target_names=list(test_gen.class_indices.keys())))

def get_server_validation_data():
    train_dir=r"D:\Major Project\Rasp\Data\train"
    
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )
    
    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=INPUT_SHAPE[:2],
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='training'
    )
    
    val_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=INPUT_SHAPE[:2],
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='validation'
    )
    return train_gen,val_gen

import tensorflow as tf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, classification_report
from sklearn.metrics import precision_score 
def evaluate_model_start(model, dataset_dir):
    # Data preparation
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)
    test_gen = datagen.flow_from_directory(
                    dataset_dir,
                    target_size=(224, 224),
                    batch_size=32,
                    class_mode='categorical',
                    shuffle=False
                )
    
    # Predictions and metrics
    start_time = time.time()
    predictions = np.argmax(model.predict(test_gen), axis=1)
    end_time = time.time()
    inference_latency = end_time - start_time
    
    true_labels = test_gen.classes
    class_names = list(test_gen.class_indices.keys())
    
    # Calculate metrics
    precision = precision_score(true_labels, predictions, average='weighted')
    test_accuracy = accuracy_score(true_labels, predictions)
    confusion_mat = confusion_matrix(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average='weighted')
    recall = recall_score(true_labels, predictions, average='weighted')
    # print("\nConfusion Matrix:")
    # sns.heatmap(confusion_mat, annot=True,fmt='g', xticklabels=['Drowsy','Not Drowsy'],yticklabels=['Drowsy','Not Drowsy'])
    # plt.ylabel('Actual', fontsize=13)
    # plt.title('Confusion Matrix', fontsize=17, pad=20)
    # plt.gca().xaxis.set_label_position('top') 
    # plt.xlabel('Prediction', fontsize=13)
    # plt.gca().xaxis.tick_top()
    # plt.gca().figure.subplots_adjust(bottom=0.2)
    # plt.gca().figure.text(0.5, 0.05, 'Prediction', ha='center', fontsize=13)
    # plt.show()
    # Print metrics
    print("Confusion Matrix:",confusion_mat)
    print(f"Test Accuracy :    {test_accuracy:.2%}")
    print(f"Precision:           {precision:.4f}")
    print(f"F1 Score:          {f1:.4f}")
    print(f"Recall:            {recall:.4f}")
    print(f"Inference Latency: {inference_latency:.4f} seconds")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(true_labels, predictions, target_names=class_names))
import os
import shutil
def get_model_size_mb(model_path: str) -> float:
    """Get the size of a saved model file in MB (works without TensorFlow)."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found!")
    return os.path.getsize(model_path) / (1024 ** 2)

def save_temp_model(model, model_path: str) -> bool:
    """Save a model temporarily if TensorFlow is available."""
    try:
        model.save(model_path)
        return True
    except NameError:  # Fallback if TensorFlow not installed
        with open(model_path, "wb") as f:
            f.write(os.urandom(1024 * 100))  # 100KB dummy file
        return False

def calculate_compression_ratio(teacher_size: float, student_size: float) -> float:
    """Calculate compression ratio between teacher and student models."""
    return teacher_size / student_size if student_size > 0 else 0

def _build_student_model(teacher_model):
    # Smaller base model with reduced parameters
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=teacher_model.input_shape[1:],
        include_top=False,
        weights='imagenet',
        alpha=0.5
    )
    base_model.trainable = False

    x = base_model.output
    x = layers.Conv2D(128, 3, activation='relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    
    # Feature alignment with proper dimension matching
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Reshape((1, 256))(x)  # Explicit single capsule input

    # First compressed capsule layer
    teacher_caps1 = teacher_model.get_layer('capsule_layer')
    # In _build_student_model()
    teacher_weights = teacher_caps1.get_weights()[0]# Explicit conversion
    x = CapsuleLayer_1(
        num_capsule=4,
        dim_capsule=8,
        routings=3,
        weight_initializer=teacher_weights[:, :, :4, :8, :]  # Use numpy array
    )(x)

    # x = CapsuleLayer_1(
    #     num_capsule=4,  # 50% reduction from teacher's 8
    #     dim_capsule=8,  # 50% reduction from teacher's 16D
    #     routings=3,
    #     weight_initializer=teacher_caps1.weights[0][:, :, :4, :8, :]  # Correct slicing
    # )(x)

    # Second compressed capsule layer
    teacher_caps2 = teacher_model.get_layer('capsule_layer_1')
    teacher_weights = teacher_caps2.get_weights()[0] # Explicit conversion
    x = CapsuleLayer_1(
        num_capsule=2,
        dim_capsule=16,
        routings=3,
        weight_initializer=teacher_weights[:, :, :2, :16, :8]  # Use numpy array
    )(x)
    # x = CapsuleLayer_1(
    #     num_capsule=2,
    #     dim_capsule=16,  # 50% reduction from teacher's 32D
    #     routings=3,
    #     weight_initializer=teacher_caps2.weights[0][:, :, :2, :16, :8]  # Adjusted slicing
    # )(x)

    outputs = Length()(x)
    return tf.keras.Model(inputs=base_model.input, outputs=outputs)

class CapsuleLayer_1(layers.Layer):
    def __init__(self, num_capsule, dim_capsule, routings=3, 
                 weight_initializer=None, **kwargs):
        super().__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.weight_initializer = weight_initializer

    def build(self, input_shape):
        self.input_num_capsule = input_shape[1]
        self.input_dim_capsule = input_shape[2]

        # Dimension-aware weight adaptation
        if self.weight_initializer is not None:
            # Preserve matrix subspace while matching dimensions
            init_weights = self.weight_initializer
            target_shape = (1, self.input_num_capsule, self.num_capsule, 
                           self.dim_capsule, self.input_dim_capsule)
            
            # Calculate zero-padding needs
            pad_dims = [
                (0, max(0, target_shape[i] - init_weights.shape[i]))
                for i in range(len(target_shape))
            ]
            
            # Apply symmetric padding and truncation
            padded_weights = np.pad(
                init_weights,
                pad_dims,
                mode='constant',
                constant_values=0
            )[:target_shape[0], :target_shape[1], :target_shape[2], 
              :target_shape[3], :target_shape[4]]
            
            initializer = tf.keras.initializers.Constant(padded_weights)
        else:
            initializer = initializers.glorot_uniform()

        self.W = self.add_weight(
            shape=(1, self.input_num_capsule, self.num_capsule,
                  self.dim_capsule, self.input_dim_capsule),
            initializer=initializer,
            name='capsule_weights'
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
            "routings": self.routings,
            "weight_initializer": self.weight_initializer
        })
        return config

    def squash(self, vectors, axis=-1):
        s_squared_norm = K.sum(K.square(vectors), axis, keepdims=True)
        scale = s_squared_norm / (1 + s_squared_norm) / K.sqrt(s_squared_norm + K.epsilon())
        return scale * vectors

INPUT_SHAPE = (224, 224, 3)
BATCH_SIZE = 32
# Server socket for federated learning
def server_socket():
    teacher_path = "temp_teacher.keras"
    student_pre_path = "temp_student_pre.keras"
    student_post_path = "temp_student_post.keras"
    # 2. Initialize Student Model
    teacher_model = load_model_from_file()
    student_model = _build_student_model(teacher_model)
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(('0.0.0.0', 5000))  # Bind to all available IPs on port 5000
    server.listen(10)  # Listen for connections

    num_clients = 2  # Number of clients participating in federated learning
    client_sockets = []
    try:
        # print("Teacher Model:")
        # evaluate_model_start(teacher_model, "D:/Major Project/Rasp/Data/test")
        # is_real_model = save_temp_model(teacher_model, teacher_path)
        # teacher_size = get_model_size_mb(teacher_path)
        # print(f"\nTeacher Model Size: {teacher_size:.2f} MB {'(Real)' if is_real_model else '(Dummy)'}")
        # flops = get_flops(teacher_model)
        # print(f"FLOPs: {flops / 1e9:.3f} GFLOPs")

        
        print("Student model before Distillation:")
        evaluate_model_start(student_model, "D:/Major Project/Rasp/Data/test")
        is_real_model = save_temp_model(student_model, student_pre_path)
        student_pre_size = get_model_size_mb(student_pre_path)
        print(f"Student Pre-Distillation Size: {student_pre_size:.2f} MB")
        #print(f"Compression Ratio: {calculate_compression_ratio(teacher_size, student_pre_size):.2f}")
        flops = get_flops(student_model)
        print(f"FLOPs: {flops / 1e9:.3f} GFLOPs")
        
        print("Student model after Distillation:")
        train_data,val_data= get_server_validation_data()
        distiller = ServerDistiller(teacher=teacher_model, student=student_model)
        test_input = tf.random.normal((1, 224, 224, 3))
        test_output = distiller(test_input)
        print(f"\nOutput shape verification: {test_output.shape} (should be (1, 2))")
        distiller.compile(
            optimizer='adam',
            grad_clip=1.0
        )
        distiller.fit(train_data,validation_data=val_data,epochs=5)
        student_model.save("student_drowsiness_model.keras")
        evaluate_model_start(student_model, "D:/Major Project/Rasp/Data/test")
        is_real_model = save_temp_model(student_model, student_post_path)
        student_post_size = get_model_size_mb(student_post_path)
        print(f"Student Post-Distillation Size: {student_post_size:.2f} MB")
        # print(f"Final Compression Ratio: {calculate_compression_ratio(teacher_size, student_post_size):.2f}")
        # flops = get_flops(student_model)
        # print(f"FLOPs: {flops / 1e9:.3f} GFLOPs")
    finally:  # Cleanup temporary files
        for path in [teacher_path, student_pre_path, student_post_path]:
            if os.path.exists(path):
                os.remove(path)

    # Accept connections from all clients once
    # print("Waiting for clients to connect...")
    # for i in range(num_clients):
    #     client, addr = server.accept()
    #     print(f"Client {i + 1} connected from {addr}.")
    #     client_sockets.append(client)

    # for round_num in range(1, 6):  # Perform 5 rounds
    #     print(f"\n==== Round {round_num} ====")
    #     client_weights = []

    #     # Step 1: Send the global model to all clients
    #     for client in client_sockets:
    #         send_global_model(client, student_model, round_num)

    #     # Collect client updates
    #     client_weights = [receive_client_weights(c) for c in client_sockets]
        
    #     # Server-side aggregation and distillation
    #     if round_num > 0:  # Start distillation after first round
    #         print("Performing server-side knowledge distillation...")
    #         distiller = ServerDistiller(teacher=teacher_model, student=student_model)
    #         distiller.compile(optimizer=tf.keras.optimizers.Adam(0.001),
    #                           temperature=3.0,
    #                           alpha=0.1) 
    #     # Use validation set for distillation (replace with your dataset)
    #         val_data,steps = get_server_validation_data("D:/Major Project/Rasp/Data/train")  
    #         #student_model.summary()
    #         start_time = time.time()
    #         distiller.fit(val_data,steps_per_epoch=steps, epochs=2, verbose=1)
    #         end_time = time.time()

    #         # Update teacher model to current student
    #         teacher_model.set_weights(student_model.get_weights())
            
    #     # Apply FedTrimmedAvg even when not distilling    
    #     aggregated_weights = aggregate_weights(client_weights)
    #     student_model.set_weights(aggregated_weights)
        
    #     evaluate_model(student_model, "D:/Major Project/Rasp/Data/test",start_time,end_time)

    # print(f"Training and weight update for all Rounds are completed.\n")
    # time_avg = sum(time_list) / len(time_list)
    # print(f"Average Time Taken for All Rounds{time_avg}.\n")

    # # Step 5: Close all client connections after all rounds
    # for client in client_sockets:
    #     client.close()
    #     print("Client connection closed.")

    # server.close()
    # print("Server shutdown.")

if __name__ == "__main__":
    server_socket()