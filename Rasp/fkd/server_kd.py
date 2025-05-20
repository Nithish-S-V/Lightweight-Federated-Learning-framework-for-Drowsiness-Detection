import socket
import struct
import zlib
import tempfile
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report
from tensorflow.keras.layers import Layer
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.saving import register_keras_serializable
from tensorflow.keras import backend as K
from sklearn.metrics import confusion_matrix, accuracy_score
from cryptography.fernet import Fernet
import base64
import hashlib
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, SeparableConv2D, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.losses import categorical_crossentropy, KLDivergence

# Custom encryption key
custom_key = "secretkey"
hashed_key = hashlib.sha256(custom_key.encode()).digest()
encoded_key = base64.urlsafe_b64encode(hashed_key)
KEY = encoded_key
cipher_suite = Fernet(KEY)

@register_keras_serializable(package="Custom")
class CapsuleLayer(Layer):
    def __init__(self, num_capsule, dim_capsule, num_routing, **kwargs):
        super(CapsuleLayer, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.num_routing = num_routing

    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=(input_shape[-1], self.num_capsule * self.dim_capsule),
            initializer="glorot_uniform",
            trainable=True,
            name="capsule_kernel",
        )
        super(CapsuleLayer, self).build(input_shape)

    def call(self, inputs):
        inputs_hat = tf.linalg.matmul(inputs, self.kernel)
        inputs_hat = tf.reshape(inputs_hat, (-1, self.num_capsule, self.dim_capsule))

        b = tf.zeros(shape=(tf.shape(inputs)[0], self.num_capsule, 1))
        for i in range(self.num_routing):
            c = tf.nn.softmax(b, axis=1)
            outputs = self.squash(tf.reduce_sum(c * inputs_hat, axis=1, keepdims=True))
            b += tf.linalg.matmul(inputs_hat, outputs, transpose_b=True)

        return tf.squeeze(outputs, axis=1)

    def squash(self, vectors, axis=-1):
        s_squared_norm = tf.reduce_sum(tf.square(vectors), axis, keepdims=True)
        scale = s_squared_norm / (1 + s_squared_norm) / tf.sqrt(s_squared_norm + K.epsilon())
        return scale * vectors

# Load the pre-trained teacher model
def load_teacher_model():
    return tf.keras.models.load_model(
        'C:/Jeeva/college/sem 8/Major project/capsule_mobilenet_best_model4.keras', 
        custom_objects={"CapsuleLayer": CapsuleLayer}
    )

# Define a smaller student model
def create_student_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    
    # First depthwise separable convolution layer
    x = SeparableConv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    # Second depthwise separable convolution layer
    x = SeparableConv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    # Third depthwise separable convolution layer
    x = SeparableConv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    # Global Average Pooling instead of Flatten
    x = GlobalAveragePooling2D()(x)
    
    # Fully connected layer with Dropout for regularization
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    
    # Output layer
    outputs = Dense(num_classes, activation='softmax')(x)
    
    # Create the model
    return Model(inputs, outputs, name="OptimizedStudentModel")

# Generate soft labels using the teacher model
def generate_soft_labels(teacher_model, val_flow):

    # Generate soft labels using the teacher model
    soft_labels = teacher_model.predict(val_flow, verbose=1)

    return soft_labels # Return teacher's predictions as labels for training

def load_image_data(base_dir):
    datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)  # 20% for validation
    
    train_flow = datagen.flow_from_directory(
        base_dir,
        target_size=(128, 128),
        batch_size=16,
        class_mode='categorical',
        subset='training',  # Specify training subset
        shuffle=True
    )

    val_flow = datagen.flow_from_directory(
        base_dir,
        target_size=(128, 128),
        batch_size=16,
        class_mode='categorical',
        subset='validation',  # Specify validation subset
        shuffle=True
    )
    
    return train_flow, val_flow

class CustomDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, train_flow, soft_labels, batch_size):
        self.train_flow = train_flow
        self.soft_labels = soft_labels
        self.batch_size = batch_size
        self.indexes = np.arange(len(train_flow))
    
    def __len__(self):
        return int(np.floor(len(self.train_flow) / self.batch_size))
    
    def __getitem__(self, index):
        # Get the image batch
        batch_data, _ = self.train_flow[index]  
        
        # Compute the correct slice for soft labels based on the index
        start_idx = index * self.batch_size
        end_idx = start_idx + self.batch_size
        
        # Ensure the slice is within bounds of the soft labels array
        soft_labels_batch = self.soft_labels[start_idx:end_idx]
        
        # # Debugging print statements
        # print(f"Index: {index}, Start: {start_idx}, End: {end_idx}")
        # print(f"Batch data shape: {batch_data.shape}")
        # print(f"Soft labels batch shape: {soft_labels_batch.shape}")
        
        return batch_data, soft_labels_batch

def distillation_loss(y_true, y_pred, soft_labels, alpha=0.7, temperature=3.0):
   
    soft_targets = tf.nn.softmax(soft_labels / temperature)  # Softened teacher outputs
    student_probs = tf.nn.softmax(y_pred / temperature)  # Softened student outputs

    # KL Divergence Loss for soft labels
    soft_loss = KLDivergence()(soft_targets, student_probs) * (temperature ** 2)

    # Crossentropy Loss for hard labels
    hard_loss = categorical_crossentropy(y_true, y_pred)

    # Weighted sum of soft and hard loss
    return alpha * soft_loss + (1 - alpha) * hard_loss

# Train student model using soft labels
def train_student_model(student_model, train_flow, soft_labels):

    student_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # student_model.compile(optimizer=Adam(learning_rate=0.001),
    # loss=lambda y_true, y_pred: distillation_loss(y_true, y_pred, soft_labels))

    cg = CustomDataGenerator(train_flow, soft_labels, batch_size=16)
    student_model.fit(cg, epochs=20, batch_size=16, verbose=0)  
    return student_model

# Send student model to client
def send_student_model(client, student_model, round_num):
    with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as tmp:
        student_model.save(tmp.name)
        tmp.seek(0)
        model_data = tmp.read()
    
    compressed_data = zlib.compress(model_data)
    encrypted_data = cipher_suite.encrypt(compressed_data)  
    data_length = struct.pack('>I', len(encrypted_data))

    client.sendall(data_length)
    with tqdm(total=len(encrypted_data), unit="B", unit_scale=True, desc="Sending model") as pbar:
        for i in range(0, len(encrypted_data), 1024 * 1024):
            client.sendall(encrypted_data[i:i + 1024 * 1024])
            pbar.update(len(encrypted_data[i:i + 1024 * 1024]))
    
    print(f"Student model for Round {round_num} sent to client.")
    client.sendall(struct.pack('>I', round_num))  

# Receive weights from client
def receive_client_weights(client):
    data_length = struct.unpack('>I', client.recv(4))[0]
    received_data = b""
    while len(received_data) < data_length:
        packet = client.recv(1024 * 1024)
        if not packet:
            break
        received_data += packet

    decrypted_data = cipher_suite.decrypt(received_data)  
    decompressed_data = zlib.decompress(decrypted_data)
    
    with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as tmp:
        tmp.write(decompressed_data)
        tmp.flush()
        model = tf.keras.models.load_model(tmp.name)
        return model.get_weights()

# Aggregation of weights
def aggregate_weights(client_weights):
    aggregated_weights = [np.zeros_like(layer) for layer in client_weights[0]]
    for weights in client_weights:
        for i, layer in enumerate(weights):
            aggregated_weights[i] += layer
    return [layer / len(client_weights) for layer in aggregated_weights]

# Evaluate the global model
def evaluate_model(model, dataset_dir):
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)
    data_flow = datagen.flow_from_directory(
        dataset_dir,
        target_size=(128, 128),
        batch_size=16,
        class_mode='categorical',
        shuffle=False,
    )
    predictions = np.argmax(model.predict(data_flow), axis=1)
    true_labels = data_flow.classes
    test_accuracy = accuracy_score(true_labels, predictions)
    print(f"Test Accuracy (global model): {test_accuracy:.2%}")
    print("\nClassification Report after updating global model:")
    print(classification_report(true_labels, predictions, target_names=list(data_flow.class_indices.keys())))


# Server function
def server_socket():
    teacher_model = load_teacher_model()
    student_model = create_student_model(input_shape=(128, 128, 3), num_classes=7)  

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(('0.0.0.0', 5000))  
    server.listen(10)

    num_clients = 2  
    client_sockets = []

    print("Waiting for clients to connect...")
    for i in range(num_clients):
        client, addr = server.accept()
        print(f"Client {i + 1} connected from {addr}.")
        client_sockets.append(client)

    for round_num in range(1, 6):
        print(f"\n==== Round {round_num} ====")
        
        dataset = 'C:/Jeeva/college/sem 8/Major project/Dataset/Fl dataset/Initial Dataset'
        train_flow, val_flow = load_image_data(dataset)
        soft_labels = generate_soft_labels(teacher_model, val_flow)
        student_model = train_student_model(student_model, train_flow, soft_labels)
        
        if round_num == 1:
            print("\nEvaluating the teacher model before sending it to clients in Round 1...")
            evaluate_model(teacher_model, 'C:/Jeeva/college/sem 8/Major project/Dataset/Fl dataset/Test')
        # Evaluate the student model before sending it to the clients in Round 1
            print("\nEvaluating the student model before sending it to clients in Round 1...")
            evaluate_model(student_model, 'C:/Jeeva/college/sem 8/Major project/Dataset/Fl dataset/Test')


        client_weights = []
        # Step 1: Send the student model to all clients
        for client in client_sockets:
            send_student_model(client, student_model, round_num)

        # Step 2: Receive updated weights from all clients
        for i, client in enumerate(client_sockets):
            print(f"Receiving weights from client {i + 1} for Round {round_num}...")
            weights = receive_client_weights(client)
            client_weights.append(weights)

        # Step 3:Aggregate weights and update the student model
        aggregated_weights = aggregate_weights(client_weights)
        student_model.set_weights(aggregated_weights)
        print(f"Student model updated after Round {round_num}.")

        # Step 4: Evaluate the updated global model
        evaluate_model(student_model, 'C:/Jeeva/college/sem 8/Major project/Dataset/Fl dataset/Test')

    # Step 5: Close all client connections after all rounds
    for client in client_sockets:
        client.close()
        print("Client connection closed.")

    server.close()
    print("Server shutdown.")

if __name__ == "__main__":
    server_socket()
