import socket
import struct
import zlib
import tempfile
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.metrics import classification_report

def load_image_data(base_dir, set_num):
    dataset_dir = f"{base_dir}/set{set_num}"
    datagen = ImageDataGenerator(rescale=1./255,
                                rotation_range=30,  
                                width_shift_range=0.2,  
                                height_shift_range=0.2,
                                shear_range=0.2,  
                                zoom_range=0.2,
                                horizontal_flip=True,
                                brightness_range=[0.8, 1.2],  
                                channel_shift_range=60.0,
                                validation_split=0.3)
    return datagen.flow_from_directory(dataset_dir ,target_size=(128, 128),batch_size=16,  class_mode='categorical',subset='training')

def evaluate_model(model, data_flow):
    predictions = np.argmax(model.predict(data_flow), axis=1)
    true_labels = data_flow.classes
    print("\nClassification Report:")
    print(classification_report(true_labels, predictions, target_names=data_flow.class_indices.keys()))

def client_socket(base_dir):
    for set_num in range(1, 6):  # Automate rounds for all 5 sets
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.connect(('192.168.137.1', 5000))  # Replace with server's IP
        print(f"Connected to server for training on set {set_num}...")

        data_length = struct.unpack('>I', client.recv(4))[0]
        received_data = b""
        while len(received_data) < data_length:
            packet = client.recv(1024 * 1024)
            if not packet:
                break
            received_data += packet
        decompressed_data = zlib.decompress(received_data)
        round_num = struct.unpack('>I', client.recv(4))[0]

        with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as tmp:
            tmp.write(decompressed_data)
            tmp.flush()
            model = tf.keras.models.load_model(tmp.name)

        data_flow = load_image_data(base_dir, set_num)
        model.fit(data_flow, epochs=10)
        evaluate_model(model, data_flow)

        with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as tmp:
            model.save(tmp.name)
            tmp.seek(0)
            updated_weights = tmp.read()

        compressed_data = zlib.compress(updated_weights)
        client.sendall(struct.pack('>I', len(compressed_data)))
        client.sendall(compressed_data)
        client.close()
        print(f"Training on set {set_num} completed.\n")

if _name_ == "_main_":
    client_socket('D:/Major Project/Fl dataset/Jammu and Kashmir')  # For Jammu
    # client_socket('/path_to_himachal')  # Uncomment for Himachal