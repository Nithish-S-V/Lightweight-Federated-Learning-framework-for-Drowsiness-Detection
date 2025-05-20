import socket
import pickle
import threading
import os
import numpy as np
import tensorflow as tf
from model import build_model
from sklearn.metrics import classification_report
from aggregation import (
    fed_avg, fed_trimmed_avg, fed_ma, fed_cda, fed_pa, fed_bn
)
from pathlib import Path
import csv
import matplotlib.pyplot as plt
from collections import defaultdict

# Constants
NUM_CLIENTS = 3
NUM_ROUNDS = 5
HOST = 'localhost'
PORT = 8888
BUFFER_SIZE = 4096
SEPARATOR = "<SEPARATOR>"
MODEL_PATH = "global_model.h5"
DATASET_DIR = "test_dataset"
INPUT_SHAPE = (224, 224, 3)
BATCH_SIZE = 32

# Aggregation methods dictionary
aggregation_methods = {
    "FedAvg": fed_avg,
    "FedTrimmedAvg": fed_trimmed_avg,
    "FedMA": fed_ma,
    "FedCDA": fed_cda,
    "FedPA": fed_pa,
    "FedBN": fed_bn,
}


def log_to_csv(method_name, round_num, accuracy):
    csv_file = "federated_accuracy_results.csv"
    file_exists = Path(csv_file).is_file()
    with open(csv_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Method", "Round", "Accuracy"])
        writer.writerow([method_name, round_num, accuracy])


def evaluate_model(model, dataset_dir, method_name, round_num):
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
    test_gen = test_datagen.flow_from_directory(
        dataset_dir,
        target_size=INPUT_SHAPE[:2],
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        color_mode='rgb',
        shuffle=False
    )
    y_pred = np.argmax(model.predict(test_gen), axis=1)
    y_true = test_gen.classes

    accuracy = np.mean(y_true == y_pred)

    with open(f"aggregation_{method_name}_results.txt", "a") as f:
        f.write(f"Round {round_num} - Accuracy: {accuracy:.4f}\n")
        f.write(classification_report(y_true, y_pred, target_names=['Eyeclose', 'Neutral', 'Yawn'], zero_division=0))
        f.write("\n\n")

    log_to_csv(method_name, round_num, accuracy)

    print(f"{method_name} - Round {round_num} Accuracy: {accuracy:.4f}")


def plot_accuracy_curve(csv_path="federated_accuracy_results.csv"):
    method_round_accuracy = defaultdict(list)

    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            method = row["Method"]
            round_num = int(row["Round"])
            acc = float(row["Accuracy"])
            method_round_accuracy[method].append((round_num, acc))

    plt.figure(figsize=(10, 6))
    for method, values in method_round_accuracy.items():
        values.sort()
        rounds, accs = zip(*values)
        plt.plot(rounds, accs, marker='o', label=method)

    plt.title("Federated Learning Accuracy Over Rounds")
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.legend()
    plt.savefig("accuracy_plot.png")
    plt.show()


def server_socket():
    global_model = build_model(INPUT_SHAPE)
    global_model.save(MODEL_PATH)

    for method_name, aggregation_func in aggregation_methods.items():
        print(f"\nStarting training with {method_name} aggregation")

        for r in range(NUM_ROUNDS):
            print(f"\n--- Round {r + 1} ---")
            client_weights = []

            def handle_client(conn, addr):
                print(f"Client {addr} connected")
                model_data = conn.recv(BUFFER_SIZE)
                client_weight = pickle.loads(model_data)
                client_weights.append(client_weight)
                conn.close()

            threads = []
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind((HOST, PORT))
                s.listen()
                print(f"Server listening on {HOST}:{PORT}")

                for _ in range(NUM_CLIENTS):
                    conn, addr = s.accept()
                    thread = threading.Thread(target=handle_client, args=(conn, addr))
                    thread.start()
                    threads.append(thread)

                for thread in threads:
                    thread.join()

            aggregated_weights = aggregation_func(client_weights)
            global_model.set_weights(aggregated_weights)
            global_model.save(MODEL_PATH)

            evaluate_model(global_model, DATASET_DIR, method_name, r + 1)

    plot_accuracy_curve()


if __name__ == '__main__':
    server_socket()
