import os
import random
import shutil
from sklearn.model_selection import train_test_split

def split_dataset(source_dir, dest_dir, train_ratio=0.4):
    # Create destination directories
    os.makedirs(os.path.join(dest_dir, 'initial_train', 'drowsy'), exist_ok=True)
    os.makedirs(os.path.join(dest_dir, 'initial_train', 'notdrowsy'), exist_ok=True)
    os.makedirs(os.path.join(dest_dir, 'client1'), exist_ok=True)
    os.makedirs(os.path.join(dest_dir, 'client2'), exist_ok=True)

    # Get all image files
    classes = ['drowsy', 'notdrowsy']
    all_files = []
    for cls in classes:
        class_files = [os.path.join(cls, f) for f in os.listdir(os.path.join(source_dir, cls))]
        all_files.extend(class_files)

    # Split into initial train and remaining
    initial_train, remaining = train_test_split(all_files, train_size=train_ratio, stratify=[f.split(os.sep)[0] for f in all_files])

    # Split remaining into client1 and client2
    client1, client2 = train_test_split(remaining, test_size=0.5, stratify=[f.split(os.sep)[0] for f in remaining])

    # Copy files to respective directories
    for file in initial_train:
        src = os.path.join(source_dir, file)
        dst = os.path.join(dest_dir, 'initial_train', file)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(src, dst)

    # Function to create imbalanced dataset
    def create_imbalanced_dataset(files, dest_dir, majority_class, num_rounds=5):
        os.makedirs(dest_dir, exist_ok=True)
        majority_files = [f for f in files if majority_class in f]
        minority_files = [f for f in files if majority_class not in f]
        
        for round in range(num_rounds):
            round_dir = os.path.join(dest_dir, f'round_{round+1}')
            os.makedirs(os.path.join(round_dir, 'drowsy'), exist_ok=True)
            os.makedirs(os.path.join(round_dir, 'notdrowsy'), exist_ok=True)
            
            # Adjust sampling to handle small minority file count
            majority_sample_size = min(int(0.8 * len(files)), len(majority_files))
            minority_sample_size = min(int(0.2 * len(files)), len(minority_files))

            
            # Use replacement if sample size is larger than population
            selected_majority = random.choices(majority_files, k=majority_sample_size)
            selected_minority = random.choices(minority_files, k=minority_sample_size)
            
            for file in selected_majority + selected_minority:
                src = os.path.join(source_dir, file)
                dst = os.path.join(round_dir, file)
                shutil.copy(src, dst)

    # Create imbalanced datasets for clients
    create_imbalanced_dataset(client1, os.path.join(dest_dir, 'client1'), 'drowsy')
    create_imbalanced_dataset(client2, os.path.join(dest_dir, 'client2'), 'notdrowsy')

# Usage
source_directory = 'D:/Major Project/Dataset'
destination_directory = 'D:/Major Project/Rasp/Data'
split_dataset(source_directory, destination_directory)
