import os
import shutil
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

class DataSplitter:
    def __init__(self, source_dir):
        """Initialize data splitter
        
        Args:
            source_dir (str): Path to main dataset directory containing 'drowsy' and 'non_drowsy' folders
        """
        self.source_dir = Path(source_dir)
        self.output_dir = Path(os.path.join(os.path.dirname(str(self.source_dir)), "federated_data"))
        
    def create_federated_splits(self, num_rounds=5, num_clients=2, initial_split=0.4, test_split=0.2):
        """Split data for federated learning
        
        Args:
            num_rounds (int): Number of federated learning rounds
            num_clients (int): Number of clients
            initial_split (float): Fraction of data to use for initial training
            test_split (float): Fraction of initial data to use for testing
            
        Returns:
            dict: Paths to all created datasets
        """
        # Clear and recreate output directory
        if self.output_dir.exists():
            shutil.rmtree(str(self.output_dir))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create initial training and test directories with class subdirectories
        for split in ['initial_train', 'initial_test']:
            split_dir = self.output_dir / split
            split_dir.mkdir(parents=True, exist_ok=True)
            for class_name in ['drowsy', 'non_drowsy']:
                (split_dir / class_name).mkdir(parents=True, exist_ok=True)
        
        # Create round directories for each client with class subdirectories
        client_dirs = {}
        for client in range(num_clients):
            client_base = self.output_dir / f"client_{client+1}"
            client_base.mkdir(parents=True, exist_ok=True)
            client_dirs[client] = []
            
            for round_num in range(num_rounds):
                round_dir = client_base / f"round_{round_num+1}"
                round_dir.mkdir(parents=True, exist_ok=True)
                for class_name in ['drowsy', 'non_drowsy']:
                    (round_dir / class_name).mkdir(parents=True, exist_ok=True)
                client_dirs[client].append(round_dir)
        
        # Process each class
        for class_name in ['drowsy', 'non_drowsy']:
            class_dir = self.source_dir / class_name
            if not class_dir.exists():
                raise ValueError(f"Class directory not found: {class_dir}")
            
            # Get all image files
            image_files = []
            for ext in ['*.jpg', '*.png']:
                image_files.extend(list(class_dir.glob(ext)))
            np.random.shuffle(image_files)
            
            # Split for initial training
            n_initial = int(len(image_files) * initial_split)
            initial_files = image_files[:n_initial]
            federated_files = image_files[n_initial:]
            
            # Further split initial data into train and test
            n_test = int(len(initial_files) * test_split)
            initial_train_files = initial_files[:-n_test]
            initial_test_files = initial_files[-n_test:]
            
            # Copy initial files
            for file in initial_train_files:
                shutil.copy2(str(file), str(self.output_dir / 'initial_train' / class_name / file.name))
            for file in initial_test_files:
                shutil.copy2(str(file), str(self.output_dir / 'initial_test' / class_name / file.name))
            
            # Split remaining files for federated rounds
            files_per_round = len(federated_files) // (num_rounds * num_clients)
            
            # Distribute files to clients and rounds
            for round_num in range(num_rounds):
                start_idx = round_num * files_per_round * num_clients
                for client in range(num_clients):
                    client_start = start_idx + (client * files_per_round)
                    client_end = client_start + files_per_round
                    round_files = federated_files[client_start:client_end]
                    
                    # Copy files for this round
                    for file in round_files:
                        dest_path = client_dirs[client][round_num] / class_name / file.name
                        shutil.copy2(str(file), str(dest_path))
        
        # Return paths
        return {
            'initial_train': str(self.output_dir / 'initial_train'),
            'initial_test': str(self.output_dir / 'initial_test'),
            'client_dirs': {
                f'client_{i+1}': {
                    f'round_{j+1}': str(client_dirs[i][j])
                    for j in range(num_rounds)
                }
                for i in range(num_clients)
            }
        }
        
    def get_split_info(self):
        """Get information about the data splits"""
        info = {'initial_train': {}, 'initial_test': {}, 'clients': {}}
        
        # Get initial training set info
        for split in ['initial_train', 'initial_test']:
            split_dir = self.output_dir / split
            if split_dir.exists():
                for class_name in ['drowsy', 'non_drowsy']:
                    class_dir = split_dir / class_name
                    if class_dir.exists():
                        info[split][class_name] = len(list(class_dir.glob('*.*')))
        
        # Get client round info
        for client_dir in self.output_dir.glob('client_*'):
            client_info = {}
            for round_dir in client_dir.glob('round_*'):
                round_info = {}
                for class_name in ['drowsy', 'non_drowsy']:
                    class_dir = round_dir / class_name
                    if class_dir.exists():
                        round_info[class_name] = len(list(class_dir.glob('*.*')))
                client_info[round_dir.name] = round_info
            info['clients'][client_dir.name] = client_info
            
        return info
