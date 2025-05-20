import os
import shutil
import numpy as np
from sklearn.model_selection import train_test_split

class NonIIDDataDistributor:
    def __init__(self, source_data_path, output_base_path):
        self.source_path = source_data_path
        self.output_path = output_base_path
        
    def create_non_iid_distribution(self, num_clients=2, num_rounds=5):
        """Create non-IID data distribution for federated learning"""
        # Get all files per class
        drowsy_files = os.listdir(os.path.join(self.source_path, 'drowsy'))
        non_drowsy_files = os.listdir(os.path.join(self.source_path, 'non_drowsy'))
        
        # Remove any non-image files
        drowsy_files = [f for f in drowsy_files if f.endswith(('.jpg', '.jpeg', '.png'))]
        non_drowsy_files = [f for f in non_drowsy_files if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        for client_id in range(num_clients):
            # Create different class ratios for each client
            # Client 1: 70% drowsy, 30% non-drowsy
            # Client 2: 30% drowsy, 70% non-drowsy
            ratio = 0.7 if client_id == 0 else 0.3
            
            # Calculate samples per class for this client
            num_drowsy = int(len(drowsy_files) * ratio)
            num_non_drowsy = int(len(non_drowsy_files) * (1 - ratio))
            
            # Sample files for this client
            client_drowsy = np.random.choice(drowsy_files, num_drowsy, replace=False)
            client_non_drowsy = np.random.choice(non_drowsy_files, num_non_drowsy, replace=False)
            
            # Update remaining files
            drowsy_files = list(set(drowsy_files) - set(client_drowsy))
            non_drowsy_files = list(set(non_drowsy_files) - set(client_non_drowsy))
            
            # Split into rounds
            drowsy_splits = np.array_split(client_drowsy, num_rounds)
            non_drowsy_splits = np.array_split(client_non_drowsy, num_rounds)
            
            # Create directory structure and copy files
            for round_idx in range(num_rounds):
                round_path = os.path.join(
                    self.output_path, 
                    f'client_{client_id}',
                    f'round_{round_idx + 1}'
                )
                
                # Create directories
                os.makedirs(os.path.join(round_path, 'drowsy'), exist_ok=True)
                os.makedirs(os.path.join(round_path, 'non_drowsy'), exist_ok=True)
                
                # Copy files for this round
                for file in drowsy_splits[round_idx]:
                    shutil.copy2(
                        os.path.join(self.source_path, 'drowsy', file),
                        os.path.join(round_path, 'drowsy', file)
                    )
                
                for file in non_drowsy_splits[round_idx]:
                    shutil.copy2(
                        os.path.join(self.source_path, 'non_drowsy', file),
                        os.path.join(round_path, 'non_drowsy', file)
                    )
                    
    def get_distribution_stats(self):
        """Get statistics about the data distribution"""
        stats = {}
        
        for client_dir in os.listdir(self.output_path):
            if not client_dir.startswith('client_'):
                continue
                
            client_stats = {'rounds': {}}
            client_path = os.path.join(self.output_path, client_dir)
            
            for round_dir in os.listdir(client_path):
                if not round_dir.startswith('round_'):
                    continue
                    
                round_path = os.path.join(client_path, round_dir)
                drowsy_count = len(os.listdir(os.path.join(round_path, 'drowsy')))
                non_drowsy_count = len(os.listdir(os.path.join(round_path, 'non_drowsy')))
                
                client_stats['rounds'][round_dir] = {
                    'drowsy': drowsy_count,
                    'non_drowsy': non_drowsy_count,
                    'total': drowsy_count + non_drowsy_count,
                    'ratio': drowsy_count / (drowsy_count + non_drowsy_count)
                }
            
            stats[client_dir] = client_stats
            
        return stats
