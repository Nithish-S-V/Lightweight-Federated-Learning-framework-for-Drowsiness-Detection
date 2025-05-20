import numpy as np
from pathlib import Path
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from encryption.paillier_manager import PaillierManager
from models.mobilenet_capsnet import MobileNetCapsNet
from edge.edge_node import EdgeNode

class CloudServer:
    def __init__(self):
        """Initialize cloud server for federated learning"""
        self.paillier = PaillierManager()
        self.global_model = MobileNetCapsNet()
        self.global_model.compile_model()
        self.clients = {}
        
    def initialize_clients(self, client_data_dirs):
        """Initialize clients with their data directories"""
        for client_id, data_dir in client_data_dirs.items():
            self.clients[client_id] = EdgeNode(client_id, data_dir)
            
        print(f"Initialized {len(self.clients)} clients: {list(self.clients.keys())}")
        
    def evaluate_model(self, data_dir, model=None, batch_size=32):
        """Evaluate model on a dataset"""
        if model is None:
            model = self.global_model.model
            
        # Create data generator
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
        generator = datagen.flow_from_directory(
            data_dir,
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode='binary',
            shuffle=False
        )
        
        # Get predictions
        predictions = model.predict(generator, batch_size=batch_size)
        y_pred = np.argmax(predictions, axis=1)  # Take argmax since we have 2 output classes
        y_true = generator.classes
        
        # Calculate metrics
        metrics = {
            'accuracy': np.mean(y_true == y_pred),
            'confusion_matrix': confusion_matrix(y_true, y_pred),
            'classification_report': classification_report(y_true, y_pred, target_names=['Not Drowsy', 'Drowsy'])
        }
        
        return metrics
        
    def train_initial_model(self, train_dir, test_dir, epochs=20, batch_size=32):
        """Train initial model on centralized data"""
        print("\n=== Training Initial Model ===")
        
        # Create data generators
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            validation_split=0.2
        )
        
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode='binary',
            subset='training',
            shuffle=True
        )
        
        valid_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode='binary',
            subset='validation',
            shuffle=True
        )
        
        # Train model
        history = self.global_model.model.fit(
            train_generator,
            validation_data=valid_generator,
            epochs=epochs,
            verbose=1
        )
        
        # Create test generator with same batch size
        test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
        test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode='binary',
            shuffle=False
        )
        
        # Evaluate on test set
        print("\nEvaluating initial model on test set...")
        test_metrics = self.evaluate_model(test_dir, batch_size=batch_size)
        
        print("\nTest Set Metrics:")
        print(f"Accuracy: {test_metrics['accuracy']:.4f}")
        print("\nClassification Report:")
        print(test_metrics['classification_report'])
        print("\nConfusion Matrix:")
        print(test_metrics['confusion_matrix'])
        
        return {
            'history': history.history,
            'test_metrics': test_metrics
        }
        
    def train_federated(self, num_rounds=5, epochs_per_round=10, batch_size=16, test_dir=None):
        """Run federated training across all clients"""
        training_history = {
            'rounds': [],
            'client_metrics': {},
            'global_metrics': []
        }
        
        for round_num in range(1, num_rounds + 1):
            print(f"\n{'='*20} Round {round_num}/{num_rounds} {'='*20}")
            round_weights = {}
            
            # Train each client
            for client_id, client in self.clients.items():
                print(f"\nTraining {client_id}")
                results = client.train_round(
                    round_num=round_num,
                    epochs=epochs_per_round,
                    batch_size=batch_size
                )
                
                if client_id not in training_history['client_metrics']:
                    training_history['client_metrics'][client_id] = []
                training_history['client_metrics'][client_id].append(results['metrics'])
                
                round_weights[client_id] = client.get_encrypted_parameters()
            
            # Aggregate weights
            global_weights = self.aggregate_weights(round_weights)
            
            # Update clients
            for client in self.clients.values():
                client.update_model(global_weights)
                
            # Evaluate global model if test set provided
            if test_dir:
                print(f"\nEvaluating global model after round {round_num}")
                metrics = self.evaluate_model(test_dir)
                training_history['global_metrics'].append({
                    'round': round_num,
                    'metrics': metrics
                })
                
                print(f"\nTest Set Metrics:")
                print(f"Accuracy: {metrics['accuracy']:.4f}")
                print("\nClassification Report:")
                print(metrics['classification_report'])
                print("\nConfusion Matrix:")
                print(metrics['confusion_matrix'])
            
            # Store round information
            training_history['rounds'].append({
                'round_num': round_num,
                'client_weights': round_weights,
                'global_weights': global_weights
            })
            
        return training_history
        
    def aggregate_weights(self, client_weights):
        """Aggregate encrypted weights from all clients"""
        if not client_weights:
            raise ValueError("No client weights available for aggregation")
            
        num_clients = len(client_weights)
        aggregated_weights = {}
        
        all_params = set()
        for weights in client_weights.values():
            all_params.update(weights.keys())
            
        for param in all_params:
            param_values = [weights[param] for weights in client_weights.values() 
                          if param in weights]
            aggregated_weights[param] = sum(param_values) / num_clients
            
        return aggregated_weights
        
    def save_global_model(self, save_path):
        """Save the global model"""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        self.global_model.model.save(str(save_path))
        print(f"Global model saved to: {save_path}")
        
    def load_global_model(self, model_path):
        """Load a saved model"""
        self.global_model.model.load_weights(model_path)
        print(f"Loaded global model from: {model_path}")
        
def run_federated_training(data_dir, output_dir='models', num_rounds=5, epochs_per_round=10, batch_size=16):
    """Run complete federated learning pipeline"""
    from utils.data_splitter import DataSplitter
    
    # Split data
    print("\nSplitting data for federated learning...")
    splitter = DataSplitter(data_dir)
    data_paths = splitter.create_federated_splits(
        num_rounds=num_rounds,
        num_clients=2,
        initial_split=0.4,
        test_split=0.2
    )
    
    # Print split info
    split_info = splitter.get_split_info()
    print("\nData Split Information:")
    print(f"Initial Training Set: {split_info['initial_train']}")
    print(f"Initial Test Set: {split_info['initial_test']}")
    for client, rounds in split_info['clients'].items():
        print(f"\n{client}:")
        for round_num, class_counts in rounds.items():
            print(f"  {round_num}: {class_counts}")
    
    # Initialize server
    print("\nInitializing cloud server...")
    server = CloudServer()
    
    # Train initial model
    print("\nTraining initial centralized model...")
    initial_results = server.train_initial_model(
        train_dir=data_paths['initial_train'],
        test_dir=data_paths['initial_test'],
        epochs=20,
        batch_size=32
    )
    
    # Initialize clients
    client_dirs = data_paths['client_dirs']
    server.initialize_clients(client_dirs)
    
    # Run federated training
    print("\nStarting federated training...")
    history = server.train_federated(
        num_rounds=num_rounds,
        epochs_per_round=epochs_per_round,
        batch_size=batch_size,
        test_dir=data_paths['initial_test']
    )
    
    # Save models
    output_path = Path(output_dir)
    server.save_global_model(output_path / 'final_federated_model.h5')
    
    return {
        'initial_results': initial_results,
        'federated_history': history
    }
