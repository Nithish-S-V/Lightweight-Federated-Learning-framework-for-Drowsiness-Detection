import os
import numpy as np
from pathlib import Path
from server.cloud_server import CloudServer

def run_federated_learning():
    print("\nInitializing federated learning...")
    
    # Initialize cloud server and load initial model
    cloud_server = CloudServer()
    
    # Load initial model weights
    initial_model_path = 'models/initial_model.h5'
    if not os.path.exists(initial_model_path):
        raise FileNotFoundError(f"Initial model not found at {initial_model_path}")
    cloud_server.global_model.model.load_weights(initial_model_path)
    print("Loaded initial model weights")
    
    # Set up clients with their data directories
    client_data_dirs = {
        'client1': 'data/federated_data/client_1',
        'client2': 'data/federated_data/client_2'
    }
    
    cloud_server.initialize_clients(client_data_dirs)
    
    # Federated learning parameters
    num_rounds = 5
    local_epochs = 10
    batch_size = 32
    
    print("\nStarting federated learning with {} clients...".format(len(client_data_dirs)))
    print("Number of rounds:", num_rounds)
    print("Local epochs per round:", local_epochs)
    print("Batch size:", batch_size)
    
    # Create results directory if it doesn't exist
    results_dir = 'results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        
    # Open results file
    results_file = os.path.join(results_dir, 'federated_learning_results.txt')
    with open(results_file, 'w') as f:
        f.write("Federated Learning Results\n")
        f.write("=========================\n\n")
        f.write(f"Number of clients: {len(client_data_dirs)}\n")
        f.write(f"Number of rounds: {num_rounds}\n")
        f.write(f"Local epochs per round: {local_epochs}\n")
        f.write(f"Batch size: {batch_size}\n\n")
    
    for round_num in range(num_rounds):
        print(f"\n=== Round {round_num + 1}/{num_rounds} ===\n")
        
        # Store results for this round
        round_results = {
            'round': round_num + 1,
            'client_results': {},
            'global_metrics': None
        }
        
        client_weights = []
        client_sizes = []
        
        # Train on each client
        for client_id, client in cloud_server.clients.items():
            print(f"\nTraining on {client_id}...")
            
            # Train on client's local data
            client.model = cloud_server.global_model
            client.model.model.set_weights(cloud_server.global_model.model.get_weights())
            client.current_round = round_num + 1
            history = client.train_round(round_num + 1, epochs=local_epochs, batch_size=batch_size)
            
            # Store client results
            round_results['client_results'][client_id] = {
                'train_history': history['history'],
                'metrics': history['metrics']
            }
            
            # Collect client's model weights and dataset size
            client_weights.append(client.model.model.get_weights())
            client_sizes.append(client.get_dataset_size())
            
            print(f"Local training completed for {client_id}")
            
        # Aggregate weights using weighted average
        total_size = sum(client_sizes)
        weighted_weights = []
        
        for i, weights in enumerate(client_weights):
            weight = client_sizes[i] / total_size
            weighted_weights.append([w * weight for w in weights])
        
        # Average the weighted weights
        avg_weights = [sum(weights) for weights in zip(*weighted_weights)]
        
        # Update global model
        cloud_server.global_model.model.set_weights(avg_weights)
        print("\nGlobal model updated with aggregated weights")
        
        # Evaluate global model on test set
        test_dir = 'data/federated_data/initial_test'
        metrics = cloud_server.evaluate_model(test_dir, batch_size=batch_size)
        round_results['global_metrics'] = metrics
        
        print(f"\nGlobal Model Metrics after Round {round_num + 1}:")
        for metric, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"{metric}: {value:.4f}")
            else:
                print(f"{metric}:\n{value}")
            
        # Save round results to file
        with open(results_file, 'a') as f:
            f.write(f"\nRound {round_num + 1} Results\n")
            f.write("-----------------\n")
            
            # Client results
            for client_id, results in round_results['client_results'].items():
                f.write(f"\n{client_id}:\n")
                f.write("Training History (Final Epoch):\n")
                for metric, values in results['train_history'].items():
                    f.write(f"- {metric}: {values[-1]:.4f}\n")
                f.write("\nEvaluation Metrics:\n")
                for metric, value in results['metrics'].items():
                    f.write(f"- {metric}: {value:.4f}\n")
                    
            # Global model results
            f.write("\nGlobal Model Metrics:\n")
            for metric, value in round_results['global_metrics'].items():
                if isinstance(value, (int, float)):
                    f.write(f"- {metric}: {value:.4f}\n")
                else:
                    f.write(f"- {metric}:\n{value}\n")
            f.write("\n")
            
    print(f"\nFederated learning completed. Results saved to {results_file}")
    
    # Save final model
    model_save_path = os.path.join(results_dir, 'final_model.h5')
    cloud_server.global_model.model.save(model_save_path)
    print(f"Final model saved to {model_save_path}")

if __name__ == '__main__':
    run_federated_learning()
