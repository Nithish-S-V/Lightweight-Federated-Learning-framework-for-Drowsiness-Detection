import argparse
from server.cloud_server import run_federated_training

def main():
    parser = argparse.ArgumentParser(description='Run federated learning for drowsiness detection')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to main dataset directory')
    parser.add_argument('--output_dir', type=str, default='models', help='Directory to save models')
    parser.add_argument('--num_rounds', type=int, default=5, help='Number of federated rounds')
    parser.add_argument('--epochs_per_round', type=int, default=10, help='Epochs per client per round')
    parser.add_argument('--batch_size', type=int, default=16, help='Training batch size')
    
    args = parser.parse_args()
    
    # Run federated training
    history = run_federated_training(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        num_rounds=args.num_rounds,
        epochs_per_round=args.epochs_per_round,
        batch_size=args.batch_size
    )
    
    print("\nFederated training completed!")

if __name__ == "__main__":
    main()
