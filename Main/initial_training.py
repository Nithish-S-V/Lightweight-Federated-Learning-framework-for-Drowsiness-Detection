from server.cloud_server import CloudServer

def main():
    # Initialize server
    print("\nInitializing cloud server...")
    server = CloudServer()
    
    # Train initial model
    print("\nTraining initial centralized model...")
    initial_results = server.train_initial_model(
        train_dir="d:/Major Project/Main/data/federated_data/initial_train",
        test_dir="d:/Major Project/Main/data/federated_data/initial_test",
        epochs=20,
        batch_size=32
    )
    
    # Save initial model
    server.save_global_model('models/initial_model.h5')
    print("\nInitial training completed!")

if __name__ == "__main__":
    main()
