from utils.data_splitter import DataSplitter
import os

def main():
    data_dir = "d:/Major Project/Main/data/train"
    
    # Create data splitter
    print("\nSplitting data for federated learning...")
    splitter = DataSplitter(data_dir)
    data_paths = splitter.create_federated_splits(
        num_rounds=5,
        num_clients=2,
        initial_split=0.4,
        test_split=0.2
    )
    
    # Print split info
    split_info = splitter.get_split_info()
    print("\nData Split Information:")
    print("\nInitial Training Set:")
    print(split_info['initial_train'])
    print("\nInitial Test Set:")
    print(split_info['initial_test'])
    
    print("\nClient Data:")
    for client, rounds in split_info['clients'].items():
        print(f"\n{client}:")
        for round_num, class_counts in rounds.items():
            print(f"  {round_num}:")
            for class_name, count in class_counts.items():
                print(f"    {class_name}: {count} images")

if __name__ == "__main__":
    main()
