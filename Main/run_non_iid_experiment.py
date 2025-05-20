import os
import shutil
import numpy as np
from lightweight_fl.utils.data_distributor import NonIIDDataDistributor
import subprocess
import time
import json

def setup_non_iid_data():
    """Create non-IID data distribution"""
    # Source data paths
    source_data = "data/train"  # Contains drowsy and non_drowsy folders
    output_base = "data/non_iid_data"
    
    # Clean previous data if exists
    if os.path.exists(output_base):
        shutil.rmtree(output_base)
    os.makedirs(output_base)
    
    # Create non-IID distribution
    distributor = NonIIDDataDistributor(source_data, output_base)
    distributor.create_non_iid_distribution(num_clients=2, num_rounds=5)
    
    # Print distribution stats
    stats = distributor.get_distribution_stats()
    print("\nData Distribution Statistics:")
    print(json.dumps(stats, indent=2))
    return stats

def run_federated_learning():
    """Run federated learning with non-IID data"""
    # Start server
    server_process = subprocess.Popen(
        ["python", "standard_fl/server/server.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True  # This makes the output strings instead of bytes
    )
    print("Server started...")
    time.sleep(5)  # Wait for server to initialize
    
    # Start clients
    client1_process = subprocess.Popen(
        [
            "python", "standard_fl/client/client.py",
            "--client_id", "1",
            "--server_host", "localhost",
            "--data_path", "data/non_iid_data/client_0"
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    client2_process = subprocess.Popen(
        [
            "python", "standard_fl/client/client.py",
            "--client_id", "2",
            "--server_host", "localhost",
            "--data_path", "data/non_iid_data/client_1"
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    print("Clients started...")
    
    try:
        # Monitor training progress
        while True:
            # Check if processes are still running
            if server_process.poll() is not None:
                print("Server process ended")
                break
                
            # Print output from server
            server_line = server_process.stdout.readline()
            if server_line:
                print("Server:", server_line.strip())
                
            # Print output from clients
            client1_line = client1_process.stdout.readline()
            if client1_line:
                print("Client 1:", client1_line.strip())
                
            client2_line = client2_process.stdout.readline()
            if client2_line:
                print("Client 2:", client2_line.strip())
                
            # Check for errors
            server_error = server_process.stderr.readline()
            if server_error:
                print("Server Error:", server_error.strip())
                
            client1_error = client1_process.stderr.readline()
            if client1_error:
                print("Client 1 Error:", client1_error.strip())
                
            client2_error = client2_process.stderr.readline()
            if client2_error:
                print("Client 2 Error:", client2_error.strip())
                
    except KeyboardInterrupt:
        print("\nStopping processes...")
    finally:
        # Cleanup processes
        for process in [server_process, client1_process, client2_process]:
            if process.poll() is None:
                process.terminate()
                process.wait()
                
        # Print any final errors
        for name, proc in [("Server", server_process), 
                         ("Client 1", client1_process), 
                         ("Client 2", client2_process)]:
            if proc.returncode != 0:
                print(f"\n{name} Error Output:")
                print(proc.stderr.read())

if __name__ == "__main__":
    print("Setting up non-IID data distribution...")
    stats = setup_non_iid_data()
    
    print("\nStarting federated learning...")
    run_federated_learning()
