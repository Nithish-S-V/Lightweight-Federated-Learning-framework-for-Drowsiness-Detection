import cv2
import numpy as np
from edge.edge_node import EdgeNode
from server.cloud_server import CloudServer
from encryption.ckks_manager import CKKSManager

def setup_system(num_edge_nodes=4):
    """Initialize the PFTL-DDD system"""
    # Initialize cloud server
    cloud_server = CloudServer()
    cloud_server.ckks.setup()
    
    # Initialize edge nodes
    edge_nodes = []
    for i in range(num_edge_nodes):
        node = EdgeNode(f"node_{i}")
        node.ckks.setup()
        cloud_server.register_edge_node(f"node_{i}")
        edge_nodes.append(node)
    
    return cloud_server, edge_nodes

def train_federated(cloud_server, edge_nodes, local_datasets, num_rounds=10):
    """Run federated training process"""
    for round_idx in range(num_rounds):
        print(f"Training round {round_idx + 1}/{num_rounds}")
        
        # Local training on each edge node
        encrypted_params_list = []
        for node_idx, node in enumerate(edge_nodes):
            # Train on local data
            node.train_local_model(local_datasets[node_idx])
            
            # Get encrypted parameters
            encrypted_params = node.get_encrypted_parameters()
            encrypted_params_list.append(encrypted_params)
        
        # Aggregate parameters on cloud server
        global_encrypted_params = cloud_server.aggregate_parameters(encrypted_params_list)
        
        # Update local models
        for node in edge_nodes:
            node.update_model(global_encrypted_params)

def main():
    # Setup system
    cloud_server, edge_nodes = setup_system()
    
    # Here you would load and distribute your datasets
    # For demonstration, we'll use dummy data
    local_datasets = [
        {'images': [], 'labels': []} for _ in range(len(edge_nodes))
    ]
    
    # Run federated training
    train_federated(cloud_server, edge_nodes, local_datasets)

if __name__ == "__main__":
    main()
