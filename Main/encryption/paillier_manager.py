import numpy as np
from phe import paillier
from typing import List, Tuple

class PaillierManager:
    def __init__(self, key_length: int = 2048):
        self.public_key = None
        self.private_key = None
        self.key_length = key_length
        
    def setup(self):
        """Generate public and private keys"""
        self.public_key, self.private_key = paillier.generate_paillier_keypair(n_length=self.key_length)
        
    def encrypt_parameters(self, parameters: List[np.ndarray]) -> List[List[float]]:
        """Encrypt model parameters using Paillier encryption"""
        encrypted_params = []
        for param in parameters:
            # Flatten array for encryption
            flat_param = param.flatten()
            # Encrypt each value
            encrypted_values = [self.public_key.encrypt(float(val)) for val in flat_param]
            encrypted_params.append({
                'values': encrypted_values,
                'shape': param.shape
            })
        return encrypted_params
    
    def decrypt_parameters(self, encrypted_parameters: List[dict]) -> List[np.ndarray]:
        """Decrypt model parameters"""
        decrypted_params = []
        for enc_param in encrypted_parameters:
            # Decrypt values
            decrypted_values = [self.private_key.decrypt(val) for val in enc_param['values']]
            # Reshape back to original shape
            decrypted_param = np.array(decrypted_values).reshape(enc_param['shape'])
            decrypted_params.append(decrypted_param)
        return decrypted_params
        
    def aggregate_encrypted_parameters(self, encrypted_params_list: List[List[dict]]) -> List[dict]:
        """Aggregate encrypted parameters from multiple nodes"""
        N = len(encrypted_params_list)
        if N == 0:
            return None
            
        # Initialize with first node's parameters
        aggregated_params = []
        for param_idx in range(len(encrypted_params_list[0])):
            shape = encrypted_params_list[0][param_idx]['shape']
            values = encrypted_params_list[0][param_idx]['values']
            
            # Add parameters from other nodes
            for node_idx in range(1, N):
                node_values = encrypted_params_list[node_idx][param_idx]['values']
                values = [v1 + v2 for v1, v2 in zip(values, node_values)]
            
            # Average the values
            values = [v * (1.0/N) for v in values]
            
            aggregated_params.append({
                'values': values,
                'shape': shape
            })
            
        return aggregated_params
