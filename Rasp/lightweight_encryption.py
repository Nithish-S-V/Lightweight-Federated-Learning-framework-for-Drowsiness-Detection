import numpy as np
from cryptography.fernet import Fernet
import base64
import json

class LightweightEncryption:
    """Lightweight encryption suitable for Raspberry Pi"""
    
    def __init__(self):
        self.key = None
        self.cipher_suite = None
        
    def generate_key(self):
        """Generate a new encryption key"""
        self.key = Fernet.generate_key()
        self.cipher_suite = Fernet(self.key)
        return self.key
        
    def set_key(self, key):
        """Set encryption key"""
        self.key = key
        self.cipher_suite = Fernet(self.key)
        
    def encrypt_array(self, arr):
        """Encrypt numpy array"""
        # Convert to bytes
        arr_bytes = json.dumps(arr.tolist()).encode()
        # Encrypt
        encrypted_data = self.cipher_suite.encrypt(arr_bytes)
        return base64.b64encode(encrypted_data).decode('utf-8')
        
    def decrypt_array(self, encrypted_str):
        """Decrypt to numpy array"""
        # Decode from base64
        encrypted_data = base64.b64decode(encrypted_str.encode('utf-8'))
        # Decrypt
        decrypted_data = self.cipher_suite.decrypt(encrypted_data)
        # Convert back to numpy array
        return np.array(json.loads(decrypted_data))
        
    def encrypt_weights(self, weights):
        """Encrypt model weights"""
        encrypted_weights = []
        for w in weights:
            encrypted_weights.append(self.encrypt_array(w))
        return encrypted_weights
        
    def decrypt_weights(self, encrypted_weights):
        """Decrypt model weights"""
        weights = []
        for ew in encrypted_weights:
            weights.append(self.decrypt_array(ew))
        return weights
