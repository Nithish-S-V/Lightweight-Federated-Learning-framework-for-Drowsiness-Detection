import numpy as np
from cryptography.fernet import Fernet

class PaillierManager:
    def __init__(self):
        """Initialize with a new Fernet key"""
        self.key = Fernet.generate_key()
        self.cipher_suite = Fernet(self.key)
        
    def get_public_key(self):
        """Return the public key for encryption"""
        return self.key.decode()
        
    def set_public_key(self, key):
        """Set the public key for encryption"""
        self.key = key.encode()
        self.cipher_suite = Fernet(self.key)
        
    def encrypt(self, value):
        """Encrypt a float value"""
        # Convert float to bytes
        value_bytes = str(value).encode()
        # Encrypt the bytes
        encrypted = self.cipher_suite.encrypt(value_bytes)
        return encrypted.decode()  # Convert to string for JSON serialization
        
    def decrypt(self, encrypted_value):
        """Decrypt an encrypted value back to float"""
        # Convert string back to bytes
        encrypted_bytes = encrypted_value.encode()
        # Decrypt the bytes
        decrypted = self.cipher_suite.decrypt(encrypted_bytes)
        return float(decrypted.decode())
