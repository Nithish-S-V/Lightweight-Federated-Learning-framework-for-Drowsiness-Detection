import tenseal as ts
import numpy as np

class CKKSManager:
    def __init__(self, security_level=128):
        self.context = None
        self.public_key = None
        self.secret_key = None
        self.security_level = security_level
        
    def setup(self):
        """Initialize CKKS context and generate keys"""
        # Create TenSEAL context
        self.context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[60, 40, 40, 60]
        )
        self.context.generate_galois_keys()
        self.context.global_scale = 2**40
        
        # Generate keys
        self.secret_key = self.context.secret_key()
        self.public_key = self.context.public_key()
        
    def encrypt_parameters(self, parameters):
        """Encrypt model parameters using CKKS"""
        encrypted_params = []
        for param in parameters:
            if isinstance(param, np.ndarray):
                # Flatten array and encrypt
                flat_param = param.flatten()
                encrypted = ts.ckks_vector(self.context, flat_param)
                encrypted_params.append(encrypted)
        return encrypted_params
    
    def decrypt_parameters(self, encrypted_parameters):
        """Decrypt model parameters"""
        decrypted_params = []
        for enc_param in encrypted_parameters:
            if isinstance(enc_param, ts.ckks_vector):
                # Decrypt and reshape
                dec_param = enc_param.decrypt()
                decrypted_params.append(np.array(dec_param))
        return decrypted_params
