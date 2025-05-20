import os
import shutil

def create_directory_structure():
    """Create and organize the implementation directories"""
    # Base directories
    os.makedirs('standard_fl', exist_ok=True)
    os.makedirs('lightweight_fl', exist_ok=True)
    
    # Standard FL structure
    standard_dirs = [
        'standard_fl/models',
        'standard_fl/encryption',
        'standard_fl/server',
        'standard_fl/client',
        'standard_fl/utils'
    ]
    
    # Create directories
    for dir_path in standard_dirs:
        os.makedirs(dir_path, exist_ok=True)
    
    # Move existing files to standard_fl
    file_moves = {
        'models/mobilenet_capsnet.py': 'standard_fl/models/',
        'models/capsule_layer.py': 'standard_fl/models/',
        'encryption/paillier_encryption.py': 'standard_fl/encryption/',
        'server/cloud_server.py': 'standard_fl/server/server.py',
        'edge/edge_node.py': 'standard_fl/client/client.py',
        'utils/data_utils.py': 'standard_fl/utils/'
    }
    
    for src, dest in file_moves.items():
        if os.path.exists(src):
            # Create destination directory if needed
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            # Copy file
            shutil.copy2(src, dest)
            print(f"Moved {src} to {dest}")
            
    print("\nDirectory structure organized successfully!")
    print("\nImplementation structure:")
    print("1. Standard FL (Laptop Implementation):")
    print("   - Full MobileNet+CapsNet model")
    print("   - Paillier homomorphic encryption")
    print("   - Original data distribution")
    print("\n2. Lightweight FL (Raspberry Pi Implementation):")
    print("   - Lightweight CNN with knowledge distillation")
    print("   - Simplified encryption")
    print("   - Non-IID data distribution")
    print("   - Memory-efficient training")
    print("   - Model quantization")

if __name__ == '__main__':
    create_directory_structure()
