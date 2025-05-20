# Implementation Structure

The project is organized into two main implementations:

## 1. Standard Implementation (./standard_fl/)
Contains the current federated learning implementation using MobileNet+CapsNet for laptops.

## 2. Lightweight Implementation (./lightweight_fl/)
Contains the Raspberry Pi optimized implementation with:
- Lightweight student model
- Simplified encryption
- Knowledge distillation from the standard model

## Directory Structure
```
.
├── standard_fl/
│   ├── models/
│   │   ├── mobilenet_capsnet.py
│   │   └── capsule_layer.py
│   ├── encryption/
│   │   └── paillier_encryption.py
│   ├── server/
│   │   └── server.py
│   └── client/
│       └── client.py
│
└── lightweight_fl/
    ├── models/
    │   ├── student_model.py      # Lightweight CNN
    │   └── knowledge_transfer.py # Distillation logic
    ├── encryption/
    │   └── lightweight_encryption.py
    ├── server/
    │   └── server.py
    ├── client/
    │   └── client.py
    └── RASPBERRY_PI_SETUP.md     # Setup instructions
```

## Key Differences

### 1. Non-Identical Dataset Distribution
- Standard: Uses balanced dataset splits
- Lightweight: Handles non-IID data distribution

### 2. Model Architecture
- Standard: MobileNet + CapsNet
- Lightweight: Custom lightweight CNN with knowledge distillation

### 3. Encryption
- Standard: Full Paillier homomorphic encryption
- Lightweight: Simplified encryption suitable for Raspberry Pi

### 4. Resource Usage
- Standard: Optimized for laptops/desktops
- Lightweight: Optimized for Raspberry Pi constraints
