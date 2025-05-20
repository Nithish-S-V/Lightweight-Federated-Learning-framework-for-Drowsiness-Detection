import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten, Dropout

class DrowsinessDetectionModel:
    def __init__(self):
        self.base_model = None
        self.model = None
        self._build_model()
    
    def _build_model(self):
        # Load VGG16 as base model
        self.base_model = VGG16(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3)
        )
        
        # Freeze the convolutional layers
        for layer in self.base_model.layers:
            layer.trainable = False
            
        # Build the complete model
        x = Flatten()(self.base_model.output)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(256, activation='relu')(x)
        output = Dense(4, activation='softmax')(x)  # 4 classes: head, eyes, mouth, drowsiness
        
        self.model = tf.keras.Model(self.base_model.input, output)
    
    def get_classification_layer_parameters(self):
        """Extract classification layer parameters for federated learning"""
        return [layer.get_weights() for layer in self.model.layers[-3:]]
    
    def set_classification_layer_parameters(self, parameters):
        """Update classification layer parameters from federated learning"""
        for layer, weights in zip(self.model.layers[-3:], parameters):
            layer.set_weights(weights)
