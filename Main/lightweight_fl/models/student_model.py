import tensorflow as tf
from tensorflow.keras import layers, Model

class LightweightCNN(Model):
    """Lightweight student model designed for Raspberry Pi"""
    
    def __init__(self):
        super(LightweightCNN, self).__init__()
        
        # Use depthwise separable convolutions for efficiency
        self.conv1 = layers.SeparableConv2D(32, 3, activation='relu', padding='same')
        self.pool1 = layers.MaxPooling2D((2, 2))
        self.batch1 = layers.BatchNormalization()
        
        self.conv2 = layers.SeparableConv2D(64, 3, activation='relu', padding='same')
        self.pool2 = layers.MaxPooling2D((2, 2))
        self.batch2 = layers.BatchNormalization()
        
        self.conv3 = layers.SeparableConv2D(128, 3, activation='relu', padding='same')
        self.pool3 = layers.MaxPooling2D((2, 2))
        self.batch3 = layers.BatchNormalization()
        
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(256, activation='relu')
        self.dropout = layers.Dropout(0.5)
        self.dense2 = layers.Dense(1, activation='sigmoid')
        
    def call(self, x, training=False):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.batch1(x, training=training)
        
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.batch2(x, training=training)
        
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.batch3(x, training=training)
        
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        return self.dense2(x)
        
    def get_logits(self, x):
        """Get logits before final activation for knowledge distillation"""
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.batch1(x)
        
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.batch2(x)
        
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.batch3(x)
        
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout(x)
        # Return logits before sigmoid
        return self.dense2.weights[0] * x + self.dense2.weights[1]
