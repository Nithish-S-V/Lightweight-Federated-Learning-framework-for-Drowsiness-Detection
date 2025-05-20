import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2

class CapsNetVisualizer:
    def __init__(self, model):
        self.model = model
        
    def visualize_capsule_activations(self, image, save_path=None):
        """Visualize capsule layer activations"""
        # Get the capsule layer outputs
        capsule_model = tf.keras.Model(
            inputs=self.model.model.input,
            outputs=self.model.model.get_layer('drowsiness_caps').output
        )
        
        # Get activations
        activations = capsule_model.predict(np.expand_dims(image, 0))
        activations = np.squeeze(activations)
        
        # Plot activations
        plt.figure(figsize=(15, 5))
        for i in range(4):  # 4 classes
            plt.subplot(1, 4, i + 1)
            plt.bar(range(16), activations[i])  # 16 dimensions per capsule
            plt.title(f'Class {i} Capsule')
            plt.xlabel('Dimension')
            plt.ylabel('Activation')
            
        if save_path:
            plt.savefig(save_path)
        plt.close()
        
    def visualize_attention(self, image, save_path=None):
        """Visualize attention maps from the model"""
        # Get feature maps before capsule layers
        feature_model = tf.keras.Model(
            inputs=self.model.model.input,
            outputs=self.model.model.get_layer('conv2d').output
        )
        
        # Get feature maps
        features = feature_model.predict(np.expand_dims(image, 0))
        features = np.mean(features, axis=-1)  # Average over channels
        
        # Create heatmap
        heatmap = np.squeeze(features)
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)
        
        # Resize heatmap to original image size
        heatmap = cv2.resize(heatmap, (224, 224))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Superimpose heatmap on original image
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        superimposed = cv2.addWeighted(image_rgb, 0.6, heatmap, 0.4, 0)
        
        if save_path:
            cv2.imwrite(save_path, cv2.cvtColor(superimposed, cv2.COLOR_RGB2BGR))
            
        return superimposed
        
    def plot_training_history(self, history, save_path=None):
        """Plot training history"""
        plt.figure(figsize=(12, 4))
        
        # Plot accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training')
        plt.plot(history.history['val_accuracy'], label='Validation')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training')
        plt.plot(history.history['val_loss'], label='Validation')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        if save_path:
            plt.savefig(save_path)
        plt.close()
