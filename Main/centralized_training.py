import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from models.mobilenet_capsnet import MobileNetCapsNet
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf

def train_centralized_model():
    # Set random seeds for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)

    # Data directories
    base_dir = os.path.join(os.path.dirname(__file__), 'data')
    train_dir = os.path.join(base_dir, 'train')
    val_dir = os.path.join(base_dir, 'validation')

    # Data generators with the same augmentation as federated learning
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    val_datagen = ImageDataGenerator(rescale=1./255)

    # Load data using generators
    batch_size = 32
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical'
    )

    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical'
    )

    # Create model
    model = MobileNetCapsNet(input_shape=(224, 224, 3)).model

    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Calculate steps per epoch
    steps_per_epoch = train_generator.samples // batch_size
    validation_steps = val_generator.samples // batch_size

    # Train model
    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=50,  # Increased epochs for better convergence
        validation_data=val_generator,
        validation_steps=validation_steps,
        verbose=1
    )

    # Evaluate model
    test_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    # Get predictions
    predictions = model.predict(test_generator)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_generator.classes  # Classes are already in correct format for binary classification

    # Calculate metrics
    conf_matrix = confusion_matrix(y_true, y_pred)
    class_report = classification_report(y_true, y_pred, 
                                      target_names=['Not Drowsy', 'Drowsy'],
                                      digits=4)

    # Save results
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    with open(os.path.join(results_dir, 'centralized_learning_results.txt'), 'w') as f:
        f.write('Centralized Learning Results\n')
        f.write('==========================\n\n')
        f.write(f'Final Training Accuracy: {history.history["accuracy"][-1]:.4f}\n')
        f.write(f'Final Training Loss: {history.history["loss"][-1]:.4f}\n')
        f.write(f'Final Validation Accuracy: {history.history["val_accuracy"][-1]:.4f}\n')
        f.write(f'Final Validation Loss: {history.history["val_loss"][-1]:.4f}\n\n')
        f.write('Confusion Matrix:\n')
        f.write(str(conf_matrix))
        f.write('\n\nClassification Report:\n')
        f.write(class_report)

    # Save model
    model.save(os.path.join(results_dir, 'centralized_model.h5'))
    print("Training completed. Results saved to results/centralized_learning_results.txt")

if __name__ == '__main__':
    train_centralized_model()
