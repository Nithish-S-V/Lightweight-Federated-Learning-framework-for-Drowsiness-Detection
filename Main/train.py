import os
import argparse
import tensorflow as tf
from models.mobilenet_capsnet import MobileNetCapsNet
from utils.data_processor import DataProcessor

class TrainingProgressCallback(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        print(f"\nEpoch {epoch + 1} starting...")
        
    def on_batch_end(self, batch, logs=None):
        if batch % 10 == 0:  # Show progress every 10 batches
            print(f"Batch {batch}: loss = {logs['loss']:.4f}, accuracy = {logs['accuracy']:.4f}")

def train_model(data_dir, batch_size=32, epochs=50, learning_rate=0.001):
    """Train the model on the dataset"""
    print("\n=== Initializing Training ===")
    print(f"Parameters: batch_size={batch_size}, epochs={epochs}, learning_rate={learning_rate}")
    
    # Initialize data processor
    print("\nLoading dataset...")
    data_processor = DataProcessor(data_dir)
    
    # Create data generators
    train_generator = data_processor.create_generator(
        subset='train',
        batch_size=batch_size,
        target_size=(224, 224)
    )
    
    valid_generator = data_processor.create_generator(
        subset='validation',
        batch_size=batch_size,
        target_size=(224, 224)
    )
    
    print(f"\nTotal training samples: {train_generator.samples}")
    print(f"Total validation samples: {valid_generator.samples}")
    
    # Initialize model
    print("\nInitializing MobileNet+CapsNet model...")
    model = MobileNetCapsNet(input_shape=(224, 224, 3))
    model.compile_model(learning_rate=learning_rate)
    
    # Setup callbacks
    print("\nSetting up callbacks...")
    checkpoint_path = os.path.join('checkpoints', 'model.h5')
    os.makedirs('checkpoints', exist_ok=True)
    
    callbacks = [
        TrainingProgressCallback(),
        # Model checkpoint
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        
        # Early stopping
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Reduce learning rate on plateau
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        
        # TensorBoard logging
        tf.keras.callbacks.TensorBoard(
            log_dir='logs',
            histogram_freq=1
        )
    ]
    
    print("\n=== Starting Training ===")
    # Train the model
    history = model.model.fit(
        train_generator,
        validation_data=valid_generator,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    print("\n=== Training Completed ===")
    # Print final metrics
    print("\nFinal Training Metrics:")
    for metric in history.history:
        final_value = history.history[metric][-1]
        print(f"{metric}: {final_value:.4f}")
    
    return model, history

def main():
    parser = argparse.ArgumentParser(description='Train drowsiness detection model')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate')
    
    args = parser.parse_args()
    
    try:
        # Train model
        model, history = train_model(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate
        )
        print("\nTraining completed successfully!")
        
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        raise

if __name__ == "__main__":
    main()
