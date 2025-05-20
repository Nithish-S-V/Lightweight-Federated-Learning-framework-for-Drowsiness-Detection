import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, initializers, backend as K
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import shutil
from tensorflow.keras.applications import VGG16, ResNet50, DenseNet121
from tensorflow.keras import layers, models

# Configuration
INPUT_SHAPE = (224, 224, 3)
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001
DATA_DIR = Path("D:/Major Project/Rasp/Data/Initial_train")
TEST_SIZE = 0.2

# Initialize model
base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(1, activation='sigmoid')
])

def prepare_datasets():
    train_dir = DATA_DIR.parent / "train"
    test_dir = DATA_DIR.parent / "test"
    
    for cls in ["drowsy", "notdrowsy"]:
        (train_dir/cls).mkdir(parents=True, exist_ok=True)
        (test_dir/cls).mkdir(parents=True, exist_ok=True)
        
        files = list((DATA_DIR/cls).glob("*"))
        if not files:
            raise ValueError(f"No files found in {DATA_DIR/cls}")
        
        train_files, test_files = train_test_split(files, test_size=TEST_SIZE, random_state=42)
        
        for f in train_files:
            shutil.copy(f, train_dir/cls/f.name)
        for f in test_files:
            shutil.copy(f, test_dir/cls/f.name)
        
    return train_dir, test_dir

def train_initial_model():
    train_dir, test_dir = prepare_datasets()
    
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )
    
    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=INPUT_SHAPE[:2],
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='training'
    )
    
    val_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=INPUT_SHAPE[:2],
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='validation'
    )
    
    model.compile(loss='binary_crossentropy',
                 optimizer='adam',
                 metrics=['accuracy'])
    
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        verbose=1
    )
    
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    test_gen = test_datagen.flow_from_directory(
        test_dir,
        target_size=INPUT_SHAPE[:2],
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )
    
    y_pred = np.argmax(model.predict(test_gen), axis=1)
    y_true = test_gen.classes
    
    print("\nTest Metrics:")
    print(f"Accuracy: {np.mean(y_true == y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Not Drowsy', 'Drowsy']))
    
    model.save(DATA_DIR.parent / "drowsiness_model_densenet.keras")
    return model

if __name__ == "__main__":
    train_initial_model()
