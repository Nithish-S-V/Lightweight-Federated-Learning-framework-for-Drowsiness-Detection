import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class DataProcessor:
    def __init__(self, data_dir):
        """Initialize DataProcessor with data directory path"""
        self.data_dir = data_dir
        self.train_dir = os.path.join(data_dir, 'train')
        self.valid_dir = os.path.join(data_dir, 'validation')
        self.test_dir = os.path.join(data_dir, 'test')
        
    def create_generator(self, subset='train', batch_size=32, target_size=(224, 224)):
        """Create data generator for the specified subset"""
        if subset == 'train':
            datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest'
            )
            directory = self.train_dir
        else:
            datagen = ImageDataGenerator(rescale=1./255)
            directory = self.valid_dir if subset == 'validation' else self.test_dir
            
        return datagen.flow_from_directory(
            directory,
            target_size=target_size,
            batch_size=batch_size,
            class_mode='binary',
            shuffle=True if subset == 'train' else False
        )
        
    def get_class_weights(self):
        """Calculate class weights to handle imbalanced dataset"""
        train_drowsy_samples = len(os.listdir(os.path.join(self.train_dir, 'drowsy')))
        train_non_drowsy_samples = len(os.listdir(os.path.join(self.train_dir, 'non_drowsy')))
        
        total_samples = train_drowsy_samples + train_non_drowsy_samples
        weight_for_0 = (1 / train_non_drowsy_samples) * (total_samples / 2.0)
        weight_for_1 = (1 / train_drowsy_samples) * (total_samples / 2.0)
        
        return {0: weight_for_0, 1: weight_for_1}
        
    @staticmethod
    def organize_data(source_dir, dest_dir):
        """Organize data into the required directory structure"""
        import shutil
        
        # Create destination directories
        for split in ['train', 'validation', 'test']:
            for class_name in ['drowsy', 'non_drowsy']:
                os.makedirs(os.path.join(dest_dir, split, class_name), exist_ok=True)
                
        # Function to distribute files
        def distribute_files(src_path, class_name):
            files = os.listdir(src_path)
            np.random.shuffle(files)
            
            n_files = len(files)
            n_train = int(0.7 * n_files)
            n_val = int(0.15 * n_files)
            
            # Distribute files
            for i, file in enumerate(files):
                if i < n_train:
                    split = 'train'
                elif i < n_train + n_val:
                    split = 'validation'
                else:
                    split = 'test'
                    
                src_file = os.path.join(src_path, file)
                dst_file = os.path.join(dest_dir, split, class_name, file)
                shutil.copy2(src_file, dst_file)
                
        # Process drowsy and non-drowsy directories
        if os.path.exists(os.path.join(source_dir, 'drowsy')):
            distribute_files(os.path.join(source_dir, 'drowsy'), 'drowsy')
        if os.path.exists(os.path.join(source_dir, 'non_drowsy')):
            distribute_files(os.path.join(source_dir, 'non_drowsy'), 'non_drowsy')
