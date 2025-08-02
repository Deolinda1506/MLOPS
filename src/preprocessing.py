import os
import numpy as np
import pandas as pd
from PIL import Image
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import albumentations as A
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

class DataPreprocessor:
    def __init__(self, data_path="data", img_size=(224, 224)):
        """
        Initialize the data preprocessor
        
        Args:
            data_path (str): Path to the data directory
            img_size (tuple): Target image size (width, height)
        """
        self.data_path = data_path
        self.img_size = img_size
        self.label_encoder = LabelEncoder()
        self.class_names = []
        self.class_distribution = {}
        
    def load_data(self):
        """
        Load and organize data from the directory structure
        
        Returns:
            tuple: (images, labels, class_names)
        """
        images = []
        labels = []
        class_names = []
        
        # Define class mapping for different directory structures
        class_mapping = {
            'adenocarcinoma': 'adenocarcinoma',
            'adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib': 'adenocarcinoma',
            'large.cell.carcinoma': 'large.cell.carcinoma',
            'large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa': 'large.cell.carcinoma',
            'squamous.cell.carcinoma': 'squamous.cell.carcinoma',
            'squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa': 'squamous.cell.carcinoma',
            'normal': 'normal'
        }
        
        # Walk through the data directory
        for root, dirs, files in os.walk(self.data_path):
            if files and any(f.lower().endswith(('.png', '.jpg', '.jpeg')) for f in files):
                # Extract class name from directory path
                original_class_name = os.path.basename(root)
                
                # Map to standardized class name
                if original_class_name in class_mapping:
                    class_name = class_mapping[original_class_name]
                else:
                    # Try to extract class name from path
                    path_parts = root.split(os.sep)
                    for part in path_parts:
                        if part in class_mapping:
                            class_name = class_mapping[part]
                            break
                    else:
                        # Use original name if no mapping found
                        class_name = original_class_name
                
                if class_name not in class_names:
                    class_names.append(class_name)
                
                print(f"Loading {class_name} images from {original_class_name}...")
                
                # Load images for this class
                for file in tqdm(files, desc=f"Loading {class_name}"):
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        file_path = os.path.join(root, file)
                        try:
                            # Load and preprocess image
                            img = self.load_and_preprocess_image(file_path)
                            if img is not None:
                                images.append(img)
                                labels.append(class_name)
                        except Exception as e:
                            print(f"Error loading {file_path}: {e}")
        
        self.class_names = sorted(class_names)  # Sort for consistency
        self.class_distribution = pd.Series(labels).value_counts().to_dict()
        
        print(f"\nDataset Summary:")
        print(f"Total images loaded: {len(images)}")
        print(f"Classes found: {self.class_names}")
        print(f"Class distribution: {self.class_distribution}")
        
        return np.array(images), np.array(labels), self.class_names
    
    def load_data_from_split(self, split='train'):
        """
        Load data from a specific split (train/test/valid)
        
        Args:
            split (str): Data split to load ('train', 'test', 'valid')
            
        Returns:
            tuple: (images, labels, class_names)
        """
        split_path = os.path.join(self.data_path, split)
        if not os.path.exists(split_path):
            print(f"Warning: {split_path} does not exist. Loading from main data directory.")
            return self.load_data()
        
        # Temporarily change data path to split directory
        original_path = self.data_path
        self.data_path = split_path
        
        # Load data from split
        images, labels, class_names = self.load_data()
        
        # Restore original path
        self.data_path = original_path
        
        return images, labels, class_names
    
    def load_and_preprocess_image(self, image_path):
        """
        Load and preprocess a single image
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            numpy.ndarray: Preprocessed image
        """
        try:
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                return None
            
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize image
            img = cv2.resize(img, self.img_size)
            
            # Normalize pixel values
            img = img.astype(np.float32) / 255.0
            
            return img
        except Exception as e:
            print(f"Error preprocessing {image_path}: {e}")
            return None
    
    def load_and_preprocess_image_from_array(self, image_array):
        """
        Preprocess image from numpy array
        
        Args:
            image_array (numpy.ndarray): Input image array
            
        Returns:
            numpy.ndarray: Preprocessed image
        """
        try:
            # Convert to RGB if needed
            if len(image_array.shape) == 3 and image_array.shape[2] == 3:
                img = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
            else:
                img = image_array.copy()
            
            # Resize image
            img = cv2.resize(img, self.img_size)
            
            # Normalize pixel values
            img = img.astype(np.float32) / 255.0
            
            return img
        except Exception as e:
            print(f"Error preprocessing image array: {e}")
            return None
    
    def create_data_generators(self, X_train, y_train, X_val, y_val, batch_size=32):
        """
        Create data generators for training and validation
        
        Args:
            X_train, y_train: Training data and labels
            X_val, y_val: Validation data and labels
            batch_size (int): Batch size for training
            
        Returns:
            tuple: (train_generator, val_generator)
        """
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        # No augmentation for validation
        val_datagen = ImageDataGenerator()
        
        # Convert labels to categorical
        y_train_cat = tf.keras.utils.to_categorical(
            self.label_encoder.fit_transform(y_train), 
            num_classes=len(self.class_names)
        )
        y_val_cat = tf.keras.utils.to_categorical(
            self.label_encoder.transform(y_val), 
            num_classes=len(self.class_names)
        )
        
        # Create generators
        train_generator = train_datagen.flow(
            X_train, y_train_cat,
            batch_size=batch_size,
            shuffle=True
        )
        
        val_generator = val_datagen.flow(
            X_val, y_val_cat,
            batch_size=batch_size,
            shuffle=False
        )
        
        return train_generator, val_generator
    
    def split_data(self, images, labels, test_size=0.2, val_size=0.2, random_state=42):
        """
        Split data into train, validation, and test sets
        
        Args:
            images, labels: Input data and labels
            test_size (float): Proportion of data for test set
            val_size (float): Proportion of remaining data for validation
            random_state (int): Random seed for reproducibility
            
        Returns:
            tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            images, labels, test_size=test_size, 
            random_state=random_state, stratify=labels
        )
        
        # Second split: separate validation set from remaining data
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size,
            random_state=random_state, stratify=y_temp
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def visualize_data_distribution(self, save_path=None):
        """
        Visualize the distribution of classes in the dataset
        
        Args:
            save_path (str): Path to save the plot
        """
        plt.figure(figsize=(12, 6))
        
        # Class distribution bar plot
        plt.subplot(1, 2, 1)
        classes = list(self.class_distribution.keys())
        counts = list(self.class_distribution.values())
        
        bars = plt.bar(classes, counts, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        plt.title('Class Distribution in Dataset')
        plt.xlabel('Classes')
        plt.ylabel('Number of Images')
        plt.xticks(rotation=45)
        
        # Add count labels on bars
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                    str(count), ha='center', va='bottom')
        
        # Pie chart
        plt.subplot(1, 2, 2)
        plt.pie(counts, labels=classes, autopct='%1.1f%%', startangle=90)
        plt.title('Class Distribution (Percentage)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def visualize_sample_images(self, images, labels, num_samples=8, save_path=None):
        """
        Visualize sample images from each class
        
        Args:
            images: Array of images
            labels: Array of labels
            num_samples (int): Number of samples per class to display
            save_path (str): Path to save the plot
        """
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.ravel()
        
        for i, class_name in enumerate(self.class_names):
            # Get indices for this class
            class_indices = np.where(labels == class_name)[0]
            
            # Select random samples
            if len(class_indices) > 0:
                sample_indices = np.random.choice(class_indices, 
                                                min(num_samples//len(self.class_names), len(class_indices)), 
                                                replace=False)
                
                for j, idx in enumerate(sample_indices):
                    if j < 2:  # Show 2 samples per class
                        axes[i*2 + j].imshow(images[idx])
                        axes[i*2 + j].set_title(f'{class_name}')
                        axes[i*2 + j].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def get_data_statistics(self):
        """
        Get statistics about the dataset
        
        Returns:
            dict: Dictionary containing dataset statistics
        """
        stats = {
            'total_images': sum(self.class_distribution.values()),
            'num_classes': len(self.class_names),
            'class_names': self.class_names,
            'class_distribution': self.class_distribution,
            'image_size': self.img_size
        }
        
        return stats 