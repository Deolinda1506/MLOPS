import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import ResNet50V2, EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime

class OverfittingMonitor(tf.keras.callbacks.Callback):
    """Custom callback to monitor and prevent overfitting"""
    
    def __init__(self, patience=5, threshold=0.1):
        super().__init__()
        self.patience = patience
        self.threshold = threshold
        self.overfitting_count = 0
        self.best_val_loss = float('inf')
        
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
            
        train_loss = logs.get('loss', 0)
        val_loss = logs.get('val_loss', 0)
        train_acc = logs.get('accuracy', 0)
        val_acc = logs.get('val_accuracy', 0)
        
        # Check for overfitting (validation loss increases while training loss decreases)
        if val_loss > self.best_val_loss and train_loss < val_loss:
            self.overfitting_count += 1
            print(f"⚠️  Potential overfitting detected! Count: {self.overfitting_count}")
            
            if self.overfitting_count >= self.patience:
                print("🛑 Overfitting threshold reached. Consider stopping training.")
                self.model.stop_training = True
        else:
            self.overfitting_count = 0
            
        # Update best validation loss
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            
        # Check for underfitting (both accuracies are low)
        if train_acc < 0.6 and val_acc < 0.6:
            print(f"⚠️  Potential underfitting detected! Train acc: {train_acc:.3f}, Val acc: {val_acc:.3f}")
            
        # Print training vs validation gap
        acc_gap = abs(train_acc - val_acc)
        if acc_gap > self.threshold:
            print(f"⚠️  Large accuracy gap detected: {acc_gap:.3f} (Train: {train_acc:.3f}, Val: {val_acc:.3f})")

class LungCancerClassifier:
    def __init__(self, num_classes=4, img_size=(224, 224), model_type='efficientnet'):
        """
        Initialize the lung cancer classifier
        
        Args:
            num_classes (int): Number of classes to predict
            img_size (tuple): Input image size
            model_type (str): Type of model architecture ('efficientnet' or 'resnet')
        """
        self.num_classes = num_classes
        self.img_size = img_size
        self.model_type = model_type
        self.model = None
        self.history = None
        self.class_names = []
        
    def build_model(self, learning_rate=0.001, dropout_rate=0.5, l2_reg=0.01):
        """
        Build the CNN model with transfer learning and regularization
        
        Args:
            learning_rate (float): Learning rate for optimizer
            dropout_rate (float): Dropout rate for regularization
            l2_reg (float): L2 regularization strength
            
        Returns:
            tensorflow.keras.Model: Compiled model
        """
        # Base model (transfer learning)
        if self.model_type == 'efficientnet':
            base_model = EfficientNetB0(
                weights='imagenet',
                include_top=False,
                input_shape=(*self.img_size, 3)
            )
        else:
            base_model = ResNet50V2(
                weights='imagenet',
                include_top=False,
                input_shape=(*self.img_size, 3)
            )
        
        # Freeze base model layers to prevent overfitting
        base_model.trainable = False
        
        # Build the complete model with regularization
        model = models.Sequential([
            # Data augmentation layer (prevents overfitting)
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
            layers.RandomBrightness(0.2),
            layers.RandomContrast(0.2),
            
            # Base model
            base_model,
            
            # Global average pooling
            layers.GlobalAveragePooling2D(),
            
            # Dense layers with regularization
            layers.Dense(512, activation='relu', 
                        kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
            layers.BatchNormalization(),
            layers.Dropout(dropout_rate),
            
            layers.Dense(256, activation='relu',
                        kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
            layers.BatchNormalization(),
            layers.Dropout(dropout_rate),
            
            # Output layer
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        # Compile model
        model.compile(
            optimizer=optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def train_model(self, train_generator, val_generator, epochs=50, 
                   early_stopping_patience=10, reduce_lr_patience=5):
        """
        Train the model with comprehensive overfitting/underfitting prevention
        
        Args:
            train_generator: Training data generator
            val_generator: Validation data generator
            epochs (int): Number of training epochs
            early_stopping_patience (int): Patience for early stopping
            reduce_lr_patience (int): Patience for learning rate reduction
            
        Returns:
            tensorflow.keras.callbacks.History: Training history
        """
        # Enhanced callbacks for overfitting/underfitting prevention
        callbacks_list = [
            # Early stopping (prevents overfitting)
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=early_stopping_patience,
                restore_best_weights=True,
                verbose=1,
                min_delta=0.001  # Minimum improvement required
            ),
            
            # Learning rate reduction (prevents overfitting)
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=reduce_lr_patience,
                min_lr=1e-7,
                verbose=1,
                cooldown=2  # Wait before reducing LR again
            ),
            
            # Model checkpoint (saves best model)
            callbacks.ModelCheckpoint(
                'models/best_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            
            # TensorBoard logging
            callbacks.TensorBoard(
                log_dir=f'logs/{datetime.now().strftime("%Y%m%d-%H%M%S")}',
                histogram_freq=1
            ),
            
            # Custom callback to monitor overfitting
            self.OverfittingMonitor()
        ]
        
        # Train the model
        self.history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=val_generator,
            callbacks=callbacks_list,
            verbose=1
        )
        
        return self.history
    
    def fine_tune_model(self, train_generator, val_generator, epochs=20, 
                       learning_rate=0.0001):
        """
        Fine-tune the model by unfreezing some layers
        
        Args:
            train_generator: Training data generator
            val_generator: Validation data generator
            epochs (int): Number of fine-tuning epochs
            learning_rate (float): Learning rate for fine-tuning
        """
        # Unfreeze the base model
        base_model = self.model.layers[4]  # Base model is at index 4
        base_model.trainable = True
        
        # Freeze early layers, unfreeze later layers
        for layer in base_model.layers[:-30]:
            layer.trainable = False
        
        # Recompile with lower learning rate
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        # Fine-tune
        fine_tune_history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=val_generator,
            callbacks=[
                callbacks.EarlyStopping(patience=5, restore_best_weights=True),
                callbacks.ReduceLROnPlateau(patience=3, factor=0.5)
            ],
            verbose=1
        )
        
        return fine_tune_history
    
    def evaluate_model(self, test_generator, class_names):
        """
        Evaluate the model on test data
        
        Args:
            test_generator: Test data generator
            class_names (list): List of class names
            
        Returns:
            dict: Evaluation metrics
        """
        # Get predictions
        predictions = self.model.predict(test_generator)
        y_pred = np.argmax(predictions, axis=1)
        y_true = test_generator.classes
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        
        # Classification report
        report = classification_report(
            y_true, y_pred, 
            target_names=class_names,
            output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Store results
        results = {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'predictions': predictions.tolist(),
            'y_true': y_true.tolist(),
            'y_pred': y_pred.tolist()
        }
        
        return results
    
    def plot_training_history(self, save_path=None):
        """
        Plot training history with overfitting/underfitting analysis
        
        Args:
            save_path (str): Path to save the plot
        """
        if self.history is None:
            print("No training history available")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Training Accuracy')
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Loss
        axes[0, 1].plot(self.history.history['loss'], label='Training Loss')
        axes[0, 1].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Accuracy Gap (Overfitting Indicator)
        train_acc = self.history.history['accuracy']
        val_acc = self.history.history['val_accuracy']
        acc_gap = [abs(t - v) for t, v in zip(train_acc, val_acc)]
        axes[0, 2].plot(acc_gap, label='Accuracy Gap', color='red')
        axes[0, 2].axhline(y=0.1, color='orange', linestyle='--', label='Overfitting Threshold')
        axes[0, 2].set_title('Training vs Validation Accuracy Gap')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Accuracy Gap')
        axes[0, 2].legend()
        axes[0, 2].grid(True)
        
        # Loss Gap
        train_loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']
        loss_gap = [abs(t - v) for t, v in zip(train_loss, val_loss)]
        axes[1, 0].plot(loss_gap, label='Loss Gap', color='purple')
        axes[1, 0].set_title('Training vs Validation Loss Gap')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss Gap')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Learning Rate (if available)
        if 'lr' in self.history.history:
            axes[1, 1].plot(self.history.history['lr'], label='Learning Rate')
            axes[1, 1].set_title('Learning Rate Schedule')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_yscale('log')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        else:
            axes[1, 1].text(0.5, 0.5, 'Learning Rate\nNot Available', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Learning Rate Schedule')
        
        # Overfitting Analysis Summary
        self.analyze_overfitting_underfitting(axes[1, 2])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_overfitting_underfitting(self, ax):
        """Analyze training history for overfitting/underfitting patterns"""
        if self.history is None:
            return
            
        train_acc = self.history.history['accuracy']
        val_acc = self.history.history['val_accuracy']
        train_loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']
        
        # Calculate metrics
        final_train_acc = train_acc[-1]
        final_val_acc = val_acc[-1]
        final_train_loss = train_loss[-1]
        final_val_loss = val_loss[-1]
        
        acc_gap = abs(final_train_acc - final_val_acc)
        loss_gap = abs(final_train_loss - final_val_loss)
        
        # Determine status
        if acc_gap > 0.1 and final_train_acc > final_val_acc:
            status = "OVERFITTING"
            color = "red"
        elif final_train_acc < 0.6 and final_val_acc < 0.6:
            status = "UNDERFITTING"
            color = "orange"
        elif acc_gap < 0.05 and final_val_acc > 0.8:
            status = "GOOD FIT"
            color = "green"
        else:
            status = "ACCEPTABLE"
            color = "blue"
        
        # Create summary text
        summary_text = f"""
        TRAINING ANALYSIS
        
        Status: {status}
        
        Final Metrics:
        Train Accuracy: {final_train_acc:.3f}
        Val Accuracy: {final_val_acc:.3f}
        Accuracy Gap: {acc_gap:.3f}
        
        Train Loss: {final_train_loss:.3f}
        Val Loss: {final_val_loss:.3f}
        Loss Gap: {loss_gap:.3f}
        
        Recommendations:
        """
        
        if status == "OVERFITTING":
            summary_text += """
        • Increase dropout rate
        • Add more regularization
        • Reduce model complexity
        • Get more training data
        • Use data augmentation
        """
        elif status == "UNDERFITTING":
            summary_text += """
        • Increase model complexity
        • Train for more epochs
        • Reduce regularization
        • Increase learning rate
        • Check data quality
        """
        else:
            summary_text += """
        • Model is well-balanced
        • Consider fine-tuning
        • Monitor performance
        """
        
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, 
               fontsize=10, verticalalignment='top', 
               bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        ax.set_title('Training Analysis Summary')
        ax.axis('off')
    
    def plot_confusion_matrix(self, confusion_matrix, class_names, save_path=None):
        """
        Plot confusion matrix
        
        Args:
            confusion_matrix: Confusion matrix array
            class_names (list): List of class names
            save_path (str): Path to save the plot
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            confusion_matrix, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names
        )
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_model(self, model_path='models/lung_cancer_model.h5'):
        """
        Save the trained model
        
        Args:
            model_path (str): Path to save the model
        """
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self.model.save(model_path)
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path='models/lung_cancer_model.h5'):
        """
        Load a trained model
        
        Args:
            model_path (str): Path to the saved model
        """
        if os.path.exists(model_path):
            self.model = tf.keras.models.load_model(model_path)
            print(f"Model loaded from {model_path}")
        else:
            print(f"Model file not found at {model_path}")
    
    def save_training_results(self, results, save_path='models/training_results.json'):
        """
        Save training results to JSON file
        
        Args:
            results (dict): Training results
            save_path (str): Path to save results
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"Training results saved to {save_path}")
    
    def get_model_summary(self):
        """
        Get model summary
        
        Returns:
            str: Model summary
        """
        if self.model is None:
            return "No model available"
        
        # Capture model summary
        summary_list = []
        self.model.summary(print_fn=lambda x: summary_list.append(x))
        return '\n'.join(summary_list) 