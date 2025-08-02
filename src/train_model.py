#!/usr/bin/env python3
"""
Lung Cancer Classification - Complete Training Pipeline

This script demonstrates the complete machine learning pipeline for lung cancer classification.
It includes data loading, preprocessing, model training, evaluation, and saving.

Usage:
    python train_model.py
"""

import os
import sys
import json
from datetime import datetime

# Add src to path
sys.path.append('src')

from preprocessing import DataPreprocessor
from model import LungCancerClassifier
from prediction import PredictionService

def main():
    """Main training pipeline"""
    print("=" * 60)
    print("LUNG CANCER CLASSIFICATION - ML PIPELINE")
    print("=" * 60)
    
    # Step 1: Data Loading and Preprocessing
    print("\n1. Loading and preprocessing data...")
    preprocessor = DataPreprocessor(data_path="data", img_size=(224, 224))
    
    try:
        # Load training data
        print("Loading training data...")
        train_images, train_labels, class_names = preprocessor.load_data_from_split('train')
        
        # Load validation data
        print("Loading validation data...")
        val_images, val_labels, _ = preprocessor.load_data_from_split('valid')
        
        # Load test data
        print("Loading test data...")
        test_images, test_labels, _ = preprocessor.load_data_from_split('test')
        
        print(f"✓ Loaded training data: {len(train_images)} images")
        print(f"✓ Loaded validation data: {len(val_images)} images")
        print(f"✓ Loaded test data: {len(test_images)} images")
        print(f"  Classes: {class_names}")
        
        # Combine all data for processing
        images = np.concatenate([train_images, val_images, test_images])
        labels = np.concatenate([train_labels, val_labels, test_labels])
        
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        print("Falling back to single data loading...")
        try:
            images, labels, class_names = preprocessor.load_data()
            print(f"✓ Loaded {len(images)} images with {len(class_names)} classes")
        except Exception as e2:
            print(f"✗ Error in fallback loading: {e2}")
            return
    
    # Step 2: Data Splitting
    print("\n2. Using pre-split data...")
    try:
        # Use the pre-split data
        X_train, y_train = train_images, train_labels
        X_val, y_val = val_images, val_labels
        X_test, y_test = test_images, test_labels
        
        print(f"✓ Training set: {X_train.shape[0]} images")
        print(f"✓ Validation set: {X_val.shape[0]} images")
        print(f"✓ Test set: {X_test.shape[0]} images")
        
        # Verify class consistency
        train_classes = set(y_train)
        val_classes = set(y_val)
        test_classes = set(y_test)
        
        print(f"  Training classes: {sorted(train_classes)}")
        print(f"  Validation classes: {sorted(val_classes)}")
        print(f"  Test classes: {sorted(test_classes)}")
        
    except Exception as e:
        print(f"✗ Error with pre-split data: {e}")
        print("Falling back to manual splitting...")
        try:
            X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(
                images, labels, test_size=0.2, val_size=0.2, random_state=42
            )
            print(f"✓ Training set: {X_train.shape[0]} images")
            print(f"✓ Validation set: {X_val.shape[0]} images")
            print(f"✓ Test set: {X_test.shape[0]} images")
        except Exception as e2:
            print(f"✗ Error in manual splitting: {e2}")
            return
    
    # Step 3: Create Data Generators
    print("\n3. Creating data generators...")
    try:
        train_generator, val_generator = preprocessor.create_data_generators(
            X_train, y_train, X_val, y_val, batch_size=32
        )
        print("✓ Data generators created successfully")
    except Exception as e:
        print(f"✗ Error creating data generators: {e}")
        return
    
    # Step 4: Build Model
    print("\n4. Building model...")
    try:
        classifier = LungCancerClassifier(
            num_classes=len(class_names),
            img_size=(224, 224),
            model_type='efficientnet'
        )
        
        model = classifier.build_model(learning_rate=0.001, dropout_rate=0.5)
        print("✓ Model built successfully")
        print(f"  Architecture: EfficientNet-B0 with transfer learning")
        print(f"  Input size: 224x224x3")
        print(f"  Output classes: {len(class_names)}")
    except Exception as e:
        print(f"✗ Error building model: {e}")
        return
    
    # Step 5: Train Model
    print("\n5. Training model...")
    try:
        print("  Starting training (this may take a while)...")
        history = classifier.train_model(
            train_generator, val_generator, 
            epochs=30, 
            early_stopping_patience=10, 
            reduce_lr_patience=5
        )
        print("✓ Training completed successfully")
    except Exception as e:
        print(f"✗ Error during training: {e}")
        return
    
    # Step 6: Evaluate Model
    print("\n6. Evaluating model...")
    try:
        test_generator, _ = preprocessor.create_data_generators(
            X_test, y_test, X_test, y_test, batch_size=32
        )
        
        results = classifier.evaluate_model(test_generator, class_names)
        
        print("✓ Evaluation completed")
        print(f"  Test Accuracy: {results['accuracy']:.4f}")
        print(f"  Weighted Precision: {results['classification_report']['weighted avg']['precision']:.4f}")
        print(f"  Weighted Recall: {results['classification_report']['weighted avg']['recall']:.4f}")
        print(f"  Weighted F1-Score: {results['classification_report']['weighted avg']['f1-score']:.4f}")
    except Exception as e:
        print(f"✗ Error during evaluation: {e}")
        return
    
    # Step 7: Save Model and Results
    print("\n7. Saving model and results...")
    try:
        # Create models directory
        os.makedirs("models", exist_ok=True)
        
        # Save model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"models/lung_cancer_model_{timestamp}.h5"
        classifier.save_model(model_path)
        
        # Save training results
        results_path = f"models/training_results_{timestamp}.json"
        classifier.save_training_results(results, results_path)
        
        # Save performance summary
        performance_summary = {
            'timestamp': timestamp,
            'model_path': model_path,
            'architecture': 'EfficientNet-B0',
            'input_size': '224x224x3',
            'classes': class_names,
            'training_samples': len(X_train),
            'validation_samples': len(X_val),
            'test_samples': len(X_test),
            'accuracy': results['accuracy'],
            'precision': results['classification_report']['weighted avg']['precision'],
            'recall': results['classification_report']['weighted avg']['recall'],
            'f1_score': results['classification_report']['weighted avg']['f1-score']
        }
        
        summary_path = f"models/performance_summary_{timestamp}.json"
        with open(summary_path, 'w') as f:
            json.dump(performance_summary, f, indent=4)
        
        print("✓ Model and results saved successfully")
        print(f"  Model: {model_path}")
        print(f"  Results: {results_path}")
        print(f"  Summary: {summary_path}")
    except Exception as e:
        print(f"✗ Error saving model: {e}")
        return
    
    # Step 8: Test Prediction Service
    print("\n8. Testing prediction service...")
    try:
        prediction_service = PredictionService(model_path=model_path, class_names=class_names)
        
        # Test with a sample image
        sample_image = X_test[0]
        result = prediction_service.predict_single(sample_image)
        
        print("✓ Prediction service working")
        print(f"  Sample prediction: {result['predicted_class']}")
        print(f"  Confidence: {result['confidence']:.4f}")
        print(f"  Actual class: {y_test[0]}")
        print(f"  Correct: {result['predicted_class'] == y_test[0]}")
    except Exception as e:
        print(f"✗ Error testing prediction service: {e}")
        return
    
    # Final Summary
    print("\n" + "=" * 60)
    print("TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"Model saved to: {model_path}")
    print(f"Test accuracy: {results['accuracy']:.4f}")
    print(f"Classes: {class_names}")
    print("\nNext steps:")
    print("1. Start the API server: python src/api.py")
    print("2. Access the web interface at: http://localhost:8000")
    print("3. Use the model for predictions")
    print("4. Monitor performance and retrain as needed")
    print("=" * 60)

if __name__ == "__main__":
    main() 