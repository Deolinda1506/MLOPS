import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
import io
import base64
import json
from datetime import datetime
import os

class PredictionService:
    def __init__(self, model_path='models/lung_cancer_model.h5', class_names=None):
        """
        Initialize the prediction service
        
        Args:
            model_path (str): Path to the trained model
            class_names (list): List of class names
        """
        self.model_path = model_path
        self.class_names = class_names or ['adenocarcinoma', 'large.cell.carcinoma', 'normal', 'squamous.cell.carcinoma']
        self.model = None
        self.img_size = (224, 224)
        self.load_model()
    
    def load_model(self):
        """
        Load the trained model
        """
        try:
            if os.path.exists(self.model_path):
                self.model = tf.keras.models.load_model(self.model_path)
                print(f"Model loaded successfully from {self.model_path}")
            else:
                print(f"Model file not found at {self.model_path}")
                self.model = None
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
    
    def preprocess_image(self, image):
        """
        Preprocess image for prediction
        
        Args:
            image: Input image (numpy array, PIL Image, or file path)
            
        Returns:
            numpy.ndarray: Preprocessed image
        """
        try:
            # Handle different input types
            if isinstance(image, str):
                # File path
                img = cv2.imread(image)
                if img is None:
                    raise ValueError(f"Could not load image from {image}")
            elif isinstance(image, np.ndarray):
                # Numpy array
                img = image.copy()
            elif isinstance(image, Image.Image):
                # PIL Image
                img = np.array(image)
            else:
                raise ValueError("Unsupported image format")
            
            # Convert BGR to RGB if needed
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize image
            img = cv2.resize(img, self.img_size)
            
            # Normalize pixel values
            img = img.astype(np.float32) / 255.0
            
            # Add batch dimension
            img = np.expand_dims(img, axis=0)
            
            return img
            
        except Exception as e:
            print(f"Error preprocessing image: {e}")
            return None
    
    def predict_single(self, image):
        """
        Predict class for a single image
        
        Args:
            image: Input image (numpy array, PIL Image, or file path)
            
        Returns:
            dict: Prediction results
        """
        if self.model is None:
            return {
                'error': 'Model not loaded',
                'success': False
            }
        
        try:
            # Preprocess image
            processed_img = self.preprocess_image(image)
            if processed_img is None:
                return {
                    'error': 'Failed to preprocess image',
                    'success': False
                }
            
            # Make prediction
            predictions = self.model.predict(processed_img, verbose=0)
            
            # Get predicted class
            predicted_class_idx = np.argmax(predictions[0])
            predicted_class = self.class_names[predicted_class_idx]
            confidence = float(predictions[0][predicted_class_idx])
            
            # Get all class probabilities
            class_probabilities = {}
            for i, class_name in enumerate(self.class_names):
                class_probabilities[class_name] = float(predictions[0][i])
            
            # Prepare result
            result = {
                'success': True,
                'predicted_class': predicted_class,
                'confidence': confidence,
                'class_probabilities': class_probabilities,
                'timestamp': datetime.now().isoformat(),
                'model_path': self.model_path
            }
            
            return result
            
        except Exception as e:
            return {
                'error': f'Prediction failed: {str(e)}',
                'success': False
            }
    
    def predict_batch(self, images):
        """
        Predict classes for multiple images
        
        Args:
            images (list): List of input images
            
        Returns:
            list: List of prediction results
        """
        if self.model is None:
            return [{'error': 'Model not loaded', 'success': False}] * len(images)
        
        results = []
        for image in images:
            result = self.predict_single(image)
            results.append(result)
        
        return results
    
    def predict_from_file(self, file_path):
        """
        Predict from image file
        
        Args:
            file_path (str): Path to image file
            
        Returns:
            dict: Prediction results
        """
        return self.predict_single(file_path)
    
    def predict_from_base64(self, base64_string):
        """
        Predict from base64 encoded image
        
        Args:
            base64_string (str): Base64 encoded image string
            
        Returns:
            dict: Prediction results
        """
        try:
            # Decode base64 string
            image_data = base64.b64decode(base64_string)
            
            # Convert to PIL Image
            image = Image.open(io.BytesIO(image_data))
            
            return self.predict_single(image)
            
        except Exception as e:
            return {
                'error': f'Failed to decode base64 image: {str(e)}',
                'success': False
            }
    
    def get_prediction_confidence_analysis(self, image):
        """
        Get detailed confidence analysis for prediction
        
        Args:
            image: Input image
            
        Returns:
            dict: Detailed confidence analysis
        """
        result = self.predict_single(image)
        
        if not result['success']:
            return result
        
        # Add confidence analysis
        probabilities = result['class_probabilities']
        sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        
        analysis = {
            'top_prediction': sorted_probs[0],
            'second_prediction': sorted_probs[1],
            'confidence_gap': sorted_probs[0][1] - sorted_probs[1][1],
            'prediction_ranking': sorted_probs,
            'uncertainty': 1 - sorted_probs[0][1]  # Higher uncertainty if top prediction is low
        }
        
        result['confidence_analysis'] = analysis
        return result
    
    def validate_prediction(self, image, expected_class=None):
        """
        Validate prediction against expected class (if provided)
        
        Args:
            image: Input image
            expected_class (str): Expected class label (optional)
            
        Returns:
            dict: Validation results
        """
        result = self.predict_single(image)
        
        if not result['success']:
            return result
        
        if expected_class is not None:
            is_correct = result['predicted_class'] == expected_class
            result['validation'] = {
                'expected_class': expected_class,
                'is_correct': is_correct,
                'accuracy': 1.0 if is_correct else 0.0
            }
        
        return result
    
    def get_model_info(self):
        """
        Get information about the loaded model
        
        Returns:
            dict: Model information
        """
        if self.model is None:
            return {
                'model_loaded': False,
                'error': 'Model not loaded'
            }
        
        return {
            'model_loaded': True,
            'model_path': self.model_path,
            'input_shape': self.model.input_shape,
            'output_shape': self.model.output_shape,
            'num_classes': len(self.class_names),
            'class_names': self.class_names,
            'total_params': self.model.count_params(),
            'trainable_params': sum([tf.keras.backend.count_params(w) for w in self.model.trainable_weights])
        }
    
    def save_prediction_log(self, prediction_result, log_file='prediction_log.json'):
        """
        Save prediction result to log file
        
        Args:
            prediction_result (dict): Prediction result
            log_file (str): Path to log file
        """
        try:
            # Load existing log
            log_data = []
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    log_data = json.load(f)
            
            # Add new prediction
            log_data.append(prediction_result)
            
            # Save updated log
            with open(log_file, 'w') as f:
                json.dump(log_data, f, indent=2)
                
        except Exception as e:
            print(f"Error saving prediction log: {e}")
    
    def get_prediction_statistics(self, log_file='prediction_log.json'):
        """
        Get statistics from prediction log
        
        Args:
            log_file (str): Path to log file
            
        Returns:
            dict: Prediction statistics
        """
        try:
            if not os.path.exists(log_file):
                return {'error': 'Log file not found'}
            
            with open(log_file, 'r') as f:
                log_data = json.load(f)
            
            if not log_data:
                return {'error': 'No prediction data available'}
            
            # Calculate statistics
            successful_predictions = [p for p in log_data if p.get('success', False)]
            
            if not successful_predictions:
                return {'error': 'No successful predictions found'}
            
            # Class distribution
            class_counts = {}
            for pred in successful_predictions:
                class_name = pred.get('predicted_class', 'unknown')
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            # Confidence statistics
            confidences = [p.get('confidence', 0) for p in successful_predictions]
            
            stats = {
                'total_predictions': len(log_data),
                'successful_predictions': len(successful_predictions),
                'success_rate': len(successful_predictions) / len(log_data),
                'class_distribution': class_counts,
                'confidence_stats': {
                    'mean': np.mean(confidences),
                    'std': np.std(confidences),
                    'min': np.min(confidences),
                    'max': np.max(confidences)
                }
            }
            
            return stats
            
        except Exception as e:
            return {'error': f'Error calculating statistics: {e}'} 