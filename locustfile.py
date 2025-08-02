import time
import random
import base64
from locust import HttpUser, task, between
from PIL import Image
import io
import numpy as np

class MLPipelineUser(HttpUser):
    wait_time = between(1, 3)  # Wait 1-3 seconds between requests
    
    def on_start(self):
        """Initialize user session"""
        # Create a dummy image for testing
        self.test_image = self.create_test_image()
    
    def create_test_image(self):
        """Create a test image for predictions"""
        # Create a random image
        img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        
        # Convert to bytes
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        return img_byte_arr.getvalue()
    
    @task(3)
    def health_check(self):
        """Health check endpoint - high frequency"""
        self.client.get("/health")
    
    @task(2)
    def get_metrics(self):
        """Get model metrics"""
        self.client.get("/metrics")
    
    @task(1)
    def predict_image(self):
        """Predict image - main functionality"""
        files = {'file': ('test_image.png', self.test_image, 'image/png')}
        
        with self.client.post("/predict", files=files, catch_response=True) as response:
            if response.status_code == 200:
                result = response.json()
                if result.get('success'):
                    response.success()
                else:
                    response.failure(f"Prediction failed: {result.get('error', 'Unknown error')}")
            else:
                response.failure(f"HTTP {response.status_code}")
    
    @task(1)
    def get_prediction_history(self):
        """Get prediction history"""
        self.client.get("/predictions?limit=10")
    
    @task(1)
    def get_training_data(self):
        """Get training data info"""
        self.client.get("/training-data")
    
    @task(1)
    def get_training_status(self):
        """Get training status"""
        self.client.get("/training-status")

class BatchPredictionUser(HttpUser):
    wait_time = between(5, 10)  # Longer wait time for batch operations
    
    def on_start(self):
        """Initialize user session"""
        self.test_images = [self.create_test_image() for _ in range(5)]
    
    def create_test_image(self):
        """Create a test image"""
        img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        return img_byte_arr.getvalue()
    
    @task(1)
    def batch_predict(self):
        """Batch prediction test"""
        files = []
        for i, img_data in enumerate(self.test_images):
            files.append(('files', (f'test_image_{i}.png', img_data, 'image/png')))
        
        with self.client.post("/predict-batch", files=files, catch_response=True) as response:
            if response.status_code == 200:
                result = response.json()
                if 'predictions' in result:
                    response.success()
                else:
                    response.failure("No predictions in response")
            else:
                response.failure(f"HTTP {response.status_code}")

class TrainingUser(HttpUser):
    wait_time = between(30, 60)  # Very long wait time for training operations
    
    def on_start(self):
        """Initialize user session"""
        self.test_images = [self.create_test_image() for _ in range(3)]
    
    def create_test_image(self):
        """Create a test image"""
        img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        return img_byte_arr.getvalue()
    
    @task(1)
    def upload_training_data(self):
        """Upload training data test"""
        files = []
        for i, img_data in enumerate(self.test_images):
            files.append(('files', (f'training_image_{i}.png', img_data, 'image/png')))
        
        data = {
            'class_names': '["adenocarcinoma", "large.cell.carcinoma", "normal", "squamous.cell.carcinoma"]'
        }
        
        with self.client.post("/upload-data", files=files, data=data, catch_response=True) as response:
            if response.status_code == 200:
                result = response.json()
                if 'message' in result:
                    response.success()
                else:
                    response.failure("Upload response missing message")
            else:
                response.failure(f"HTTP {response.status_code}")
    
    @task(1)
    def start_retraining(self):
        """Start model retraining test"""
        with self.client.post("/retrain", catch_response=True) as response:
            if response.status_code == 200:
                result = response.json()
                if result.get('status') == 'training':
                    response.success()
                else:
                    response.failure("Training not started")
            else:
                response.failure(f"HTTP {response.status_code}")

# Custom events for monitoring
from locust import events

@events.request.add_listener
def my_request_handler(request_type, name, response_time, response_length, response, context, exception, start_time, url, **kwargs):
    """Custom request handler for detailed monitoring"""
    if exception:
        print(f"Request failed: {name} - {exception}")
    else:
        print(f"Request successful: {name} - {response_time}ms")

@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Called when a test is starting"""
    print("Load test starting...")

@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Called when a test is ending"""
    print("Load test ending...") 