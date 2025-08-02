import os
import time
import base64
import json
from typing import List, Optional, Dict, Any
from datetime import datetime
import asyncio
import threading

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from PIL import Image
import io

# Import our modules
from preprocessing import DataPreprocessor
from model import LungCancerClassifier
from prediction import PredictionService
from database import DatabaseManager

# Initialize FastAPI app
app = FastAPI(
    title="Lung Cancer Classification ML Pipeline",
    description="A complete ML pipeline for lung cancer classification with retraining capabilities",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# Initialize services
db_manager = DatabaseManager()
prediction_service = PredictionService()

# Global variables for model training
training_in_progress = False
training_status = {"status": "idle", "progress": 0, "message": ""}

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global prediction_service
    
    # Load the latest model
    latest_model = db_manager.get_latest_model()
    if latest_model:
        model_path = latest_model['model_path']
        if os.path.exists(model_path):
            prediction_service = PredictionService(
                model_path=model_path,
                class_names=latest_model['class_names']
            )
            print(f"Loaded model: {latest_model['model_name']}")
        else:
            print("Model file not found, using default prediction service")

@app.get("/", response_class=HTMLResponse)
async def home():
    """Serve the main web interface"""
    return templates.TemplateResponse("index.html", {"request": {}})

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    model_info = prediction_service.get_model_info()
    db_stats = db_manager.get_database_statistics()
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model_info.get('model_loaded', False),
        "database_stats": db_stats
    }

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    """
    Predict lung cancer type from uploaded image
    """
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read image file
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to base64 for storage
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        # Make prediction
        start_time = time.time()
        result = prediction_service.predict_single(image)
        processing_time = time.time() - start_time
        
        if result['success']:
            # Save prediction to database
            latest_model = db_manager.get_latest_model()
            model_id = latest_model['id'] if latest_model else None
            
            db_manager.save_prediction(
                model_id=model_id,
                input_data=img_str,
                predicted_class=result['predicted_class'],
                confidence=result['confidence'],
                processing_time=processing_time
            )
            
            # Add processing time to result
            result['processing_time'] = processing_time
            
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict-batch")
async def predict_batch(files: List[UploadFile] = File(...)):
    """
    Predict multiple images at once
    """
    try:
        results = []
        
        for file in files:
            if not file.content_type.startswith('image/'):
                continue
            
            # Read image file
            image_data = await file.read()
            image = Image.open(io.BytesIO(image_data))
            
            # Make prediction
            result = prediction_service.predict_single(image)
            result['filename'] = file.filename
            results.append(result)
        
        return {"predictions": results}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.post("/upload-data")
async def upload_training_data(
    files: List[UploadFile] = File(...),
    class_names: str = Form(...)
):
    """
    Upload training data for model retraining
    """
    try:
        # Parse class names
        class_names_list = json.loads(class_names)
        
        if len(files) != len(class_names_list):
            raise HTTPException(
                status_code=400, 
                detail="Number of files must match number of class names"
            )
        
        # Save uploaded files
        saved_paths = []
        for file in files:
            if not file.content_type.startswith('image/'):
                continue
            
            # Create upload directory
            upload_dir = "static/uploads"
            os.makedirs(upload_dir, exist_ok=True)
            
            # Save file
            file_path = os.path.join(upload_dir, file.filename)
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            saved_paths.append(file_path)
        
        # Save to database
        data_ids = db_manager.save_training_data(saved_paths, class_names_list)
        
        return {
            "message": f"Successfully uploaded {len(saved_paths)} files",
            "data_ids": data_ids,
            "files": saved_paths
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/retrain")
async def retrain_model(background_tasks: BackgroundTasks):
    """
    Trigger model retraining with uploaded data
    """
    global training_in_progress, training_status
    
    if training_in_progress:
        raise HTTPException(status_code=400, detail="Training already in progress")
    
    # Start training in background
    background_tasks.add_task(train_model_background)
    
    return {"message": "Model retraining started", "status": "training"}

@app.get("/training-status")
async def get_training_status():
    """Get current training status"""
    return training_status

@app.get("/metrics")
async def get_model_metrics():
    """Get model performance metrics"""
    try:
        # Get latest model
        latest_model = db_manager.get_latest_model()
        if not latest_model:
            return {"error": "No model found"}
        
        # Get prediction history
        predictions = db_manager.get_prediction_history(limit=1000)
        
        # Calculate metrics
        if predictions:
            total_predictions = len(predictions)
            correct_predictions = sum(
                1 for p in predictions 
                if p.get('actual_class') and p['predicted_class'] == p['actual_class']
            )
            
            accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
            
            # Confidence statistics
            confidences = [p['confidence'] for p in predictions if p.get('confidence')]
            avg_confidence = np.mean(confidences) if confidences else 0
            
            metrics = {
                "model_info": {
                    "name": latest_model['model_name'],
                    "accuracy": latest_model['accuracy'],
                    "precision": latest_model['precision'],
                    "recall": latest_model['recall'],
                    "f1_score": latest_model['f1_score'],
                    "training_date": latest_model['training_date']
                },
                "prediction_metrics": {
                    "total_predictions": total_predictions,
                    "accuracy": accuracy,
                    "average_confidence": avg_confidence
                },
                "database_stats": db_manager.get_database_statistics()
            }
        else:
            metrics = {
                "model_info": {
                    "name": latest_model['model_name'],
                    "accuracy": latest_model['accuracy'],
                    "precision": latest_model['precision'],
                    "recall": latest_model['recall'],
                    "f1_score": latest_model['f1_score'],
                    "training_date": latest_model['training_date']
                },
                "prediction_metrics": {
                    "total_predictions": 0,
                    "accuracy": 0,
                    "average_confidence": 0
                },
                "database_stats": db_manager.get_database_statistics()
            }
        
        return metrics
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")

@app.get("/predictions")
async def get_prediction_history(limit: int = 100):
    """Get prediction history"""
    try:
        predictions = db_manager.get_prediction_history(limit=limit)
        return {"predictions": predictions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get predictions: {str(e)}")

@app.get("/training-data")
async def get_training_data(class_name: Optional[str] = None):
    """Get training data information"""
    try:
        data = db_manager.get_training_data(class_name=class_name)
        return {"training_data": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get training data: {str(e)}")

def train_model_background():
    """Background task for model training"""
    global training_in_progress, training_status, prediction_service
    
    try:
        training_in_progress = True
        training_status = {"status": "training", "progress": 0, "message": "Initializing..."}
        
        # Update status
        def update_status(progress, message):
            training_status["progress"] = progress
            training_status["message"] = message
        
        update_status(10, "Loading data...")
        
        # Load training data
        preprocessor = DataPreprocessor(data_path="../data")
        images, labels, class_names = preprocessor.load_data()
        
        update_status(20, "Splitting data...")
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(
            images, labels, test_size=0.2, val_size=0.2
        )
        
        update_status(30, "Creating data generators...")
        
        # Create data generators
        train_generator, val_generator = preprocessor.create_data_generators(
            X_train, y_train, X_val, y_val, batch_size=32
        )
        
        update_status(40, "Building model...")
        
        # Build and train model
        classifier = LungCancerClassifier(
            num_classes=len(class_names),
            img_size=(224, 224),
            model_type='efficientnet'
        )
        
        model = classifier.build_model(learning_rate=0.001, dropout_rate=0.5)
        
        update_status(50, "Training model...")
        
        # Train model
        history = classifier.train_model(
            train_generator, val_generator, epochs=30
        )
        
        update_status(80, "Evaluating model...")
        
        # Evaluate model
        test_generator, _ = preprocessor.create_data_generators(
            X_test, y_test, X_test, y_test, batch_size=32
        )
        
        results = classifier.evaluate_model(test_generator, class_names)
        
        update_status(90, "Saving model...")
        
        # Save model
        model_path = f"models/lung_cancer_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5"
        classifier.save_model(model_path)
        
        # Save to database
        metrics = {
            'accuracy': results['accuracy'],
            'precision': results['classification_report']['weighted avg']['precision'],
            'recall': results['classification_report']['weighted avg']['recall'],
            'f1_score': results['classification_report']['weighted avg']['f1-score']
        }
        
        parameters = {
            'model_type': 'efficientnet',
            'img_size': (224, 224),
            'learning_rate': 0.001,
            'dropout_rate': 0.5,
            'epochs': 30
        }
        
        model_id = db_manager.save_model(
            model_name=f"Lung Cancer Classifier {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            model_path=model_path,
            model_type='efficientnet',
            metrics=metrics,
            parameters=parameters,
            class_names=class_names
        )
        
        # Update prediction service
        prediction_service = PredictionService(model_path=model_path, class_names=class_names)
        
        update_status(100, "Training completed successfully!")
        
        # Save training results
        classifier.save_training_results(results)
        
    except Exception as e:
        training_status = {"status": "error", "progress": 0, "message": f"Training failed: {str(e)}"}
        print(f"Training error: {e}")
    
    finally:
        training_in_progress = False

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True) 