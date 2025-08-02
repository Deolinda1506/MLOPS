import sqlite3
import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd

class DatabaseManager:
    def __init__(self, db_path='database/lung_cancer_ml.db'):
        """
        Initialize database manager
        
        Args:
            db_path (str): Path to SQLite database file
        """
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """
        Initialize database tables
        """
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create tables
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS models (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL,
                    model_path TEXT NOT NULL,
                    model_type TEXT NOT NULL,
                    accuracy REAL,
                    precision REAL,
                    recall REAL,
                    f1_score REAL,
                    training_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    parameters TEXT,
                    class_names TEXT
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS training_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_path TEXT NOT NULL,
                    class_name TEXT NOT NULL,
                    upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    file_size INTEGER,
                    image_dimensions TEXT,
                    used_in_training BOOLEAN DEFAULT FALSE
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id INTEGER,
                    input_data TEXT,
                    predicted_class TEXT,
                    confidence REAL,
                    actual_class TEXT,
                    prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    processing_time REAL,
                    FOREIGN KEY (model_id) REFERENCES models (id)
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS training_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id INTEGER,
                    start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    end_time TIMESTAMP,
                    epochs INTEGER,
                    batch_size INTEGER,
                    learning_rate REAL,
                    final_accuracy REAL,
                    final_loss REAL,
                    status TEXT DEFAULT 'running',
                    FOREIGN KEY (model_id) REFERENCES models (id)
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id INTEGER,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (model_id) REFERENCES models (id)
                )
            ''')
            
            conn.commit()
    
    def save_model(self, model_name: str, model_path: str, model_type: str, 
                   metrics: Dict[str, float], parameters: Dict[str, Any], 
                   class_names: List[str]) -> int:
        """
        Save model information to database
        
        Args:
            model_name (str): Name of the model
            model_path (str): Path to saved model file
            model_type (str): Type of model architecture
            metrics (dict): Model performance metrics
            parameters (dict): Model training parameters
            class_names (list): List of class names
            
        Returns:
            int: Model ID
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO models (model_name, model_path, model_type, accuracy, 
                                  precision, recall, f1_score, parameters, class_names)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                model_name, model_path, model_type,
                metrics.get('accuracy', 0.0),
                metrics.get('precision', 0.0),
                metrics.get('recall', 0.0),
                metrics.get('f1_score', 0.0),
                json.dumps(parameters),
                json.dumps(class_names)
            ))
            
            model_id = cursor.lastrowid
            conn.commit()
            return model_id
    
    def get_latest_model(self) -> Optional[Dict[str, Any]]:
        """
        Get the latest trained model
        
        Returns:
            dict: Model information or None
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM models ORDER BY training_date DESC LIMIT 1
            ''')
            
            row = cursor.fetchone()
            if row:
                columns = [description[0] for description in cursor.description]
                model_data = dict(zip(columns, row))
                
                # Parse JSON fields
                if model_data['parameters']:
                    model_data['parameters'] = json.loads(model_data['parameters'])
                if model_data['class_names']:
                    model_data['class_names'] = json.loads(model_data['class_names'])
                
                return model_data
            
            return None
    
    def save_training_data(self, file_paths: List[str], class_names: List[str]) -> List[int]:
        """
        Save training data information
        
        Args:
            file_paths (list): List of file paths
            class_names (list): List of corresponding class names
            
        Returns:
            list: List of inserted data IDs
        """
        data_ids = []
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            for file_path, class_name in zip(file_paths, class_names):
                # Get file size
                file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
                
                # Get image dimensions (if it's an image file)
                image_dimensions = "unknown"
                if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                    try:
                        from PIL import Image
                        with Image.open(file_path) as img:
                            image_dimensions = f"{img.width}x{img.height}"
                    except:
                        pass
                
                cursor.execute('''
                    INSERT INTO training_data (file_path, class_name, file_size, image_dimensions)
                    VALUES (?, ?, ?, ?)
                ''', (file_path, class_name, file_size, image_dimensions))
                
                data_ids.append(cursor.lastrowid)
            
            conn.commit()
        
        return data_ids
    
    def get_training_data(self, class_name: Optional[str] = None, 
                         used_in_training: Optional[bool] = None) -> List[Dict[str, Any]]:
        """
        Get training data information
        
        Args:
            class_name (str): Filter by class name
            used_in_training (bool): Filter by training usage
            
        Returns:
            list: List of training data records
        """
        with sqlite3.connect(self.db_path) as conn:
            query = "SELECT * FROM training_data WHERE 1=1"
            params = []
            
            if class_name:
                query += " AND class_name = ?"
                params.append(class_name)
            
            if used_in_training is not None:
                query += " AND used_in_training = ?"
                params.append(used_in_training)
            
            query += " ORDER BY upload_date DESC"
            
            df = pd.read_sql_query(query, conn, params=params)
            return df.to_dict('records')
    
    def save_prediction(self, model_id: int, input_data: str, predicted_class: str,
                       confidence: float, actual_class: Optional[str] = None,
                       processing_time: Optional[float] = None) -> int:
        """
        Save prediction record
        
        Args:
            model_id (int): ID of the model used
            input_data (str): Input data (file path or base64)
            predicted_class (str): Predicted class
            confidence (float): Prediction confidence
            actual_class (str): Actual class (if known)
            processing_time (float): Time taken for prediction
            
        Returns:
            int: Prediction ID
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO predictions (model_id, input_data, predicted_class, 
                                       confidence, actual_class, processing_time)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (model_id, input_data, predicted_class, confidence, 
                  actual_class, processing_time))
            
            prediction_id = cursor.lastrowid
            conn.commit()
            return prediction_id
    
    def get_prediction_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get prediction history
        
        Args:
            limit (int): Maximum number of records to return
            
        Returns:
            list: List of prediction records
        """
        with sqlite3.connect(self.db_path) as conn:
            query = '''
                SELECT p.*, m.model_name 
                FROM predictions p 
                LEFT JOIN models m ON p.model_id = m.id 
                ORDER BY p.prediction_date DESC 
                LIMIT ?
            '''
            
            df = pd.read_sql_query(query, conn, params=[limit])
            return df.to_dict('records')
    
    def start_training_session(self, model_id: int, epochs: int, 
                              batch_size: int, learning_rate: float) -> int:
        """
        Start a new training session
        
        Args:
            model_id (int): ID of the model being trained
            epochs (int): Number of training epochs
            batch_size (int): Training batch size
            learning_rate (float): Learning rate
            
        Returns:
            int: Training session ID
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO training_sessions (model_id, epochs, batch_size, learning_rate)
                VALUES (?, ?, ?, ?)
            ''', (model_id, epochs, batch_size, learning_rate))
            
            session_id = cursor.lastrowid
            conn.commit()
            return session_id
    
    def end_training_session(self, session_id: int, final_accuracy: float, 
                           final_loss: float, status: str = 'completed'):
        """
        End a training session
        
        Args:
            session_id (int): Training session ID
            final_accuracy (float): Final training accuracy
            final_loss (float): Final training loss
            status (str): Session status
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE training_sessions 
                SET end_time = CURRENT_TIMESTAMP, final_accuracy = ?, 
                    final_loss = ?, status = ?
                WHERE id = ?
            ''', (final_accuracy, final_loss, status, session_id))
            
            conn.commit()
    
    def save_model_metrics(self, model_id: int, metrics: Dict[str, float]):
        """
        Save model metrics
        
        Args:
            model_id (int): Model ID
            metrics (dict): Dictionary of metric names and values
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            for metric_name, metric_value in metrics.items():
                cursor.execute('''
                    INSERT INTO model_metrics (model_id, metric_name, metric_value)
                    VALUES (?, ?, ?)
                ''', (model_id, metric_name, metric_value))
            
            conn.commit()
    
    def get_model_metrics_history(self, model_id: int, metric_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get model metrics history
        
        Args:
            model_id (int): Model ID
            metric_name (str): Specific metric name to filter
            
        Returns:
            list: List of metric records
        """
        with sqlite3.connect(self.db_path) as conn:
            query = "SELECT * FROM model_metrics WHERE model_id = ?"
            params = [model_id]
            
            if metric_name:
                query += " AND metric_name = ?"
                params.append(metric_name)
            
            query += " ORDER BY timestamp DESC"
            
            df = pd.read_sql_query(query, conn, params=params)
            return df.to_dict('records')
    
    def get_database_statistics(self) -> Dict[str, Any]:
        """
        Get database statistics
        
        Returns:
            dict: Database statistics
        """
        with sqlite3.connect(self.db_path) as conn:
            stats = {}
            
            # Count records in each table
            tables = ['models', 'training_data', 'predictions', 'training_sessions', 'model_metrics']
            for table in tables:
                cursor = conn.cursor()
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                stats[f'{table}_count'] = cursor.fetchone()[0]
            
            # Get latest model info
            latest_model = self.get_latest_model()
            if latest_model:
                stats['latest_model'] = {
                    'name': latest_model['model_name'],
                    'accuracy': latest_model['accuracy'],
                    'training_date': latest_model['training_date']
                }
            
            # Get prediction accuracy (if actual classes are available)
            cursor = conn.cursor()
            cursor.execute('''
                SELECT COUNT(*) as total, 
                       SUM(CASE WHEN predicted_class = actual_class THEN 1 ELSE 0 END) as correct
                FROM predictions 
                WHERE actual_class IS NOT NULL
            ''')
            result = cursor.fetchone()
            if result[0] > 0:
                stats['prediction_accuracy'] = result[1] / result[0]
            
            return stats
    
    def cleanup_old_data(self, days_to_keep: int = 30):
        """
        Clean up old data
        
        Args:
            days_to_keep (int): Number of days of data to keep
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Delete old predictions
            cursor.execute('''
                DELETE FROM predictions 
                WHERE prediction_date < datetime('now', '-{} days')
            '''.format(days_to_keep))
            
            # Delete old metrics
            cursor.execute('''
                DELETE FROM model_metrics 
                WHERE timestamp < datetime('now', '-{} days')
            '''.format(days_to_keep))
            
            conn.commit() 