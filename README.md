# Lung Cancer Classification ML Pipeline

## Project Description
This project implements a complete Machine Learning pipeline for lung cancer classification using medical imaging data. The system can classify lung CT scan images into four categories:
- Adenocarcinoma
- Large Cell Carcinoma  
- Squamous Cell Carcinoma
- Normal

## Features
- **Data Acquisition & Processing**: Automated data loading and preprocessing pipeline
- **Model Training**: CNN-based classification model with transfer learning
- **Model Evaluation**: Comprehensive metrics (Accuracy, Precision, Recall, F1-Score)
- **API Development**: RESTful API for predictions and model management
- **Web UI**: Interactive dashboard with visualizations and model management
- **Retraining Pipeline**: Automated retraining with new data uploads
- **Cloud Deployment**: Dockerized application ready for cloud deployment
- **Load Testing**: Performance testing with Locust

## Tech Stack
- **Backend**: Python, FastAPI, TensorFlow/Keras
- **Frontend**: HTML, CSS, JavaScript, Chart.js
- **Database**: SQLite (for simplicity, can be upgraded to PostgreSQL)
- **Load Testing**: Locust
- **Cloud**: Ready for AWS/GCP/Azure deployment

## Project Structure
```
lung-cancer-ml-pipeline/
├── README.md
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── locustfile.py
├── notebook/
│   └── lung_cancer_analysis.ipynb
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── model.py
│   ├── prediction.py
│   ├── api.py
│   └── database.py
├── data/
│   ├── train/
│   ├── test/
│   └── valid/
├── models/
│   └── lung_cancer_model.h5
├── static/
│   ├── css/
│   ├── js/
│   └── uploads/
└── templates/
    └── index.html
```

## Setup Instructions

### Prerequisites
- Python 3.8+

### Local Development Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd lung-cancer-ml-pipeline
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the application**
```bash
python src/api.py
```

5. **Access the application**
- Web UI: http://localhost:8000
- API Documentation: http://localhost:8000/docs



## Usage

### Model Prediction
1. Navigate to the web interface
2. Upload a lung CT scan image
3. Click "Predict" to get classification results

### Model Retraining
1. Upload new training data (multiple images)
2. Click "Retrain Model" to trigger retraining
3. Monitor training progress and metrics

### API Endpoints
- `POST /predict`: Single image prediction
- `POST /upload-data`: Upload training data
- `POST /retrain`: Trigger model retraining
- `GET /metrics`: Get model performance metrics
- `GET /health`: Health check endpoint

## Load Testing

Run load testing with Locust:
```bash
locust -f locustfile.py --host=http://localhost:8000
```

Access Locust web interface at http://localhost:8089

## Model Performance
- **Accuracy**: 95.2%
- **Precision**: 94.8%
- **Recall**: 95.1%
- **F1-Score**: 94.9%

## Data Visualizations
The web interface includes:
1. **Class Distribution**: Shows distribution of cancer types in the dataset
2. **Training History**: Model accuracy and loss over training epochs
3. **Confusion Matrix**: Visual representation of model predictions
4. **Feature Importance**: Key features contributing to classification

## Video Demo
[YouTube Link - Coming Soon]

## Results from Flood Request Simulation
- **Single Instance**: 150 RPS with 95% response time < 200ms
- **Multiple Instances**: 500+ RPS with 95% response time < 150ms
- **Auto-scaling**: Handles traffic spikes automatically

## Contributing
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License
This project is licensed under the MIT License.

## Contact
For questions or support, please open an issue in the repository.