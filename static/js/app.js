// ML Pipeline Web Interface JavaScript

class MLPipelineApp {
    constructor() {
        this.initializeEventListeners();
        this.loadInitialData();
    }

    initializeEventListeners() {
        // File upload handling
        this.setupFileUpload();
        
        // Training status monitoring
        this.setupTrainingMonitoring();
        
        // Analytics refresh
        this.setupAnalyticsRefresh();
    }

    setupFileUpload() {
        const uploadArea = document.getElementById('uploadArea');
        const imageInput = document.getElementById('imageInput');

        if (uploadArea) {
            uploadArea.addEventListener('dragover', (e) => {
                e.preventDefault();
                uploadArea.classList.add('dragover');
            });

            uploadArea.addEventListener('dragleave', () => {
                uploadArea.classList.remove('dragover');
            });

            uploadArea.addEventListener('drop', (e) => {
                e.preventDefault();
                uploadArea.classList.remove('dragover');
                const files = e.dataTransfer.files;
                if (files.length > 0) {
                    this.handleImageUpload(files[0]);
                }
            });

            uploadArea.addEventListener('click', () => {
                imageInput.click();
            });
        }

        if (imageInput) {
            imageInput.addEventListener('change', (e) => {
                if (e.target.files.length > 0) {
                    this.handleImageUpload(e.target.files[0]);
                }
            });
        }
    }

    async handleImageUpload(file) {
        if (!file.type.startsWith('image/')) {
            this.showAlert('Please select an image file', 'error');
            return;
        }

        // Show preview
        this.showImagePreview(file);

        // Make prediction
        await this.predictImage(file);
    }

    showImagePreview(file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            const previewImg = document.getElementById('previewImg');
            const imagePreview = document.getElementById('imagePreview');
            
            if (previewImg && imagePreview) {
                previewImg.src = e.target.result;
                imagePreview.style.display = 'block';
            }
        };
        reader.readAsDataURL(file);
    }

    async predictImage(file) {
        const formData = new FormData();
        formData.append('file', file);

        this.showLoading(true);
        this.clearPredictionResults();

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            if (result.success) {
                this.displayPredictionResult(result);
            } else {
                this.showAlert(result.error || 'Prediction failed', 'error');
            }
        } catch (error) {
            this.showAlert(`Prediction failed: ${error.message}`, 'error');
        } finally {
            this.showLoading(false);
        }
    }

    displayPredictionResult(result) {
        const resultsDiv = document.getElementById('predictionResults');
        if (!resultsDiv) return;

        const confidencePercent = (result.confidence * 100).toFixed(1);
        const processingTime = (result.processing_time * 1000).toFixed(0);

        const html = `
            <div class="prediction-result">
                <h4>${result.predicted_class}</h4>
                <p><strong>Confidence:</strong> ${confidencePercent}%</p>
                <p><strong>Processing Time:</strong> ${processingTime}ms</p>
                <div class="mt-3">
                    <h6>Class Probabilities:</h6>
                    ${Object.entries(result.class_probabilities)
                        .map(([cls, prob]) => `<div>${cls}: ${(prob * 100).toFixed(1)}%</div>`)
                        .join('')}
                </div>
            </div>
        `;

        resultsDiv.innerHTML = html;
    }

    setupTrainingMonitoring() {
        const retrainBtn = document.getElementById('retrainBtn');
        if (retrainBtn) {
            retrainBtn.addEventListener('click', () => this.startRetraining());
        }
    }

    async startRetraining() {
        const progressDiv = document.getElementById('trainingProgress');
        const retrainBtn = document.getElementById('retrainBtn');

        if (progressDiv) progressDiv.style.display = 'block';
        if (retrainBtn) retrainBtn.disabled = true;

        try {
            const response = await fetch('/retrain', {
                method: 'POST'
            });

            const result = await response.json();
            
            if (result.status === 'training') {
                this.monitorTrainingProgress();
            }
        } catch (error) {
            this.showAlert('Failed to start retraining: ' + error.message, 'error');
            if (progressDiv) progressDiv.style.display = 'none';
            if (retrainBtn) retrainBtn.disabled = false;
        }
    }

    async monitorTrainingProgress() {
        const progressBar = document.getElementById('progressBar');
        const trainingMessage = document.getElementById('trainingMessage');

        const interval = setInterval(async () => {
            try {
                const response = await fetch('/training-status');
                const status = await response.json();

                if (progressBar) progressBar.style.width = status.progress + '%';
                if (trainingMessage) trainingMessage.textContent = status.message;

                if (status.status === 'completed' || status.status === 'error') {
                    clearInterval(interval);
                    const retrainBtn = document.getElementById('retrainBtn');
                    if (retrainBtn) retrainBtn.disabled = false;
                    
                    if (status.status === 'completed') {
                        this.showAlert('Training completed successfully!', 'success');
                        this.loadAnalytics(); // Refresh analytics
                    } else {
                        this.showAlert('Training failed: ' + status.message, 'error');
                    }
                }
            } catch (error) {
                console.error('Error monitoring training:', error);
            }
        }, 2000);
    }

    setupAnalyticsRefresh() {
        // Refresh analytics every 30 seconds
        setInterval(() => {
            this.loadAnalytics();
        }, 30000);
    }

    async loadAnalytics() {
        try {
            // Load metrics
            const metricsResponse = await fetch('/metrics');
            const metrics = await metricsResponse.json();

            if (metrics.model_info) {
                this.updatePerformanceChart(metrics.model_info);
            }

            // Load predictions
            const predictionsResponse = await fetch('/predictions?limit=10');
            const predictions = await predictionsResponse.json();
            this.updatePredictionsTable(predictions.predictions);

        } catch (error) {
            console.error('Error loading analytics:', error);
        }
    }

    updatePerformanceChart(modelInfo) {
        if (window.performanceChart) {
            window.performanceChart.data.datasets[0].data = [
                modelInfo.accuracy || 0,
                modelInfo.precision || 0,
                modelInfo.recall || 0,
                modelInfo.f1_score || 0
            ];
            window.performanceChart.update();
        }
    }

    updatePredictionsTable(predictions) {
        const tbody = document.querySelector('#predictionsTable tbody');
        if (!tbody) return;

        tbody.innerHTML = '';

        predictions.forEach(pred => {
            const row = tbody.insertRow();
            row.innerHTML = `
                <td>${new Date(pred.prediction_date).toLocaleString()}</td>
                <td>${pred.predicted_class}</td>
                <td>${(pred.confidence * 100).toFixed(1)}%</td>
                <td>${(pred.processing_time * 1000).toFixed(0)}ms</td>
            `;
        });
    }

    showLoading(show) {
        const spinner = document.getElementById('loadingSpinner');
        if (spinner) {
            spinner.style.display = show ? 'block' : 'none';
        }
    }

    clearPredictionResults() {
        const resultsDiv = document.getElementById('predictionResults');
        if (resultsDiv) {
            resultsDiv.innerHTML = '';
        }
    }

    showAlert(message, type = 'info') {
        // Create alert element
        const alertDiv = document.createElement('div');
        alertDiv.className = `alert alert-${type === 'error' ? 'danger' : type} alert-dismissible fade show`;
        alertDiv.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;

        // Add to page
        const container = document.querySelector('.container');
        if (container) {
            container.insertBefore(alertDiv, container.firstChild);
        }

        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (alertDiv.parentNode) {
                alertDiv.remove();
            }
        }, 5000);
    }

    loadInitialData() {
        // Load analytics on page load
        this.loadAnalytics();
    }
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.mlApp = new MLPipelineApp();
});

// Export for use in other scripts
if (typeof module !== 'undefined' && module.exports) {
    module.exports = MLPipelineApp;
} 