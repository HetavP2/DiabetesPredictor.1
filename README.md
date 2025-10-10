# Diabetes Predictor

## Introduction

A machine learning project designed to predict diabetes onset in patients based on medical diagnostic measurements. This project utilizes the famous Pima Indians Diabetes Database to build and compare two different predictive models: Logistic Regression and Neural Networks. The system helps healthcare professionals assess diabetes risk using key medical indicators, providing accurate binary classification results.

## ✨ Features 

### **Machine Learning Models**
- **Logistic Regression**: Classical binary classification approach with high interpretability
- **Neural Network**: Deep learning model using TensorFlow/Keras for enhanced prediction accuracy
- **Model Comparison**: Side-by-side evaluation of both approaches to determine optimal performance

### **Data Processing & Analysis**
- **Data Preprocessing**: Comprehensive data cleaning and preparation pipeline
- **Feature Scaling**: StandardScaler implementation for normalized feature distributions
- **Class Balance Handling**: RandomOverSampler to address dataset imbalances
- **Data Visualization**: Statistical analysis and trend visualization capabilities

### **Model Training & Optimization**
- **Train/Validation/Test Split**: Proper 70/20/10 data separation for robust evaluation
- **Hyperparameter Tuning**: Grid search optimization for neural network parameters
- **Cross-Validation**: Multiple model configurations tested for optimal performance
- **Performance Metrics**: Comprehensive evaluation using accuracy, precision, recall, and F1-score

### **Predictive Analytics**
- **Binary Classification**: Predicts diabetes presence (0: No Diabetes, 1: Diabetes)
- **Probability Scoring**: Confidence levels for predictions
- **Medical Feature Analysis**: Evaluation based on 8 key medical indicators:
  - Pregnancies
  - Glucose Level
  - Blood Pressure
  - Skin Thickness
  - Insulin Level
  - BMI (Body Mass Index)
  - Diabetes Pedigree Function
  - Age

### **Model Performance Tracking**
- **Training Metrics**: Real-time monitoring of model training progress
- **Validation Performance**: Continuous evaluation on validation dataset
- **Loss Visualization**: Graphical representation of training and validation loss
- **Accuracy Tracking**: Performance monitoring across epochs

## 🛠️ Tech Stack

### **Core Machine Learning**
- **Python 3.11+**: Primary programming language for data science and ML
- **Jupyter Notebook**: Interactive development environment for experimentation and analysis
- **NumPy**: Numerical computing and array operations
- **Pandas**: Data manipulation, analysis, and CSV processing

### **Machine Learning Libraries**
- **Scikit-learn**: Classical machine learning algorithms and preprocessing tools
  - `LogisticRegression`: Binary classification model
  - `StandardScaler`: Feature normalization
  - `accuracy_score`, `classification_report`: Performance evaluation metrics
- **TensorFlow/Keras**: Deep learning framework for neural network implementation
  - Sequential model architecture
  - Dense layers with ReLU and sigmoid activation
  - Adam optimizer with configurable learning rates
  - Binary crossentropy loss function
- **imbalanced-learn**: Handles class imbalance issues
  - `RandomOverSampler`: Synthetic data generation for minority class

### **API & Web Framework**
- **FastAPI**: Modern, high-performance web framework for building APIs
  - Automatic API documentation with OpenAPI/Swagger
  - Type validation with Pydantic models
  - Asynchronous request handling
  - Built-in data validation and serialization
- **Uvicorn**: ASGI server for running FastAPI applications
- **Pydantic**: Data validation using Python type annotations

### **Model Persistence & Deployment**
- **Joblib**: Efficient model serialization for scikit-learn components
- **TensorFlow SavedModel**: Native TensorFlow model persistence format
- **JSON Configuration**: Runtime configuration management
- **Docker**: Containerization for consistent deployment environments
- **Docker Compose**: Multi-container orchestration and development setup

### **Data Processing Pipeline**
- **Custom Train/Test Split**: 70/20/10 train/validation/test separation
- **Feature Scaling Pipeline**: Standardized preprocessing workflow
- **Class Imbalance Handling**: Automated minority class oversampling

### **Visualization & Analysis**
- **Matplotlib**: Statistical plotting and model performance visualization
  - Loss curves and accuracy plots
  - Feature distribution histograms
  - Training progress monitoring

### **Dataset**
- **Pima Indians Diabetes Database**: Medical diagnostic dataset from Kaggle
  - 768 instances with 8 medical predictor features
  - Binary target variable (diabetes outcome)
  - Well-established benchmark dataset for diabetes prediction

## 🚀 Model Architecture

### **Logistic Regression Model**
- Linear classification approach with regularization
- Maximum iterations: 100
- Random state: 42 for reproducible results
- Performance: ~73% validation accuracy

### **Neural Network Architecture**
```
Input Layer (8 features) 
    ↓
Dropout Layer (configurable rate: 0.0-0.2)
    ↓
Dense Layer (17 or 64 nodes, ReLU activation)
    ↓
Dense Layer (17 or 64 nodes, ReLU activation)
    ↓
Output Layer (1 node, Sigmoid activation)
```

### **Hyperparameter Grid Search**
- **Node Configurations**: 17 nodes (2n+1 rule) or 64 nodes
- **Dropout Rates**: 0.0 (no dropout) or 0.2 (20% dropout)
- **Learning Rates**: 0.001, 0.005, 0.01
- **Batch Sizes**: 64 or 128
- **Training Epochs**: 100
- **Optimization**: Best model selected based on lowest validation loss

## 📁 Performance Results

### **Final Neural Network Model Performance**
The optimized TensorFlow model (selected from grid search) achieved the following metrics:

#### **Training Set Results**
- **Accuracy**: 78.29%
- **Precision**: 76.47%
- **Recall**: 81.71%
- **F1-Score**: 79.01%
- **AUC**: 86.82%

#### **Validation Set Results**
- **Accuracy**: 77.27%
- **Precision**: 64.18%
- **Recall**: 79.63%
- **F1-Score**: 71.07%
- **AUC**: 85.61%

#### **Test Set Results**
- **Accuracy**: 77.92%
- **Precision**: 63.89%
- **Recall**: 85.19%
- **F1-Score**: 73.02%
- **AUC**: 83.78%

### **Confusion Matrix Analysis**
**Test Set Performance:**
```
                Predicted
           No Diabetes  Diabetes
Actual No      37         13
       Yes      4         23
```

### **Model Optimization Results**
- **Best Hyperparameters**: Selected from 24 different configurations
- **Optimization Metric**: Minimum validation loss (0.475)
- **Architecture**: 17 nodes per hidden layer with 0.2 dropout
- **Training**: 0.001 learning rate, 128 batch size, 100 epochs
- **Model Selection**: Systematic grid search across multiple parameters

### **Logistic Regression Baseline**
- **Training Accuracy**: 73.86%
- **Validation Accuracy**: 79.22%
- **Test Accuracy**: 77.92%
- **Comparison**: Neural network shows improved recall for diabetes detection

## 🌐 FastAPI Implementation

### **API Architecture**
The project includes a production-ready FastAPI service that provides real-time diabetes prediction capabilities.

#### **Available Endpoints**

##### **Health Check - `GET /health`**
Returns API status and model information
```json
{
  "status": "ok",
  "model": "diabetes_tf_model.keras",
  "threshold": 0.5
}
```

##### **Single Prediction - `POST /predict`**
Makes diabetes prediction for individual patient data

**Request Body:**
```json
{
  "Pregnancies": 6.0,
  "Glucose": 148.0,
  "BloodPressure": 72.0,
  "SkinThickness": 35.0,
  "Insulin": 0.0,
  "BMI": 33.6,
  "DiabetesPedigreeFunction": 0.627,
  "Age": 50.0
}
```

**Response:**
```json
{
  "probability": 0.8245,
  "prediction": 1,
  "threshold": 0.5
}
```

### **Input Validation**
- **Pydantic Models**: Automatic type validation and conversion
- **Required Fields**: All 8 medical features must be provided
- **Data Types**: Automatic float conversion with error handling
- **Missing Values**: Comprehensive validation prevents null inputs

### **Model Loading & Inference**
- **Automatic Model Loading**: TensorFlow model loaded at startup
- **Feature Scaling**: Integrated StandardScaler preprocessing
- **Error Handling**: Comprehensive exception management
- **Performance**: Optimized for low-latency predictions

## 🐳 Docker Deployment

### **Container Architecture**
The application is fully containerized for consistent deployment across environments.

#### **Dockerfile Features**
- **Base Image**: Python 3.11 slim for minimal footprint
- **Working Directory**: `/app` for organized file structure
- **Dependency Installation**: Automated requirements.txt installation
- **Port Exposure**: FastAPI service on port 8000
- **Startup Command**: Uvicorn ASGI server with production settings

#### **Docker Compose Setup**
```yaml
services:
  diabetes-api:
    build: .
    container_name: diabetes_api
    ports:
      - "8000:8000"
    volumes:
      - .:/app
```

### **Quick Start Commands**
```bash
# Build and run with Docker Compose
docker-compose up --build

# Access API documentation
http://localhost:8000/docs

# Health check
curl http://localhost:8000/health
```

## 📁 Project Structure

```
DiabetesPredictor.1/
│
├── 📓 Data & Training
│   ├── diabetes.csv              # Pima Indians dataset
│   └── diabetespred.ipynb        # Complete ML workflow notebook
│
├── 🤖 Models & Artifacts
│   ├── models/
│   │   ├── diabetes_tf_model.keras   # Trained TensorFlow model
│   │   ├── feature_scaler.joblib     # Fitted StandardScaler
│   │   ├── inference_config.json     # Runtime configuration
│   │   └── metrics.json              # Performance metrics
│
├── 🌐 API & Deployment
│   ├── main.py                   # FastAPI application
│   ├── requirements.txt          # Python dependencies
│   ├── Dockerfile               # Container configuration
│   └── docker-compose.yml       # Multi-container orchestration
│
└── 📄 Documentation
    ├── README.md                # This comprehensive guide
    └── .gitignore               # Git exclusion rules
```

## 🚀 Getting Started

### **Option 1: Jupyter Notebook (Training & Experimentation)**
```bash
# Clone and navigate to project
git clone <repository-url>
cd DiabetesPredictor.1

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter and open diabetespred.ipynb
jupyter notebook diabetespred.ipynb

# Run all cells to train models and generate artifacts
```

### **Option 2: FastAPI Service (Production Deployment)**
```bash
# Using Docker (Recommended)
docker-compose up --build

# OR run locally
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000

# Access interactive API docs
open http://localhost:8000/docs
```

### **Option 3: Development Setup**
```bash
# Install in development mode
pip install -r requirements.txt

# Train models (if not already done)
jupyter nbconvert --execute diabetespred.ipynb

# Start API in development mode
uvicorn main:app --reload
```

## 📊 Dataset Information
**Source**: Pima Indians Diabetes Database (Kaggle)
- **URL**: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
- **Instances**: 768 patient records
- **Features**: 8 medical diagnostic measurements
- **Target**: Binary diabetes outcome (0: No Diabetes, 1: Diabetes)
- **Imbalance**: ~65% negative, ~35% positive class distribution

## 🎯 Use Cases

### **Healthcare Applications**
- **Early Screening**: Automated diabetes risk assessment in clinical settings
- **Population Health**: Large-scale screening programs for at-risk populations
- **Clinical Decision Support**: AI-assisted diagnostic tool for healthcare professionals
- **Remote Monitoring**: Telemedicine integration for remote patient assessment

### **Research & Development**
- **Model Comparison**: Benchmarking traditional ML vs deep learning approaches
- **Feature Engineering**: Analysis of medical indicator importance and correlations
- **Hyperparameter Studies**: Grid search methodology for neural network optimization
- **Performance Analysis**: Comprehensive evaluation of binary classification metrics

### **Educational & Training**
- **ML Pipeline Demonstration**: End-to-end machine learning project workflow
- **API Development**: FastAPI implementation for ML model deployment
- **Containerization**: Docker-based deployment strategies
- **Healthcare AI**: Application of AI in medical diagnostic scenarios

### **Production Deployment**
- **Microservice Architecture**: Scalable API service for healthcare systems
- **Batch Processing**: High-throughput prediction capabilities
- **Integration Ready**: RESTful API for seamless system integration
- **Monitoring & Logging**: Production-ready error handling and health checks
