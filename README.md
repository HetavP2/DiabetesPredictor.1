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
- **Python 3.12+**: Primary programming language for data science and ML
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

### **Data Preprocessing**
- **imbalanced-learn**: Handles class imbalance issues
  - `RandomOverSampler`: Synthetic data generation for minority class
- **Data Splitting**: Custom 70/20/10 train/validation/test separation

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

## 📊 Performance Results

### **Logistic Regression Results**
- **Training Accuracy**: 75.22%
- **Validation Accuracy**: 73.38%
- **Test Accuracy**: 68.83%

### **Neural Network Results**
- **Best Configuration**: Optimized through grid search
- **Test Accuracy**: ~66% (best performing model)
- **Model Selection**: Based on minimum validation loss

### **Classification Metrics**
Both models provide detailed classification reports including:
- Precision scores for each class
- Recall rates for diabetes detection
- F1-scores for balanced performance assessment
- Support values showing class distribution

### **Running the Project**
1. Clone the repository and navigate to the project directory
2. Download libraries
3. Ensure `diabetes.csv` is in the same directory as the notebook
4. Open `diabetespred.ipynb` in Jupyter Notebook or JupyterLab
5. Run all cells sequentially to train and evaluate both models

### **Dataset Source**
The project uses the Pima Indians Diabetes Database available at:
https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database

## 🎯 Use Cases

- **Healthcare Risk Assessment**: Early diabetes screening and risk evaluation
- **Medical Research**: Comparative analysis of traditional vs. deep learning approaches
- **Educational Purpose**: Understanding binary classification techniques in healthcare
- **Clinical Decision Support**: Supplementary tool for medical professionals
