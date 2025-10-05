import json
from pathlib import Path
from typing import List
import joblib, numpy
import tensorflow
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

# setup paths to models
MODEL_PATH = Path("models") / "diabetes_tf_model.keras"  
SCALER_PATH = Path("models") / "feature_scaler.joblib"
CONFIG_PATH = Path("models") / "inference_config.json"

# validation check for models
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Missing model file: {MODEL_PATH}")
if not SCALER_PATH.exists():
    raise FileNotFoundError(f"Missing scaler file: {SCALER_PATH}")

# load models
model = tensorflow.keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

threshold = 0.5

# use fastapi (build web app) and setup model
app = FastAPI(title="Diabetes Prediction API", version="1.0")

# initialize 8 input features and ensure post request to predict has these values
class PatientData(BaseModel):
    Pregnancies: float = Field(..., description="Number of pregnancies")
    Glucose: float = Field(..., description="Plasma glucose concentration")
    BloodPressure: float = Field(..., description="Diastolic blood pressure (mm Hg)")
    SkinThickness: float = Field(..., description="Triceps skinfold thickness (mm)")
    Insulin: float = Field(..., description="2-Hour serum insulin (mu U/ml)")
    BMI: float = Field(..., description="Body mass index")
    DiabetesPedigreeFunction: float = Field(..., description="Diabetes pedigree function")
    Age: float = Field(..., description="Age (years)")

    @field_validator("*", mode="before")
    def ensure_num(cls, v):
        if v is None:
            raise ValueError("missing value")
        return float(v)

