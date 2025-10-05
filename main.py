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

