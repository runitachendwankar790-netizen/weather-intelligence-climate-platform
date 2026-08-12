
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import os

# ============================================
# LOAD MODEL
# ============================================

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "best_weather_model.pkl")
FEATURE_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "feature_columns.pkl")

model = joblib.load(MODEL_PATH)
feature_columns = joblib.load(FEATURE_PATH)


# ============================================
# CREATE FASTAPI APPLICATION
# ============================================

app = FastAPI(
    title="Weather Intelligence API",
    description="AI-based Southwest Monsoon Rainfall Prediction API",
    version="1.0.0"
)


# ============================================
# HEALTH CHECK
# ============================================

@app.get("/")
def home():

    return {
        "message": "Weather Intelligence API is running",
        "status": "success"
    }


# ============================================
# MODEL INFORMATION
# ============================================

@app.get("/model-info")
def model_info():

    return {
        "model": "Best Weather Prediction Model",
        "number_of_features": len(feature_columns),
        "features": feature_columns
    }


# ============================================
# PREDICTION REQUEST
# ============================================

class PredictionRequest(BaseModel):

    features: dict


# ============================================
# PREDICTION ENDPOINT
# ============================================

@app.post("/predict")
def predict(request: PredictionRequest):

    try:

        input_data = request.features

        missing_features = [
            feature
            for feature in feature_columns
            if feature not in input_data
        ]

        if missing_features:

            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Missing features",
                    "missing_features": missing_features
                }
            )

        df = pd.DataFrame(
            [input_data]
        )

        df = df[
            feature_columns
        ]

        prediction = model.predict(df)

        return {
            "predicted_monsoon_rainfall_mm":
                float(prediction[0])
        }

    except HTTPException:

        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
