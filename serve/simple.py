"""
Simple FastAPI server for the good-old-iris-model
"""

import os
import torch
import numpy as np
from fastapi import FastAPI, HTTPException

from pydantic import BaseModel
from sklearn.datasets import load_iris
from huggingface_hub import login as hf_login
from huggingface_hub import snapshot_download

# custom import
from dev.model import IrisModel


# fetch and load the model
def initialize_huggingface_model():
    hf_login(token=os.getenv("HF_TOKEN"))

    REPO_ID = os.getenv("HUGGINGFACE_MODEL_REPO")
    FETCHED_ARTIFACTS_PATH = "./fetched_artifacts"
    os.makedirs(FETCHED_ARTIFACTS_PATH, exist_ok=True)

    snapshot_download(repo_id=REPO_ID, local_dir=FETCHED_ARTIFACTS_PATH)

    MODEL_PATH = os.path.join(FETCHED_ARTIFACTS_PATH, "model.pth")
    SCALER_PATH = os.path.join(FETCHED_ARTIFACTS_PATH, "scaler.joblib")

    iris = load_iris()
    CLASS_NAMES = iris.target_names.tolist()
    MODEL = IrisModel.load(FETCHED_ARTIFACTS_PATH)

    return MODEL, CLASS_NAMES


# initialize the model
MODEL, CLASS_NAMES = initialize_huggingface_model()


app = FastAPI()


class InputFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


@app.post("/predict")
def predict(features: InputFeatures):
    try:
        input_features = np.array(
            [
                features.sepal_length,
                features.sepal_width,
                features.petal_length,
                features.petal_width,
            ]
        ).reshape(1, -1)

        prediction = MODEL.predict(input_features)[0]
        probabilities = MODEL.predict_proba(input_features)[0]
        confidence = probabilities[prediction]

        return {
            "predicted_class": int(prediction),
            "predicted_class_name": CLASS_NAMES[prediction],
            "confidence": round(float(confidence), 2),
            "probabilities": {
                class_name: round(float(prob), 2)
                for class_name, prob in zip(CLASS_NAMES, probabilities)
            },
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/")
def read_root():
    return {"message": "Hello, good-old-iris-model API!"}


@app.get("/env")
def check_hf_login():
    return {
        "env_hf_token": os.getenv("HF_TOKEN", None) is not None,
        "env_hf_repo": os.getenv("HUGGINGFACE_MODEL_REPO", None) is not None,
        "is_gpu_available": torch.cuda.is_available(),
    }


# Add health check endpoint
@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.get("/is-gpu-available")
def check_gpu():
    available = torch.cuda.is_available()
    return {"is_gpu_available": available}


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
