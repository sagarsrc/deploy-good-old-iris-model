import os
import torch
import numpy as np

from pydantic import BaseModel
from sklearn.datasets import load_iris
from huggingface_hub import snapshot_download
from fastapi import FastAPI, HTTPException

# from dotenv import load_dotenv

from dev.model import IrisModel

# while deploying on fly.io, we don't need to load the .env file
# Load environment variables from .env file
# load_dotenv()

# Get the repo_id from environment variable
REPO_ID = os.getenv("HUGGINGFACE_MODEL_REPO")

# Create a directory to fetch artifacts
FETCHED_ARTIFACTS_PATH = "./fetched_artifacts"
os.makedirs(FETCHED_ARTIFACTS_PATH, exist_ok=True)

# Download model and scaler from Hugging Face Hub
snapshot_download(repo_id=REPO_ID, local_dir=FETCHED_ARTIFACTS_PATH)

MODEL_PATH = os.path.join(FETCHED_ARTIFACTS_PATH, "model.pth")
SCALER_PATH = os.path.join(FETCHED_ARTIFACTS_PATH, "scaler.joblib")

# Load the model and iris dataset once at startup
iris = load_iris()
CLASS_NAMES = iris.target_names.tolist()
MODEL = IrisModel.load(FETCHED_ARTIFACTS_PATH)  # Load model once at startup

app = FastAPI()


class InputFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


@app.get("/")
def read_root():
    return {"message": "Hello good old iris model API ;)"}


@app.get("/is-gpu-available")
def check_gpu():
    available = torch.cuda.is_available()
    return {"is_gpu_available": available}


@app.post("/predict")
def predict(features: InputFeatures):
    try:
        # Convert input features to numpy array
        input_features = np.array(
            [
                features.sepal_length,
                features.sepal_width,
                features.petal_length,
                features.petal_width,
            ]
        ).reshape(
            1, -1
        )  # Reshape for single prediction

        # Make prediction using the pre-loaded model
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
