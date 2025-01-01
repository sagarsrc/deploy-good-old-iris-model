import os
import torch
from fastapi import FastAPI

app = FastAPI()


@app.get("/hello")
def read_root():
    return {"message": "Hello, World!"}


@app.get("/is-hf-logged-in")
def check_hf_login():
    return {
        "env_hf_token": os.getenv("HF_TOKEN"),
        "env_hf_repo": os.getenv("HUGGINGFACE_MODEL_REPO"),
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
