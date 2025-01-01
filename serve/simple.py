import os
from fastapi import FastAPI

app = FastAPI()


@app.get("/hello")
def read_root():
    return {"message": "Hello, World!"}


@app.get("/is-hf-logged-in")
def check_hf_login():
    return {"is_hf_logged_in": os.getenv("HF_TOKEN") is not None}


# Add health check endpoint
@app.get("/health")
def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
