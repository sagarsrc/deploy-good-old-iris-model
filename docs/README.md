# Iris Model Development and Deployment Guide

## Local Development

### Model Training

Train and save model on iris dataset:

```bash
python dev.model.py > ./local_artifacts/train_log.txt 2>&1
```

### Model Inference

Run inference tests on iris dataset and custom inputs:

```bash
python dev.inference.py > ./local_artifacts/inference_log.txt 2>&1
```

## Local Testing

### FastAPI Server

Start the server:

```bash
uvicorn serve.inference_api:app --reload
```

### Sample API Requests

Predict Setosa

```bash
curl -X POST "http://localhost:8000/predict" \
-H "Content-Type: application/json" \
-d '{
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
}'
```

### Docker Testing

```bash
sudo docker build -t iris .
sudo docker run -d -p 8000:8000 iris
```

Use the same curl requests as above for testing.

## Fly.io Deployment

### Deployment Commands

CPU Version:

```bash
flyctl deploy --remote-only --config cpu.fly.toml --dockerfile ./Dockerfile.cpu
```

GPU Version:

```bash
flyctl deploy --remote-only --config gpu.fly.toml --dockerfile ./Dockerfile.gpu
```

### API Testing

Health Check:

```bash
curl -X GET "https://good-old-iris-model.fly.dev/env" \
-H "Content-Type: application/json"
```

Expected Response:

```json
{ "env_hf_token": true, "env_hf_repo": true, "is_gpu_available": false }
```

Model Inference:

1. Predict Versicolor

```bash
curl -X POST "https://good-old-iris-model.fly.dev/predict" \
-H "Content-Type: application/json" \
-d '{
    "sepal_length": 6.4,
    "sepal_width": 2.9,
    "petal_length": 4.3,
    "petal_width": 1.3
}'
```

2. Predict Virginica

```bash
curl -X POST "https://good-old-iris-model.fly.dev/predict" \
-H "Content-Type: application/json" \
-d '{
    "sepal_length": 7.7,
    "sepal_width": 3.8,
    "petal_length": 6.7,
    "petal_width": 2.2
}'
```

Expected Response example:

```json
{
  "predicted_class": 2,
  "predicted_class_name": "virginica",
  "confidence": 0.54,
  "probabilities": {
    "setosa": 0.1,
    "versicolor": 0.36,
    "virginica": 0.54
  }
}
```

## Known Issues

1. GPU deployment requires manual account review from Fly.io:

```
✖ Failed: error creating a new machine: failed to launch VM: Your organization is not allowed to use GPU machines. Please contact billing@fly.io
Please contact billing@fly.io (Request ID: 01JGHZ65QWG5FNV757MFYJTBQ7-iad) (Trace ID: 7c92684e1b16fd263ee75b2b5b34e7e9)
```

See [forum discussion](https://community.fly.io/t/your-organization-is-not-allowed-to-use-gpu-machines/19166).

2. CI/CD is functional for both CPU and GPU versions, but GPU machine creation fails due to Fly.io restrictions.

## References

1. [Fly.io Resource Pricing](https://fly.io/docs/about/pricing/#machines)
2. [Fly.io Continuous Deployment with Github Actions](https://fly.io/docs/launch/continuous-deployment-with-github-actions/)
3. [Fly.io GPU Quickstart](https://fly.io/docs/gpus/gpu-quickstart/)
