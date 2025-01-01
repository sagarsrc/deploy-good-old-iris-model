# development

## model training

- train model on iris dataset
- save model and training history

```bash
python dev.model.py > ./local_artifacts/train_log.txt 2>&1
```

## model inference

- tests on iris dataset
- has a function to test custom input

```bash
python dev.inference.py > ./local_artifacts/inference_log.txt 2>&1
```

# testing local server

## start server

```bash
uvicorn serve.inference_api:app --reload
```

## test local curl requests to fastapi server

Example 1 - Likely Setosa

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

Example 2 - Likely Versicolor

```bash
curl -X POST "http://localhost:8000/predict" \
-H "Content-Type: application/json" \
-d '{
    "sepal_length": 6.4,
    "sepal_width": 2.9,
    "petal_length": 4.3,
    "petal_width": 1.3
}'
```

Example 3 - Likely Virginica

```bash
curl -X POST "http://localhost:8000/predict" \
-H "Content-Type: application/json" \
-d '{
    "sepal_length": 7.7,
    "sepal_width": 3.8,
    "petal_length": 6.7,
    "petal_width": 2.2
}'
```

# testing on docker locally

```bash
sudo docker build -t iris .
sudo docker run -d -p 8000:8000 iris
```

same curl requests as above

# deployment fly.io

API health check

```bash
curl -X GET "https://good-old-iris-model.fly.dev" \
-H "Content-Type: application/json"
```

Expected output

```bash
{"message":"Hello good old iris model API ;)"}
```

End point to test model inference hit fly.io

- test curl request to deployed model on fly.io
- endpoint: https://good-old-iris-model.fly.dev/predict

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

Expected output

```bash
{"predicted_class":2,"predicted_class_name":"virginica","confidence":0.54,"probabilities":{"setosa":0.1,"versicolor":0.36,"virginica":0.54}}
```

# Todo

1. use GPU machine

# References

1. [Fly.io Resource Pricing](https://fly.io/docs/about/pricing/#machines)
2. [Fly.io Continuous Deployment with Github Actions](https://fly.io/docs/launch/continuous-deployment-with-github-actions/)
3. [Fly.io GPU Quickstart](https://fly.io/docs/gpus/gpu-quickstart/)
