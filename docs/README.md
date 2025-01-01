# development

## model training

- train model on iris dataset
- save model and training history

```
python dev.model.py > ./local_artifacts/train_log.txt 2>&1
```

## model inference

- tests on iris dataset
- has a function to test custom input

```
python dev.inference.py > ./local_artifacts/inference_log.txt 2>&1
```

# testing server

## start server

```
uvicorn serve.inference_api:app --reload
```

## test curl requests

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

# deployment docker

```
sudo docker build -t iris .
sudo docker run -d -p 8000:8000 iris
```

# deployment fly

- API health check

```
curl -X GET "https://good-old-iris-model.fly.dev" \
-H "Content-Type: application/json"

{"message":"Hello good old iris model API ;)"}
```

- test curl request to deployed model on fly.io
- endpoint: https://good-old-iris-model.fly.dev/predict

```
curl -X POST "https://good-old-iris-model.fly.dev/predict" \
-H "Content-Type: application/json" \
-d '{
"sepal_length": 6.4,
"sepal_width": 2.9,
"petal_length": 4.3,
"petal_width": 1.3
}'

```
