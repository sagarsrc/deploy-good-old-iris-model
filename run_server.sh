#!/bin/bash
# source ./.env
uvicorn serve.inference_api:app --host 0.0.0.0 --port 8000

# uvicorn main:app --host 0.0.0.0 --port 8000
