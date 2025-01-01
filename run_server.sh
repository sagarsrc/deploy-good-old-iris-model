#!/bin/bash
echo $HF_TOKEN
uvicorn serve.simple:app --host 0.0.0.0 --port "${PORT:-8000}"