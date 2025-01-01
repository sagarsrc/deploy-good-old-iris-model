#!/bin/bash
# source ./.env
uvicorn serve.simple:app --host 0.0.0.0 --port 8000
