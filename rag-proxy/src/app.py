import requests
import logging
import json
import sys
import os

import fastapi

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)


OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://ollama:11434")
RAG_URL = os.environ.get("RAG_URL", "http://rag:8000")
CODE_EXPERT_MODELS = os.environ.get("CODE_EXPERT_MODEL", "code-expert:latest,gemma3:4b").split(",")


app = fastapi.FastAPI()

@app.get("/health")
def get_health():
    return "green"
