import os
import time
import json
import logging
import uvicorn
import requests
from pathlib import Path
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel

# Import your existing modules and classes
from code_indexer import CodebaseIndexer
from rag_engine import CodeRAGEngine

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get environment variables
ELASTICSEARCH_HOST = os.getenv("ELASTICSEARCH_HOST", "http://elasticsearch:9200")
ELASTICSEARCH_INDEX = os.getenv("ELASTICSEARCH_INDEX", "code-embeddings")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")
CODE_DIR = os.getenv("CODE_DIR", "/code")
AUTO_INGEST = os.getenv("AUTO_INGEST_ON_STARTUP", "true").lower() == "true"

# Use Gemma which is likely already available for embeddings
EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL", "all-minilm:l6-v2") 
LLM_MODEL = os.getenv("LLM_MODEL", "gemma3:4b")

# Initialize FastAPI
app = FastAPI(title="Code RAG API")

def check_model_availability(model_name, ollama_url):
    """Check if a model is available in Ollama and pull it if not"""
    try:
        # Check if model exists
        response = requests.get(f"{ollama_url}/api/tags", timeout=10)
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_exists = any(m.get("name") == model_name for m in models)
            
            if model_exists:
                logger.info(f"Model {model_name} is available")
                return True
            else:
                logger.warning(f"Model {model_name} not found, attempting to pull...")
                
                # Pull the model
                pull_response = requests.post(
                    f"{ollama_url}/api/pull", 
                    json={"name": model_name}, 
                    timeout=300  # Longer timeout for model pulling
                )
                
                if pull_response.status_code == 200:
                    logger.info(f"Successfully pulled model {model_name}")
                    return True
                else:
                    logger.error(f"Failed to pull model {model_name}: {pull_response.text}")
                    return False
        else:
            logger.error(f"Failed to check available models: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"Error checking/pulling model {model_name}: {e}")
        return False

# Check models and initialize your components
logger.info(f"Checking availability of models: {EMBEDDINGS_MODEL} and {LLM_MODEL}")

# Initialize your components after checking model availability
indexer = CodebaseIndexer(
    es_url=ELASTICSEARCH_HOST,
    index_name=ELASTICSEARCH_INDEX,
    code_dir=CODE_DIR,
    ollama_url=OLLAMA_HOST,
    embeddings_model=EMBEDDINGS_MODEL
)

rag_engine = CodeRAGEngine(
    es_url=ELASTICSEARCH_HOST,
    index_name=ELASTICSEARCH_INDEX,
    ollama_url=OLLAMA_HOST,
    embeddings_model=EMBEDDINGS_MODEL,
    llm_model=LLM_MODEL
)

def check_elasticsearch_health(es_url, logger):
    """Check if Elasticsearch is ready with better status detection"""
    try:
        # Simple health check that accepts yellow status
        response = requests.get(f"{es_url}/_cluster/health", timeout=10)
        if response.status_code == 200:
            health_data = response.json()
            status = health_data.get("status")
            logger.info(f"Elasticsearch health status: {status}")
            
            # Both green and yellow are operational statuses
            if status in ["green", "yellow"]:
                return True, health_data
            else:
                logger.warning(f"Elasticsearch status is {status}, not ready yet")
                return False, health_data
        else:
            logger.warning(f"Elasticsearch health check failed with status code: {response.status_code}")
            return False, {"status": "unknown", "error": f"status code {response.status_code}"}
    except Exception as e:
        logger.warning(f"Error checking Elasticsearch health: {str(e)}")
        return False, {"status": "error", "error": str(e)}

def auto_ingest_on_startup():
    """Improved startup function that properly handles Elasticsearch status and model availability"""
    if AUTO_INGEST:
        logger.info(f"Auto-ingest enabled for code directory: {CODE_DIR}")
        
        # Initial pause to let services initialize
        time.sleep(10)
        
        # Check Elasticsearch with improved detection
        max_retries = 10
        retry_interval = 10
        
        for attempt in range(max_retries):
            is_ready, health_data = check_elasticsearch_health(ELASTICSEARCH_HOST, logger)
            
            if is_ready:
                logger.info("Elasticsearch is ready for ingestion")
                break
                
            if attempt < max_retries - 1:
                logger.info(f"Waiting for Elasticsearch... (Attempt {attempt+1}/{max_retries})")
                time.sleep(retry_interval)
        
        # Check if embedding model is available
        if not check_model_availability(EMBEDDINGS_MODEL, OLLAMA_HOST):
            logger.error(f"Required embedding model {EMBEDDINGS_MODEL} not available. Skipping ingestion.")
            return
        
        # Proceed with ingestion
        try:
            logger.info(f"Starting automatic code ingestion from {CODE_DIR}")
            result = indexer.ingest_codebase(force=True)
            logger.info(f"Automatic code ingestion completed with status: {result.get('status', 'unknown')}")
        except Exception as e:
            logger.error(f"Error during code ingestion: {str(e)}")

@app.get("/")
def read_root():
    return {
        "status": "Code RAG service is running",
        "endpoints": {
            "GET /": "This information",
            "GET /status": "Check indexing status",
            "POST /ingest": "Start code ingestion",
            "POST /query": "Query the codebase",
            "POST /force-ingest": "Force ingestion regardless of ES status"
        }
    }

@app.get("/status")
def get_status():
    """Get current system status and Elasticsearch health"""
    try:
        # Check Elasticsearch
        is_ready, health_data = check_elasticsearch_health(ELASTICSEARCH_HOST, logger)
        
        # Check index
        index_status = indexer.check_index() if hasattr(indexer, "check_index") else {"error": "check_index method not available"}
        
        # Check Ollama and models
        try:
            ollama_response = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
            models = []
            if ollama_response.status_code == 200:
                models = [m.get("name") for m in ollama_response.json().get("models", [])]
            
            ollama_status = {
                "available": ollama_response.status_code == 200,
                "status_code": ollama_response.status_code,
                "models": models,
                "embeddings_model_available": EMBEDDINGS_MODEL in models,
                "llm_model_available": LLM_MODEL in models
            }
        except Exception as e:
            ollama_status = {"available": False, "error": str(e)}
        
        # Check code directory
        try:
            code_files = len(list(Path(CODE_DIR).rglob("*"))) if Path(CODE_DIR).exists() else 0
            code_dir_status = {"exists": Path(CODE_DIR).exists(), "file_count": code_files}
        except Exception as e:
            code_dir_status = {"exists": "error", "error": str(e)}
        
        return {
            "elasticsearch": {
                "ready": is_ready,
                "health": health_data,
                "index": index_status
            },
            "ollama": ollama_status,
            "code_directory": code_dir_status,
            "indexer_status": indexer.status if hasattr(indexer, "status") else "unknown",
            "ingestion_stats": indexer.stats if hasattr(indexer, "stats") else {},
            "config": {
                "embeddings_model": EMBEDDINGS_MODEL,
                "llm_model": LLM_MODEL
            }
        }
    except Exception as e:
        return {"error": str(e)}

@app.post("/force-ingest")
async def force_ingest(background_tasks: BackgroundTasks):
    """Force code ingestion regardless of Elasticsearch status"""
    # Check if embedding model is available
    if not check_model_availability(EMBEDDINGS_MODEL, OLLAMA_HOST):
        return {
            "status": "error", 
            "message": f"Required embedding model {EMBEDDINGS_MODEL} not available"
        }
    
    background_tasks.add_task(
        lambda: indexer.ingest_codebase(force=True)
    )
    return {"status": "ingestion_started", "message": "Forced ingestion started in background"}

# Add a query endpoint
@app.post("/query")
async def query_codebase(request: dict):
    """Query the codebase using RAG"""
    try:
        # Check if models are available
        if not check_model_availability(EMBEDDINGS_MODEL, OLLAMA_HOST):
            return {"error": f"Embedding model {EMBEDDINGS_MODEL} not available"}
        
        if not check_model_availability(LLM_MODEL, OLLAMA_HOST):
            return {"error": f"LLM model {LLM_MODEL} not available"}
        
        query = request.get("query")
        if not query:
            return {"error": "Query text is required"}
            
        k = request.get("k", 5)
        filters = request.get("filters")
        
        result = rag_engine.query(query=query, k=k, filters=filters)
        return result
    except Exception as e:
        logger.error(f"Error during query: {e}")
        return {"error": str(e)}

# If you have code like this at the bottom of your file,
# keep it as is or replace it with this updated version
if __name__ == "__main__":
    # Start auto-ingestion in background
    if AUTO_INGEST:
        import threading
        threading.Thread(target=auto_ingest_on_startup, daemon=True).start()
    
    # Start the API server
    uvicorn.run("main:app", host="0.0.0.0", port=8000, log_level="info")
