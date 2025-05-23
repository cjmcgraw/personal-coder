import requests
import logging
import json
import sys
import os
from typing import Any, Dict

import fastapi
from fastapi import Request, Response
from fastapi.responses import StreamingResponse

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://ollama:11434")
RAG_URL = os.environ.get("RAG_URL", "http://rag:8000")
CODE_EXPERT_MODELS = os.environ.get("CODE_EXPERT_MODEL", "code-expert:latest,gemma3:4b").split(",")

app = fastapi.FastAPI()


@app.get("/health")
def get_health():
    return {"status": "green"}


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"])
async def proxy_to_ollama(path: str, request: Request):
    """
    Proxy all requests to Ollama, preserving method, headers, params, and body.
    """
    # Build the target URL
    target_url = f"{OLLAMA_URL}/{path}"
    
    # Get query parameters
    query_params = dict(request.query_params)
    
    # Get headers and remove host-related headers
    headers = dict(request.headers)
    headers.pop("host", None)
    headers.pop("content-length", None)  # Let requests calculate this
    
    # Get request body
    body = await request.body()
    
    # Log the passthrough
    logger.info(f"Proxying {request.method} request: {request.url.path} -> {target_url}")
    if query_params:
        logger.info(f"  Query params: {query_params}")
    if body:
        try:
            # Try to parse as JSON for pretty logging
            body_json = json.loads(body)
            logger.info(f"  Body: {json.dumps(body_json, indent=2)}")
        except:
            logger.info(f"  Body: {body[:200]}..." if len(body) > 200 else f"  Body: {body}")
    
    # Make the request to Ollama
    try:
        response = requests.request(
            method=request.method,
            url=target_url,
            headers=headers,
            params=query_params,
            data=body,
            stream=True,  # Stream for large responses
            allow_redirects=False
        )
        
        # Log response status
        logger.info(f"  Response: {response.status_code}")
        
        # Create response headers, excluding some that FastAPI will set
        response_headers = dict(response.headers)
        response_headers.pop("content-encoding", None)
        response_headers.pop("content-length", None)
        response_headers.pop("transfer-encoding", None)
        
        # Stream the response back
        def generate():
            for chunk in response.iter_content(chunk_size=8192):
                yield chunk
        
        return StreamingResponse(
            generate(),
            status_code=response.status_code,
            headers=response_headers,
            media_type=response.headers.get("content-type", "application/octet-stream")
        )
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error proxying request: {e}")
        return Response(
            content=f"Proxy error: {str(e)}",
            status_code=502,
            media_type="text/plain"
        )
