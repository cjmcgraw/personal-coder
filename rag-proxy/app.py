import os
import json
import logging
import requests
from flask import Flask, request, Response, jsonify, stream_with_context

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://ollama:11434")
RAG_URL = os.environ.get("RAG_URL", "http://rag:8000")
CODE_EXPERT_MODELS = os.environ.get("CODE_EXPERT_MODEL", "code-expert:latest,gemma3:4b").split(",")

@app.route('/api/<path:subpath>', methods=['GET', 'POST', 'PUT', 'DELETE'])
def ollama_proxy(subpath):
    """Handle paths with /ollama prefix that OpenWebUI uses"""
    url = f"{OLLAMA_URL}/api/{subpath}"
    logger.info(f"Proxying /ollama request to {url}")
    
    try:
        # Forward the request to Ollama
        resp = requests.request(
            method=request.method,
            url=url,
            headers={key: value for key, value in request.headers if key != 'Host'},
            data=request.get_data(),
            cookies=request.cookies,
            params=request.args,
            stream=True
        )
        
        return Response(
            stream_with_context(resp.iter_content(chunk_size=1024)),
            status=resp.status_code,
            content_type=resp.headers.get('Content-Type', 'application/json')
        )
    except Exception as e:
        logger.error(f"Error proxying /ollama request: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Special case for generate with RAG enhancement
@app.route('/api/generate', methods=['POST'])
def ollama_generate():
    """Handle /ollama/api/generate with RAG enhancement"""
    try:
        data = request.json
        model = data.get('model', '')
        prompt = data.get('prompt', '')
        
        logger.info(f"Processing /ollama/api/generate for model {model}")
        
        # Only enhance RAG-enabled models
        print(model)
        if model in CODE_EXPERT_MODELS:
            logger.info(f"Enhancing prompt with RAG for model {model}")
            # Query RAG for code context
            try:
                rag_response = requests.post(
                    f"{RAG_URL}/query",
                    json={"query": prompt},
                    timeout=30
                )
                
                if rag_response.status_code == 200:
                    rag_data = rag_response.json()
                    sources = rag_data.get("sources", [])
                    
                    if sources:
                        # Format code context
                        context = "RELEVANT CODE FROM THE CODEBASE:\n\n"
                        for i, source in enumerate(sources, 1):
                            file_path = source.get('file', 'unknown')
                            content = source.get('content', '')
                            context += f"File: {file_path}\n```\n{content}\n```\n\n"
                        
                        # Create enhanced prompt with code context
                        enhanced_prompt = f"""Here are relevant code snippets from the codebase:

{context}

Using the code snippets above, please answer this question:
{prompt}
"""
                        # Update prompt
                        data['prompt'] = enhanced_prompt
                        logger.info(f"Enhanced prompt with {len(sources)} code snippets")
                    else:
                        logger.info("No relevant code snippets found")
                else:
                    logger.warning(f"RAG query failed: {rag_response.text}")
            except Exception as e:
                logger.error(f"Error querying RAG: {str(e)}")
        
        # Forward to Ollama
        logger.info(f"Forwarding to Ollama: {OLLAMA_URL}/api/generate")
        
        # Handle streaming properly
        ollama_response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json=data,
            stream=True
        )
        
        return Response(
            stream_with_context(ollama_response.iter_content(chunk_size=1024)),
            status=ollama_response.status_code,
            content_type=ollama_response.headers.get('Content-Type', 'application/json')
        )
            
    except Exception as e:
        logger.error(f"Error processing /ollama/api/generate: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Catch-all handler for direct API endpoints (without /ollama prefix)
@app.route('/api/<path:subpath>', methods=['GET', 'POST', 'PUT', 'DELETE'])
def direct_api_proxy(subpath):
    url = f"{OLLAMA_URL}/api/{subpath}"
    logger.info(f"Proxying direct API request to {url}")
    
    try:
        resp = requests.request(
            method=request.method,
            url=url,
            headers={key: value for key, value in request.headers if key != 'Host'},
            data=request.get_data(),
            cookies=request.cookies,
            params=request.args,
            stream=True
        )
        
        return Response(
            stream_with_context(resp.iter_content(chunk_size=1024)),
            status=resp.status_code,
            content_type=resp.headers.get('Content-Type', 'application/json')
        )
    except Exception as e:
        logger.error(f"Error proxying direct API request: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Debug endpoint
@app.route('/debug', methods=['GET'])
def debug():
    """Debug endpoint to test connectivity"""
    try:
        # Test RAG
        try:
            rag_status = requests.get(f"{RAG_URL}/status", timeout=5).json()
        except Exception as e:
            rag_status = {"error": str(e)}
        
        # Test Ollama
        try:
            ollama_models = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5).json()
        except Exception as e:
            ollama_models = {"error": str(e)}
        
        # Test proxy routes
        routes = [str(rule) for rule in app.url_map.iter_rules()]
        
        return jsonify({
            "rag_status": rag_status,
            "ollama_models": ollama_models,
            "configured_rag_models": CODE_EXPERT_MODELS,
            "proxy_routes": routes,
            "env": {
                "OLLAMA_URL": OLLAMA_URL,
                "RAG_URL": RAG_URL
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Root path
@app.route('/')
def index():
    return jsonify({
        "status": "running",
        "endpoints": {
            "/ollama/api/*": "OpenWebUI compatible Ollama API endpoints",
            "/api/*": "Direct Ollama API endpoints",
            "/debug": "Debug endpoint for connectivity testing"
        }
    })

@app.route('/test-rag', methods=['GET'])
def test_rag():
    """Test the RAG integration with a simple query"""
    query = request.args.get('query', 'Show me the main.py file')
    
    try:
        # Call RAG directly
        rag_response = requests.post(
            f"{RAG_URL}/query",
            json={"query": query},
            timeout=30
        )
        
        # Return RAG response along with connection info
        return jsonify({
            "rag_url": RAG_URL,
            "query": query,
            "rag_status_code": rag_response.status_code,
            "rag_response": rag_response.json() if rag_response.status_code == 200 else None,
            "has_snippets": len(rag_response.json().get("sources", [])) > 0 if rag_response.status_code == 200 else False
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    logger.info(f"Starting proxy server - Ollama URL: {OLLAMA_URL}, RAG URL: {RAG_URL}")
    logger.info(f"RAG-enabled models: {CODE_EXPERT_MODELS}")
    app.run(host='0.0.0.0', port=3000)
