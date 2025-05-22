from uuid import uuid4
import subprocess
import asyncio
import pathlib
import logging
import json
import time
import sys
import os
import io
from typing import List, Optional, Dict, Any

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
import fastapi
from fastapi import HTTPException, Query
from pydantic import BaseModel
import numpy as np

from .rag import ollama_util
from .rag import tokenizers
from .rag import es

log = logging.getLogger(__name__)
app = fastapi.FastAPI()


# Pydantic models for requests/responses
class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    min_score: Optional[float] = None
    filters: Optional[Dict[str, Any]] = None


class EmbeddingQueryRequest(BaseModel):
    embedding: List[float]
    top_k: int = 5
    min_score: Optional[float] = None
    filters: Optional[Dict[str, Any]] = None


class CodeSearchResult(BaseModel):
    content: str
    file_path: str
    line_start: int
    line_end: int
    score: float
    kind: Optional[str] = None
    source: Optional[str] = None
    highlights: Optional[List[str]] = None


class QueryResponse(BaseModel):
    results: List[CodeSearchResult]
    total_results: int
    query_time_ms: float


@app.get("/health")
def health():
    return "green"


@app.post("/query", response_model=QueryResponse)
async def query_codebase(request: QueryRequest):
    """Query the codebase using semantic search with text input.
    
    This endpoint:
    1. Converts your text query to embeddings
    2. Performs hybrid search (vector + text matching)
    3. Returns the most relevant code snippets
    """
    start_time = time.time()
    
    try:
        # Get embeddings for the query
        log.info(f"Processing query: {request.query}")
        query_embedding = ollama_util.get_embeddings(request.query)
        
        # Perform hybrid search
        results = es.hybrid_search(
            query_text=request.query,
            query_embedding=query_embedding,
            top_k=request.top_k,
            filters=request.filters
        )
        
        # Format results
        formatted_results = []
        for result in results:
            metadata = result["metadata"]
            formatted_results.append(CodeSearchResult(
                content=result["content"],
                file_path=metadata["file_path"],
                line_start=metadata["line_start"],
                line_end=metadata["line_end"],
                score=result["score"],
                kind=metadata.get("kind"),
                source=metadata.get("source"),
                highlights=result.get("highlights", [])
            ))
        
        query_time = (time.time() - start_time) * 1000
        
        return QueryResponse(
            results=formatted_results,
            total_results=len(formatted_results),
            query_time_ms=query_time
        )
        
    except Exception as e:
        log.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query/embedding", response_model=QueryResponse)
async def query_by_embedding(request: EmbeddingQueryRequest):
    """Query the codebase using a pre-computed embedding vector.
    
    This is useful when you already have embeddings or want to find
    similar code to a specific code snippet.
    """
    start_time = time.time()
    
    try:
        # Validate embedding dimensions
        if len(request.embedding) != es.DEFAULT_EMBEDDING_DIMS:
            raise ValueError(f"Embedding must have {es.DEFAULT_EMBEDDING_DIMS} dimensions, got {len(request.embedding)}")
        
        # Perform vector search
        results = es.search_similar_code(
            query_embedding=request.embedding,
            top_k=request.top_k,
            min_score=request.min_score,
            filters=request.filters
        )
        
        # Format results
        formatted_results = []
        for result in results:
            metadata = result["metadata"]
            formatted_results.append(CodeSearchResult(
                content=result["content"],
                file_path=metadata["file_path"],
                line_start=metadata["line_start"],
                line_end=metadata["line_end"],
                score=result["score"],
                kind=metadata.get("kind"),
                source=metadata.get("source")
            ))
        
        query_time = (time.time() - start_time) * 1000
        
        return QueryResponse(
            results=formatted_results,
            total_results=len(formatted_results),
            query_time_ms=query_time
        )
        
    except Exception as e:
        log.error(f"Embedding query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/query/file/{filepath:path}", response_model=QueryResponse)
async def query_by_filepath(
    filepath: str,
    fuzzy: bool = Query(True, description="Use fuzzy matching for filepath"),
    limit: int = Query(20, description="Maximum results to return")
):
    """Query the codebase by file path pattern.
    
    Returns all indexed code chunks from files matching the given path pattern.
    """
    start_time = time.time()
    
    try:
        # Build the query based on fuzzy or exact matching
        if fuzzy:
            # Use wildcard query for fuzzy matching
            query_body = {
                "size": limit,
                "query": {
                    "wildcard": {
                        "metadata.file_path": f"*{filepath}*"
                    }
                },
                "_source": ["content", "metadata"],
                "sort": [
                    {"metadata.file_path": "asc"},
                    {"metadata.line_start": "asc"}
                ]
            }
        else:
            # Exact match
            query_body = {
                "size": limit,
                "query": {
                    "term": {
                        "metadata.file_path": filepath
                    }
                },
                "_source": ["content", "metadata"],
                "sort": [
                    {"metadata.line_start": "asc"}
                ]
            }
        
        # Execute search
        results = es.es_client.search(index=es.CODE_INDEX_NAME, body=query_body)
        
        # Format results
        formatted_results = []
        for hit in results["hits"]["hits"]:
            metadata = hit["_source"]["metadata"]
            formatted_results.append(CodeSearchResult(
                content=hit["_source"]["content"],
                file_path=metadata["file_path"],
                line_start=metadata["line_start"],
                line_end=metadata["line_end"],
                score=hit["_score"],
                kind=metadata.get("kind"),
                source=metadata.get("source")
            ))
        
        query_time = (time.time() - start_time) * 1000
        
        return QueryResponse(
            results=formatted_results,
            total_results=len(formatted_results),
            query_time_ms=query_time
        )
        
    except Exception as e:
        log.error(f"File path query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/query/fuzzy", response_model=QueryResponse)
async def query_fuzzy_text(
    text: str = Query(..., description="Text to search for"),
    fuzziness: str = Query("AUTO", description="Fuzziness level (AUTO, 0, 1, 2)"),
    top_k: int = Query(10, description="Number of results to return")
):
    """Query using fuzzy text matching (handles typos and variations).
    
    This is useful for finding code even with spelling mistakes or variations.
    """
    start_time = time.time()
    
    try:
        # Build fuzzy query
        query_body = {
            "size": top_k,
            "query": {
                "match": {
                    "content": {
                        "query": text,
                        "fuzziness": fuzziness,
                        "prefix_length": 1  # First character must match
                    }
                }
            },
            "_source": ["content", "metadata"],
            "highlight": {
                "fields": {
                    "content": {
                        "fragment_size": 150,
                        "number_of_fragments": 3
                    }
                }
            }
        }
        
        # Execute search
        results = es.es_client.search(index=es.CODE_INDEX_NAME, body=query_body)
        
        # Format results
        formatted_results = []
        for hit in results["hits"]["hits"]:
            metadata = hit["_source"]["metadata"]
            formatted_results.append(CodeSearchResult(
                content=hit["_source"]["content"],
                file_path=metadata["file_path"],
                line_start=metadata["line_start"],
                line_end=metadata["line_end"],
                score=hit["_score"],
                kind=metadata.get("kind"),
                source=metadata.get("source"),
                highlights=hit.get("highlight", {}).get("content", [])
            ))
        
        query_time = (time.time() - start_time) * 1000
        
        return QueryResponse(
            results=formatted_results,
            total_results=len(formatted_results),
            query_time_ms=query_time
        )
        
    except Exception as e:
        log.error(f"Fuzzy text query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/query/stats")
async def get_query_stats():
    """Get statistics about the indexed codebase."""
    try:
        stats = es.get_index_stats()
        return stats
    except Exception as e:
        log.exception(e)
        log.error(f"Failed to get stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/query/kinds")
async def list_token_kinds():
    """List all available token kinds (function, class, method, etc.)."""
    try:
        stats = es.get_tokenizer_stats()
        return {
            "kinds": list(stats["kinds"].keys()),
            "counts": stats["kinds"]
        }
    except Exception as e:
        log.error(f"Failed to get token kinds: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/query/by-kind/{kind}", response_model=QueryResponse)
async def query_by_token_kind(
    kind: str,
    limit: int = Query(20, description="Maximum results to return")
):
    """Query for all code snippets of a specific kind (function, class, etc.)."""
    start_time = time.time()
    
    try:
        results = es.search_by_kind(kind=kind, limit=limit)
        
        # Format results
        formatted_results = []
        for result in results:
            metadata = result["metadata"]
            formatted_results.append(CodeSearchResult(
                content=result["content"],
                file_path=metadata["file_path"],
                line_start=metadata["line_start"],
                line_end=metadata["line_end"],
                score=1.0,  # No score for term queries
                kind=metadata.get("kind"),
                source=metadata.get("source")
            ))
        
        query_time = (time.time() - start_time) * 1000
        
        return QueryResponse(
            results=formatted_results,
            total_results=len(formatted_results),
            query_time_ms=query_time
        )
        
    except Exception as e:
        log.error(f"Query by kind failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest-code-base")
def ingest_code_base(force: bool=False, background_tasks: fastapi.BackgroundTasks=None):
    runid = uuid4().hex
    background_tasks.add_task(background_rag, runid=runid, force=force)
    return dict(
        message="started background task",
        runid=runid
    )


running = False
async def background_rag(runid, force=False):
    prefix = f"({runid=}):background_rag - "
    global running
    if running:
        await asyncio.sleep(0.1)
        if running:
            log.error(f"{prefix} cannot start task. Its already running!")
            raise ValueError(f"{prefix} cannot start task. Its already running!")
    cmd = f"cd /workdir && python -u -m src.rag.run_rag --runid {runid}"
    if force:
        cmd += " --force-reload"
    running = True
    try:
        await asyncio.sleep(0.250)
        log.info(f"{prefix} running command: {cmd}")
        await asyncio.to_thread(subprocess.run, cmd, shell=True, check=True, stdout=sys.stdout, stderr=sys.stderr)
        log.info(f"{prefix} successfully finished command: {cmd}")
    except Exception as err:
        log.error(f"{prefix} rag process unexpectedly error {cmd=}!")
        log.error(err)
    finally:
        running = False



# Testing endpoints
@app.get("/test/embeddings")
def test_embeddings(text: str):
    embeddings = ollama_util.get_embeddings(text)
    return embeddings


@app.get("/test/embeddings/distance")
def test_embeddings_difference(a: str, b:str):
    a_vec = np.array(ollama_util.get_embeddings(a))
    b_vec = np.array(ollama_util.get_embeddings(b))
    dot = np.dot(a_vec, b_vec)
    norms = np.linalg.norm(a_vec) * np.linalg.norm(b_vec)
    return {
        "euclidean": np.linalg.norm(b_vec - a_vec),
        "cosine": 1 - (dot / norms),
    }


code_dir = pathlib.Path("/code")
@app.get("/test/tokenizer/{filepath:path}", response_class=fastapi.responses.PlainTextResponse)
def test_process_file(filepath: str, tokenizer: str=None):
    logging.info(f"searching for a file named: {filepath}")
    files = [
        file
        for file in code_dir.rglob(f"**/*{filepath}*")
        if file.is_file()
    ]
    if len(files) <= 0:
        logging.warning(f"file not found matching: {filepath}")
        return {}, 404
    logging.info(f"found {len(files)} matching files")
    choice = files[0]
    logging.info(f"using {choice}")
    
    limit = set([tokenizer]) if tokenizer else None
    kwargs = dict(limit=limit)
    t = time.time()
    with io.StringIO() as buf:
        for i, token in enumerate(tokenizers.tokenize(choice, **kwargs)):
            data = token.model_dump()
            data['file_path'] = str(token.file_path)
            content = data.pop('content')
            print(f"chunk={i}", file=buf)
            print(f"{json.dumps(data, indent=4)}", file=buf)
            for n, line in enumerate(content.split("\n")):
                print(f"  {n:<3} : {token.line_start + n:<3} | {line}", file=buf)
            print("\n", file=buf)
        
        log.info(f"took {time.time() - t:,.4f}s to process {filepath}")
        
        return fastapi.responses.StreamingResponse(
            iter([buf.getvalue()]),
            media_type='text/plain'
        )