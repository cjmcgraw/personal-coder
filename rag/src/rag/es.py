import itertools as its
import logging
import time
from typing import List, Dict, Optional, Any, Iterator
import os

import elasticsearch
import elasticsearch.helpers

from . import CodeDocument

log = logging.getLogger(__name__)

# Constants
CODE_INDEX_NAME = "code"
DEFAULT_EMBEDDING_DIMS = 384
DEFAULT_BATCH_SIZE = 500

# Elasticsearch connection
ES_HOST = os.environ.get("ES_HOST", "http://elasticsearch:9200")
es_client = elasticsearch.Elasticsearch(hosts=ES_HOST)

# Index mapping configuration - includes all Token fields
DEFAULT_MAPPING = {
    "mappings": {
        "properties": {
            "content": {
                "type": "text",
                "analyzer": "standard",
                "fields": {
                    "keyword": {
                        "type": "keyword",
                        "ignore_above": 10000
                    }
                }
            },
            "metadata": {
                "properties": {
                    "file_path": {"type": "keyword"},
                    "file_type": {"type": "keyword"},
                    "repo": {"type": "keyword"},
                    "line_start": {"type": "integer"},
                    "line_end": {"type": "integer"},
                    "source": {"type": "keyword"},  # tokenizer source (tree-sitter, regex, etc.)
                    "kind": {"type": "keyword"},      # token kind (function, class, method, etc.)
                    "indexed_at": {"type": "date", "format": "epoch_second"}
                }
            },
            "vector": {
                "type": "dense_vector",
                "dims": DEFAULT_EMBEDDING_DIMS,
                "index": True,
                "similarity": "cosine"
            }
        }
    }
}


def ensure_index_exists():
    """Ensure the code index exists with proper mapping."""
    if not es_client.indices.exists(index=CODE_INDEX_NAME):
        log.warning(f"Index '{CODE_INDEX_NAME}' not found. Creating...")
        es_client.indices.create(index=CODE_INDEX_NAME, body=DEFAULT_MAPPING)
        log.info(f"Created index '{CODE_INDEX_NAME}'")


class BulkIndexer:
    """Context manager for bulk indexing code documents.
    
    Usage:
        with BulkIndexer() as indexer:
            for doc in documents:
                indexer.add(doc)
        
        # Or without context manager:
        indexer = BulkIndexer()
        for doc in documents:
            indexer.add(doc)
        indexer.close()  # Don't forget to close!
    """
    
    def __init__(self, batch_size: int = DEFAULT_BATCH_SIZE):
        self.batch_size = batch_size
        self.buffer: List[CodeDocument] = []
        self.stats = {
            "total": 0,
            "batches": 0,
            "start_time": time.time(),
            "errors": 0
        }
        ensure_index_exists()
    
    def add(self, document: CodeDocument) -> None:
        """Add a document to the buffer and flush if needed."""
        self.buffer.append(document)
        if len(self.buffer) >= self.batch_size:
            self.flush()
    
    def flush(self) -> Dict[str, Any]:
        """Flush the current buffer to Elasticsearch.
        
        Returns:
            Dict with success count and any failures
        """
        if not self.buffer:
            return {"success": 0, "failed": []}
        
        batch_start = time.time()
        operations = []
        
        for record in self.buffer:
            operations.append({
                "_index": CODE_INDEX_NAME,
                "_source": {
                    "content": record.content,
                    "vector": record.embedding,
                    "metadata": {
                        "file_path": str(record.file_path),
                        "file_type": record.file_type,
                        "repo": record.repo,
                        "line_start": record.line_start,
                        "line_end": record.line_end,
                        "source": record.source,
                        "kind": getattr(record, 'kind', ''),
                        "indexed_at": int(time.time())
                    }
                }
            })
        
        try:
            success, failed = elasticsearch.helpers.bulk(
                es_client,
                operations,
                stats_only=False,
                raise_on_error=False  # Don't raise, return errors
            )
            
            batch_time = time.time() - batch_start
            self.stats["batches"] += 1
            self.stats["total"] += success
            
            if failed:
                self.stats["errors"] += len(failed)
                log.error(f"Failed to index {len(failed)} documents in batch {self.stats['batches']}")
                for error in failed[:5]:  # Log first 5 errors
                    log.error(f"Error: {error}")
            
            log.debug(
                f"Indexed batch {self.stats['batches']}: "
                f"{success}/{len(self.buffer)} docs in {batch_time:.2f}s"
            )
            
            result = {"success": success, "failed": failed}
            
        except Exception as e:
            log.error(f"Bulk indexing failed: {e}")
            self.stats["errors"] += len(self.buffer)
            result = {"success": 0, "failed": [str(e)]}
        
        finally:
            self.buffer.clear()
        
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current indexing statistics."""
        elapsed = time.time() - self.stats["start_time"]
        return {
            **self.stats,
            "elapsed_time": elapsed,
            "docs_per_second": self.stats["total"] / elapsed if elapsed > 0 else 0,
            "buffer_size": len(self.buffer)
        }
    
    def close(self) -> Dict[str, Any]:
        """Flush remaining documents and return final statistics."""
        self.flush()
        stats = self.get_stats()
        
        log.info(
            f"Bulk indexing complete: {stats['total']} docs "
            f"in {stats['batches']} batches, {stats['elapsed_time']:.2f}s total "
            f"({stats['docs_per_second']:.1f} docs/s)"
        )
        
        if stats['errors'] > 0:
            log.warning(f"Encountered {stats['errors']} errors during indexing")
        
        return stats
    
    def __enter__(self) -> 'BulkIndexer':
        """Enter the context manager."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the context manager, ensuring all documents are flushed."""
        self.close()
    
    def __len__(self) -> int:
        """Return the current buffer size."""
        return len(self.buffer)


def index_documents(
    documents: Iterator[CodeDocument], 
    batch_size: int = DEFAULT_BATCH_SIZE,
    progress_callback: Optional[callable] = None
) -> Dict[str, Any]:
    """Index documents using the bulk indexer.
    
    Args:
        documents: Iterator of documents to index
        batch_size: Number of documents per batch
        progress_callback: Optional callback function(stats) called after each batch
        
    Returns:
        Final indexing statistics
    """
    with BulkIndexer(batch_size) as indexer:
        for doc in documents:
            indexer.add(doc)
            
            # Call progress callback if provided
            if progress_callback and indexer.stats["total"] % batch_size == 0:
                progress_callback(indexer.get_stats())
        
        return indexer.get_stats()


def search_similar_code(
    query_embedding: List[float],
    top_k: int = 5,
    min_score: Optional[float] = None,
    filters: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """Search for similar code snippets using vector similarity.
    
    Args:
        query_embedding: The embedding vector for the query
        top_k: Number of results to return
        min_score: Minimum similarity score threshold
        filters: Additional filters (e.g., {"kind": "function", "file_type": "py"})
        
    Returns:
        List of matching documents with scores
    """
    ensure_index_exists()
    
    # Build the base query
    base_query = {"match_all": {}}
    
    # Add filters if provided
    if filters:
        filter_clauses = []
        for field, value in filters.items():
            filter_clauses.append({"term": {f"metadata.{field}": value}})
        
        base_query = {
            "bool": {
                "must": [{"match_all": {}}],
                "filter": filter_clauses
            }
        }
    
    search_body = {
        "size": top_k,
        "query": {
            "script_score": {
                "query": base_query,
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'vector') + 1.0",
                    "params": {"query_vector": query_embedding}
                }
            }
        },
        "_source": ["content", "metadata"]
    }
    
    if min_score:
        search_body["min_score"] = min_score
    
    results = es_client.search(index=CODE_INDEX_NAME, body=search_body)
    
    return [
        {
            "content": hit["_source"]["content"],
            "metadata": hit["_source"]["metadata"],
            "score": hit["_score"]
        }
        for hit in results["hits"]["hits"]
    ]


def hybrid_search(
    query_text: str,
    query_embedding: List[float],
    top_k: int = 5,
    vector_weight: float = 0.7,
    text_weight: float = 0.3,
    filters: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """Perform hybrid search combining vector and text similarity.
    
    Args:
        query_text: The text query
        query_embedding: The embedding vector for the query
        top_k: Number of results to return
        vector_weight: Weight for vector similarity (0-1)
        text_weight: Weight for text similarity (0-1)
        filters: Additional filters (e.g., {"kind": "function", "source": "tree-sitter"})
        
    Returns:
        List of matching documents with combined scores
    """
    ensure_index_exists()
    
    # Build the should clauses
    should_clauses = [
        {
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    "source": f"""
                        double vector_score = cosineSimilarity(params.query_vector, 'vector') + 1.0;
                        return vector_score * {vector_weight};
                    """,
                    "params": {"query_vector": query_embedding}
                }
            }
        },
        {
            "match": {
                "content": {
                    "query": query_text,
                    "boost": text_weight
                }
            }
        }
    ]
    
    # Build the query with optional filters
    query = {"bool": {"should": should_clauses}}
    
    if filters:
        filter_clauses = []
        for field, value in filters.items():
            filter_clauses.append({"term": {f"metadata.{field}": value}})
        query["bool"]["filter"] = filter_clauses
    
    search_body = {
        "size": top_k,
        "query": query,
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
    
    results = es_client.search(index=CODE_INDEX_NAME, body=search_body)
    
    return [
        {
            "content": hit["_source"]["content"],
            "metadata": hit["_source"]["metadata"],
            "score": hit["_score"],
            "highlights": hit.get("highlight", {}).get("content", [])
        }
        for hit in results["hits"]["hits"]
    ]


def search_by_kind(kind: str, limit: int = 20) -> List[Dict[str, Any]]:
    """Search for code snippets by their kind (e.g., 'function', 'class', 'method').
    
    Args:
        kind: The kind of code snippet to search for
        limit: Maximum number of results to return
        
    Returns:
        List of matching documents
    """
    ensure_index_exists()
    
    search_body = {
        "size": limit,
        "query": {
            "term": {"metadata.kind": kind}
        },
        "_source": ["content", "metadata"],
        "sort": [
            {"metadata.file_path": "asc"},
            {"metadata.line_start": "asc"}
        ]
    }
    
    results = es_client.search(index=CODE_INDEX_NAME, body=search_body)
    
    return [
        {
            "content": hit["_source"]["content"],
            "metadata": hit["_source"]["metadata"]
        }
        for hit in results["hits"]["hits"]
    ]


def get_tokenizer_stats() -> Dict[str, int]:
    """Get statistics about tokenizers used in the index."""
    ensure_index_exists()
    
    agg_body = {
        "size": 0,
        "aggs": {
            "tokenizers": {
                "terms": {
                    "field": "metadata.source",
                    "size": 50
                }
            },
            "kinds": {
                "terms": {
                    "field": "metadata.kind",
                    "size": 100
                }
            },
            "file_types": {
                "terms": {
                    "field": "metadata.file_type",
                    "size": 50
                }
            }
        }
    }
    
    results = es_client.search(index=CODE_INDEX_NAME, body=agg_body)
    
    return {
        "tokenizers": {
            bucket["key"]: bucket["doc_count"]
            for bucket in results["aggregations"]["tokenizers"]["buckets"]
        },
        "kinds": {
            bucket["key"]: bucket["doc_count"]
            for bucket in results["aggregations"]["kinds"]["buckets"]
        },
        "file_types": {
            bucket["key"]: bucket["doc_count"]
            for bucket in results["aggregations"]["file_types"]["buckets"]
        }
    }


def delete_by_file_path(file_path: str) -> int:
    """Delete all documents for a given file path.
    
    Args:
        file_path: The file path to delete documents for
        
    Returns:
        Number of documents deleted
    """
    ensure_index_exists()
    
    response = es_client.delete_by_query(
        index=CODE_INDEX_NAME,
        body={
            "query": {
                "term": {"metadata.file_path": file_path}
            }
        }
    )
    
    deleted = response.get("deleted", 0)
    log.info(f"Deleted {deleted} documents for file: {file_path}")
    return deleted


def get_index_stats() -> Dict[str, Any]:
    """Get statistics about the code index."""
    ensure_index_exists()
    
    stats = es_client.indices.stats(index=CODE_INDEX_NAME)
    count = es_client.count(index=CODE_INDEX_NAME)
    tokenizer_stats = get_tokenizer_stats()
    
    # Get file count
    file_count_agg = es_client.search(
        index=CODE_INDEX_NAME,
        body={
            "size": 0,
            "aggs": {
                "unique_files": {
                    "cardinality": {
                        "field": "metadata.file_path"
                    }
                }
            }
        }
    )

    return {
        "total_documents": count["count"],
        "unique_files": file_count_agg["aggregations"]["unique_files"]["value"],
        "index_size": stats["indices"][CODE_INDEX_NAME]["total"]["store"]["size_in_bytes"],
        "index_size_human": f"{stats["indices"][CODE_INDEX_NAME]["total"]["store"]["size_in_bytes"] * 1e-6:,.2f} mb",
        "stats_by_tokenizer": tokenizer_stats
    }


def clear_index() -> bool:
    """Delete and recreate the index. USE WITH CAUTION!"""
    try:
        if es_client.indices.exists(index=CODE_INDEX_NAME):
            es_client.indices.delete(index=CODE_INDEX_NAME)
            log.warning(f"Deleted index '{CODE_INDEX_NAME}'")
        
        ensure_index_exists()
        return True
    except Exception as e:
        log.error(f"Error clearing index: {e}")
        return False


# Backward compatibility function
def bulk_index_code(data: List[CodeDocument], batch_size: int = DEFAULT_BATCH_SIZE):
    """Legacy bulk index function for backward compatibility."""
    log.warning(f"Beginning insert of {len(data)} records")
    start = time.time()
    
    stats = index_documents(iter(data), batch_size)
    
    elapsed = time.time() - start
    log.warning(f"Finished bulk insert of {len(data)} records in {elapsed:.2f}s")