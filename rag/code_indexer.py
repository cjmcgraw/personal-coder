import os
import glob
import logging
import time
from typing import List, Dict, Any, Optional
import json
from pathlib import Path

from elasticsearch import Elasticsearch
from elasticsearch.exceptions import NotFoundError
import requests

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_elasticsearch import ElasticsearchStore

logger = logging.getLogger(__name__)

class CodebaseIndexer:
    def __init__(
        self,
        es_url: str,
        index_name: str,
        code_dir: str,
        ollama_url: str,
        embeddings_model: str
    ):
        self.es_url = es_url
        self.index_name = index_name
        self.code_dir = code_dir
        self.ollama_url = ollama_url
        self.embeddings_model = embeddings_model
        self.status = "idle"
        self.stats = {
            "processed_files": 0,
            "chunks": 0,
            "indexing_time": 0,
            "last_indexed": None
        }
        
        # Initialize Elasticsearch client
        self.es_client = Elasticsearch(es_url)
        
    def check_elasticsearch(self) -> Dict[str, Any]:
        """Check if Elasticsearch is available"""
        try:
            health = self.es_client.cluster.health()
            return {
                "available": True,
                "status": health["status"],
                "version": self.es_client.info()["version"]["number"]
            }
        except Exception as e:
            logger.error(f"Error connecting to Elasticsearch: {e}")
            return {
                "available": False,
                "error": str(e)
            }
    
    def check_index(self) -> Dict[str, Any]:
        """Check if index exists and get document count"""
        try:
            index_exists = self.es_client.indices.exists(index=self.index_name)
            if index_exists:
                stats = self.es_client.indices.stats(index=self.index_name)
                doc_count = stats["indices"][self.index_name]["total"]["docs"]["count"]
                return {
                    "exists": True,
                    "doc_count": doc_count
                }
            else:
                return {"exists": False}
        except Exception as e:
            logger.error(f"Error checking index: {e}")
            return {
                "exists": False,
                "error": str(e)
            }
    
    def create_index_if_not_exists(self) -> bool:
        """Create the Elasticsearch index with appropriate mappings if it doesn't exist"""
        try:
            if not self.es_client.indices.exists(index=self.index_name):
                # Create index with mappings for code and embeddings
                mappings = {
                    "mappings": {
                        "properties": {
                            "text": {"type": "text"},
                            "metadata": {
                                "properties": {
                                    "source": {"type": "keyword"},
                                    "file_path": {"type": "keyword"},
                                    "file_type": {"type": "keyword"},
                                    "line_start": {"type": "integer"},
                                    "line_end": {"type": "integer"},
                                    "repo": {"type": "keyword"}
                                }
                            },
                            "vector": {
                                "type": "dense_vector",
                                "dims": 384,  # Adjust based on your embedding model
                                "index": True,
                                "similarity": "cosine"
                            }
                        }
                    }
                }
                self.es_client.indices.create(index=self.index_name, body=mappings)
                logger.info(f"Created index {self.index_name} with vector search mappings")
                return True
            return True
        except Exception as e:
            logger.error(f"Error creating index: {e}")
            return False
    
    def delete_index(self) -> bool:
        """Delete the Elasticsearch index if it exists"""
        try:
            if self.es_client.indices.exists(index=self.index_name):
                self.es_client.indices.delete(index=self.index_name)
                logger.info(f"Deleted index {self.index_name}")
            return True
        except Exception as e:
            logger.error(f"Error deleting index: {e}")
            return False
    
    def should_process_file(self, file_path: str, file_extensions: List[str]) -> bool:
        """Determine if a file should be processed based on extension and other criteria"""
        _, ext = os.path.splitext(file_path)
        
        # Skip files that are too large (>1MB)
        if os.path.getsize(file_path) > 1_000_000:
            return False
            
        # Skip hidden files and directories
        if any(part.startswith('.') for part in Path(file_path).parts):
            return False
            
        # Skip common non-code directories
        ignored_dirs = ['node_modules', 'venv', '.git', '__pycache__', 'dist', 'build']
        if any(ignored_dir in Path(file_path).parts for ignored_dir in ignored_dirs):
            return False
            
        return ext in file_extensions

    def extract_code_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract metadata from the code file"""
        rel_path = os.path.relpath(file_path, self.code_dir)
        _, ext = os.path.splitext(file_path)
        
        return {
            "source": rel_path,
            "file_path": file_path,
            "file_type": ext.lstrip('.'),
            "repo": os.path.basename(os.path.abspath(self.code_dir))
        }

    def ingest_codebase(self, force: bool = False, file_extensions: List[str] = None) -> Dict[str, Any]:
        """Index the entire codebase into Elasticsearch"""
        if file_extensions is None:
            file_extensions = ['.py', '.js', '.ts', '.java', '.c', '.cpp', '.go', '.rs', '.sh', '.jsx', '.tsx']
            
        start_time = time.time()
        self.status = "indexing"
        self.stats = {
            "processed_files": 0,
            "chunks": 0,
            "indexing_time": 0,
            "last_indexed": None
        }
        
        try:
            # Check if Elasticsearch is available
            es_status = self.check_elasticsearch()
            if not es_status["available"]:
                self.status = "failed"
                return {
                    "status": "failed",
                    "error": f"Elasticsearch not available: {es_status.get('error')}"
                }
            
            # Delete index if it exists and force is true
            if force:
                self.delete_index()
                
            # Create index if it doesn't exist
            if not self.create_index_if_not_exists():
                self.status = "failed"
                return {
                    "status": "failed",
                    "error": "Failed to create Elasticsearch index"
                }
                
            # Initialize embeddings
            embeddings = OllamaEmbeddings(
                model=self.embeddings_model,
                base_url=self.ollama_url
            )
            
            # Initialize vector store
            vector_store = ElasticsearchStore(
                es_url=self.es_url,
                index_name=self.index_name,
                embedding=embeddings
            )
            
            # Set up text splitter for code
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", "class ", "def ", "function ", "app.", "router.", "if ", "for "]
            )
            
            # Process all code files
            documents = []
            for ext in file_extensions:
                files = glob.glob(f"{self.code_dir}/**/*{ext}", recursive=True)
                for file_path in files:
                    if self.should_process_file(file_path, file_extensions):
                        try:
                            # Load the file
                            loader = TextLoader(file_path, encoding='utf-8')
                            file_docs = loader.load()
                            
                            # Add metadata
                            metadata = self.extract_code_metadata(file_path)
                            for doc in file_docs:
                                doc.metadata.update(metadata)
                                
                                # Count lines for positioning context
                                line_count = len(doc.page_content.splitlines())
                                doc.metadata["line_start"] = 1
                                doc.metadata["line_end"] = line_count
                            
                            # Split the document
                            splits = text_splitter.split_documents(file_docs)
                            
                            # Update line numbers in metadata for each chunk
                            for i, split in enumerate(splits):
                                text_lines = split.page_content.splitlines()
                                if i > 0 and "line_start" in splits[i-1].metadata:
                                    prev_end = splits[i-1].metadata["line_end"]
                                    split.metadata["line_start"] = prev_end
                                    split.metadata["line_end"] = prev_end + len(text_lines)
                                
                            documents.extend(splits)
                            self.stats["processed_files"] += 1
                            
                            # Log progress periodically
                            if self.stats["processed_files"] % 20 == 0:
                                logger.info(f"Processed {self.stats['processed_files']} files so far...")
                                
                        except Exception as e:
                            logger.error(f"Error processing {file_path}: {e}")
            
            # Log total files found
            logger.info(f"Found {len(documents)} document chunks from {self.stats['processed_files']} files")
            self.stats["chunks"] = len(documents)
            
            # Store documents in batches to avoid timeouts
            batch_size = 100
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i+batch_size]
                vector_store.add_documents(batch)
                logger.info(f"Indexed batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}")
            
            # Update status and stats
            self.status = "completed"
            self.stats["indexing_time"] = time.time() - start_time
            self.stats["last_indexed"] = time.strftime("%Y-%m-%d %H:%M:%S")
            
            logger.info(f"Indexing completed in {self.stats['indexing_time']:.2f} seconds")
            logger.info(f"Processed {self.stats['processed_files']} files into {self.stats['chunks']} chunks")
            
            return {
                "status": self.status,
                "stats": self.stats
            }
            
        except Exception as e:
            self.status = "failed"
            logger.error(f"Indexing failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "stats": self.stats
            }