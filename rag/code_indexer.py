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

import traceback
import sys

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
        # Get embedding dimensions from model
        try:
            embedding_dims = self.get_embedding_dimensions()
            # Use this in the mapping
            mappings = {
                "mappings": {
                    "properties": {
                        # ... other fields
                        "vector": {
                            "type": "dense_vector",
                            "dims": embedding_dims,  # Dynamic based on model
                            "index": True,
                            "similarity": "cosine"
                        }
                    }
                }
            }
            # ... rest of the function
        except Exception as e:
            logger.error(f"Error creating index: {e}")
            return False

    def get_embedding_dimensions(self) -> int:
        """Determine embedding dimensions by testing the model"""
        try:
            logger.debug(f"Getting embedding dimensions from model: {self.embeddings_model}")
            logger.debug(f"Using Ollama URL: {self.ollama_url}")

            # Create an embedding for a simple test string
            response = requests.post(
                f"{self.ollama_url}/api/embeddings",
                json={"model": self.embeddings_model, "prompt": "test"}
            )

            logger.debug(f"Embedding API status code: {response.status_code}")

            if response.status_code == 200:
                embedding = response.json().get("embedding", [])
                logger.debug(f"Embedding dimension: {len(embedding)}")
                return len(embedding)
            else:
                logger.warning(f"Error from embedding API: {response.text}")
                # Default to 384 if we can't determine
                logger.warning(f"Could not determine embedding dimensions, using default (384)")
                return 384
        except Exception as e:
            logger.error(f"Error determining embedding dimensions: {e}")
            logger.error(traceback.format_exc())
            return 384  # Default fallback

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

        # Debug logs for file checking
        logger.debug(f"Checking file: {file_path}, extension: {ext}")

        # Skip files that are too large (>1MB)
        file_size = os.path.getsize(file_path)
        if file_size > 1_000_000:
            logger.debug(f"Skipping large file: {file_path} ({file_size} bytes)")
            return False

        # Skip hidden files and directories
        if any(part.startswith('.') for part in Path(file_path).parts):
            logger.debug(f"Skipping hidden path: {file_path}")
            return False

        # Skip common non-code directories
        ignored_dirs = ['node_modules', 'venv', '.git', '__pycache__', 'dist', 'build']
        if any(ignored_dir in Path(file_path).parts for ignored_dir in ignored_dirs):
            logger.debug(f"Skipping ignored directory: {file_path}")
            return False

        result = ext in file_extensions
        logger.debug(f"File {file_path} will be processed: {result}")
        return result

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
            # Add comprehensive debug logs
            logger.debug(f"Starting codebase ingestion with params: force={force}, extensions={file_extensions}")
            logger.debug(f"Elasticsearch URL: {self.es_url}, Index: {self.index_name}")
            logger.debug(f"Code directory: {self.code_dir}")
            logger.debug(f"Ollama URL: {self.ollama_url}, Embeddings model: {self.embeddings_model}")

            # Validate code directory
            import os
            if not os.path.exists(self.code_dir):
                error_msg = f"Code directory {self.code_dir} does not exist!"
                logger.error(error_msg)
                self.status = "failed"
                return {"status": "failed", "error": error_msg}

            if not os.path.isdir(self.code_dir):
                error_msg = f"{self.code_dir} is not a directory!"
                logger.error(error_msg)
                self.status = "failed"
                return {"status": "failed", "error": error_msg}

            # Log directory contents for debugging
            try:
                dir_contents = os.listdir(self.code_dir)
                logger.debug(f"Code directory contents: {dir_contents}")
                logger.debug(f"Total files/folders in directory: {len(dir_contents)}")
            except Exception as e:
                logger.error(f"Error reading directory contents: {e}")

            # Check if Elasticsearch is available
            es_status = self.check_elasticsearch()
            logger.debug(f"Elasticsearch status: {es_status}")
            if not es_status["available"]:
                self.status = "failed"
                return {
                    "status": "failed",
                    "error": f"Elasticsearch not available: {es_status.get('error')}"
                }

            # Delete index if it exists and force is true
            if force:
                logger.info("Force flag is True, deleting existing index if present")
                deleted = self.delete_index()
                logger.debug(f"Index deletion result: {deleted}")

            # Create index if it doesn't exist
            logger.debug("Attempting to create index if it doesn't exist")
            if not self.create_index_if_not_exists():
                self.status = "failed"
                return {
                    "status": "failed",
                    "error": "Failed to create Elasticsearch index"
                }
            logger.debug("Index creation check completed")

            # Direct test of Ollama API before using langchain
            logger.debug(f"Testing Ollama API directly for embeddings with model: {self.embeddings_model}")
            try:
                import requests
                import json

                test_url = f"{self.ollama_url}/api/embeddings"
                test_payload = {"model": self.embeddings_model, "prompt": "test"}

                logger.debug(f"Sending request to: {test_url}")
                logger.debug(f"With payload: {test_payload}")

                test_response = requests.post(
                    test_url,
                    json=test_payload,
                    timeout=20
                )

                logger.debug(f"Ollama embeddings API response status: {test_response.status_code}")

                if test_response.status_code != 200:
                    error_msg = f"Ollama embeddings API returned {test_response.status_code}"
                    try:
                        error_msg += f": {test_response.text}"
                    except:
                        pass
                    raise Exception(error_msg)

                # Make sure we got a valid JSON response
                try:
                    resp_json = test_response.json()
                    logger.debug(f"Ollama API response type: {type(resp_json)}")

                    if not resp_json:
                        raise Exception("Ollama API returned empty JSON response")

                    if not isinstance(resp_json, dict):
                        raise Exception(f"Ollama API returned unexpected response type: {type(resp_json)}")

                    if "embedding" not in resp_json:
                        logger.debug(f"Response keys: {resp_json.keys()}")
                        raise Exception(f"Ollama API response does not contain 'embedding' field: {json.dumps(resp_json)[:200]}")

                    embedding_dim = len(resp_json["embedding"])
                    logger.debug(f"Embedding test successful, dimension: {embedding_dim}")
                except json.JSONDecodeError as je:
                    logger.error(f"Failed to parse Ollama API response as JSON: {je}")
                    raise Exception(f"Invalid JSON response from Ollama: {test_response.text[:200]}")
            except Exception as e:
                logger.error(f"Direct Ollama API test failed: {e}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                self.status = "failed"
                return {
                    "status": "failed",
                    "error": f"Failed to communicate with Ollama embeddings API: {str(e)}",
                    "stats": self.stats
                }

            # Initialize embeddings
            logger.debug(f"Initializing langchain OllamaEmbeddings with model: {self.embeddings_model}")
            try:
                embeddings = OllamaEmbeddings(
                    model=self.embeddings_model,
                    base_url=self.ollama_url
                )
                logger.debug("OllamaEmbeddings model initialized successfully")

                # Test embedding to verify it works with langchain
                logger.debug("Testing langchain embedding with a sample query")
                test_embedding = embeddings.embed_query("Test embedding with langchain")

                if test_embedding is None:
                    raise Exception("Langchain embedding test returned None")

                logger.debug(f"Langchain test embedding created successfully, dimension: {len(test_embedding)}")
            except Exception as e:
                logger.error(f"Error initializing langchain embeddings: {e}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                self.status = "failed"
                return {
                    "status": "failed",
                    "error": f"Failed to initialize langchain embeddings: {str(e)}",
                    "stats": self.stats
                }

            # Initialize vector store
            logger.debug("Initializing ElasticsearchStore")
            try:
                vector_store = ElasticsearchStore(
                    es_url=self.es_url,
                    index_name=self.index_name,
                    embedding=embeddings
                )
                logger.debug("ElasticsearchStore initialized successfully")
            except Exception as e:
                logger.error(f"Error initializing vector store: {e}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                self.status = "failed"
                return {
                    "status": "failed",
                    "error": f"Failed to initialize vector store: {str(e)}",
                    "stats": self.stats
                }

            # Set up text splitter for code
            logger.debug("Setting up text splitter")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", "class ", "def ", "function ", "app.", "router.", "if ", "for "]
            )

            # Process all code files
            documents = []
            for ext in file_extensions:
                logger.info(f"Searching for files with extension: {ext}")
                files = glob.glob(f"{self.code_dir}/**/*{ext}", recursive=True)
                logger.info(f"Found {len(files)} files with extension {ext}")

                for file_path in files:
                    logger.debug(f"Examining file: {file_path}")

                    if self.should_process_file(file_path, file_extensions):
                        try:
                            # Log file details
                            file_size = os.path.getsize(file_path)
                            logger.debug(f"Processing file: {file_path} ({file_size} bytes)")

                            # Load the file
                            try:
                                loader = TextLoader(file_path, encoding='utf-8')
                                file_docs = loader.load()
                                logger.debug(f"Loaded file {file_path} - {len(file_docs)} documents, total content length: {sum(len(doc.page_content) for doc in file_docs)}")
                            except UnicodeDecodeError:
                                logger.warning(f"Unicode decode error in {file_path}, trying with 'latin1' encoding")
                                loader = TextLoader(file_path, encoding='latin1')
                                file_docs = loader.load()

                            # Add metadata
                            metadata = self.extract_code_metadata(file_path)
                            logger.debug(f"Extracted metadata: {metadata}")

                            for doc in file_docs:
                                doc.metadata.update(metadata)

                                # Count lines for positioning context
                                line_count = len(doc.page_content.splitlines())
                                doc.metadata["line_start"] = 1
                                doc.metadata["line_end"] = line_count

                            # Split the document
                            logger.debug(f"Splitting document into chunks")
                            splits = text_splitter.split_documents(file_docs)
                            logger.debug(f"Created {len(splits)} chunks from file")

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
                            import traceback
                            logger.error(f"Traceback: {traceback.format_exc()}")
                            # Continue with next file instead of failing completely
                            continue

            # Log total files found
            logger.info(f"Found {len(documents)} document chunks from {self.stats['processed_files']} files")

            if len(documents) == 0:
                logger.warning("No documents were processed! Check file extensions and filters.")
                self.status = "completed"  # Still mark as completed, just with 0 docs
                self.stats["chunks"] = 0
                self.stats["indexing_time"] = time.time() - start_time
                self.stats["last_indexed"] = time.strftime("%Y-%m-%d %H:%M:%S")

                return {
                    "status": self.status,
                    "stats": self.stats,
                    "warning": "No documents were processed"
                }

            self.stats["chunks"] = len(documents)

            # Store documents in batches to avoid timeouts
            batch_size = 100
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i+batch_size]
                try:
                    logger.info(f"Indexing batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1} ({len(batch)} documents)")
                    vector_store.add_documents(batch)
                    logger.debug(f"Successfully indexed batch {i//batch_size + 1}")
                except Exception as e:
                    logger.error(f"Error indexing batch {i//batch_size + 1}: {e}")
                    import traceback
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    # Try with a smaller batch size if the batch is too large
                    if len(batch) > 10:
                        logger.info("Trying with a smaller batch size")
                        small_batch_size = max(1, len(batch) // 5)
                        for j in range(0, len(batch), small_batch_size):
                            small_batch = batch[j:j+small_batch_size]
                            try:
                                logger.info(f"Indexing small batch {j//small_batch_size + 1} ({len(small_batch)} documents)")
                                vector_store.add_documents(small_batch)
                            except Exception as small_e:
                                logger.error(f"Error indexing small batch: {small_e}")
                                # Continue with next small batch

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
            import traceback
            self.status = "failed"
            logger.error(f"Indexing failed with exception: {str(e)}")
            logger.error(f"Exception type: {type(e).__name__}")
            logger.error(f"Traceback: {traceback.format_exc()}")

            # Check for specific common errors
            if "ConnectionError" in str(e):
                logger.error("Connection error detected. Check Elasticsearch is running and accessible.")
            elif "AuthenticationException" in str(e) or "AuthorizationException" in str(e):
                logger.error("Authentication error with Elasticsearch. Check credentials.")
            elif "No such file or directory" in str(e):
                logger.error("File system error detected. Check path and permissions.")
            elif "MemoryError" in str(e):
                logger.error("Memory error. The process may need more memory.")
            elif "TimeoutError" in str(e):
                logger.error("Timeout error. Elasticsearch may be overloaded or unreachable.")
            elif "NoneType" in str(e):
                logger.error("NoneType error. A response or object may be unexpectedly None.")

            return {
                "status": "failed",
                "error": str(e),
                "traceback": traceback.format_exc(),
                "stats": self.stats
            }