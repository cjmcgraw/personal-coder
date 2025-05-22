import logging
from typing import Dict, Any, List, Optional
import time

from elasticsearch import Elasticsearch
from langchain_community.embeddings import OllamaEmbeddings
from langchain_elasticsearch import ElasticsearchStore
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

logger = logging.getLogger(__name__)

class CodeRAGEngine:
    def __init__(
        self,
        es_url: str,
        index_name: str,
        ollama_url: str,
        embeddings_model: str,
        llm_model: str
    ):
        self.es_url = es_url
        self.index_name = index_name
        self.ollama_url = ollama_url
        self.embeddings_model = embeddings_model
        self.llm_model = llm_model
        
        # Initialize Elasticsearch client
        self.es_client = Elasticsearch(es_url)
        
    def get_prompt_template(self) -> str:
        """Get the RAG prompt template for code questions"""
        return """
        You are an expert software developer assistant with deep knowledge of this codebase.
        Use the following code snippets to answer the question.

        RELEVANT CODE SNIPPETS:
        {context}

        USER QUESTION: {question}

        When answering:
        1. If you reference code files, include their paths
        2. Focus on the specific code above, not general programming knowledge
        3. If you're uncertain, say so rather than guessing
        4. Show relevant code snippets when explaining
        5. Be specific about where implementations can be found
        6. Consider code organization patterns in your response

        DETAILED ANSWER:
        """
        
    def query(
        self, 
        query: str, 
        k: int = 5, 
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Query the code using the RAG engine"""
        start_time = time.time()
        
        try:
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
            
            # Create metadata filters if provided
            search_kwargs = {"k": k}
            if filters:
                filter_clauses = []
                for key, value in filters.items():
                    if isinstance(value, list):
                        filter_clauses.append({"terms": {f"metadata.{key}": value}})
                    else:
                        filter_clauses.append({"term": {f"metadata.{key}": value}})
                
                if filter_clauses:
                    search_kwargs["filter"] = {"bool": {"must": filter_clauses}}
            
            # Get retriever with filters
            retriever = vector_store.as_retriever(search_kwargs=search_kwargs)
            
            # Initialize LLM
            llm = Ollama(
                model=self.llm_model,
                base_url=self.ollama_url,
                temperature=0.1  # Lower temperature for code questions
            )
            
            # Create prompt
            prompt = PromptTemplate(
                template=self.get_prompt_template(),
                input_variables=["context", "question"]
            )
            
            # Create RAG chain
            rag_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True,
                chain_type_kwargs={"prompt": prompt}
            )
            
            # Execute query
            result = rag_chain({"query": query})
            
            # Format sources
            sources = []
            for doc in result["source_documents"]:
                source_info = {
                    "file": doc.metadata.get("source", "Unknown"),
                    "file_type": doc.metadata.get("file_type", ""),
                    "lines": f"{doc.metadata.get('line_start', '?')}-{doc.metadata.get('line_end', '?')}",
                    "content": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
                }
                sources.append(source_info)
            
            query_time = time.time() - start_time
            logger.info(f"Query completed in {query_time:.2f} seconds")
            
            return {
                "answer": result["result"],
                "sources": sources,
                "query_time": query_time
            }
            
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return {
                "answer": f"Error querying the codebase: {str(e)}",
                "sources": [],
                "error": str(e)
            }