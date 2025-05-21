# Personal Coder

My personal coder used to pretrain an LLM on a codebase using RAG (Retrieval Augmented Generation).

## Quick Start

Getting started with Personal Coder is simple:

```bash
# Set the path to the codebase you want to analyze
export RAG_CODE_BASE=/path/to/your/codebase

# Start the service (with GPU support)
docker-compose up

# Access the web interface
# Open your browser and navigate to http://localhost:8999
```

That's it! The web interface will guide you through interacting with the AI assistant.

## Overview

Personal Coder is a tool that enhances Large Language Models (LLMs) with knowledge of your specific codebase. It implements a Retrieval Augmented Generation (RAG) pipeline that indexes your code and retrieves relevant snippets when you ask questions, resulting in more accurate and contextual responses.

## Features

- **Easy Docker-based Deployment**: Uses docker-compose for simple setup
- **Web-based UI**: Built on Open WebUI for a seamless chat experience
- **GPU Acceleration**: Configured for NVIDIA GPU support
- **Full RAG Pipeline**: Automatic code indexing, embedding, and retrieval
- **Built on Open Source Models**: Uses Gemma 3 (4B) as the base model
- **Elasticsearch Vector Store**: Efficient semantic search of your codebase

## System Architecture

Personal Coder consists of several components:

1. **Ollama**: Hosts and serves the LLM (Gemma 3:4B with a specialized prompt)
2. **WebUI**: Provides the chat interface at `localhost:8999`
3. **RAG-Proxy**: Intercepts queries to code-relevant models and enhances them with context
4. **Elasticsearch**: Stores code embeddings for semantic search
5. **RAG Engine**: Handles code indexing, embedding, and retrieval

## Requirements

- Docker and Docker Compose
- NVIDIA GPU with CUDA support (recommended)
- At least 8GB of RAM
- A codebase to analyze and query

## Advanced Usage

### Environment Variables

- `RAG_CODE_BASE`: The path to the codebase you want to analyze (required)

### Customizing Prompt Templates

The system prompt for the code expert model can be customized by editing the `ollama/modelfiles/code-expert` file.

### Reindexing Your Codebase

If you change the codebase or want to force a reindex:

```bash
# With the system running
curl -X POST http://localhost:8000/force-ingest
```

### Supported File Types

The system automatically indexes common code file types:
`.py`, `.js`, `.ts`, `.java`, `.c`, `.cpp`, `.go`, `.rs`, `.sh`, `.jsx`, `.tsx`

## How It Works

1. Your codebase is indexed and chunked into semantically meaningful pieces
2. Code chunks are embedded using the all-minilm:l6-v2 model
3. Embeddings are stored in Elasticsearch for efficient retrieval
4. When you ask a question, the RAG system:
   - Encodes your question into an embedding
   - Searches for the most relevant code snippets
   - Prepends these snippets to your query for context
   - Feeds the enhanced prompt to the LLM
5. The LLM (Gemma 3:4B) generates a response grounded in your actual code

## License

This project is licensed under the MIT License - see the LICENSE file for details.
