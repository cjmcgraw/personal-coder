Here's a README.md formatted specifically for the RAG container:

```markdown
# Code RAG Service

A high-performance Retrieval-Augmented Generation (RAG) service for semantic code search. This service indexes your codebase using advanced tokenization and embeddings, enabling intelligent code discovery through natural language queries.

## Features

- 🔍 **Semantic Search** - Find code by meaning, not just keywords
- 🌳 **Tree-sitter Tokenization** - Language-aware code parsing
- 🔀 **Hybrid Search** - Combines vector similarity with text matching
- 📁 **Multi-file Support** - Handles Python, JavaScript, Java, Go, and more
- 🚀 **Fast Indexing** - Bulk indexing with intelligent caching
- 🎯 **Precise Filtering** - Search by file type, token kind, or custom filters
- 🔧 **RESTful API** - Easy integration with any application

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Code Files    │────▶│   RAG Service   │────▶│ Elasticsearch   │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                               │                          │
                               ▼                          ▼
                        ┌─────────────────┐     ┌─────────────────┐
                        │     Ollama      │     │     Redis       │
                        │  (Embeddings)   │     │   (Caching)     │
                        └─────────────────┘     └─────────────────┘
```

## Quick Start

### Using Docker Compose

The RAG service is designed to run as part of a Docker Compose stack:

```yaml
services:
  rag:
    build: ./rag
    volumes:
      - /path/to/your/code:/code:ro
    environment:
      - OLLAMA_HOST=http://ollama:11434
      - ELASTICSEARCH_HOST=http://elasticsearch:9200
      - REDIS_HOST=redis
      - CODE_DIR=/code
      - AUTO_INGEST_ON_STARTUP=true
    ports:
      - "8000:8000"
    depends_on:
      elasticsearch:
        condition: service_healthy
      redis:
        condition: service_healthy
      ollama:
        condition: service_started
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OLLAMA_HOST` | Ollama service URL | `http://ollama:11434` |
| `ELASTICSEARCH_HOST` | Elasticsearch URL | `http://elasticsearch:9200` |
| `REDIS_HOST` | Redis hostname | `redis` |
| `CODE_DIR` | Directory to index | `/code` |
| `AUTO_INGEST_ON_STARTUP` | Auto-index on startup | `false` |
| `EMBEDDINGS_MODEL` | Ollama embedding model | `all-minilm:l6-v2` |

## API Reference

### Health Check

```bash
GET /health
```

Returns `"green"` when service is healthy.

### Indexing

#### Start Indexing
```bash
POST /ingest-code-base?force=false
```

Starts background indexing of the codebase.

**Parameters:**
- `force` (bool): Force complete re-indexing

**Response:**
```json
{
  "message": "started background task",
  "runid": "d27dd04c59df4bf58a5302866f7263bf"
}
```

### Querying

#### Semantic Search
```bash
POST /query
Content-Type: application/json

{
  "query": "user authentication and password validation",
  "top_k": 5,
  "min_score": 1.0,
  "filters": {
    "file_type": "py",
    "kind": "function"
  }
}
```

Find code using natural language queries.

**Request Body:**
- `query` (string, required): Natural language search query
- `top_k` (int): Number of results to return (default: 5)
- `min_score` (float): Minimum similarity score
- `filters` (object): Additional filters
  - `file_type`: Filter by file extension
  - `kind`: Filter by token type (function, class, method, etc.)
  - `source`: Filter by tokenizer source

**Response:**
```json
{
  "results": [
    {
      "content": "def authenticate_user(username: str, password: str)...",
      "file_path": "src/auth/authentication.py",
      "line_start": 45,
      "line_end": 55,
      "score": 1.87,
      "kind": "function:authenticate_user",
      "source": "tree-sitter",
      "highlights": ["authenticate user with <em>password</em>"]
    }
  ],
  "total_results": 5,
  "query_time_ms": 42.5
}
```

#### File Path Search
```bash
GET /query/file/{filepath}?fuzzy=true&limit=20
```

Search for code within specific files.

**Parameters:**
- `filepath` (string): File path pattern
- `fuzzy` (bool): Use wildcard matching (default: true)
- `limit` (int): Maximum results (default: 20)

#### Fuzzy Text Search
```bash
GET /query/fuzzy?text=authentcation&fuzziness=AUTO&top_k=10
```

Find code with fuzzy text matching (handles typos).

**Parameters:**
- `text` (string): Search text
- `fuzziness` (string): AUTO, 0, 1, or 2
- `top_k` (int): Number of results

#### Search by Token Type
```bash
GET /query/by-kind/{kind}?limit=20
```

Get all code snippets of a specific type.

**Parameters:**
- `kind` (string): Token type (function, class, method, etc.)
- `limit` (int): Maximum results

#### Direct Embedding Search
```bash
POST /query/embedding
Content-Type: application/json

{
  "embedding": [0.123, -0.456, ...],
  "top_k": 5,
  "min_score": 1.5
}
```

Search using pre-computed embedding vectors.

### Statistics & Metadata

#### Index Statistics
```bash
GET /query/stats
```

Returns index statistics including document count, file count, and tokenizer distribution.

#### Available Token Types
```bash
GET /query/kinds
```

Lists all available token types and their counts.

### Testing Endpoints

#### Test Tokenization
```bash
GET /test/tokenizer/{filepath}?tokenizer=tree-sitter
```

Shows how a file is tokenized for debugging.

#### Test Embeddings
```bash
GET /test/embeddings?text=your+code+here
```

Generate embeddings for text.

#### Compare Embeddings
```bash
GET /test/embeddings/distance?a=text1&b=text2
```

Calculate similarity between two texts.

## Configuration

### Gitignore Support

The service respects `.gitignore` files in your codebase. Additionally, create `rag/src/rag/default-ignores.json` to define default ignore patterns:

```json
{
  "python": ["__pycache__/", "*.pyc", "*.pyo"],
  "virtualenv": ["venv/", ".env"],
  "vcs": [".git/", ".svn/"],
  "ide": [".idea/", ".vscode/"],
  "build": ["dist/", "build/"]
}
```

### File Size Limits

- Maximum file size: 5MB (configurable in code)
- Binary files are automatically skipped
- Empty files are ignored

### Supported File Types

The tokenizer automatically detects language based on file extension:
- Python: `.py`
- JavaScript: `.js`, `.jsx`
- TypeScript: `.ts`, `.tsx`
- Java: `.java`
- Go: `.go`
- Rust: `.rs`
- C/C++: `.c`, `.cpp`, `.h`
- And many more...

## Examples

### Finding Authentication Code

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "user authentication with JWT tokens",
    "top_k": 5,
    "filters": {"file_type": "py"}
  }'
```

### Finding All API Endpoints

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "@app.route decorator Flask FastAPI",
    "top_k": 20,
    "filters": {"kind": "decorated_function"}
  }'
```

### Finding Error Handling

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "try except catch error handling logging",
    "top_k": 10
  }'
```

### Building a Code Assistant

```python
import requests

class CodeRAG:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    def search(self, query, top_k=5, **filters):
        response = requests.post(
            f"{self.base_url}/query",
            json={
                "query": query,
                "top_k": top_k,
                "filters": filters
            }
        )
        return response.json()
    
    def get_context_for_llm(self, query):
        results = self.search(query, top_k=3)
        
        context = []
        for r in results["results"]:
            context.append(f"""
File: {r['file_path']} (lines {r['line_start']}-{r['line_end']})
```{r['file_path'].split('.')[-1]}
{r['content']}
```
""")
        return "\n".join(context)

# Usage
rag = CodeRAG()
context = rag.get_context_for_llm("how does authentication work?")
print(context)
```

## Performance Optimization

### Indexing Performance

- Bulk indexing processes files in batches of 5,000 documents
- Redis caches file hashes to avoid re-indexing unchanged files
- Parallel tokenization for large codebases

### Query Performance

- Vector search typically returns results in <50ms
- Hybrid search may take 50-200ms depending on text complexity
- Use filters to reduce search space and improve performance

### Resource Requirements

- **Memory**: 2-4GB recommended
- **CPU**: Benefits from multiple cores during indexing
- **Storage**: ~100-200MB per 1000 files (varies by code density)

## Troubleshooting

### No Results Found

1. Check indexing status:
   ```bash
   curl http://localhost:8000/query/stats
   ```

2. Verify files aren't ignored:
   ```bash
   docker exec rag-container cat /workdir/src/rag/default-ignores.json
   ```

3. Try broader queries or fuzzy search

### Slow Indexing

1. Check Ollama is running and accessible
2. Verify Elasticsearch health
3. Reduce batch size if memory constrained
4. Check Redis connectivity

### Memory Issues

1. Reduce `batch_size` in `BulkIndexer`
2. Index in smaller chunks
3. Increase container memory limits

## Development

### Running Tests

```bash
# Inside container
pytest tests/

# From host
docker exec rag-container pytest tests/
```

### Adding New Tokenizers

Create a new file in `rag/src/rag/tokenizers/`:

```python
from . import Tokenizer, Token

class MyTokenizer(Tokenizer):
    def __init__(self):
        self.name = "my-tokenizer"
    
    def tokenize(self, file_path):
        # Your tokenization logic
        yield Token(...)
```

### Debugging

Enable debug logs:
```python
logging.basicConfig(level=logging.DEBUG)
```

View container logs:
```bash
docker compose logs -f rag
```

## License

[Your License Here]

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## Support

For issues and questions:
- Check the [troubleshooting](#troubleshooting) section
- Review container logs
- Open an issue on GitHub
```

This README is specifically tailored for the RAG container, including Docker-specific setup, configuration details, and comprehensive API documentation suitable for developers who will be using or contributing to the service.
