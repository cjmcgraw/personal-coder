import pathlib
import json
import logging
import importlib
import time
import os
from typing import Iterator

import pydantic

# Configure logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

dir = pathlib.Path(__file__).parent


# Token model at the very top
class Token(pydantic.BaseModel):
    file_path: pathlib.Path
    file_type: str
    repo: str
    line_start: int
    line_end: int
    content: str
    source: str
    kind: str = ""

# I am feeling fucking loco. Its like this. Tokenize the fuck outta the documents and use
# what contexts make sense and dont. Just keep adding contexts until you're good
class Tokenizer:
    """Base class for all tokenizers."""
    def tokenize(self, file_path, **kwargs) -> Iterator[Token]:
        """Generate tokens from a file."""
        raise NotImplementedError("Tokenizers must implement tokenize method")


tokenizers: dict[str, Tokenizer] = {}

# Main API function - prominently placed after Token
def tokenize(*documents, limit: set[str]=None) -> Iterator[Token]:
    """
    Main API function that tokenizes documents using all available tokenizers.
    
    Args:
        *documents: One or more file paths to tokenize
        
    Returns:
        Iterator yielding Token objects
    """
    # Load all tokenizers from the package
    if len(tokenizers) < 1:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        for filename in os.listdir(current_dir):
            if filename.endswith('.py') and '__' not in filename:
                module_name = filename[:-3]  # Remove .py extension
                module = importlib.import_module(f".{module_name}", package=__name__)
                tokenizer = getattr(module, 'Tokenizer')()
                tokenizers[tokenizer.name] = tokenizer
    
    # Process each document with all tokenizers
    for doc in documents:
        file_path = pathlib.Path(doc) if isinstance(doc, str) else doc
        log.info(f"Processing {file_path}")

        t = time.time()
        for tokenizer in tokenizers.values():
            if not limit or tokenizer.name in limit:
                log.info(f"  Using {tokenizer.name} tokenizer")
                for i, token in enumerate(tokenizer.tokenize(file_path)):
                    yield token
                log.debug(f"  {tokenizer.name} generated tokens={i} from {file_path} in {time.time() - t:,.5f}s")
        


# Helper functions
def get_language(file_path) -> str:
    """Determine the language for a file based on its suffix or name."""
    suffix = file_path.suffix.lower()
    return LANGUAGE_MAP.get(suffix) or LANGUAGE_MAP.get(file_path.name)


def merge_overlapping_tokens(tokens: list[Token]) -> list[Token]:
    """Merge tokens that overlap or are adjacent, preserving the most semantic ones."""
    if not tokens:
        return []
    
    # Sort tokens by line start
    tokens.sort(key=lambda t: (t.line_start, -t.line_end))
    
    # First, deduplicate identical tokens
    unique_tokens = []
    seen_content = set()
    
    for token in tokens:
        content_hash = hash(token.content)
        if content_hash not in seen_content:
            unique_tokens.append(token)
            seen_content.add(content_hash)
    
    merged = [unique_tokens[0]]
    
    for current in unique_tokens[1:]:
        previous = merged[-1]
        
        # Check for overlap or adjacency
        if current.line_start <= previous.line_end + 1:
            # If tokens overlap or are adjacent, merge them
            # Keep whichever content is longer (likely more semantic)
            if len(current.content) > len(previous.content):
                content = current.content
            else:
                content = previous.content
                
            merged[-1] = Token(
                file_path=previous.file_path,
                file_type=previous.file_type,
                repo=previous.repo,
                line_start=min(previous.line_start, current.line_start),
                line_end=max(previous.line_end, current.line_end),
                content=content
            )
        else:
            # No overlap, add as separate token
            merged.append(current)
    
    return merged


# Load language configurations
with (dir / "suffixes.json").open('r') as f:
    LANGUAGE_MAP = json.load(f)
