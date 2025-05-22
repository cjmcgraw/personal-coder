from itertools import batched
import logging
import io

from . import Tokenizer, Token

log = logging.getLogger(__name__)

class Tokenizer(Tokenizer):
    """Tokenizer that splits code into line-based chunks."""
    def __init__(self):
        self.name = 'line-based'
    
    def tokenize(self, file_path, max_lines=None):
        """
        Generate tokens from a file by splitting into chunks.
        
        Args:
            file_path: Path to the file to tokenize
            max_lines: Optional number of lines per token. If None, uses multiple chunking strategies.
        """
        file_type = file_path.suffix.lstrip('.')
        
        # If max_lines is specified, use it; otherwise use multiple chunking strategies
        chunk_sizes = [max_lines] if max_lines else [1000, 500, 250, 100, 75, 50, 25, 10, 5]
        
        # Process the file once for each chunking strategy
        for chunk_size in chunk_sizes:
            with open(file_path, 'r', encoding='utf-8') as f:
                line_num = 1
                
                # Process the file in batches of lines
                for chunk in batched(f, chunk_size):
                    # Convert chunk to a list to get its length
                    chunk_list = list(chunk)
                    chunk_len = len(chunk_list)
                    
                    if chunk_len > 0:
                        end_line = line_num + chunk_len - 1
                        content = ''.join(chunk_list)
                        
                        yield Token(
                            file_path=file_path,
                            file_type=file_type,
                            repo="default",
                            line_start=line_num,
                            line_end=end_line,
                            content=content,
                            source=self.name,
                        )
                        
                        line_num = end_line + 1