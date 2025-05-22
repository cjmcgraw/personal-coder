import logging
import random
from . import Tokenizer, Token

log = logging.getLogger(__name__)

class Tokenizer(Tokenizer):
    """Tokenizer that extracts random samples from code."""
    name = "random-sample"
    
    def tokenize(self, file_path, num_samples=5, min_lines=10, max_lines=100):
        """
        Generate random samples of lines from a file.
        
        Args:
            file_path: Path to the file to tokenize
            num_samples: Number of random samples to generate
            min_lines: Minimum number of lines per sample
            max_lines: Maximum number of lines per sample
        """
        file_type = file_path.suffix.lstrip('.')
        
        # Read all lines from the file
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = list(f)
            
        total_lines = len(lines)
        
        # If file is too small, just return one token with the whole file
        if total_lines <= min_lines:
            yield Token(
                file_path=file_path,
                file_type=file_type,
                repo="default",
                line_start=1,
                line_end=total_lines,
                content=''.join(lines),
                source=self.name,
            )
            return
            
        # Generate random samples
        for _ in range(num_samples):
            # Pick a random starting line
            max_start = total_lines - min_lines
            if max_start <= 0:
                start_line = 1
            else:
                start_line = random.randint(1, max_start)
                
            # Pick a random length
            remaining_lines = total_lines - start_line + 1
            sample_length = min(
                random.randint(min_lines, max_lines),
                remaining_lines
            )
            
            # Calculate end line
            end_line = start_line + sample_length - 1
            
            # Extract the content
            content = ''.join(lines[start_line-1:end_line])
            
            yield Token(
                file_path=file_path,
                file_type=file_type,
                repo="default",
                line_start=start_line,
                line_end=end_line,
                content=content,
                source=self.name,
            )