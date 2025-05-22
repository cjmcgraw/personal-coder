import re
import logging

from . import Tokenizer, Token, get_language

log = logging.getLogger(__name__)

class Tokenizer(Tokenizer):
    """Tokenizer that uses regex patterns to identify code structures."""

    def __init__(self):
        self.name = "regex-pattern"
    
    def tokenize(self, file_path, max_size=15000):
        """Generate tokens from a file using regex patterns."""
        language = get_language(file_path)
        if not language:
            return []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        
        file_type = file_path.suffix.lstrip('.')
        tokens = []
        
        # Define patterns for different languages
        patterns = {
            "python": r'^(def\s+\w+|class\s+\w+)',
            "javascript": r'^(function\s+\w+|class\s+\w+|\w+\s*=\s*function|\w+\s*=\s*\(.*?\)\s*=>|const\s+\w+\s*=\s*\()',
            "typescript": r'^(function\s+\w+|class\s+\w+|interface\s+\w+|\w+\s*=\s*function|\w+\s*=\s*\(.*?\)\s*=>|const\s+\w+\s*=\s*\()',
            "jsx": r'^(function\s+\w+|class\s+\w+|const\s+\w+\s*=\s*\(|const\s+\w+\s*=\s*React\.)',
            "tsx": r'^(function\s+\w+|class\s+\w+|interface\s+\w+|const\s+\w+\s*=\s*\(|const\s+\w+\s*=\s*React\.)',
            "java": r'^(public|private|protected|static|final|abstract|class|interface|enum).*?(class|interface|enum)\s+\w+',
            "c": r'^(\w+\s+\w+\s*\(.*?\)\s*\{)',
            "cpp": r'^(class|struct|enum|namespace|template|void|int|float|double|bool|char|auto|inline)\s+\w+',
            "go": r'^(func|type|var|const|import|package)\s+',
            "rust": r'^(fn|struct|enum|impl|trait|const|static|let|use|mod|pub)\s+',
            "ruby": r'^(def|class|module|if|unless|case|while|until|for|begin)\s+',
            "php": r'^(function|class|interface|trait|namespace)\s+',
            "csharp": r'^(public|private|protected|internal|static|abstract|class|interface|enum|struct|void|delegate)\s+',
            "bash": r'^(function\s+\w+|\w+\s*\(\s*\)\s*\{|if\s+|for\s+|while\s+|case\s+)',
            "html": r'^<(html|head|body|div|span|h1|h2|h3|p|a|img|script|style|link)',
            "css": r'^([.#]?\w+|\*|\[.*?\])\s*\{',
            "yaml": r'^(\w+:|\s*-\s+\w+:)',
            "json": r'^(\s*\{\s*|\s*\[\s*|\s*"\w+"\s*:)',
            "dockerfile": r'^(FROM|RUN|CMD|LABEL|MAINTAINER|EXPOSE|ENV|ADD|COPY|ENTRYPOINT|VOLUME|USER|WORKDIR|ARG)',
        }
        
        # Get the appropriate regex pattern
        pattern_str = patterns.get(language)
        if not pattern_str:
            return []
        
        pattern = re.compile(pattern_str, re.MULTILINE)
        matches = list(pattern.finditer(code))
        
        if matches:
            for i, match in enumerate(matches):
                start_pos = match.start()
                end_pos = matches[i + 1].start() if i < len(matches) - 1 else len(code)
                
                chunk_code = code[start_pos:end_pos]
                
                # Skip if chunk is too large
                if len(chunk_code) > max_size:
                    continue
                    
                line_start = code[:start_pos].count('\n') + 1
                line_end = code[:end_pos].count('\n') + 1
                
                tokens.append(Token(
                    file_path=file_path,
                    file_type=file_type,
                    repo="default",
                    line_start=line_start,
                    line_end=line_end,
                    content=chunk_code,
                    source=self.name
                ))
        
        return tokens