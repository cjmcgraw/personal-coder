import pathlib
import logging
import json
import re
from typing import Iterator, Dict, Set, List, Optional, Tuple

from tree_sitter_language_pack import get_parser

from . import Tokenizer, Token, get_language

log = logging.getLogger(__name__)

dir = pathlib.Path(__file__).parent

with (dir / "treesitter.definitions.json").open("r", encoding='utf-8') as f:
    _raw = json.load(f)
    assert hasattr(_raw, 'items') and len(_raw.items()) > 1

DEFINITION_TYPES = {k: set(v) for k, v in _raw.items()}

# Define token kinds that are particularly relevant for RAG
RAG_TOKEN_KINDS = {
    # High-level semantic structures
    "function": "function",
    "method": "method",
    "class": "class",
    "interface": "interface",
    "module": "module",
    
    # Control flow structures
    "if_statement": "conditional",
    "for_statement": "loop",
    "while_statement": "loop",
    "switch_statement": "conditional",
    "try_statement": "error_handling",
    "catch_clause": "error_handling",
    
    # Data and type definitions
    "struct": "data_structure",
    "enum": "data_structure",
    "type_declaration": "type_definition",
    "variable_declaration": "variable",
    
    # Imports and exports
    "import_statement": "import",
    "import_from_statement": "import",
    "export_statement": "export",
    
    # Other meaningful chunks
    "docstring": "documentation",
    "comment_block": "documentation",
    "decorated_definition": "decorated_function",
    "assignment": "assignment",
}

# Language-specific docstring patterns
DOCSTRING_PATTERNS = {
    "python": r'""".*?"""|\'\'\'.*?\'\'\'',
    "javascript": r'/\*\*[\s\S]*?\*/',
    "typescript": r'/\*\*[\s\S]*?\*/',
    "java": r'/\*\*[\s\S]*?\*/',
    "go": r'/\*[\s\S]*?\*/',
    "rust": r'///(.*?)$|/\*\*[\s\S]*?\*/',
}


class Tokenizer(Tokenizer):
    """Tokenizer that uses tree-sitter to extract semantic tokens from code for RAG applications."""

    def __init__(self):
        self.name = "tree-sitter"
    
    def tokenize(self, file_path) -> Iterator[Token]:
        """Generate semantically meaningful tokens from a file using tree-sitter.
        
        For RAG applications, we want to extract chunks that preserve the semantic
        meaning of the code and include sufficient context for retrieval.
        """
        # Determine language
        language = get_language(file_path)
        if not language:
            log.warning(f"No language mapping for {file_path}")
            return
        
        # Get parser
        parser = get_parser(language)
        if not parser:
            log.warning(f"No parser available for {language}")
            return
        
        log.info(f"Processing {file_path} with tree-sitter {language} parser")
        
        # Read file
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
        except UnicodeDecodeError:
            try:
                # Try with a different encoding
                with open(file_path, 'r', encoding='latin-1') as f:
                    code = f.read()
            except Exception as e:
                log.error(f"Failed to read {file_path}: {e}")
                return
        
        file_type = file_path.suffix.lstrip('.')
        
        # Parse the file with tree-sitter
        tree = parser.parse(bytes(code, 'utf-8'))
        
        # Track what tokens we've yielded to avoid duplicates
        yielded_spans = set()
        tokens_yielded = 0
        
        # Extract docstrings and comments first - they can be used as separate tokens
        # and also attached to related code elements
        docstring_tokens = list(self._extract_docstrings(file_path, code, file_type, language))
        for token in docstring_tokens:
            yielded_spans.add((token.line_start, token.line_end))
            yield token
            tokens_yielded += 1
        
        # Extract module-level docstring if present (for Python)
        if language == "python":
            module_docstring = self._extract_module_docstring(file_path, code, file_type)
            if module_docstring and (module_docstring.line_start, module_docstring.line_end) not in yielded_spans:
                yielded_spans.add((module_docstring.line_start, module_docstring.line_end))
                yield module_docstring
                tokens_yielded += 1
        
        # Extract initial module-level code (before the first function/class)
        module_token = self._extract_initial_module_code(file_path, code, file_type)
        if module_token and (module_token.line_start, module_token.line_end) not in yielded_spans:
            yielded_spans.add((module_token.line_start, module_token.line_end))
            yield module_token
            tokens_yielded += 1
        
        # Extract language-specific tokens (imports, statements, etc.)
        statement_tokens = []
        if language in ["python", "javascript", "typescript", "go"]:
            statement_tokens = list(self._extract_statement_tokens(file_path, code, file_type, tree.root_node, language))
        
        for token in statement_tokens:
            if (token.line_start, token.line_end) not in yielded_spans:
                yielded_spans.add((token.line_start, token.line_end))
                yield token
                tokens_yielded += 1
        
        # Find all semantic definition nodes (functions, classes, methods, etc.)
        definition_nodes = self._find_definition_nodes(tree.root_node, code, language)
        
        # Process definition nodes into tokens
        for node in definition_nodes:
            start_byte = node.start_byte
            end_byte = node.end_byte
            
            # Calculate line numbers
            line_start = code[:start_byte].count('\n') + 1
            line_end = code[:end_byte].count('\n') + 1
            
            # Skip if we've already yielded this span
            if (line_start, line_end) in yielded_spans:
                continue
            
            # Get node content
            node_content = code[start_byte:end_byte]
            
            # Determine the appropriate kind
            kind = self._determine_token_kind(node.type, language)
            
            # Extract node name if possible
            node_name = self._extract_node_name(node, code, language)
            if node_name:
                kind = f"{kind}:{node_name}"
            
            # Create token
            token = Token(
                file_path=file_path,
                file_type=file_type,
                repo="default",
                line_start=line_start,
                line_end=line_end,
                content=node_content,
                source="tree-sitter",
                kind=kind
            )
            
            yielded_spans.add((line_start, line_end))
            yield token
            tokens_yielded += 1
            
            # For classes and large functions, also extract their methods/nested functions separately
            if node.type in ["class_definition", "class_declaration"] or len(node_content.split('\n')) > 20:
                nested_tokens = self._extract_nested_definitions(node, code, file_path, file_type, language)
                for nested_token in nested_tokens:
                    if (nested_token.line_start, nested_token.line_end) not in yielded_spans:
                        yielded_spans.add((nested_token.line_start, nested_token.line_end))
                        yield nested_token
                        tokens_yielded += 1
        
        # If nothing was found, fall back to a whole-file token
        if tokens_yielded == 0:
            log.warning(f"No definitions found in {file_path}, returning whole file")
            yield Token(
                file_path=file_path,
                file_type=file_type,
                repo="default",
                line_start=1,
                line_end=code.count('\n') + 1,
                content=code,
                source="tree-sitter",
                kind="whole_file"
            )
    
    def _extract_initial_module_code(self, file_path, code, file_type):
        """Extract the initial module-level code (before first function/class)."""
        pattern = re.compile(r'^(?:\s*def\s+\w+|\s*class\s+\w+|\s*@\w+)', re.MULTILINE)
        match = pattern.search(code)
        
        if match and match.start() > 0:
            module_text = code[:match.start()].rstrip()
            if module_text.strip():
                module_end_line = module_text.count('\n') + 1
                
                return Token(
                    file_path=file_path,
                    file_type=file_type,
                    repo="default",
                    line_start=1,
                    line_end=module_end_line,
                    content=module_text,
                    source="tree-sitter",
                    kind="module"
                )
        
        return None
    
    def _extract_module_docstring(self, file_path, code, file_type):
        """Extract module-level docstring for Python files."""
        lines = code.split('\n')
        docstring_pattern = re.compile(r'^(\s*)(""".*?"""|\'\'\'.*?\'\'\')(?:\s*$|\s)', re.DOTALL | re.MULTILINE)
        
        # Check for module docstring - should be at the beginning of the file
        # after imports and possibly a shebang or encoding declaration
        start_idx = 0
        
        # Skip shebang and encoding lines
        while start_idx < len(lines) and (lines[start_idx].startswith('#!') or 
                                          lines[start_idx].startswith('# -*- coding') or
                                          lines[start_idx].strip() == ''):
            start_idx += 1
        
        # Skip import statements
        while start_idx < len(lines) and (lines[start_idx].startswith('import ') or 
                                          lines[start_idx].startswith('from ') or
                                          lines[start_idx].strip() == ''):
            start_idx += 1
        
        # Now check for docstring
        if start_idx < len(lines):
            # Join remaining lines to allow for multiline docstring matching
            remaining_code = '\n'.join(lines[start_idx:])
            match = docstring_pattern.search(remaining_code)
            
            if match:
                docstring = match.group(2)
                # Calculate line numbers
                docstring_start = start_idx + remaining_code[:match.start()].count('\n')
                docstring_end = docstring_start + docstring.count('\n')
                
                return Token(
                    file_path=file_path,
                    file_type=file_type,
                    repo="default",
                    line_start=docstring_start + 1,  # +1 because line numbers are 1-indexed
                    line_end=docstring_end + 1,
                    content=docstring,
                    source="tree-sitter",
                    kind="module_docstring"
                )
        
        return None
    
    def _extract_docstrings(self, file_path, code, file_type, language):
        """Extract docstrings and comment blocks as separate tokens."""
        if language not in DOCSTRING_PATTERNS:
            return
            
        pattern = re.compile(DOCSTRING_PATTERNS[language], re.DOTALL)
        
        for match in pattern.finditer(code):
            docstring = match.group(0)
            if len(docstring.strip()) < 10:  # Skip very short docstrings
                continue
                
            # Calculate line numbers
            start_pos = match.start()
            end_pos = match.end()
            line_start = code[:start_pos].count('\n') + 1
            line_end = code[:end_pos].count('\n') + 1
            
            # Try to identify what this docstring is for
            docstring_kind = "docstring"
            
            # Look for function/class/method definition after the docstring
            post_docstring = code[end_pos:end_pos + 200]  # Look ahead a bit
            if re.search(r'def\s+\w+', post_docstring):
                docstring_kind = "function_docstring"
            elif re.search(r'class\s+\w+', post_docstring):
                docstring_kind = "class_docstring"
            
            yield Token(
                file_path=file_path,
                file_type=file_type,
                repo="default",
                line_start=line_start,
                line_end=line_end,
                content=docstring,
                source="tree-sitter",
                kind=docstring_kind
            )
    
    def _extract_statement_tokens(self, file_path, code, file_type, root_node, language):
        """Extract individual tokens for imports, assignments, and other statements."""
        # Define language-specific statement types
        statement_types = {
            "python": {
                "import_statement", 
                "import_from_statement", 
                "assignment",
                "global_statement",
                "call",
                "return_statement"
            },
            "javascript": {
                "import_statement",
                "export_statement",
                "variable_declaration",
                "lexical_declaration",
                "return_statement"
            },
            "typescript": {
                "import_statement",
                "export_statement",
                "variable_declaration",
                "lexical_declaration",
                "type_alias_declaration",
                "interface_declaration",
                "return_statement"
            },
            "go": {
                "import_declaration",
                "var_declaration",
                "const_declaration",
                "return_statement"
            }
        }
        
        current_statement_types = statement_types.get(language, set())
        
        def traverse(node):
            if node.type in current_statement_types:
                start_byte = node.start_byte
                end_byte = node.end_byte
                
                # Calculate line numbers
                line_start = code[:start_byte].count('\n') + 1
                line_end = code[:end_byte].count('\n') + 1
                
                # Skip very small nodes
                node_text = code[start_byte:end_byte]
                if len(node_text.strip()) < 3:
                    return
                
                # Check if this statement is inside a function/method
                is_in_function = False
                parent = node.parent
                while parent:
                    if parent.type in [
                        "function_definition", 
                        "function_declaration",
                        "method_definition",
                        "method_declaration",
                        "arrow_function"
                    ]:
                        is_in_function = True
                        break
                    parent = parent.parent
                
                if not is_in_function:
                    # Map the node type to a RAG-appropriate kind
                    if node.type in ["import_statement", "import_from_statement", "import_declaration"]:
                        kind = "import"
                    elif node.type in ["assignment", "variable_declaration", "lexical_declaration", "var_declaration", "const_declaration"]:
                        kind = "declaration"
                    elif node.type == "export_statement":
                        kind = "export"
                    elif node.type == "return_statement":
                        kind = "return"
                    else:
                        kind = node.type
                    
                    yield Token(
                        file_path=file_path,
                        file_type=file_type,
                        repo="default",
                        line_start=line_start,
                        line_end=line_end,
                        content=node_text,
                        source="tree-sitter",
                        kind=kind
                    )
            
            # Recursively check children
            for child in node.children:
                yield from traverse(child)
        
        yield from traverse(root_node)
    
    def _find_definition_nodes(self, root_node, code, language):
        """Find all function and class definition nodes."""
        # Get language-specific definition types
        definition_types = DEFINITION_TYPES.get(language, DEFINITION_TYPES.get("default", set()))
        
        # Collect definitions
        definitions = []
        
        def traverse(node, parent_type=None):
            # Special case for handling decorated functions in Python
            is_decorated = False
            if language == "python":
                # Check for decorated_definition parent
                if parent_type == "decorated_definition":
                    is_decorated = True
                # Also check decorator pattern in node text
                elif node.type == "function_definition":
                    node_text = code[node.start_byte:node.end_byte]
                    if re.search(r'@\w+', node_text, re.MULTILINE):
                        is_decorated = True
            
            # Check if this node is a definition we're interested in
            is_definition = node.type in definition_types
            
            if is_definition or is_decorated:
                # Skip very small nodes
                node_text = code[node.start_byte:node.end_byte]
                if len(node_text.strip()) >= 10:
                    # For Python decorated functions, include the decorator
                    if is_decorated and language == "python" and parent_type == "decorated_definition":
                        parent_node = node.parent
                        if parent_node and parent_node.type == "decorated_definition":
                            definitions.append(parent_node)
                            return  # Skip processing children
                    
                    definitions.append(node)
            
            # Continue traversing children
            for child in node.children:
                traverse(child, node.type)
        
        traverse(root_node)
        return definitions
    
    def _extract_nested_definitions(self, parent_node, code, file_path, file_type, language):
        """Extract nested definitions like methods inside classes."""
        nested_tokens = []
        
        # Get language-specific method/function types
        method_types = {
            "python": {"function_definition", "method_definition"},
            "javascript": {"function_declaration", "method_definition", "arrow_function"},
            "typescript": {"function_declaration", "method_definition", "arrow_function"},
            "java": {"method_declaration"},
            "go": {"function_declaration", "method_declaration"},
            "rust": {"function_item"},
            "c": {"function_definition"},
            "cpp": {"function_definition"},
        }.get(language, {"function", "method"})
        
        def traverse(node):
            if node.type in method_types:
                start_byte = node.start_byte
                end_byte = node.end_byte
                
                # Calculate line numbers
                line_start = code[:start_byte].count('\n') + 1
                line_end = code[:end_byte].count('\n') + 1
                
                # Get node content
                node_content = code[start_byte:end_byte]
                
                # Extract method name
                method_name = self._extract_node_name(node, code, language)
                kind = "method" if method_name else node.type
                if method_name:
                    kind = f"{kind}:{method_name}"
                
                nested_tokens.append(Token(
                    file_path=file_path,
                    file_type=file_type,
                    repo="default",
                    line_start=line_start,
                    line_end=line_end,
                    content=node_content,
                    source="tree-sitter",
                    kind=kind
                ))
            
            # Continue traversing
            for child in node.children:
                traverse(child)
        
        traverse(parent_node)
        return nested_tokens
    
    def _determine_token_kind(self, node_type, language):
        """Map tree-sitter node types to RAG-appropriate token kinds."""
        # First check our RAG_TOKEN_KINDS mapping
        if node_type in RAG_TOKEN_KINDS:
            return RAG_TOKEN_KINDS[node_type]
        
        # Then handle language-specific mappings
        if language == "python":
            if node_type == "function_definition":
                return "function"
            elif node_type == "class_definition":
                return "class"
            elif node_type == "decorated_definition":
                return "decorated_function"
        elif language in ["javascript", "typescript"]:
            if node_type == "function_declaration":
                return "function"
            elif node_type == "class_declaration":
                return "class"
            elif node_type == "method_definition":
                return "method"
            elif node_type == "arrow_function":
                return "function"
        elif language == "java":
            if node_type == "method_declaration":
                return "method"
            elif node_type == "class_declaration":
                return "class"
        
        # Fall back to the original node type
        return node_type
    
    def _extract_node_name(self, node, code, language):
        """Extract the name of a function, class, or method from its node."""
        # Different approaches based on language
        if language == "python":
            # For Python, look for an identifier child node
            for child in node.children:
                if child.type == "identifier":
                    return code[child.start_byte:child.end_byte]
                
        elif language in ["javascript", "typescript"]:
            # For JS/TS, check node type
            if node.type in ["function_declaration", "class_declaration"]:
                for child in node.children:
                    if child.type == "identifier":
                        return code[child.start_byte:child.end_byte]
            elif node.type == "method_definition":
                for child in node.children:
                    if child.type == "property_identifier":
                        return code[child.start_byte:child.end_byte]
        
        elif language == "java":
            if node.type in ["class_declaration", "method_declaration"]:
                for child in node.children:
                    if child.type == "identifier":
                        return code[child.start_byte:child.end_byte]
        
        # Fall back to first part of content for a name
        node_text = code[node.start_byte:node.end_byte]
        first_line = node_text.split('\n', 1)[0].strip()
        
        # Extract name with regex based on language
        if language == "python":
            match = re.search(r'(def|class)\s+(\w+)', first_line)
            if match:
                return match.group(2)
        elif language in ["javascript", "typescript"]:
            match = re.search(r'(function|class)\s+(\w+)', first_line)
            if match:
                return match.group(2)
        elif language == "java":
            match = re.search(r'(class|interface)\s+(\w+)', first_line)
            if match:
                return match.group(2)
        
        return ""