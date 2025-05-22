import pathlib
import logging
import json
import re
from typing import Iterator, Dict, Set, List, Optional, Tuple
from dataclasses import dataclass

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
    "trait": "trait",
    
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



@dataclass
class ImportInfo:
    """Information about an import statement."""
    statement: str
    imported_names: Set[str]
    line_start: int
    line_end: int
    import_type: str  # 'module', 'from', 'namespace', etc.


@dataclass
class ModuleScopeInfo:
    """Information about module-level declarations."""
    statement: str
    name: str
    line_start: int
    line_end: int
    scope_type: str  # 'constant', 'variable', 'type', etc.

class Tokenizer(Tokenizer):
    """Tokenizer that uses tree-sitter to extract semantic tokens from code for RAG applications."""

    def __init__(self, include_context=True, context_mode="minimal", include_module_scope=True):
        self.name = "tree-sitter"
        self.include_context = include_context
        self.context_mode = context_mode  # "none", "minimal", "full"
        self.include_module_scope = include_module_scope
        
        # Language-specific context extractors
        self.context_extractors = {
            "python": PythonContextExtractor(),
            "java": JavaContextExtractor(),
            "javascript": JavaScriptContextExtractor(),
            "typescript": TypeScriptContextExtractor(),
            "php": PhpContextExtractor(),
            "perl": PerlContextExtractor(),
            "c": CContextExtractor(),
            "cpp": CppContextExtractor(),
            "rust": RustContextExtractor(),
            "go": GoContextExtractor(),
        }
    
    def _extract_imports_for_language(self, code: str, language: str) -> List[ImportInfo]:
        """Extract imports for a specific language."""
        extractor = self.context_extractors.get(language)
        if extractor:
            return extractor.extract_imports(code)
        return []
    
    def _extract_module_scope_for_language(self, code: str, language: str) -> List[ModuleScopeInfo]:
        """Extract module-scope declarations for a specific language."""
        extractor = self.context_extractors.get(language)
        if extractor and hasattr(extractor, 'extract_module_scope'):
            return extractor.extract_module_scope(code)
        return []
    
    def _get_relevant_context(
        self, 
        node_text: str, 
        imports: List[ImportInfo], 
        module_scope: List[ModuleScopeInfo],
        language: str
    ) -> Tuple[List[str], List[str]]:
        """Get imports and module scope items that are actually used in the node text."""
        extractor = self.context_extractors.get(language)
        if not extractor:
            return [], []
        
        # Get relevant imports
        if self.context_mode == "none":
            relevant_imports = []
        elif self.context_mode == "full":
            relevant_imports = [imp.statement for imp in imports]
        else:  # minimal
            relevant_imports = extractor.get_used_names(node_text, imports)
        
        # Get relevant module scope
        if self.include_module_scope and self.context_mode != "none":
            if self.context_mode == "full":
                relevant_scope = [item.statement for item in module_scope]
            else:  # minimal
                relevant_scope = extractor.get_used_module_scope(node_text, module_scope)
        else:
            relevant_scope = []
        
        return relevant_imports, relevant_scope
    
    def _create_contextualized_token(
        self, 
        node, 
        code: str, 
        file_path: pathlib.Path,
        file_type: str,
        language: str,
        imports: List[ImportInfo],
        module_scope: List[ModuleScopeInfo]
    ) -> Token:
        """Create a token with appropriate context."""
        start_byte = node.start_byte
        end_byte = node.end_byte
        
        # Get the main content
        node_content = code[start_byte:end_byte]
        
        # Calculate line numbers for the original content
        line_start = code[:start_byte].count('\n') + 1
        line_end = code[:end_byte].count('\n') + 1
        
        # Get relevant context if enabled
        if self.include_context:
            relevant_imports, relevant_scope = self._get_relevant_context(
                node_content, imports, module_scope, language
            )
            
            # Build context
            context_parts = []
            
            # Add imports
            if relevant_imports:
                context_parts.extend(relevant_imports)
                context_parts.append('')  # Empty line
            
            # Add module scope
            if relevant_scope:
                context_parts.extend(relevant_scope)
                context_parts.append('')  # Empty line
            
            # Add the main content
            if context_parts:
                content_with_context = '\n'.join(context_parts) + node_content
            else:
                content_with_context = node_content
        else:
            content_with_context = node_content
        
        # Determine the appropriate kind
        kind = self._determine_token_kind(node.type, language)
        
        # Extract node name if possible
        node_name = self._extract_node_name(node, code, language)
        if node_name:
            kind = f"{kind}:{node_name}"
        
        return Token(
            file_path=file_path,
            file_type=file_type,
            repo="default",
            line_start=line_start,
            line_end=line_end,
            content=content_with_context,
            source="tree-sitter",
            kind=kind
        )
    
    def tokenize(self, file_path) -> Iterator[Token]:
        """Generate semantically meaningful tokens from a file using tree-sitter."""
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
        
        log.info(f"Processing {file_path} with tree-sitter {language} parser (context={self.context_mode}, module_scope={self.include_module_scope})")
        
        # Read file
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
        except UnicodeDecodeError:
            try:
                with open(file_path, 'r', encoding='latin-1') as f:
                    code = f.read()
            except Exception as e:
                log.error(f"Failed to read {file_path}: {e}")
                return
        
        file_type = file_path.suffix.lstrip('.')
        
        # Extract context elements
        imports = self._extract_imports_for_language(code, language)
        module_scope = self._extract_module_scope_for_language(code, language)
        
        log.debug(f"Found {len(imports)} imports and {len(module_scope)} module-scope items in {file_path}")
        
        # Parse the file with tree-sitter
        tree = parser.parse(bytes(code, 'utf-8'))
        
        # Track what tokens we've yielded to avoid duplicates
        yielded_spans = set()
        tokens_yielded = 0
        
        # Always yield imports as separate tokens for flexibility
        for imp in imports:
            # Create import token
            import_content = imp.statement
            
            token = Token(
                file_path=file_path,
                file_type=file_type,
                repo="default",
                line_start=imp.line_start,
                line_end=imp.line_end,
                content=import_content,
                source="tree-sitter",
                kind="import"
            )
            
            yielded_spans.add((imp.line_start, imp.line_end))
            yield token
            tokens_yielded += 1
        
        # Yield module scope items as separate tokens
        if self.include_module_scope:
            for item in module_scope:
                # Skip if already yielded
                if (item.line_start, item.line_end) in yielded_spans:
                    continue
                
                token = Token(
                    file_path=file_path,
                    file_type=file_type,
                    repo="default",
                    line_start=item.line_start,
                    line_end=item.line_end,
                    content=item.statement,
                    source="tree-sitter",
                    kind=f"module_{item.scope_type}:{item.name}"
                )
                
                yielded_spans.add((item.line_start, item.line_end))
                yield token
                tokens_yielded += 1
        
        # Extract docstrings and comments
        docstring_tokens = list(self._extract_docstrings(file_path, code, file_type, language))
        for token in docstring_tokens:
            if (token.line_start, token.line_end) not in yielded_spans:
                yielded_spans.add((token.line_start, token.line_end))
                yield token
                tokens_yielded += 1
        
        # Find all semantic definition nodes
        definition_nodes = self._find_definition_nodes(tree.root_node, code, language)
        
        # Process definition nodes into tokens with context
        for node in definition_nodes:
            start_byte = node.start_byte
            end_byte = node.end_byte
            
            # Calculate line numbers
            line_start = code[:start_byte].count('\n') + 1
            line_end = code[:end_byte].count('\n') + 1
            
            # Skip if we've already yielded this span
            if (line_start, line_end) in yielded_spans:
                continue
            
            # Create contextualized token
            token = self._create_contextualized_token(
                node, code, file_path, file_type, language, imports, module_scope
            )
            
            yielded_spans.add((line_start, line_end))
            yield token
            tokens_yielded += 1
            
            # For classes and large functions, also extract their methods/nested functions separately
            if node.type in ["class_definition", "class_declaration"] or len(token.content.split('\n')) > 20:
                nested_tokens = self._extract_nested_definitions(
                    node, code, file_path, file_type, language, imports, module_scope
                )
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
    
    def _extract_nested_definitions(
        self, 
        parent_node, 
        code: str, 
        file_path: pathlib.Path,
        file_type: str,
        language: str,
        imports: List[ImportInfo],
        module_scope: List[ModuleScopeInfo]
    ) -> List[Token]:
        """Extract nested definitions like methods inside classes."""
        nested_tokens = []
        
        # Get language-specific method/function types
        method_types = {
            "python": {"function_definition", "method_definition"},
            "javascript": {"function_declaration", "method_definition", "arrow_function"},
            "typescript": {"function_declaration", "method_definition", "arrow_function"},
            "java": {"method_declaration"},
            "php": {"method_declaration"},
            "perl": {"subroutine_declaration", "method_declaration"},
            "c": {"function_definition"},
            "cpp": {"function_definition", "method_declaration"},
            "rust": {"function_item"},
            "go": {"function_declaration", "method_declaration"},
        }.get(language, {"function", "method"})
        
        def traverse(node):
            if node.type in method_types:
                # Create contextualized token for nested definition
                token = self._create_contextualized_token(
                    node, code, file_path, file_type, language, imports, module_scope
                )
                nested_tokens.append(token)
            
            # Continue traversing
            for child in node.children:
                traverse(child)
        
        traverse(parent_node)
        return nested_tokens
    
    def _extract_docstrings(self, file_path, code, file_type, language):
        """Extract docstrings and comment blocks as separate tokens."""
        # Language-specific docstring patterns
        docstring_patterns = {
            "python": r'""".*?"""|\'\'\'.*?\'\'\'',
            "javascript": r'/\*\*[\s\S]*?\*/',
            "typescript": r'/\*\*[\s\S]*?\*/',
            "java": r'/\*\*[\s\S]*?\*/',
            "php": r'/\*\*[\s\S]*?\*/',
            "perl": r'=pod[\s\S]*?=cut',
            "c": r'/\*\*[\s\S]*?\*/',
            "cpp": r'/\*\*[\s\S]*?\*/',
            "rust": r'///.*?$|/\*\*[\s\S]*?\*/',
            "go": r'/\*\*[\s\S]*?\*/',
        }
        
        pattern_str = docstring_patterns.get(language)
        if not pattern_str:
            return
            
        pattern = re.compile(pattern_str, re.DOTALL | re.MULTILINE)
        
        for match in pattern.finditer(code):
            docstring = match.group(0)
            if len(docstring.strip()) < 10:  # Skip very short docstrings
                continue
                
            # Calculate line numbers
            start_pos = match.start()
            end_pos = match.end()
            line_start = code[:start_pos].count('\n') + 1
            line_end = code[:end_pos].count('\n') + 1
            
            yield Token(
                file_path=file_path,
                file_type=file_type,
                repo="default",
                line_start=line_start,
                line_end=line_end,
                content=docstring,
                source="tree-sitter",
                kind="docstring"
            )
    
    def _find_definition_nodes(self, root_node, code, language):
        """Find all function and class definition nodes."""
        # Get language-specific definition types
        definition_types = DEFINITION_TYPES.get(language, DEFINITION_TYPES.get("default", set()))
        
        definitions = []
        
        def traverse(node):
            # Check if this node is a definition we're interested in
            if node.type in definition_types:
                # Skip very small nodes
                node_text = code[node.start_byte:node.end_byte]
                if len(node_text.strip()) >= 10:
                    definitions.append(node)
            
            # Continue traversing children
            for child in node.children:
                traverse(child)
        
        traverse(root_node)
        return definitions
    
    def _determine_token_kind(self, node_type, language):
        """Map tree-sitter node types to RAG-appropriate token kinds."""
        # First check our RAG_TOKEN_KINDS mapping
        if node_type in RAG_TOKEN_KINDS:
            return RAG_TOKEN_KINDS[node_type]
        
        # Language-specific mappings
        language_mappings = {
            "python": {
                "function_definition": "function",
                "class_definition": "class",
                "decorated_definition": "decorated_function"
            },
            "javascript": {
                "function_declaration": "function",
                "class_declaration": "class",
                "method_definition": "method",
                "arrow_function": "function"
            },
            "typescript": {
                "function_declaration": "function",
                "class_declaration": "class",
                "method_definition": "method",
                "arrow_function": "function",
                "interface_declaration": "interface"
            },
            "java": {
                "method_declaration": "method",
                "class_declaration": "class",
                "interface_declaration": "interface"
            },
            "php": {
                "function_definition": "function",
                "method_declaration": "method",
                "class_declaration": "class"
            },
            "perl": {
                "subroutine_declaration": "function",
                "method_declaration": "method",
                "package_declaration": "module"
            },
            "c": {
                "function_definition": "function",
                "struct_specifier": "struct",
                "enum_specifier": "enum"
            },
            "cpp": {
                "function_definition": "function",
                "class_specifier": "class",
                "struct_specifier": "struct"
            },
            "rust": {
                "function_item": "function",
                "struct_item": "struct",
                "enum_item": "enum",
                "trait_item": "trait",
                "impl_item": "impl"
            },
            "go": {
                "function_declaration": "function",
                "method_declaration": "method",
                "type_declaration": "type"
            }
        }
        
        if language in language_mappings and node_type in language_mappings[language]:
            return language_mappings[language][node_type]
        
        # Fall back to the original node type
        return node_type
    
    def _extract_node_name(self, node, code, language):
        """Extract the name of a function, class, or method from its node."""
        # Language-specific name extraction
        if language == "python":
            for child in node.children:
                if child.type == "identifier":
                    return code[child.start_byte:child.end_byte]
                    
        elif language in ["javascript", "typescript"]:
            if node.type in ["function_declaration", "class_declaration"]:
                for child in node.children:
                    if child.type == "identifier":
                        return code[child.start_byte:child.end_byte]
            elif node.type == "method_definition":
                for child in node.children:
                    if child.type == "property_identifier":
                        return code[child.start_byte:child.end_byte]
        
        elif language == "java":
            if node.type in ["class_declaration", "method_declaration", "interface_declaration"]:
                for child in node.children:
                    if child.type == "identifier":
                        return code[child.start_byte:child.end_byte]
        
        elif language == "php":
            if node.type in ["function_definition", "method_declaration", "class_declaration"]:
                for child in node.children:
                    if child.type == "name":
                        return code[child.start_byte:child.end_byte]
        
        elif language == "perl":
            # Perl is more complex, try regex
            node_text = code[node.start_byte:node.end_byte]
            if node.type == "subroutine_declaration":
                match = re.search(r'sub\s+(\w+)', node_text)
                if match:
                    return match.group(1)
            elif node.type == "package_declaration":
                match = re.search(r'package\s+([\w:]+)', node_text)
                if match:
                    return match.group(1)
        
        elif language in ["c", "cpp"]:
            # For C/C++, look for function names
            node_text = code[node.start_byte:node.end_byte]
            if node.type == "function_definition":
                # Match function name pattern
                match = re.search(r'(?:^|\s)(\w+)\s*\([^)]*\)\s*{', node_text)
                if match:
                    return match.group(1)
        
        elif language == "rust":
            # For Rust, find identifier after fn/struct/enum/trait
            for child in node.children:
                if child.type == "identifier":
                    return code[child.start_byte:child.end_byte]
        
        elif language == "go":
            # For Go, find identifier in type or function declaration
            for child in node.children:
                if child.type == "identifier":
                    return code[child.start_byte:child.end_byte]
        
        return ""


class LanguageContextExtractor:
    """Base class for language-specific context extraction."""
    
    def extract_imports(self, code: str) -> List[ImportInfo]:
        """Extract all imports from code."""
        return []
    
    def extract_module_scope(self, code: str) -> List[ModuleScopeInfo]:
        """Extract module-level declarations."""
        return []
    
    def get_used_names(self, code_snippet: str, imports: List[ImportInfo]) -> List[str]:
        """Determine which imports are actually used in a code snippet."""
        used_imports = []
        for imp in imports:
            for name in imp.imported_names:
                # Check if the name is used as a word boundary
                if re.search(r'\b' + re.escape(name) + r'\b', code_snippet):
                    used_imports.append(imp.statement)
                    break
        return list(dict.fromkeys(used_imports))  # Remove duplicates while preserving order
    
    def get_used_module_scope(self, code_snippet: str, module_scope: List[ModuleScopeInfo]) -> List[str]:
        """Determine which module-scope items are used in a code snippet."""
        used_scope = []
        for item in module_scope:
            if re.search(r'\b' + re.escape(item.name) + r'\b', code_snippet):
                used_scope.append(item.statement)
        return list(dict.fromkeys(used_scope))  # Remove duplicates


class PythonContextExtractor(LanguageContextExtractor):
    """Extract context from Python code."""
    
    def extract_imports(self, code: str) -> List[ImportInfo]:
        imports = []
        lines = code.split('\n')
        
        # Regex patterns for Python imports
        import_pattern = re.compile(r'^import\s+([\w\.,\s]+)')
        from_pattern = re.compile(r'^from\s+([\w\.]+)\s+import\s+(.+)')
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            
            # Skip if inside string or comment
            if line_stripped.startswith('#') or line_stripped.startswith('"""') or line_stripped.startswith("'''"):
                continue
            
            # Standard import
            match = import_pattern.match(line_stripped)
            if match:
                modules = [m.strip() for m in match.group(1).split(',')]
                imported_names = set()
                for module in modules:
                    # Handle 'import x as y'
                    if ' as ' in module:
                        _, alias = module.split(' as ')
                        imported_names.add(alias.strip())
                    else:
                        # Use the first part of module name
                        imported_names.add(module.split('.')[0])
                
                imports.append(ImportInfo(
                    statement=line.rstrip(),
                    imported_names=imported_names,
                    line_start=i + 1,
                    line_end=i + 1,
                    import_type='module'
                ))
            
            # From import
            match = from_pattern.match(line_stripped)
            if match:
                module = match.group(1)
                items = match.group(2)
                
                if items.strip() == '*':
                    # Can't determine names for star imports
                    imported_names = {'*'}
                else:
                    # Parse imported items
                    imported_names = set()
                    # Handle parentheses for multi-line imports
                    if '(' in items:
                        # Multi-line import, need to handle continuation
                        j = i
                        while ')' not in lines[j] and j < len(lines) - 1:
                            j += 1
                            items += ' ' + lines[j].strip()
                    
                    # Parse the items
                    items = items.replace('(', '').replace(')', '')
                    for item in items.split(','):
                        item = item.strip()
                        if ' as ' in item:
                            _, alias = item.split(' as ')
                            imported_names.add(alias.strip())
                        else:
                            imported_names.add(item)
                
                imports.append(ImportInfo(
                    statement=line.rstrip(),
                    imported_names=imported_names,
                    line_start=i + 1,
                    line_end=i + 1,
                    import_type='from'
                ))
        
        return imports
    
    def extract_module_scope(self, code: str) -> List[ModuleScopeInfo]:
        """Extract module-level variables, constants, and type annotations."""
        module_scope = []
        lines = code.split('\n')
        
        # Track indentation to identify module-level
        in_class = False
        in_function = False
        indent_stack = [0]
        
        # Patterns for module-level declarations
        constant_pattern = re.compile(r'^([A-Z_][A-Z0-9_]*)\s*=\s*(.+)')
        variable_pattern = re.compile(r'^([a-z_][a-zA-Z0-9_]*)\s*=\s*(.+)')
        type_alias_pattern = re.compile(r'^(\w+)\s*:\s*(?:Type\[.+\]|type)\s*=\s*(.+)')
        typed_var_pattern = re.compile(r'^(\w+)\s*:\s*([^=]+)(?:\s*=\s*(.+))?')
        
        for i, line in enumerate(lines):
            # Skip empty lines and comments
            if not line.strip() or line.strip().startswith('#'):
                continue
            
            # Calculate indentation
            indent = len(line) - len(line.lstrip())
            
            # Check if we're entering/leaving a class or function
            if re.match(r'^(async\s+)?def\s+', line.strip()):
                in_function = True
                indent_stack.append(indent)
            elif re.match(r'^class\s+', line.strip()):
                in_class = True
                indent_stack.append(indent)
            elif indent <= indent_stack[-1] and (in_class or in_function):
                # We've left the class/function
                indent_stack.pop()
                if len(indent_stack) == 1:
                    in_class = False
                    in_function = False
            
            # Only process module-level (indent 0) declarations
            if indent == 0 and not in_class and not in_function:
                line_stripped = line.strip()
                
                # Skip imports and docstrings
                if (line_stripped.startswith('import ') or 
                    line_stripped.startswith('from ') or
                    line_stripped.startswith('"""') or
                    line_stripped.startswith("'''")):
                    continue
                
                # Check for type alias
                match = type_alias_pattern.match(line_stripped)
                if match:
                    name = match.group(1)
                    module_scope.append(ModuleScopeInfo(
                        statement=line.rstrip(),
                        name=name,
                        line_start=i + 1,
                        line_end=i + 1,
                        scope_type='type_alias'
                    ))
                    continue
                
                # Check for typed variable
                match = typed_var_pattern.match(line_stripped)
                if match and ':' in line_stripped and '(' not in line_stripped.split(':')[0]:
                    name = match.group(1)
                    # Skip if it's a function definition
                    if not re.match(r'^(async\s+)?def\s+', line_stripped):
                        module_scope.append(ModuleScopeInfo(
                            statement=line.rstrip(),
                            name=name,
                            line_start=i + 1,
                            line_end=i + 1,
                            scope_type='typed_variable'
                        ))
                        continue
                
                # Check for constant (UPPER_CASE)
                match = constant_pattern.match(line_stripped)
                if match:
                    name = match.group(1)
                    module_scope.append(ModuleScopeInfo(
                        statement=line.rstrip(),
                        name=name,
                        line_start=i + 1,
                        line_end=i + 1,
                        scope_type='constant'
                    ))
                    continue
                
                # Check for regular variable
                match = variable_pattern.match(line_stripped)
                if match:
                    name = match.group(1)
                    # Skip if it's __name__, __file__, etc.
                    if not name.startswith('__'):
                        module_scope.append(ModuleScopeInfo(
                            statement=line.rstrip(),
                            name=name,
                            line_start=i + 1,
                            line_end=i + 1,
                            scope_type='variable'
                        ))
        
        return module_scope


class JavaScriptContextExtractor(LanguageContextExtractor):
    """Extract context from JavaScript/TypeScript code."""
    
    def extract_imports(self, code: str) -> List[ImportInfo]:
        imports = []
        lines = code.split('\n')
        
        # Various import patterns in JS/TS
        patterns = [
            # import { x, y } from 'module'
            (re.compile(r'^import\s*\{([^}]+)\}\s*from\s*[\'"]([^\'\"]+)[\'"];?'), 'named'),
            # import * as x from 'module'
            (re.compile(r'^import\s*\*\s*as\s+(\w+)\s*from\s*[\'"]([^\'\"]+)[\'"];?'), 'namespace'),
            # import x from 'module'
            (re.compile(r'^import\s+(\w+)\s*from\s*[\'"]([^\'\"]+)[\'"];?'), 'default'),
            # import 'module'
            (re.compile(r'^import\s*[\'"]([^\'\"]+)[\'"];?'), 'side-effect'),
            # const x = require('module')
            (re.compile(r'^(?:const|let|var)\s+(\w+)\s*=\s*require\s*\([\'"]([^\'\"]+)[\'"]\)'), 'require'),
            # const { x, y } = require('module')
            (re.compile(r'^(?:const|let|var)\s+\{([^}]+)\}\s*=\s*require\s*\([\'"]([^\'\"]+)[\'"]\)'), 'require-destructure'),
        ]
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            
            for pattern, import_type in patterns:
                match = pattern.match(line_stripped)
                if match:
                    if import_type == 'named':
                        # Parse named imports
                        names_str = match.group(1)
                        imported_names = set()
                        for name in names_str.split(','):
                            name = name.strip()
                            if ' as ' in name:
                                _, alias = name.split(' as ')
                                imported_names.add(alias.strip())
                            else:
                                imported_names.add(name)
                    elif import_type in ['namespace', 'default', 'require']:
                        imported_names = {match.group(1)}
                    elif import_type == 'require-destructure':
                        names_str = match.group(1)
                        imported_names = {n.strip() for n in names_str.split(',')}
                    else:
                        imported_names = set()
                    
                    imports.append(ImportInfo(
                        statement=line.rstrip(),
                        imported_names=imported_names,
                        line_start=i + 1,
                        line_end=i + 1,
                        import_type=import_type
                    ))
                    break
        
        return imports
    
    def extract_module_scope(self, code: str) -> List[ModuleScopeInfo]:
        """Extract module-level constants, variables, types, and interfaces."""
        module_scope = []
        lines = code.split('\n')
        
        # Track block depth to identify module-level
        brace_count = 0
        in_function = False
        in_class = False
        
        # Patterns for module-level declarations
        const_pattern = re.compile(r'^(?:export\s+)?const\s+([A-Z_][A-Z0-9_]*)\s*=\s*(.+)')
        var_pattern = re.compile(r'^(?:export\s+)?(?:const|let|var)\s+([a-zA-Z_]\w*)\s*=\s*(.+)')
        type_pattern = re.compile(r'^(?:export\s+)?type\s+(\w+)\s*=\s*(.+)')
        interface_pattern = re.compile(r'^(?:export\s+)?interface\s+(\w+)')
        enum_pattern = re.compile(r'^(?:export\s+)?enum\s+(\w+)')
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            
            # Skip empty lines and comments
            if not line_stripped or line_stripped.startswith('//'):
                continue
            
            # Track braces to determine scope
            brace_count += line.count('{') - line.count('}')
            
            # Check if we're entering a function or class
            if re.match(r'^(async\s+)?function\s+', line_stripped) or re.match(r'^\w+\s*\([^)]*\)\s*{', line_stripped):
                in_function = True
            elif re.match(r'^class\s+', line_stripped):
                in_class = True
            
            # Reset flags when we exit blocks
            if brace_count == 0:
                in_function = False
                in_class = False
            
            # Only process module-level declarations
            if brace_count == 0 and not in_function and not in_class:
                # Skip imports
                if line_stripped.startswith('import ') or 'require(' in line_stripped:
                    continue
                
                # Check for TypeScript type
                match = type_pattern.match(line_stripped)
                if match:
                    name = match.group(1)
                    module_scope.append(ModuleScopeInfo(
                        statement=line.rstrip(),
                        name=name,
                        line_start=i + 1,
                        line_end=i + 1,
                        scope_type='type'
                    ))
                    continue
                
                # Check for interface
                match = interface_pattern.match(line_stripped)
                if match:
                    name = match.group(1)
                    # Find the end of the interface
                    j = i
                    interface_brace_count = line.count('{') - line.count('}')
                    while interface_brace_count > 0 and j < len(lines) - 1:
                        j += 1
                        interface_brace_count += lines[j].count('{') - lines[j].count('}')
                    
                    # Get full interface
                    interface_lines = lines[i:j+1]
                    module_scope.append(ModuleScopeInfo(
                        statement='\n'.join(interface_lines),
                        name=name,
                        line_start=i + 1,
                        line_end=j + 1,
                        scope_type='interface'
                    ))
                    continue
                
                # Check for enum
                match = enum_pattern.match(line_stripped)
                if match:
                    name = match.group(1)
                    module_scope.append(ModuleScopeInfo(
                        statement=line.rstrip(),
                        name=name,
                        line_start=i + 1,
                        line_end=i + 1,
                        scope_type='enum'
                    ))
                    continue
                
                # Check for constant (UPPER_CASE)
                match = const_pattern.match(line_stripped)
                if match:
                    name = match.group(1)
                    module_scope.append(ModuleScopeInfo(
                        statement=line.rstrip(),
                        name=name,
                        line_start=i + 1,
                        line_end=i + 1,
                        scope_type='constant'
                    ))
                    continue
                
                # Check for regular variable
                match = var_pattern.match(line_stripped)
                if match:
                    name = match.group(1)
                    module_scope.append(ModuleScopeInfo(
                        statement=line.rstrip(),
                        name=name,
                        line_start=i + 1,
                        line_end=i + 1,
                        scope_type='variable'
                    ))
        
        return module_scope


class CContextExtractor(LanguageContextExtractor):
    """Extract context from C code."""
    
    def extract_imports(self, code: str) -> List[ImportInfo]:
        imports = []
        lines = code.split('\n')
        
        # C include patterns
        include_pattern = re.compile(r'^#include\s*[<"]([^>"]+)[>"]')
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            
            match = include_pattern.match(line_stripped)
            if match:
                header = match.group(1)
                # Extract just the filename without path
                header_name = header.split('/')[-1].split('.')[0]
                
                imports.append(ImportInfo(
                    statement=line.rstrip(),
                    imported_names={header_name},
                    line_start=i + 1,
                    line_end=i + 1,
                    import_type='include'
                ))
        
        return imports
    
    def extract_module_scope(self, code: str) -> List[ModuleScopeInfo]:
        """Extract module-level defines, typedefs, structs, and global variables."""
        module_scope = []
        lines = code.split('\n')
        
        # Track if we're inside a function
        brace_count = 0
        in_function = False
        
        # Patterns for C module-level declarations
        define_pattern = re.compile(r'^#define\s+(\w+)')
        typedef_pattern = re.compile(r'^typedef\s+.*\s+(\w+);')
        struct_pattern = re.compile(r'^(?:typedef\s+)?struct\s+(\w*)\s*{')
        enum_pattern = re.compile(r'^(?:typedef\s+)?enum\s+(\w*)\s*{')
        global_var_pattern = re.compile(r'^(?:static\s+)?(?:const\s+)?(?:extern\s+)?(?:unsigned\s+)?(?:signed\s+)?(?:short\s+)?(?:long\s+)?(\w+(?:\s*\*)*)\s+(\w+)(?:\s*\[.*?\])?\s*(?:=.*)?;')
        function_proto_pattern = re.compile(r'^(?:static\s+)?(?:inline\s+)?(?:const\s+)?(?:unsigned\s+)?(?:signed\s+)?(?:short\s+)?(?:long\s+)?(\w+(?:\s*\*)*)\s+(\w+)\s*\([^)]*\)\s*;')
        
        i = 0
        while i < len(lines):
            line = lines[i]
            line_stripped = line.strip()
            
            # Skip empty lines and single-line comments
            if not line_stripped or line_stripped.startswith('//'):
                i += 1
                continue
            
            # Skip multi-line comments
            if '/*' in line_stripped:
                while i < len(lines) and '*/' not in lines[i]:
                    i += 1
                i += 1
                continue
            
            # Track braces
            brace_count += line.count('{') - line.count('}')
            
            # Check if we're in a function
            if re.match(r'^(?:static\s+)?(?:inline\s+)?.*\s+\w+\s*\([^)]*\)\s*{', line_stripped):
                in_function = True
            
            if brace_count == 0:
                in_function = False
            
            # Only process module-level declarations
            if not in_function:
                # Check for #define
                match = define_pattern.match(line_stripped)
                if match:
                    name = match.group(1)
                    # Handle multi-line defines
                    full_define = line.rstrip()
                    j = i
                    while j < len(lines) - 1 and lines[j].rstrip().endswith('\\'):
                        j += 1
                        full_define += '\n' + lines[j].rstrip()
                    
                    module_scope.append(ModuleScopeInfo(
                        statement=full_define,
                        name=name,
                        line_start=i + 1,
                        line_end=j + 1,
                        scope_type='macro'
                    ))
                    i = j + 1
                    continue
                
                # Check for typedef
                match = typedef_pattern.match(line_stripped)
                if match:
                    name = match.group(1)
                    module_scope.append(ModuleScopeInfo(
                        statement=line.rstrip(),
                        name=name,
                        line_start=i + 1,
                        line_end=i + 1,
                        scope_type='typedef'
                    ))
                    i += 1
                    continue
                
                # Check for struct
                match = struct_pattern.match(line_stripped)
                if match:
                    name = match.group(1) or "anonymous_struct"
                    # Find the end of the struct
                    j = i
                    struct_brace_count = line.count('{') - line.count('}')
                    while struct_brace_count > 0 and j < len(lines) - 1:
                        j += 1
                        struct_brace_count += lines[j].count('{') - lines[j].count('}')
                    
                    # Include typedef name if present
                    if j < len(lines) - 1 and 'typedef' in line_stripped:
                        typedef_match = re.search(r'}\s*(\w+)\s*;', lines[j])
                        if typedef_match:
                            name = typedef_match.group(1)
                    
                    struct_lines = lines[i:j+1]
                    module_scope.append(ModuleScopeInfo(
                        statement='\n'.join(struct_lines),
                        name=name,
                        line_start=i + 1,
                        line_end=j + 1,
                        scope_type='struct'
                    ))
                    i = j + 1
                    continue
                
                # Check for enum
                match = enum_pattern.match(line_stripped)
                if match:
                    name = match.group(1) or "anonymous_enum"
                    # Find the end of the enum
                    j = i
                    enum_brace_count = line.count('{') - line.count('}')
                    while enum_brace_count > 0 and j < len(lines) - 1:
                        j += 1
                        enum_brace_count += lines[j].count('{') - lines[j].count('}')
                    
                    enum_lines = lines[i:j+1]
                    module_scope.append(ModuleScopeInfo(
                        statement='\n'.join(enum_lines),
                        name=name,
                        line_start=i + 1,
                        line_end=j + 1,
                        scope_type='enum'
                    ))
                    i = j + 1
                    continue
                
                # Check for function prototype
                match = function_proto_pattern.match(line_stripped)
                if match:
                    name = match.group(2)
                    module_scope.append(ModuleScopeInfo(
                        statement=line.rstrip(),
                        name=name,
                        line_start=i + 1,
                        line_end=i + 1,
                        scope_type='function_prototype'
                    ))
                    i += 1
                    continue
                
                # Check for global variable
                match = global_var_pattern.match(line_stripped)
                if match and not re.match(r'.*\s+\w+\s*\([^)]*\)', line_stripped):  # Not a function
                    name = match.group(2)
                    module_scope.append(ModuleScopeInfo(
                        statement=line.rstrip(),
                        name=name,
                        line_start=i + 1,
                        line_end=i + 1,
                        scope_type='global_variable'
                    ))
            
            i += 1
        
        return module_scope


class CppContextExtractor(CContextExtractor):
    """Extract context from C++ code."""
    
    def extract_imports(self, code: str) -> List[ImportInfo]:
        imports = []
        lines = code.split('\n')
        
        # C++ include and using patterns
        include_pattern = re.compile(r'^#include\s*[<"]([^>"]+)[>"]')
        using_pattern = re.compile(r'^using\s+(?:namespace\s+)?(.+);')
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            
            # Check for includes
            match = include_pattern.match(line_stripped)
            if match:
                header = match.group(1)
                header_name = header.split('/')[-1].split('.')[0]
                
                imports.append(ImportInfo(
                    statement=line.rstrip(),
                    imported_names={header_name},
                    line_start=i + 1,
                    line_end=i + 1,
                    import_type='include'
                ))
                continue
            
            # Check for using statements
            match = using_pattern.match(line_stripped)
            if match:
                namespace = match.group(1)
                namespace_parts = namespace.split('::')
                
                imports.append(ImportInfo(
                    statement=line.rstrip(),
                    imported_names={namespace_parts[-1]},
                    line_start=i + 1,
                    line_end=i + 1,
                    import_type='using'
                ))
        
        return imports
    
    def extract_module_scope(self, code: str) -> List[ModuleScopeInfo]:
        """Extract C++ module-level declarations including namespaces, templates, etc."""
        # First get C-style declarations
        module_scope = super().extract_module_scope(code)
        
        lines = code.split('\n')
        brace_count = 0
        namespace_stack = []
        
        # Additional C++ patterns
        namespace_pattern = re.compile(r'^namespace\s+(\w+)\s*{')
        class_pattern = re.compile(r'^(?:template\s*<.*?>\s*)?class\s+(\w+)')
        template_pattern = re.compile(r'^template\s*<(.+?)>')
        constexpr_pattern = re.compile(r'^constexpr\s+.*\s+(\w+)\s*=')
        
        i = 0
        while i < len(lines):
            line = lines[i]
            line_stripped = line.strip()
            
            # Skip empty lines and comments
            if not line_stripped or line_stripped.startswith('//'):
                i += 1
                continue
            
            # Track namespace context
            if namespace_pattern.match(line_stripped):
                match = namespace_pattern.match(line_stripped)
                namespace_stack.append(match.group(1))
            
            # Track braces
            brace_count += line.count('{') - line.count('}')
            
            # Check for class declaration
            match = class_pattern.match(line_stripped)
            if match and brace_count == len(namespace_stack):
                name = match.group(1)
                # Check if it's a template
                if i > 0 and template_pattern.match(lines[i-1].strip()):
                    start_line = i - 1
                    statement = lines[i-1].rstrip() + '\n' + line.rstrip()
                else:
                    start_line = i
                    statement = line.rstrip()
                
                module_scope.append(ModuleScopeInfo(
                    statement=statement,
                    name=name,
                    line_start=start_line + 1,
                    line_end=i + 1,
                    scope_type='class_declaration'
                ))
            
            # Check for constexpr
            match = constexpr_pattern.match(line_stripped)
            if match:
                name = match.group(1)
                module_scope.append(ModuleScopeInfo(
                    statement=line.rstrip(),
                    name=name,
                    line_start=i + 1,
                    line_end=i + 1,
                    scope_type='constexpr'
                ))
            
            i += 1
        
        return module_scope


class RustContextExtractor(LanguageContextExtractor):
    """Extract context from Rust code."""
    
    def extract_imports(self, code: str) -> List[ImportInfo]:
        imports = []
        lines = code.split('\n')
        
        # Rust use statements
        use_pattern = re.compile(r'^(?:pub\s+)?use\s+(.+);')
        extern_pattern = re.compile(r'^extern\s+crate\s+(\w+);')
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            
            # Check for use statements
            match = use_pattern.match(line_stripped)
            if match:
                use_path = match.group(1)
                # Handle complex imports like use std::{io, fmt};
                if '{' in use_path:
                    base_path, items = use_path.split('{')
                    base_path = base_path.strip().rstrip('::')
                    items = items.rstrip('}').strip()
                    imported_names = set()
                    for item in items.split(','):
                        item = item.strip()
                        if ' as ' in item:
                            _, alias = item.split(' as ')
                            imported_names.add(alias.strip())
                        else:
                            imported_names.add(item)
                else:
                    # Simple import
                    parts = use_path.split('::')
                    if ' as ' in parts[-1]:
                        _, alias = parts[-1].split(' as ')
                        imported_names = {alias.strip()}
                    else:
                        imported_names = {parts[-1]}
                
                imports.append(ImportInfo(
                    statement=line.rstrip(),
                    imported_names=imported_names,
                    line_start=i + 1,
                    line_end=i + 1,
                    import_type='use'
                ))
            
            # Check for extern crate
            match = extern_pattern.match(line_stripped)
            if match:
                crate_name = match.group(1)
                imports.append(ImportInfo(
                    statement=line.rstrip(),
                    imported_names={crate_name},
                    line_start=i + 1,
                    line_end=i + 1,
                    import_type='extern_crate'
                ))
        
        return imports
    
    def extract_module_scope(self, code: str) -> List[ModuleScopeInfo]:
        """Extract Rust module-level items: consts, statics, types, structs, enums, traits."""
        module_scope = []
        lines = code.split('\n')
        
        # Track if we're inside a function or impl block
        brace_count = 0
        in_function = False
        in_impl = False
        
        # Rust patterns
        const_pattern = re.compile(r'^(?:pub(?:\(.*?\))?\s+)?const\s+(\w+)\s*:')
        static_pattern = re.compile(r'^(?:pub(?:\(.*?\))?\s+)?static\s+(?:mut\s+)?(\w+)\s*:')
        type_pattern = re.compile(r'^(?:pub(?:\(.*?\))?\s+)?type\s+(\w+)')
        struct_pattern = re.compile(r'^(?:pub(?:\(.*?\))?\s+)?struct\s+(\w+)')
        enum_pattern = re.compile(r'^(?:pub(?:\(.*?\))?\s+)?enum\s+(\w+)')
        trait_pattern = re.compile(r'^(?:pub(?:\(.*?\))?\s+)?trait\s+(\w+)')
        mod_pattern = re.compile(r'^(?:pub(?:\(.*?\))?\s+)?mod\s+(\w+)')
        
        i = 0
        while i < len(lines):
            line = lines[i]
            line_stripped = line.strip()
            
            # Skip empty lines and comments
            if not line_stripped or line_stripped.startswith('//'):
                i += 1
                continue
            
            # Track braces
            brace_count += line.count('{') - line.count('}')
            
            # Check if we're entering a function or impl block
            if re.match(r'^(?:pub(?:\(.*?\))?\s+)?(?:async\s+)?fn\s+', line_stripped):
                in_function = True
            elif re.match(r'^impl(?:<.*?>)?\s+', line_stripped):
                in_impl = True
            
            # Reset flags when we exit blocks
            if brace_count == 0:
                in_function = False
                in_impl = False
            
            # Only process module-level declarations
            if not in_function and not in_impl:
                # Check for const
                match = const_pattern.match(line_stripped)
                if match:
                    name = match.group(1)
                    # Find the end of the const declaration
                    j = i
                    while j < len(lines) - 1 and not lines[j].rstrip().endswith(';'):
                        j += 1
                    
                    const_lines = lines[i:j+1]
                    module_scope.append(ModuleScopeInfo(
                        statement='\n'.join(const_lines),
                        name=name,
                        line_start=i + 1,
                        line_end=j + 1,
                        scope_type='const'
                    ))
                    i = j + 1
                    continue
                
                # Check for static
                match = static_pattern.match(line_stripped)
                if match:
                    name = match.group(1)
                    j = i
                    while j < len(lines) - 1 and not lines[j].rstrip().endswith(';'):
                        j += 1
                    
                    static_lines = lines[i:j+1]
                    module_scope.append(ModuleScopeInfo(
                        statement='\n'.join(static_lines),
                        name=name,
                        line_start=i + 1,
                        line_end=j + 1,
                        scope_type='static'
                    ))
                    i = j + 1
                    continue
                
                # Check for type alias
                match = type_pattern.match(line_stripped)
                if match:
                    name = match.group(1)
                    module_scope.append(ModuleScopeInfo(
                        statement=line.rstrip(),
                        name=name,
                        line_start=i + 1,
                        line_end=i + 1,
                        scope_type='type_alias'
                    ))
                    i += 1
                    continue
                
                # Check for struct
                match = struct_pattern.match(line_stripped)
                if match:
                    name = match.group(1)
                    # Check if it's a tuple struct or unit struct
                    if line_stripped.endswith(';') or '(' in line_stripped:
                        module_scope.append(ModuleScopeInfo(
                            statement=line.rstrip(),
                            name=name,
                            line_start=i + 1,
                            line_end=i + 1,
                            scope_type='struct'
                        ))
                        i += 1
                    else:
                        # Find the end of the struct
                        j = i
                        struct_brace_count = line.count('{') - line.count('}')
                        while struct_brace_count > 0 and j < len(lines) - 1:
                            j += 1
                            struct_brace_count += lines[j].count('{') - lines[j].count('}')
                        
                        struct_lines = lines[i:j+1]
                        module_scope.append(ModuleScopeInfo(
                            statement='\n'.join(struct_lines),
                            name=name,
                            line_start=i + 1,
                            line_end=j + 1,
                            scope_type='struct'
                        ))
                        i = j + 1
                    continue
                
                # Check for enum
                match = enum_pattern.match(line_stripped)
                if match:
                    name = match.group(1)
                    j = i
                    enum_brace_count = line.count('{') - line.count('}')
                    while enum_brace_count > 0 and j < len(lines) - 1:
                        j += 1
                        enum_brace_count += lines[j].count('{') - lines[j].count('}')
                    
                    enum_lines = lines[i:j+1]
                    module_scope.append(ModuleScopeInfo(
                        statement='\n'.join(enum_lines),
                        name=name,
                        line_start=i + 1,
                        line_end=j + 1,
                        scope_type='enum'
                    ))
                    i = j + 1
                    continue
                
                # Check for trait
                match = trait_pattern.match(line_stripped)
                if match:
                    name = match.group(1)
                    j = i
                    trait_brace_count = line.count('{') - line.count('}')
                    while trait_brace_count > 0 and j < len(lines) - 1:
                        j += 1
                        trait_brace_count += lines[j].count('{') - lines[j].count('}')
                    
                    trait_lines = lines[i:j+1]
                    module_scope.append(ModuleScopeInfo(
                        statement='\n'.join(trait_lines),
                        name=name,
                        line_start=i + 1,
                        line_end=j + 1,
                        scope_type='trait'
                    ))
                    i = j + 1
                    continue
                
                # Check for module declaration
                match = mod_pattern.match(line_stripped)
                if match:
                    name = match.group(1)
                    module_scope.append(ModuleScopeInfo(
                        statement=line.rstrip(),
                        name=name,
                        line_start=i + 1,
                        line_end=i + 1,
                        scope_type='module'
                    ))
                    i += 1
                    continue
            
            i += 1
        
        return module_scope


class GoContextExtractor(LanguageContextExtractor):
    """Extract context from Go code."""
    
    def extract_imports(self, code: str) -> List[ImportInfo]:
        imports = []
        lines = code.split('\n')
        
        # Go import patterns
        single_import_pattern = re.compile(r'^import\s+"([^"]+)"')
        multi_import_start = re.compile(r'^import\s*\(')
        import_line_pattern = re.compile(r'^\s*(?:(\w+)\s+)?"([^"]+)"')
        
        i = 0
        while i < len(lines):
            line = lines[i]
            line_stripped = line.strip()
            
            # Single import
            match = single_import_pattern.match(line_stripped)
            if match:
                package = match.group(1)
                package_name = package.split('/')[-1]
                
                imports.append(ImportInfo(
                    statement=line.rstrip(),
                    imported_names={package_name},
                    line_start=i + 1,
                    line_end=i + 1,
                    import_type='import'
                ))
                i += 1
                continue
            
            # Multi-line imports
            if multi_import_start.match(line_stripped):
                import_block = [line.rstrip()]
                j = i + 1
                while j < len(lines) and ')' not in lines[j]:
                    import_match = import_line_pattern.match(lines[j])
                    if import_match:
                        alias = import_match.group(1)
                        package = import_match.group(2)
                        if alias:
                            imported_names = {alias}
                        else:
                            imported_names = {package.split('/')[-1]}
                        
                        imports.append(ImportInfo(
                            statement=lines[j].rstrip(),
                            imported_names=imported_names,
                            line_start=j + 1,
                            line_end=j + 1,
                            import_type='import'
                        ))
                    j += 1
                i = j + 1
                continue
            
            i += 1
        
        return imports
    
    def extract_module_scope(self, code: str) -> List[ModuleScopeInfo]:
        """Extract Go package-level declarations: consts, vars, types, interfaces."""
        module_scope = []
        lines = code.split('\n')
        
        # Track if we're inside a function
        brace_count = 0
        in_function = False
        
        # Go patterns
        package_pattern = re.compile(r'^package\s+(\w+)')
        const_pattern = re.compile(r'^const\s+(?:\(|(\w+))')
        var_pattern = re.compile(r'^var\s+(?:\(|(\w+))')
        type_pattern = re.compile(r'^type\s+(\w+)')
        
        i = 0
        while i < len(lines):
            line = lines[i]
            line_stripped = line.strip()
            
            # Skip empty lines and comments
            if not line_stripped or line_stripped.startswith('//'):
                i += 1
                continue
            
            # Track braces
            brace_count += line.count('{') - line.count('}')
            
            # Check if we're entering a function
            if re.match(r'^func\s+(?:\(\w+\s+\*?\w+\)\s+)?(\w+)', line_stripped):
                in_function = True
            
            if brace_count == 0:
                in_function = False
            
            # Only process package-level declarations
            if not in_function:
                # Check for const declaration
                match = const_pattern.match(line_stripped)
                if match:
                    if match.group(1):  # Single const
                        name = match.group(1)
                        module_scope.append(ModuleScopeInfo(
                            statement=line.rstrip(),
                            name=name,
                            line_start=i + 1,
                            line_end=i + 1,
                            scope_type='const'
                        ))
                        i += 1
                    else:  # Const block
                        j = i + 1
                        while j < len(lines) and ')' not in lines[j]:
                            const_line = lines[j].strip()
                            if const_line and not const_line.startswith('//'):
                                name_match = re.match(r'^\s*(\w+)', const_line)
                                if name_match:
                                    name = name_match.group(1)
                                    module_scope.append(ModuleScopeInfo(
                                        statement=lines[j].rstrip(),
                                        name=name,
                                        line_start=j + 1,
                                        line_end=j + 1,
                                        scope_type='const'
                                    ))
                            j += 1
                        i = j + 1
                    continue
                
                # Check for var declaration
                match = var_pattern.match(line_stripped)
                if match:
                    if match.group(1):  # Single var
                        name = match.group(1)
                        module_scope.append(ModuleScopeInfo(
                            statement=line.rstrip(),
                            name=name,
                            line_start=i + 1,
                            line_end=i + 1,
                            scope_type='var'
                        ))
                        i += 1
                    else:  # Var block
                        j = i + 1
                        while j < len(lines) and ')' not in lines[j]:
                            var_line = lines[j].strip()
                            if var_line and not var_line.startswith('//'):
                                name_match = re.match(r'^\s*(\w+)', var_line)
                                if name_match:
                                    name = name_match.group(1)
                                    module_scope.append(ModuleScopeInfo(
                                        statement=lines[j].rstrip(),
                                        name=name,
                                        line_start=j + 1,
                                        line_end=j + 1,
                                        scope_type='var'
                                    ))
                            j += 1
                        i = j + 1
                    continue
                
                # Check for type declaration
                match = type_pattern.match(line_stripped)
                if match:
                    name = match.group(1)
                    # Check what kind of type
                    if 'struct' in line_stripped:
                        # Find the end of the struct
                        j = i
                        struct_brace_count = line.count('{') - line.count('}')
                        while struct_brace_count > 0 and j < len(lines) - 1:
                            j += 1
                            struct_brace_count += lines[j].count('{') - lines[j].count('}')
                        
                        struct_lines = lines[i:j+1]
                        module_scope.append(ModuleScopeInfo(
                            statement='\n'.join(struct_lines),
                            name=name,
                            line_start=i + 1,
                            line_end=j + 1,
                            scope_type='struct'
                        ))
                        i = j + 1
                    elif 'interface' in line_stripped:
                        # Find the end of the interface
                        j = i
                        interface_brace_count = line.count('{') - line.count('}')
                        while interface_brace_count > 0 and j < len(lines) - 1:
                            j += 1
                            interface_brace_count += lines[j].count('{') - lines[j].count('}')
                        
                        interface_lines = lines[i:j+1]
                        module_scope.append(ModuleScopeInfo(
                            statement='\n'.join(interface_lines),
                            name=name,
                            line_start=i + 1,
                            line_end=j + 1,
                            scope_type='interface'
                        ))
                        i = j + 1
                    else:
                        # Simple type alias
                        module_scope.append(ModuleScopeInfo(
                            statement=line.rstrip(),
                            name=name,
                            line_start=i + 1,
                            line_end=i + 1,
                            scope_type='type_alias'
                        ))
                        i += 1
                    continue
            
            i += 1
        
        return module_scope


class JavaContextExtractor(LanguageContextExtractor):
    """Extract context from Java code."""
    
    def extract_imports(self, code: str) -> List[ImportInfo]:
        imports = []
        lines = code.split('\n')
        
        import_pattern = re.compile(r'^import\s+(static\s+)?([\w\.]+)(\.\*)?;')
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            
            match = import_pattern.match(line_stripped)
            if match:
                is_static = bool(match.group(1))
                package = match.group(2)
                is_wildcard = bool(match.group(3))
                
                if is_wildcard:
                    imported_names = {package.split('.')[-1] + '.*'}
                else:
                    # Use the class name (last part)
                    imported_names = {package.split('.')[-1]}
                
                imports.append(ImportInfo(
                    statement=line.rstrip(),
                    imported_names=imported_names,
                    line_start=i + 1,
                    line_end=i + 1,
                    import_type='static' if is_static else 'import'
                ))
        
        return imports


class TypeScriptContextExtractor(JavaScriptContextExtractor):
    """TypeScript uses the same extraction as JavaScript."""
    pass


class PhpContextExtractor(LanguageContextExtractor):
    """Extract context from PHP code."""
    
    def extract_imports(self, code: str) -> List[ImportInfo]:
        imports = []
        lines = code.split('\n')
        
        # PHP use statements
        use_pattern = re.compile(r'^use\s+([\w\\]+)(?:\s+as\s+(\w+))?;')
        namespace_pattern = re.compile(r'^namespace\s+([\w\\]+);')
        
        current_namespace = None
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            
            # Track namespace
            match = namespace_pattern.match(line_stripped)
            if match:
                current_namespace = match.group(1)
                continue
            
            # Use statements
            match = use_pattern.match(line_stripped)
            if match:
                full_class = match.group(1)
                alias = match.group(2)
                
                if alias:
                    imported_names = {alias}
                else:
                    # Use the class name (last part)
                    imported_names = {full_class.split('\\')[-1]}
                
                imports.append(ImportInfo(
                    statement=line.rstrip(),
                    imported_names=imported_names,
                    line_start=i + 1,
                    line_end=i + 1,
                    import_type='use'
                ))
        
        return imports


class PerlContextExtractor(LanguageContextExtractor):
    """Extract context from Perl code."""
    
    def extract_imports(self, code: str) -> List[ImportInfo]:
        imports = []
        lines = code.split('\n')
        
        # Perl use/require statements
        use_pattern = re.compile(r'^use\s+([\w:]+)(?:\s+qw\(([^)]+)\))?(?:\s+(.+))?;')
        require_pattern = re.compile(r'^require\s+([\w:]+|[\'"][\w:]+[\'"])(?:\s+(.+))?;')
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            
            # use statements
            match = use_pattern.match(line_stripped)
            if match:
                module = match.group(1)
                qw_imports = match.group(2)
                other_imports = match.group(3)
                
                imported_names = {module}
                if qw_imports:
                    # Parse qw() imports
                    imported_names.update(qw_imports.split())
                
                imports.append(ImportInfo(
                    statement=line.rstrip(),
                    imported_names=imported_names,
                    line_start=i + 1,
                    line_end=i + 1,
                    import_type='use'
                ))
            
            # require statements
            match = require_pattern.match(line_stripped)
            if match:
                module = match.group(1).strip('\'"')
                imported_names = {module}
                
                imports.append(ImportInfo(
                    statement=line.rstrip(),
                    imported_names=imported_names,
                    line_start=i + 1,
                    line_end=i + 1,
                    import_type='require'
                ))
        
        return imports

