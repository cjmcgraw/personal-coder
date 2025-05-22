import logging
import pathlib
import json
import re
import os

from tree_sitter_language_pack import get_binding, get_language, get_parser

from .. import CodeChunk

log = logging.getLogger(__name__)
lang_lookup_file = pathlib.Path(__file__).with_name("suffix-language-mapping.json")
assert lang_lookup_file.exists()
with lang_lookup_file.open('r') as f:
    language_map = json.load(f)

log.info(f"tree sitter loaded a language map of {len(language_map)} keys to languages")


def _get_language(filepath: pathlib.Path) -> str:
    if filepath.name in language_map:
        return language_map[filepath.name]

    sfx = filepath.suffix.lower()
    return language_map.get(sfx)


def process_file(filepath: pathlib.Path, max_size=1500) -> list[CodeChunk]:
    language = _get_language(filepath)
    if not language:
        log.error(f"warning: unsupported file: {filepath}")
        return []


    parser = get_parser(language)
    if not parser:
        log.error(f"Failed to find parser for: {filepath}")
        return []

    with filepath.open('r', encoding='utf-8') as f:
        code = f.read()

    kwargs = dict(
        file_path=str(filepath),
        repo_name="???",
        code=code,
        parser=parser,
        max_size=max_size,
    )

    return [chunk for chunk in process_chunk(**kwargs)]

# Update your definition_types and complete_types sets to include YAML nodes
definition_types = {
    'function_definition', 'class_definition', 'method_definition',
    'function_declaration', 'class_declaration', 'method_declaration',
    'function', 'class', 'method',
    # YAML specific types
    'block_mapping_pair', 'block_sequence_item', 'block_mapping', 'block_sequence'
}

complete_types = {
    'statement', 'expression_statement', 'return_statement',
    'if_statement', 'for_statement', 'while_statement',
    # YAML specific types
    'block_mapping_pair', 'block_sequence_item', 'flow_mapping', 'flow_sequence'
}

import_pattern = re.compile('^(?:import|from)[^\n]+$')


def _extract_context(code):
    """Extract context like imports that should be included with a chunk"""
    import_lines = re.findall(import_pattern, code)
    return '\n'.join(import_lines) + '\n\n' if import_lines else ''


def _get_line_numbers(code, start_byte, end_byte):
    """Calculate line numbers based on byte positions"""
    # Get the code up to the start and end positions
    code_before_start = code[:start_byte]
    code_before_end = code[:end_byte]

    # Count newlines to determine line numbers
    line_start = code_before_start.count('\n') + 1
    line_end = code_before_end.count('\n') + 1

    return line_start, line_end


def process_chunk(file_path, repo_name, code, parser, max_size=150_00):
    """
    Process a file and yield CodeChunk objects

    Args:
        file_path: Path to the file
        repo_name: Name of the repository
        code: Source code content
        parser: Initialized Tree-sitter parser
        max_size: Maximum chunk size in characters
    """
    # Parse the code with Tree-sitter
    tree = parser.parse(bytes(code, 'utf-8'))

    # Extract file type from path
    file_type = pathlib.Path(file_path).suffix.lstrip('.')

    # Extract global context (imports)
    global_context = _extract_context(code)

    # Convert path to pathlib.Path
    path_obj = pathlib.Path(file_path)

    # Process the syntax tree
    yield from _process_node(
        node=tree.root_node,
        code=code,
        max_size=max_size,
        file_path=path_obj,
        file_type=file_type,
        repo=repo_name,
        global_context=global_context
    )


def _process_node(node, code, max_size, file_path, file_type, repo, global_context, current_chunk=""):
    """
    Recursively process nodes in the syntax tree to create semantically meaningful chunks.

    Args:
        node: Current tree-sitter node
        code: Original source code string
        max_size: Maximum chunk size in characters
        file_path: Path to the file
        file_type: Type of the file (extension)
        repo: Repository name
        global_context: Global context like imports
        current_chunk: Accumulated text for the current chunk
    """
    # Get node text and positions
    start_byte = node.start_byte
    end_byte = node.end_byte
    node_text = code[start_byte:end_byte]
    node_size = len(node_text)

    # Calculate line numbers
    line_start, line_end = _get_line_numbers(code, start_byte, end_byte)

    # If node is a top-level definition (function, class, etc.)
    if node.type in definition_types:
        # If this node is too big for a single chunk, process its children
        if node_size > max_size:
            # Process children individually
            for child in node.children:
                yield from _process_node(
                    child, code, max_size, file_path, file_type,
                    repo, global_context, current_chunk=""
                )
        else:
            # This definition fits in a chunk, yield it as a CodeChunk
            yield CodeChunk(
                file_path=file_path,
                file_type=file_type,
                repo=repo,
                line_start=line_start,
                line_end=line_end,
                content=node_text,
            )

    # If not a definition, try to add to current chunk
    else:
        # If adding this node would exceed max size, yield current chunk and start new one
        if len(current_chunk) + node_size > max_size:
            if current_chunk:  # Only yield non-empty chunks
                # We need start/end lines for the accumulated chunk, which is tricky
                # For now, use the node's line_end as an approximation
                yield CodeChunk(
                    file_path=file_path,
                    file_type=file_type,
                    repo=repo,
                    line_start=line_start - len(current_chunk.split('\n')),
                    line_end=line_start,
                    content=current_chunk,
                )

            # If the node itself is too big, recursively process its children
            if node_size > max_size:
                for child in node.children:
                    yield from _process_node(
                        child, code, max_size, file_path, file_type,
                        repo, global_context, current_chunk=""
                    )
            else:
                # Start a new chunk with this node
                yield CodeChunk(
                    file_path=file_path,
                    file_type=file_type,
                    repo=repo,
                    line_start=line_start,
                    line_end=line_end,
                    content=node_text,
                )
        else:
            # Add to current chunk
            new_chunk = current_chunk + node_text

            # If this completes a logical unit, yield it as a chunk
            if node.type in complete_types:
                yield CodeChunk(
                    file_path=file_path,
                    file_type=file_type,
                    repo=repo,
                    line_start=line_start - len(current_chunk.split('\n')),
                    line_end=line_end,
                    content=new_chunk,
                )
            else:
                # Return updated chunk for accumulation by caller
                return new_chunk
