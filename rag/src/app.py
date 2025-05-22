import pathlib
import logging
import sys
import os

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

import fastapi
import numpy as np

from . import ollama_util
from .rag.tokenizers import treesitter_processor

log = logging.getLogger(__name__)

app = fastapi.FastAPI()


@app.get("/health")
def health():
    return "green"


# some testing endpoints to check that things are working as expected
@app.get("/test/embeddings")
def test_embeddings(text: str):
    embeddings = ollama_util.get_embeddings(text)
    return embeddings


@app.get("/test/embeddings/distance")
def test_embeddings_difference(a: str, b:str):
    a_vec = np.array(ollama_util.get_embeddings(a))
    b_vec = np.array(ollama_util.get_embeddings(b))

    dot = np.dot(a_vec, b_vec)
    norms = np.linalg.norm(a_vec) * np.linalg.norm(b_vec)
    return {
        "euclidean": np.linalg.norm(b_vec - a_vec),
        "cosine": 1 - (dot / norms),
    }


code_dir = pathlib.Path("/code")


@app.get("/test/process/tree-sitter/{filename}", response_class=fastapi.responses.PlainTextResponse)
def test_process_file(filename: str):
    logging.info(f"searching for a file named: {filename}")
    files = [
        file
        for file in code_dir.rglob(f"**/*{filename}*")
        if file.is_file()
    ]

    if len(files) <= 0:
        logging.warning(f"file not found matching: {filename}")
        return {}, 404

    logging.info(f"found {len(files)} matching files")
    choice = files[0]

    logging.info(f"using {choice}")
    chunks = treesitter_processor.process_file(choice)
    chunk_data = ""
    content = ""
    for i, chunk in enumerate(chunks):
        chunk_data += f"chunk={i} {chunk.repo} {chunk.file_path} {chunk.file_type} {chunk.line_start} {chunk.line_end}\n"
        content += f"""
 (chunk={i})
{chunk.content}
"""

    return "\n".join([chunk_data, content])
