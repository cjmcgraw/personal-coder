import pathlib
import logging
import json
import time
import sys
import os
import io

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

import fastapi
import numpy as np

from .rag import ollama_util
from .rag import tokenizers

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


@app.get("/test/tokenizer/{filepath:path}", response_class=fastapi.responses.PlainTextResponse)
def test_process_file(filepath: str, tokenizer: str=None):
    logging.info(f"searching for a file named: {filepath}")
    files = [
        file
        for file in code_dir.rglob(f"**/*{filepath}*")
        if file.is_file()
    ]

    if len(files) <= 0:
        logging.warning(f"file not found matching: {filepath}")
        return {}, 404

    logging.info(f"found {len(files)} matching files")
    choice = files[0]

    logging.info(f"using {choice}")
    
    limit = set([tokenizer]) if tokenizer else None
    kwargs = dict(limit=limit)
    t = time.time()
    with io.StringIO() as buf:
        for i, token in enumerate(tokenizers.tokenize(choice, **kwargs)):
            data = token.model_dump()
            data['file_path'] = str(token.file_path)
            content = data.pop('content')

            print(f"chunk={i}", file=buf)
            print(f"{json.dumps(data, indent=4)}", file=buf)
            for n, line in enumerate(content.split("\n")):
                print(f"  {n:<3} : {token.line_start + n:<3} | {line}", file=buf)
            print("\n", file=buf)
        
        log.info(f"took {time.time() - t:,.4f}s to process {filepath}")

        
        return fastapi.responses.StreamingResponse(
            iter([buf.getvalue()]),
            media_type='text/plain'
        )
