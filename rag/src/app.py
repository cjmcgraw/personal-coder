from uuid import uuid4
import subprocess
import asyncio
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



@app.post("/ingest-code-base")
def ingest_code_base(force: bool=False, background_tasks: fastapi.BackgroundTasks=None):
    runid = uuid4().hex
    background_tasks.add_task(background_rag, runid=runid, force=force)
    return dict(
        message="started background task",
        runid=runid
    )


running = False
async def background_rag(runid, force=False):
    prefix = f"({runid=}):background_rag - "
    global running
    if running:
        await asyncio.sleep(0.1)
        if running:
            log.error(f"{prefix} cannot start task. Its already running!")
            raise ValueError(f"{prefix} cannot start task. Its already running!")

    cmd = f"cd /workdir && python -u -m src.rag.run_rag --runid {runid}"
    if force:
        cmd += " --force-reload"

    running = True
    try:
        await asyncio.sleep(0.250)
        log.info(f"{prefix} running command: {cmd}")
        await asyncio.to_thread(subprocess.run, cmd, shell=True, check=True, stdout=sys.stdout, stderr=sys.stderr)
        log.info(f"{prefix} successfully finished command: {cmd}")
    except Exception as err:
        log.error(f"{prefix} rag process unexpectedly error {cmd=}!")
        log.error(err)
    finally:
        running = False

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
