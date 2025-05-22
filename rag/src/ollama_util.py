import requests
import traceback
import logging
import time
import os

from . import CodeDocument

logger = logging.getLogger(__name__)

EMBEDDINGS_MODEL = os.environ.get("EMBEDDINGS_MODEL", "all-minilm:l6-v2")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://ollama:11434")


def get_embeddings(text, retries=5, **kwargs):
    "retrieve the embeddings from ollama via http"
    for attempt in range(retries + 1):
        logger.debug(f"requesting embedding ({attempt=}/{retries})")
        try:
            return _get_embeddings(text, **kwargs)
        except Exception as err:
            logger.exception(err)
            time.sleep(2 ** attempt)

    raise Exception(f"failed to get embeddings after {retries=}")


def _get_embeddings(text, timeout=60):
    logger.debug(f"text length={len(text)}")
    logger.debug(f"text preview: {text[:25]}...")

    response = requests.post(
        f"{OLLAMA_URL}/api/embeddings",
        json={"model": EMBEDDINGS_MODEL, "prompt": text},
        timeout=timeout
    )

    response.raise_for_status()
    result = response.json()
    assert 'embedding' in result, f"""
        Missing embedding from ollama response!!

        {response.status_code=}
        {response=}
        {result=}
    """
    embedding = result['embedding']
    return embedding
