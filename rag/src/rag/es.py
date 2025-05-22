import itertools as its
import logging
import pathlib
import time

import pydantic
import elasticsearch
import elasticsearch.helpers

from .. import CodeDocument

log = logging.getLogger(__name__)

es_client = elasticsearch.Elasticsearch(hosts="elasticsearch:9200")
CODE_INDEX_NAME = "code"

DEFAULT_MAPPING = {
    "mappings": {
        "properties": {
            "content": {"type": "text"},
            "metadata": {
                "properties": {
                    "file_path": {"type": "keyword"},
                    "file_type": {"type": "keyword"},
                    "repo": {"type": "keyword"},
                    "line_start": {"type": "integer"},
                    "line_end": {"type": "integer"},
                }
            },
            "vector": {
                "type": "dense_vector",
                "dims": 384,
                "index": True,
                "similarity": "cosine"
            }
        }
    }
}



def bulk_index_code(data: list[CodeDocument], batch_size=500):
    if not es_client.indices.exists(index=CODE_INDEX_NAME):
        log.warning(f"didn't find index='{CODE_INDEX_NAME}' Creating.. this should only happen at start up ideally!")
        es_client.indices.create(index=CODE_INDEX_NAME, body=DEFAULT_MAPPING)

    log.warning(f"beginning insert of {len(data)} records")
    start = time.time()

    for i, batch in enumerate(its.batched(data, batch_size)):
        log.debug(f"processing batch={i} of {len(batch)=}")
        batch_start = time.time()
        operations = [{
            "_index": CODE_INDEX_NAME,
            "_source": dict(
                context=record.context,
                vector=record.embedding,
                metadata=dict(
                    file_path=str(record.file_path),
                    file_type=record.file_type,
                    repo=record.repo,
                    line_start=record.line_start,
                    line_end=record.line_end,
                )
            )
        } for record in batch]

        elasticsearch.helpers.bulk(
            es_client,
            operations,
            stats_only=False,  # Get full error details
            raise_on_error=True  # Don't stop on errors
        )
        log.debug(f"finished batch={i} of {len(batch)=} in {(time.time() - batch_start):,.2f}s")

    log.warning(f"finished all bulk inserts of {len(data)} records. Took={(time.time() - start):,.2f}s")