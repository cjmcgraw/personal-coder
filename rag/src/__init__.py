import pathlib

import pydantic


class CodeChunk(pydantic.BaseModel):
    file_path: pathlib.Path
    file_type: str
    repo: str
    line_start: int
    line_end: int
    content: str


class CodeDocument(CodeChunk):
    embedding: list[float]

