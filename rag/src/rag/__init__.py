import pathlib
import pydantic


# Token model at the very top
class Token(pydantic.BaseModel):
    file_path: pathlib.Path
    file_type: str
    repo: str
    line_start: int
    line_end: int
    content: str
    source: str
    kind: str = ""

class CodeDocument(Token):
    embedding: list[float]

