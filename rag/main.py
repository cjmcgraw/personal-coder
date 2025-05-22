import logging
import pathlib
import sys
import os

import fastapi

from .

app = fastapi.FastAPI()


app.get("/health")
def health_check():
    return "green"


app.get("/status")