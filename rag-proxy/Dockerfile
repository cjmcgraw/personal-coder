FROM python:3.10-slim
WORKDIR /app

RUN pip install --no-cache-dir flask requests

COPY app.py .

CMD ["python", "app.py"]
