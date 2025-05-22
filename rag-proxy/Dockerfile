FROM python:3.10-slim

RUN --mount=type=cache,sharing=locked,target=/var/lib/apt/lists \
    --mount=type=cache,sharing=locked,target=/var/cache/apt,type=cache \
    rm -f /etc/apt/apt.conf.d/docker-clean \
    && apt-get update -yqq \
    && apt-get install -yqq curl build-essential telnet iputils-ping jq vim

WORKDIR /app
HEALTHCHECK --timeout=1s --interval=10s --retries=3 --start-period=10s CMD ["curl", "-f", "http://localhost:3000/"]

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app.py .
CMD ["python", "app.py"]
