FROM python:3.13

RUN --mount=type=cache,sharing=locked,target=/var/lib/apt/lists \
    --mount=type=cache,sharing=locked,target=/var/cache/apt,type=cache \
    rm -f /etc/apt/apt.conf.d/docker-clean \
    && apt-get update -yqq \
    && apt-get install -yqq curl build-essential telnet iputils-ping jq vim


WORKDIR /workdir
HEALTHCHECK --timeout=1s --interval=10s --retries=3 --start-period=10s CMD ["curl", "-f", "http://localhost:8000/health"]

# Install Python dependencies
COPY requirements.txt .
RUN pip install  -r requirements.txt

# Copy application code
COPY src ./src

# Run the service
ENTRYPOINT ["python", "-m", "debugpy", "--listen", "0.0.0.0:5678", "-m", "fastapi", "dev", "--host", "0.0.0.0", "--port", "8000", "--reload", "src/app.py"]
