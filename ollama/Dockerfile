FROM ollama/ollama

RUN --mount=type=cache,sharing=locked,target=/var/lib/apt/lists \
    --mount=type=cache,sharing=locked,target=/var/cache/apt,type=cache \
    rm -f /etc/apt/apt.conf.d/docker-clean \
    && apt-get update -yqq \
    && apt-get install -yqq curl build-essential telnet iputils-ping jq vim

EXPOSE 11434

ENV OLLAMA_MODELS=/ollama-models
ENV OLLAMA_HOST=0.0.0.0:11434
ENV OLLAMA_DEBUG=1
ENV OLLAMA_NUM_PARALLEL=5
ENV OLLAMA_LOAD_TIMEOUT=600m

RUN mkdir /ollama-models
WORKDIR /workdir
COPY healthcheck.sh .
HEALTHCHECK --interval=30s --timeout=15s --start-period=120s --retries=3 CMD ["/bin/bash", "/workdir/healthcheck.sh"]

COPY docker-entrypoint.sh .
COPY modelfiles modelfiles/
ENTRYPOINT ["/bin/bash", "docker-entrypoint.sh"]

