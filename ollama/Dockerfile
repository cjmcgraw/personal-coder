FROM ollama/ollama
WORKDIR /workdir
RUN mkdir /ollama-models

EXPOSE 11434

ENV OLLAMA_MODELS=/ollama-models
ENV OLLAMA_HOST=0.0.0.0:11434
ENV OLLAMA_DEBUG=1
ENV OLLAMA_NUM_PARALLEL=5
ENV OLLAMA_LOAD_TIMEOUT=600m


COPY docker-entrypoint.sh .
COPY modelfiles modelfiles/
ENTRYPOINT ["/bin/bash", "docker-entrypoint.sh"]

