#! /usr/bin/env bash
env

ollama start & 
PID=$! 
sleep 10
ollama pull all-minilm:l6-v2
ollama create code-expert -f modelfiles/code-expert

wait ${PID}
