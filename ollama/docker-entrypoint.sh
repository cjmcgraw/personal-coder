#! /usr/bin/env bash
set -eu

env

# Start Ollama
ollama start &
PID=$!

# Function to check if Ollama is ready
check_ollama_ready() {
  curl -s --fail "http://localhost:11434/api/tags" > /dev/null
}

# Wait for Ollama to be ready with retries
echo "Waiting for Ollama to become available..."
for i in {1..30}; do
  if check_ollama_ready; then
    echo "Ollama is ready!"
    break
  fi
  echo "Waiting for Ollama to start (attempt $i/30)..."
  sleep 2
done

# Check if Ollama is ready
if ! check_ollama_ready; then
  echo "ERROR: Ollama failed to become available in time"
  exit 1
fi

# Pull and create models
echo "Pulling embedding model..."
ollama pull all-minilm:l6-v2 || {
  echo "ERROR: Failed to pull all-minilm:l6-v2"
  exit 1
}

echo "Creating code-expert model..."
ollama create code-expert -f modelfiles/code-expert || {
  echo "ERROR: Failed to create code-expert model"
  exit 1
}

echo "Models ready!"

# Wait for the Ollama process
wait ${PID}
