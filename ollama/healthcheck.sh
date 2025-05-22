#!/bin/bash
set -e

# Get models to check from environment variable
# Default to "all-minilm:l6-v2,code-expert:latest,gemma3:4b" if not specified
REQUIRED_MODELS=${REQUIRED_MODELS:-"all-minilm:l6-v2,code-expert:latest,gemma3:4b"}
OLLAMA_API_URL=${OLLAMA_API_URL:-"http://localhost:11434"}

# First check if Ollama API is responsive
echo "Checking Ollama API..."
if ! curl -s --fail "${OLLAMA_API_URL}/api/tags" > /dev/null; then
  echo "❌ Ollama API is not responding"
  exit 1
fi
echo "✅ Ollama API is responsive"

# Check if all required models are available
echo "Checking for required models: ${REQUIRED_MODELS}"
# Get the available models from Ollama
RESPONSE=$(curl -s "${OLLAMA_API_URL}/api/tags")

# Check if response is valid JSON
if ! echo "$RESPONSE" | jq -e . >/dev/null 2>&1; then
  echo "❌ Invalid JSON response from Ollama API"
  exit 1
fi

# Extract model names from response
AVAILABLE_MODELS=$(echo "$RESPONSE" | jq -r '.models[].name')

# Check each required model
MISSING_MODELS=""
IFS=',' read -ra MODEL_ARRAY <<< "$REQUIRED_MODELS"
for MODEL in "${MODEL_ARRAY[@]}"; do
  # Trim whitespace
  MODEL=$(echo "$MODEL" | xargs)

  # Check if model exists
  if ! echo "$AVAILABLE_MODELS" | grep -q "^$MODEL$"; then
    MISSING_MODELS="${MISSING_MODELS} ${MODEL}"
  fi
done

# Report results
if [ -n "$MISSING_MODELS" ]; then
  echo "❌ Missing required models:${MISSING_MODELS}"
  exit 1
else
  echo "✅ All required models are available"
fi

# All checks passed
echo "✅ Ollama healthcheck successful"
exit 0