#!/bin/sh

# Exit on any error
set -e

# Check if GEMINI_API_KEY is provided as an environment variable
if [ -z "${GEMINI_API_KEY}" ]; then
    echo "Error: GEMINI_API_KEY environment variable is not set."
    exit 1
fi

# Add API key to .env file
echo "GEMINI_API_KEY=${GEMINI_API_KEY}" > .env

# Create arxiv_papers directory if it doesn't exist
mkdir -p /app/arxiv_papers

# Start the application with PORT from Render if available, otherwise default to 10000
PORT="${PORT:-10000}"
exec uvicorn fastapi_app:app --host 0.0.0.0 --port $PORT
