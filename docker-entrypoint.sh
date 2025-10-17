#!/bin/sh

# Exit on any error
set -e

# Check if GEMINI_API_KEY is provided as an environment variable
if [ -z "${GEMINI_API_KEY}" ]; then
    echo "Error: GEMINI_API_KEY environment variable is not set."
    exit 1
fi

# Set environment variables for MCP configuration
export PYTHON_PATH="/usr/local/bin/python3"
export APP_PATH="/app"

# Add API key and environment variables to .env file
cat > .env << EOF
GEMINI_API_KEY=${GEMINI_API_KEY}
PYTHON_PATH=${PYTHON_PATH}
APP_PATH=${APP_PATH}
EOF

# Create arxiv_papers directory if it doesn't exist
mkdir -p /app/arxiv_papers

# Verify MCP server files exist
echo "Checking MCP server files..."
ls -la /app/mcp-server/

# Test Python path
echo "Python executable: $(which python3)"
echo "Python version: $(python3 --version)"

# Test MCP server files can be executed
echo "Testing MCP server files..."
for server in /app/mcp-server/*.py; do
    if [ -f "$server" ]; then
        echo "Testing $server..."
        python3 -m py_compile "$server" && echo "✓ $server compiles" || echo "✗ $server has syntax errors"
    fi
done

# Start the application with PORT from Render if available, otherwise default to 10000
PORT="${PORT:-10000}"
exec uvicorn main:app --host 0.0.0.0 --port $PORT
