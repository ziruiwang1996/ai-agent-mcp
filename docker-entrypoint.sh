#!/bin/sh

# Exit on any error
set -e

# Require GOOGLE_API_KEY (used by google_genai).
API_KEY="${GOOGLE_API_KEY}"
if [ -z "${API_KEY}" ]; then
    echo "Error: GOOGLE_API_KEY must be set."
    exit 1
fi

# Set environment variables for MCP configuration
export PYTHON_PATH="/usr/local/bin/python3"
export APP_PATH="/app"

# Add API key and environment variables to .env file
cat > .env << EOF
GOOGLE_API_KEY=${API_KEY}
PYTHON_PATH=${PYTHON_PATH}
APP_PATH=${APP_PATH}
EOF

echo "Checking MCP server files..."
ls -la /app/mcp_servers/

# Test Python path
echo "Python executable: $(which python3)"
echo "Python version: $(python3 --version)"

# Test MCP server files can be executed
echo "Testing MCP server files..."
for server in /app/mcp_servers/*.py; do
    if [ -f "$server" ]; then
        echo "Testing $server..."
        python3 -m py_compile "$server" && echo "✓ $server compiles" || echo "✗ $server has syntax errors"
    fi
done

# Start the application with PORT from Render if available, otherwise default to 10000
PORT="${PORT:-10000}"
exec uvicorn main:app --host 0.0.0.0 --port $PORT
