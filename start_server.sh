#!/usr/bin/env bash
# start_server.sh — Run the Cross-Modal Cartographer backend
# Usage: ./start_server.sh [port]
#
# For device testing on the same WiFi, point the mobile app at your machine's
# LAN IP in Mobile-Frontend/lib/services/api_service.dart:
#   const String _baseUrl = 'http://<YOUR-LAN-IP>:8000';

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${1:-8000}"

# Prefer a repo-local virtualenv (.venv), else fall back to uvicorn on PATH.
if [ -x "${SCRIPT_DIR}/.venv/bin/uvicorn" ]; then
    UVICORN="${SCRIPT_DIR}/.venv/bin/uvicorn"
elif command -v uvicorn >/dev/null 2>&1; then
    UVICORN="uvicorn"
else
    echo "Error: uvicorn not found." >&2
    echo "Create the environment first:" >&2
    echo "    python3 -m venv .venv && source .venv/bin/activate" >&2
    echo "    pip install -r backend/requirements.txt" >&2
    exit 1
fi

echo "Starting Cross-Modal Cartographer backend on port ${PORT}..."
echo "Web app      → http://localhost:${PORT}/"
echo "Health check → http://localhost:${PORT}/health"
echo "API docs     → http://localhost:${PORT}/docs"
echo ""

cd "${SCRIPT_DIR}"
exec "${UVICORN}" backend.main:app \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --reload
