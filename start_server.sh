#!/usr/bin/env zsh
# start_server.sh — Run the Cross-Modal Cartographer backend
# Usage: ./start_server.sh [port]
#
# For device testing on the same WiFi, use your machine's LAN IP in
# Mobile-Frontend/lib/services/api_service.dart:
#   const String _baseUrl = 'http://<YOUR-LAN-IP>:8000';

set -e
SCRIPT_DIR="${0:A:h}"
VENV="${SCRIPT_DIR}/../V0/.venv/bin/uvicorn"
PORT="${1:-8000}"

echo "Starting Cross-Modal Cartographer backend on port ${PORT}..."
echo "Health check → http://localhost:${PORT}/health"
echo "API docs     → http://localhost:${PORT}/docs"
echo ""

cd "${SCRIPT_DIR}"
exec "${VENV}" backend.main:app \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --reload
