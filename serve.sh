#!/usr/bin/env bash
# Serve the demo with caching disabled (see serve.py).
set -euo pipefail
cd "$(dirname "$0")"
PORT="${1:-8080}"
exec python3 serve.py "$PORT"
