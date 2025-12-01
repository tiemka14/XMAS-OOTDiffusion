#!/bin/bash
# Use PORT env var if provided, default to 8000
PORT=${PORT:-8000}
# Increase keep-alive timeout and enable logging
exec uvicorn app.main:app --host 0.0.0.0 --port "$PORT" --timeout-keep-alive 120 --log-level info
