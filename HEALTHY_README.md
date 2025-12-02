# Troubleshooting: Workers Unhealthy

This file summarizes the common causes and fixes for workers that become unhealthy in hosted environments (e.g., Runpod).

Common causes
- The container exits/crashes on startup (often due to GPU/driver mismatch or missing packages).
- The app blocks on long-running tasks during import (e.g., model downloads) and never binds to the HTTP port.
- The platform's health checks expect a specific endpoint or a quick HTTP 200 response.

What we changed
- Load the heavy model in a FastAPI `startup` event (background) instead of during import. This keeps the HTTP server binding fast and allows health checks to succeed quicker.
- Added `/health` and `/ready` endpoints:
  - `/health`: Returns 200 whenever the service HTTP server is running.
  - `/ready`: Returns `"ready"` only when the model finished loading.
- Add fallback to CPU if `cuda` is requested but not available.
- Added error handling to keep the process running if model loading fails (the error surface will be in /health and /ready), which prevents immediate process exit and allows better logging.
- Start script uses environment `PORT` and runs uvicorn as PID 1 to make logs and process signals behave properly.
- `Dockerfile` now sets `PORT`, marks the start script as executable, and adds a Docker `HEALTHCHECK`.

How to use
- If target environment has GPU, set `DEVICE=cuda` at runtime. If not, use `DEVICE=cpu` to avoid CUDA errors.
- Build and run locally to verify:

```bash
# Build
docker build -t xtype-your-name/xmas-ootdiffusion:latest .
docker build -t xtype-your-name/xmas-ootdiffusion:latest . \
  --build-arg BUILD_DATE="$(date -u +"%Y-%m-%dT%H:%M:%SZ")" \
  --build-arg VCS_REF="$(git rev-parse --short HEAD || echo unknown)" \
  --build-arg BUILD_VERSION="1.0"
# Run locally (use GPU if available; set DEVICE=cpu to avoid GPU requirement)
docker run -e DEVICE=cpu -p 8000:8000 xtype-your-name/xmas-ootdiffusion:latest
# Check health
curl http://localhost:8000/health
curl http://localhost:8000/ready
# Try the service
curl -v -F "person=@./assets/person.png" -F "cloth=@./assets/sweater.png" http://localhost:8000/tryon
```

Debugging tips
- Check container logs in your platform or with `docker logs`.
- If container crashes on startup, look for exceptions about CUDA versions or missing dependencies.
- If health shows `model_error` in `/health` or `/ready`, set `DEVICE=cpu` to verify whether GPU is causing issues.

If you still get "all workers unhealthy":
- Share the container logs, and we'll examine specific tracebacks to suggest more targeted fixes.

Tip: Pin your Python package versions in `requirements.txt` to ensure reproducible builds. Example:

```
fastapi==0.99.1
uvicorn==0.21.1
torch==2.1.0
# Add exact versions for the rest of your dependencies
```
