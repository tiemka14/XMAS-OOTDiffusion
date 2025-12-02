FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Build metadata can be injected from the build system using --build-arg
ARG BUILD_DATE="unknown"
ARG VCS_REF="unknown"
ARG BUILD_VERSION="unknown"

LABEL org.opencontainers.image.created=$BUILD_DATE \
	  org.opencontainers.image.revision=$VCS_REF \
	  org.opencontainers.image.version=$BUILD_VERSION

# Install only git (pip is already in the base image) and reduce layer size by
# cleaning apt lists; use --no-install-recommends to reduce additional packages
RUN apt-get update && \
	apt-get install -y --no-install-recommends git curl && \
	rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only requirements first to leverage Docker cache when deps don't change
COPY requirements.txt ./
# Use --no-cache-dir to avoid leaving package wheel caches in the image
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the sources
COPY . .

# Minimal runtime env vars
ENV PORT=8000
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Create a non-root user for runtime and set permissions
RUN useradd --create-home --shell /bin/bash appuser && \
	chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

# Make start script executable and provide a lightweight healthcheck
RUN chmod +x ./start.sh

HEALTHCHECK --interval=15s --timeout=3s --start-period=20s --retries=3 \
	CMD curl -f http://127.0.0.1:${PORT}/health || exit 1

CMD ["bash", "start.sh"]
