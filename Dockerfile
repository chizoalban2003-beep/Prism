FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg curl && rm -rf /var/lib/apt/lists/*
WORKDIR /app

# Copy dependency manifests first for layer caching
COPY pyproject.toml .
COPY requirements.txt .
RUN pip install --no-cache-dir -e ".[full]"

COPY . .
EXPOSE 8742
ENV KDE_CONFIG=/app/prism_config.toml
RUN useradd -m prismuser && mkdir -p /home/prismuser/.prism && chown -R prismuser:prismuser /home/prismuser
USER prismuser
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8742/_health || exit 1
CMD ["python", "prism_daemon.py"]
