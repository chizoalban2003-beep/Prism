FROM python:3.11-slim
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg curl \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /app

# Copy the dep manifest first to keep the install layer cached when only
# source changes. requirements.txt is intentionally not copied — it's a
# pointer to pyproject.toml, not a real lock file.
COPY pyproject.toml .
COPY README.md .
RUN pip install --no-cache-dir -e ".[full]"

COPY . .
# Ownership + data dir are set BEFORE switching user so the bind mount at
# /home/prismuser/.prism (if any) inherits the right uid.
RUN useradd -m prismuser \
 && mkdir -p /home/prismuser/.prism \
 && chown -R prismuser:prismuser /home/prismuser /app
USER prismuser
EXPOSE 8742
ENV KDE_CONFIG=/app/prism_config.toml
# Inside the container the app must bind all interfaces or the published
# port maps to nothing; compose publishes only to the host's loopback,
# so the local-only model is preserved.
ENV PRISM_HOST=0.0.0.0
ENV PRISM_BIND_ALL_INTERFACES=1
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8742/_health || exit 1
CMD ["python", "prism_daemon.py"]
