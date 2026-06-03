FROM python:3.11-slim
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8742
ENV KDE_CONFIG=/app/prism_config.toml
RUN useradd -m prismuser && mkdir -p /home/prismuser/.prism && chown -R prismuser:prismuser /home/prismuser
USER prismuser
CMD ["python", "kde_cli.py", "server", "--port", "8742"]
