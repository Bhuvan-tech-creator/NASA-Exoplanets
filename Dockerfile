FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

# Create default data directories (can be overridden at runtime)
ENV DATA_DIR=/app \
    MODELS_DIR=/app/models \
    UPLOADS_DIR=/app/uploads
RUN mkdir -p $MODELS_DIR $UPLOADS_DIR

# Expose default port
EXPOSE 5000

# Default start command (can be overridden by platform)
CMD ["gunicorn", "app:app", "--workers=2", "--threads=4", "--timeout=120", "--bind=0.0.0.0:5000", "--preload"]
