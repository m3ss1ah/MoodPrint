# ── Stage 1: Build React frontend ─────────────────────────────────────────
FROM node:20-alpine AS frontend-builder

WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install --legacy-peer-deps
COPY frontend/ .
RUN npm run build
# Output goes to /app/backend/static (per vite.config.js)

# ── Stage 2: Python backend ────────────────────────────────────────────────
FROM python:3.11-slim

# System deps for librosa / soundfile
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps
COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend
COPY backend/ ./backend/

# Copy built frontend into static folder
COPY --from=frontend-builder /app/backend/static ./backend/static

# Copy model artefacts (must be trained before building image)
COPY models/ ./models/

# Serve static files from FastAPI
RUN pip install --no-cache-dir aiofiles

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
