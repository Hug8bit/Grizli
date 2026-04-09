# ── Backend (Python 3.11 + FastAPI) ──────────────────────────────────────────
FROM python:3.11-slim AS backend

WORKDIR /app

# System deps needed by pandapower / numpy
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
