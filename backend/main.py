"""
GRIZLI FastAPI Backend — entry point.

Run with:
    uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import os
import logging

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()
logging.basicConfig(level=logging.INFO)

from backend.routers import simulate, ml, data  # noqa: E402  (after load_dotenv)

app = FastAPI(
    title="GRIZLI API",
    description="Grid Reconfiguration Intelligence for Zero-Loss Integration — REST API",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ──────────────────────────────────────────────────────────────────────
_origins = os.getenv(
    "CORS_ORIGINS", "http://localhost:3000,http://localhost:5173"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(simulate.router, prefix="/api")
app.include_router(ml.router,       prefix="/api/ml")
app.include_router(data.router,     prefix="/api/data")


# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/", tags=["Meta"])
def root():
    return {"service": "GRIZLI API", "version": "2.0.0", "docs": "/docs"}


@app.get("/health", tags=["Meta"])
def health():
    return {"status": "ok"}
