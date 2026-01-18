"""
Main entry point for Turkish Sentence Emotion Analysis API

This module provides the FastAPI application with sentence-level analysis endpoints.
It integrates the existing word-level model with the new sentence analysis pipeline.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
from pathlib import Path

from sentence_analysis.api import create_sentence_analysis_app


# Configuration
# Get the project root directory (parent of Backend)
PROJECT_ROOT = Path(__file__).parent.parent
MODEL_PATH = str(PROJECT_ROOT / "Demo" / "models" / "best_emotion_model.pkl")
CORS_ORIGINS = ["*"]  # Configure appropriately for production
MAX_QUEUE_SIZE = 50
RATE_LIMIT_REQUESTS = 10
RATE_LIMIT_WINDOW = 60  # seconds
PROCESSING_TIMEOUT = 30  # seconds


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application
    
    Returns:
        Configured FastAPI application
    """
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}\n"
            f"Please ensure the trained model is available at this path."
        )
    
    # Create sentence analysis app
    app = create_sentence_analysis_app(
        model_path=MODEL_PATH,
        cors_origins=CORS_ORIGINS,
        max_queue_size=MAX_QUEUE_SIZE,
        rate_limit_requests=RATE_LIMIT_REQUESTS,
        rate_limit_window=RATE_LIMIT_WINDOW,
        processing_timeout=PROCESSING_TIMEOUT
    )
    
    return app


# Create the application
app = create_app()


if __name__ == "__main__":
    import uvicorn
    
    print("=" * 60)
    print("Turkish Sentence Emotion Analysis API")
    print("=" * 60)
    print(f"Model: {MODEL_PATH}")
    print(f"Max Queue Size: {MAX_QUEUE_SIZE}")
    print(f"Rate Limit: {RATE_LIMIT_REQUESTS} requests per {RATE_LIMIT_WINDOW}s")
    print(f"Processing Timeout: {PROCESSING_TIMEOUT}s")
    print("=" * 60)
    print("\nAPI Endpoints:")
    print("  POST /api/analyze-sentence - Analyze Turkish sentence audio")
    print("  GET  /api/status/{job_id}  - Get analysis job status")
    print("  GET  /health               - Health check")
    print("=" * 60)
    print("\nStarting server on http://0.0.0.0:8000")
    print("Press Ctrl+C to stop\n")
    
    uvicorn.run(
        "main_sentence_analysis:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )

