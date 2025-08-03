#!/usr/bin/env python3
"""
Simple AI Service for KnowledgeHub
Provides embeddings without database dependency
"""

import os
import logging
from typing import List
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="KnowledgeHub AI Service", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model
MODEL_NAME = os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
logger.info(f"Loading model: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME)
logger.info(f"Model loaded successfully")

# Pydantic models
class EmbeddingRequest(BaseModel):
    texts: List[str]

class HealthResponse(BaseModel):
    status: str
    model: str
    model_loaded: bool
    embedding_dim: int
    timestamp: str

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health():
    return {
        "status": "healthy",
        "model": MODEL_NAME,
        "model_loaded": True,
        "embedding_dim": model.get_sentence_embedding_dimension(),
        "timestamp": datetime.utcnow().isoformat()
    }

# Embeddings endpoint
@app.post("/embeddings")
async def generate_embeddings(request: EmbeddingRequest):
    try:
        embeddings = model.encode(request.texts)
        return {
            "embeddings": embeddings.tolist(),
            "model": MODEL_NAME,
            "dim": model.get_sentence_embedding_dimension()
        }
    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)