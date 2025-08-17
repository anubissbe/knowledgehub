#!/usr/bin/env python3
"""
Update existing chunks with real embeddings from AI service
"""

import asyncio
import logging
import sys
import json
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import httpx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATABASE_URL = "postgresql://knowledgehub:knowledgehub123@localhost:5433/knowledgehub"
AI_SERVICE_URL = "http://localhost:8002"

async def update_chunk_embeddings():
    """Update all chunks with real embeddings"""
    engine = create_engine(DATABASE_URL)
    Session = sessionmaker(bind=engine)
    
    # Get chunks that have zero embeddings
    with Session() as session:
        result = session.execute(text("""
            SELECT id, content 
            FROM chunks 
            WHERE embedding IS NOT NULL 
            AND array_length(embedding, 1) > 0
            AND embedding[1] = 0.0  -- Zero vector check
            ORDER BY id 
            LIMIT 100
        """))
        chunks = result.fetchall()
    
    logger.info(f"Found {len(chunks)} chunks to update with embeddings")
    
    updated_count = 0
    async with httpx.AsyncClient(timeout=30.0) as client:
        for chunk_id, content in chunks:
            try:
                # Generate real embedding
                response = await client.post(
                    f"{AI_SERVICE_URL}/api/ai/embed",
                    json={"text": content[:1000]}  # Limit text length
                )
                response.raise_for_status()
                result = response.json()
                embedding = result["embedding"]
                
                # Update in database
                with Session() as session:
                    session.execute(text("""
                        UPDATE chunks 
                        SET embedding = :embedding
                        WHERE id = :id
                    """), {
                        "embedding": embedding,
                        "id": chunk_id
                    })
                    session.commit()
                
                updated_count += 1
                if updated_count % 10 == 0:
                    logger.info(f"Updated {updated_count} chunks")
                    
            except Exception as e:
                logger.error(f"Failed to update chunk {chunk_id}: {e}")
                continue
    
    logger.info(f"Updated {updated_count} chunks with real embeddings")
    return updated_count

if __name__ == "__main__":
    result = asyncio.run(update_chunk_embeddings())
    print(f"✅ Updated {result} chunks with real embeddings")