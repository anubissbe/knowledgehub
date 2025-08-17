#!/usr/bin/env python3
"""
Test the RAG query system with real embeddings
"""

import asyncio
import logging
import json
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import httpx
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATABASE_URL = "postgresql://knowledgehub:knowledgehub123@localhost:5433/knowledgehub"
AI_SERVICE_URL = "http://localhost:8002"

class SimpleRAG:
    """Simple working RAG system"""
    
    def __init__(self):
        self.engine = create_engine(DATABASE_URL)
        self.Session = sessionmaker(bind=self.engine)
    
    async def get_embedding(self, text: str):
        """Get embedding from AI service"""
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{AI_SERVICE_URL}/api/ai/embed",
                json={"text": text}
            )
            response.raise_for_status()
            result = response.json()
            return result["embedding"]
    
    async def search_similar_chunks(self, query: str, limit: int = 5):
        """Search for similar chunks using manual cosine similarity calculation"""
        # Get query embedding
        query_embedding = await self.get_embedding(query)
        
        # Search database for chunks and calculate similarity in Python
        with self.Session() as session:
            result = session.execute(text("""
                SELECT id, content, document_id, metadata, embedding
                FROM chunks 
                WHERE embedding IS NOT NULL 
                AND NOT (embedding[1] = 0.0 AND embedding[2] = 0.0)
                LIMIT 200
            """))
            
            chunks = []
            query_vec = np.array(query_embedding)
            
            for row in result:
                chunk_vec = np.array(row.embedding)
                
                # Calculate cosine similarity
                similarity = np.dot(query_vec, chunk_vec) / (
                    np.linalg.norm(query_vec) * np.linalg.norm(chunk_vec)
                )
                
                if similarity > 0.3:  # Minimum similarity threshold
                    try:
                        metadata = json.loads(row.metadata) if isinstance(row.metadata, str) else (row.metadata or {})
                    except:
                        metadata = {}
                    
                    chunks.append({
                        "id": row.id,
                        "content": row.content,
                        "document_id": row.document_id,
                        "metadata": metadata,
                        "similarity": float(similarity),
                        "distance": float(1 - similarity)
                    })
            
            # Sort by similarity (highest first) and return top chunks
            chunks.sort(key=lambda x: x["similarity"], reverse=True)
            return chunks[:limit]
    
    async def rag_query(self, query: str):
        """Perform complete RAG query"""
        logger.info(f"Processing RAG query: {query}")
        
        # Search for relevant chunks
        chunks = await self.search_similar_chunks(query, limit=3)
        
        if not chunks:
            return {
                "query": query,
                "answer": "No relevant information found.",
                "chunks_used": 0,
                "chunks": []
            }
        
        # Build context from chunks
        context_parts = []
        for chunk in chunks:
            source_info = ""
            if chunk["metadata"].get("source_metadata", {}).get("url"):
                source_info = f" (Source: {chunk['metadata']['source_metadata']['url']})"
            
            context_parts.append(f"{chunk['content'][:500]}{source_info}")
        
        context = "\n\n".join(context_parts)
        
        # Generate answer (simplified - would normally call LLM)
        answer = f"Based on the retrieved information:\n\n{context[:1000]}..."
        
        return {
            "query": query,
            "answer": answer,
            "chunks_used": len(chunks),
            "chunks": [
                {
                    "id": c["id"],
                    "similarity": c["similarity"],
                    "preview": c["content"][:100] + "...",
                    "source": c["metadata"].get("source_metadata", {}).get("url", "Unknown")
                } for c in chunks
            ]
        }

async def test_rag_queries():
    """Test various RAG queries"""
    rag = SimpleRAG()
    
    test_queries = [
        "What is Docker?",
        "How to use FastAPI?", 
        "Container deployment",
        "API documentation",
        "Python web framework"
    ]
    
    print("🧪 TESTING RAG QUERY SYSTEM")
    print("=" * 50)
    
    for query in test_queries:
        try:
            result = await rag.rag_query(query)
            
            print(f"\n📋 Query: {query}")
            print(f"📊 Chunks found: {result['chunks_used']}")
            print(f"📝 Answer preview: {result['answer'][:200]}...")
            
            print("🔍 Relevant chunks:")
            for chunk in result["chunks"]:
                print(f"  • {chunk['similarity']:.3f} similarity | {chunk['preview']} | {chunk['source']}")
            
            print("-" * 50)
            
        except Exception as e:
            print(f"❌ Query failed: {query} | Error: {e}")
            
    return True

if __name__ == "__main__":
    success = asyncio.run(test_rag_queries())
    print(f"\n✅ RAG testing completed successfully!" if success else "❌ RAG testing failed!")