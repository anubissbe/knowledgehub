#!/usr/bin/env python3
"""
FINAL RAG VERIFICATION REPORT
"""

import asyncio
from sqlalchemy import create_engine, text
import httpx
import time

DATABASE_URL = "postgresql://knowledgehub:knowledgehub123@localhost:5433/knowledgehub"
AI_SERVICE_URL = "http://localhost:8002"

async def full_verification():
    """Complete RAG system verification"""
    
    print("🔧 RAG PIPELINE REDEMPTION VERIFICATION REPORT")
    print("=" * 60)
    
    # Database verification
    engine = create_engine(DATABASE_URL)
    with engine.connect() as conn:
        # Documents count
        result = conn.execute(text("SELECT COUNT(*) FROM documents"))
        doc_count = result.scalar()
        
        # Total chunks
        result = conn.execute(text("SELECT COUNT(*) FROM chunks"))
        chunk_count = result.scalar()
        
        # Real embeddings (non-zero)
        result = conn.execute(text("""
            SELECT COUNT(*) FROM chunks 
            WHERE embedding IS NOT NULL 
            AND NOT (embedding[1] = 0.0 AND embedding[2] = 0.0)
        """))
        real_embeddings = result.scalar()
        
        # Sample embedding to verify it's real
        result = conn.execute(text("""
            SELECT embedding[1:3] as sample, LEFT(content, 50) as content_preview
            FROM chunks 
            WHERE embedding IS NOT NULL 
            AND NOT (embedding[1] = 0.0 AND embedding[2] = 0.0)
            LIMIT 1
        """))
        sample = result.fetchone()
    
    print(f"📊 DATABASE STATUS:")
    print(f"  Documents in database: {doc_count}")
    print(f"  Chunks created: {chunk_count}")
    print(f"  Real embeddings: {real_embeddings}")
    print(f"  Embedding completion: {(real_embeddings/chunk_count*100):.1f}%" if chunk_count > 0 else "No chunks!")
    if sample:
        print(f"  Sample embedding: {sample[0]} | Content: {sample[1]}...")
    print()
    
    # AI Service verification
    try:
        start_time = time.time()
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                f"{AI_SERVICE_URL}/api/ai/embed",
                json={"text": "test query"}
            )
            response.raise_for_status()
            result = response.json()
        
        embed_time = (time.time() - start_time) * 1000
        embed_works = "embedding" in result and len(result["embedding"]) == 384
        
        print(f"🤖 AI SERVICE STATUS:")
        print(f"  Embedding endpoint: {'✅ WORKING' if embed_works else '❌ BROKEN'}")
        print(f"  Response time: {embed_time:.0f}ms")
        print(f"  Embedding dimension: {len(result.get('embedding', []))}")
        print()
        
    except Exception as e:
        print(f"❌ AI SERVICE ERROR: {e}")
        return False
    
    # RAG Query verification
    try:
        from test_rag_query import SimpleRAG
        rag = SimpleRAG()
        
        start_time = time.time()
        result = await rag.rag_query("What is Docker?")
        query_time = (time.time() - start_time) * 1000
        
        print(f"🧠 RAG QUERY STATUS:")
        print(f"  Query processing: {'✅ WORKING' if result['chunks_used'] > 0 else '❌ BROKEN'}")
        print(f"  Query response time: {query_time:.0f}ms")
        print(f"  Chunks retrieved: {result['chunks_used']}")
        if result['chunks_used'] > 0:
            avg_similarity = sum(c['similarity'] for c in result['chunks']) / len(result['chunks'])
            print(f"  Average similarity: {avg_similarity:.3f}")
            print(f"  Top result similarity: {result['chunks'][0]['similarity']:.3f}")
        print()
        
    except Exception as e:
        print(f"❌ RAG QUERY ERROR: {e}")
        return False
    
    # Overall assessment
    rag_working = (
        doc_count > 0 and 
        chunk_count > 0 and 
        real_embeddings > 0 and 
        embed_works and 
        result['chunks_used'] > 0
    )
    
    print("🎯 FINAL ASSESSMENT:")
    print(f"  RAG Pipeline Status: {'🟢 FULLY WORKING' if rag_working else '🔴 BROKEN'}")
    print(f"  Documents processed: {doc_count > 0}")
    print(f"  Chunking functional: {chunk_count > 0}")
    print(f"  Embeddings generated: {real_embeddings > 0}")
    print(f"  AI service operational: {embed_works}")
    print(f"  Queries returning results: {result['chunks_used'] > 0}")
    print()
    
    if rag_working:
        print("✅ REDEMPTION SUCCESS!")
        print("The RAG pipeline is now fully functional and can:")
        print("  • Process documents into chunks")
        print("  • Generate real embeddings")
        print("  • Perform semantic similarity searches")
        print("  • Return relevant content for queries")
        print("  • Provide measurable performance metrics")
    else:
        print("❌ REDEMPTION FAILED!")
        print("The RAG pipeline is still broken.")
    
    return rag_working

if __name__ == "__main__":
    success = asyncio.run(full_verification())
    exit(0 if success else 1)