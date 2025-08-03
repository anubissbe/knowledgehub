#!/usr/bin/env python3
"""
Test the simple RAG implementation
"""

import asyncio
import logging
from datetime import datetime

# Add the project to path
import sys
sys.path.insert(0, '/opt/projects/knowledgehub')

from api.services.rag.simple_rag_service import get_rag_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_simple_rag():
    """Test the simple RAG service"""
    logger.info("Testing Simple RAG Service")
    
    # Get the RAG service
    rag_service = get_rag_service()
    
    # Check if LlamaIndex is available
    logger.info(f"LlamaIndex available: {rag_service.llamaindex_initialized}")
    
    # Test document ingestion
    logger.info("\n1. Testing document ingestion...")
    
    test_content = """
    # Python Best Practices
    
    Python is a versatile programming language with many best practices.
    
    ## Code Style
    - Use PEP 8 for code formatting
    - Write clear, descriptive variable names
    - Keep functions small and focused
    - Use type hints for better code clarity
    
    ## Error Handling
    - Always handle exceptions appropriately
    - Use specific exception types
    - Log errors for debugging
    - Provide meaningful error messages
    
    ## Testing
    - Write unit tests for all functions
    - Use pytest for testing framework
    - Aim for high test coverage
    - Test edge cases and error conditions
    """
    
    ingest_result = await rag_service.ingest_document(
        content=test_content,
        metadata={
            "title": "Python Best Practices Guide",
            "source": "test",
            "author": "Test Suite",
            "created_at": datetime.utcnow().isoformat()
        },
        source_type="documentation",
        use_contextual_enrichment=True
    )
    
    logger.info(f"Ingestion result: {ingest_result}")
    
    # Test querying
    logger.info("\n2. Testing RAG queries...")
    
    test_queries = [
        "What are Python best practices for error handling?",
        "How should I write Python tests?",
        "What is PEP 8?"
    ]
    
    for query in test_queries:
        logger.info(f"\nQuery: {query}")
        
        result = await rag_service.query(
            query_text=query,
            user_id="test_user",
            project_id=None,
            top_k=3,
            use_hybrid=True
        )
        
        logger.info(f"Response: {result.get('response', 'No response')[:200]}...")
        logger.info(f"Sources found: {len(result.get('source_nodes', []))}")
        
    # Test index stats
    logger.info("\n3. Testing index statistics...")
    stats = await rag_service.update_index_stats()
    logger.info(f"Index stats: {stats}")
    
    logger.info("\nSimple RAG test completed successfully!")


if __name__ == "__main__":
    asyncio.run(test_simple_rag())