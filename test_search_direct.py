#!/usr/bin/env python3
"""Test search functionality directly"""

import sys
sys.path.insert(0, '/opt/projects/knowledgehub')

from api.services.search_service import SearchService
from api.schemas.search import SearchQuery, SearchType
from api.models import get_db
from sqlalchemy.orm import Session
import asyncio

async def test_search():
    """Test search functionality"""
    # Get database session
    from api.models.base import engine, SessionLocal
    db = SessionLocal()
    
    # Create search service
    search_service = SearchService()
    
    # Create search query
    query = SearchQuery(
        query="authentication",
        search_type=SearchType.HYBRID,
        limit=5
    )
    
    try:
        # Execute search
        print("Executing search for 'authentication'...")
        results = await search_service.search(db, query)
        
        print(f"\nSearch completed in {results['search_time_ms']}ms")
        print(f"Found {results['total']} results")
        
        for i, result in enumerate(results['results']):
            print(f"\nResult {i+1}:")
            print(f"  Content: {result['content'][:100]}...")
            print(f"  Score: {result['score']}")
            print(f"  Type: {result.get('chunk_type', 'unknown')}")
            print(f"  Source: {result['source_name']}")
            
    except Exception as e:
        print(f"Error during search: {e}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()

if __name__ == "__main__":
    asyncio.run(test_search())