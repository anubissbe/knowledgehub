#!/usr/bin/env python3
"""Fix uppercase chunk_type values in database"""

import sys
sys.path.insert(0, '/opt/projects/knowledgehub')

from api.models.base import SessionLocal
from api.models.document import DocumentChunk
from sqlalchemy import text

def fix_chunk_types():
    """Fix uppercase chunk_type values"""
    db = SessionLocal()
    
    try:
        # First check what enum values are defined in the database
        enum_values = db.execute(text("""
            SELECT unnest(enum_range(NULL::chunk_type))::text as value
        """)).fetchall()
        
        print("Defined enum values in database:")
        for (value,) in enum_values:
            print(f"  - {value}")
        
        # Check current values in the table
        distinct_types = db.execute(text("SELECT DISTINCT chunk_type FROM document_chunks")).fetchall()
        print("\nDistinct chunk_type values in table:")
        for (chunk_type,) in distinct_types:
            print(f"  - {chunk_type}")
            
    except Exception as e:
        print(f"Error: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    fix_chunk_types()