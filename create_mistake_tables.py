#!/usr/bin/env python3
"""
Create mistake tracking tables in the database
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import create_engine
from api.models import Base, MistakeTracking, ErrorPattern
from api.config import settings

def create_tables():
    """Create the mistake tracking tables"""
    # Create engine
    engine = create_engine(settings.DATABASE_URL)
    
    # Create tables
    print("Creating mistake tracking tables...")
    Base.metadata.create_all(bind=engine, tables=[
        MistakeTracking.__table__,
        ErrorPattern.__table__
    ])
    
    print("Tables created successfully!")
    
    # Verify tables exist
    from sqlalchemy import inspect
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    
    if 'mistake_tracking' in tables:
        print("✓ mistake_tracking table exists")
    else:
        print("✗ mistake_tracking table NOT found")
        
    if 'error_patterns' in tables:
        print("✓ error_patterns table exists")
    else:
        print("✗ error_patterns table NOT found")

if __name__ == "__main__":
    create_tables()