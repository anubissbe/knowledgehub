#!/usr/bin/env python3
"""Create missing database tables"""

from sqlalchemy import create_engine, text
from api.models.base import Base
from api.models.mistake_tracking import MistakeTracking, ErrorPattern
from api.models.decision import Decision, DecisionAlternative
import os

# Database connection
DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://knowledgehub:knowledgehub@localhost:5433/knowledgehub')

def create_tables():
    """Create all missing tables"""
    engine = create_engine(DATABASE_URL)
    
    # Check which tables exist
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name IN ('mistake_tracking', 'error_patterns', 'decisions', 'decision_alternatives')
        """))
        existing_tables = [row[0] for row in result]
        
    print(f"Existing tables: {existing_tables}")
    
    # Create all tables from the models
    Base.metadata.create_all(engine, checkfirst=True)
    
    # Check again
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name IN ('mistake_tracking', 'error_patterns', 'decisions', 'decision_alternatives')
        """))
        new_tables = [row[0] for row in result]
        
    print(f"Tables after creation: {new_tables}")
    
    # Show created tables
    created = set(new_tables) - set(existing_tables)
    if created:
        print(f"Created tables: {created}")
    else:
        print("No new tables created")

if __name__ == "__main__":
    create_tables()