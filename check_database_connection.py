#!/usr/bin/env python3
"""
Script to check database connectivity and configuration.
"""

import os
import sys
import asyncio
import logging
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import create_async_engine
import psycopg2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_database_config():
    """Check database configuration."""
    
    # Get database configuration from environment
    db_config = {
        "host": os.getenv("DATABASE_HOST", "localhost"),
        "port": os.getenv("DATABASE_PORT", "5433"),
        "database": os.getenv("DATABASE_NAME", "knowledgehub"),
        "user": os.getenv("DATABASE_USER", "knowledgehub"),
        "password": os.getenv("DATABASE_PASSWORD", "knowledgehub123")
    }
    
    # Build connection strings
    sync_url = f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
    async_url = f"postgresql+asyncpg://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
    
    logger.info("Database Configuration:")
    logger.info(f"  Host: {db_config['host']}")
    logger.info(f"  Port: {db_config['port']}")
    logger.info(f"  Database: {db_config['database']}")
    logger.info(f"  User: {db_config['user']}")
    logger.info(f"  Connection URL: {sync_url.replace(db_config['password'], '***')}")
    
    return sync_url, async_url, db_config

def test_psycopg2_connection(db_config):
    """Test direct psycopg2 connection."""
    try:
        logger.info("\nTesting psycopg2 connection...")
        conn = psycopg2.connect(
            host=db_config['host'],
            port=db_config['port'],
            database=db_config['database'],
            user=db_config['user'],
            password=db_config['password']
        )
        
        cursor = conn.cursor()
        cursor.execute("SELECT version();")
        version = cursor.fetchone()[0]
        logger.info(f"✅ psycopg2 connection successful!")
        logger.info(f"   PostgreSQL version: {version}")
        
        # Check if database exists and has tables
        cursor.execute("""
            SELECT count(*) 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
        """)
        table_count = cursor.fetchone()[0]
        logger.info(f"   Tables in database: {table_count}")
        
        cursor.close()
        conn.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ psycopg2 connection failed: {e}")
        return False

def test_sync_connection(sync_url):
    """Test synchronous SQLAlchemy connection."""
    try:
        logger.info("\nTesting synchronous SQLAlchemy connection...")
        engine = create_engine(sync_url, echo=False)
        
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            logger.info(f"✅ Sync SQLAlchemy connection successful!")
            
            # Check for tables
            result = conn.execute(text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                ORDER BY table_name
                LIMIT 10
            """))
            tables = [row[0] for row in result]
            if tables:
                logger.info(f"   Sample tables: {', '.join(tables)}")
            else:
                logger.info("   No tables found in database")
        
        engine.dispose()
        return True
        
    except Exception as e:
        logger.error(f"❌ Sync SQLAlchemy connection failed: {e}")
        return False

async def test_async_connection(async_url):
    """Test asynchronous SQLAlchemy connection."""
    try:
        logger.info("\nTesting asynchronous SQLAlchemy connection...")
        engine = create_async_engine(async_url, echo=False)
        
        async with engine.connect() as conn:
            result = await conn.execute(text("SELECT 1"))
            logger.info(f"✅ Async SQLAlchemy connection successful!")
            
            # Check for required tables
            result = await conn.execute(text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name IN ('sources', 'documents', 'document_chunks', 'memories', 'ai_memories')
                ORDER BY table_name
            """))
            tables = [row[0] for row in result]
            if tables:
                logger.info(f"   Core tables found: {', '.join(tables)}")
            else:
                logger.warning("   ⚠️  Core tables not found - database may need initialization")
        
        await engine.dispose()
        return True
        
    except Exception as e:
        logger.error(f"❌ Async SQLAlchemy connection failed: {e}")
        return False

async def main():
    """Main function to run all database tests."""
    logger.info("=== KnowledgeHub Database Connection Test ===\n")
    
    # Get configuration
    sync_url, async_url, db_config = check_database_config()
    
    # Run tests
    psycopg2_ok = test_psycopg2_connection(db_config)
    sync_ok = test_sync_connection(sync_url)
    async_ok = await test_async_connection(async_url)
    
    # Summary
    logger.info("\n=== Test Summary ===")
    logger.info(f"psycopg2 connection: {'✅ OK' if psycopg2_ok else '❌ Failed'}")
    logger.info(f"Sync SQLAlchemy: {'✅ OK' if sync_ok else '❌ Failed'}")
    logger.info(f"Async SQLAlchemy: {'✅ OK' if async_ok else '❌ Failed'}")
    
    if all([psycopg2_ok, sync_ok, async_ok]):
        logger.info("\n✅ All database connections successful!")
        return 0
    else:
        logger.error("\n❌ Some database connections failed!")
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))