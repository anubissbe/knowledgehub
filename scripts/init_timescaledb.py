#!/usr/bin/env python3
"""
Initialize TimescaleDB with proper extension and permissions.
"""

import asyncio
import sys
import os
import logging
from datetime import datetime

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import asyncpg
from shared.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def init_timescaledb():
    """Initialize TimescaleDB with extension and proper setup."""
    config = Config()
    
    # Parse TimescaleDB URL
    timescale_url = config.TIMESCALE_URL
    logger.info(f"Connecting to TimescaleDB at: {timescale_url}")
    
    # First connect to postgres database to enable extension
    postgres_url = timescale_url.replace('/knowledgehub_analytics', '/postgres')
    
    try:
        # Connect to postgres database first
        conn = await asyncpg.connect(postgres_url)
        
        logger.info("🔧 Checking TimescaleDB extension...")
        
        # Check if TimescaleDB extension exists
        result = await conn.fetch("""
            SELECT EXISTS (
                SELECT 1 FROM pg_extension WHERE extname = 'timescaledb'
            );
        """)
        
        extension_exists = result[0]['exists']
        
        if not extension_exists:
            logger.info("📦 Installing TimescaleDB extension...")
            try:
                await conn.execute("CREATE EXTENSION IF NOT EXISTS timescaledb;")
                logger.info("✅ TimescaleDB extension installed successfully")
            except Exception as e:
                logger.error(f"❌ Failed to install TimescaleDB extension: {e}")
                logger.info("💡 Make sure you're running TimescaleDB container, not regular PostgreSQL")
                return False
        else:
            logger.info("✅ TimescaleDB extension already installed")
        
        # Verify TimescaleDB is working by checking extension info
        try:
            version_result = await conn.fetch("SELECT extversion FROM pg_extension WHERE extname = 'timescaledb';")
            if version_result:
                version = version_result[0]['extversion']
                logger.info(f"🎉 TimescaleDB extension version: {version}")
            else:
                logger.error("❌ TimescaleDB extension not found")
                return False
        except Exception as e:
            logger.error(f"❌ TimescaleDB extension check failed: {e}")
            return False
        
        # Database should already exist based on environment variables
        logger.info("✅ Using existing knowledgehub_analytics database")
        
        await conn.close()
        
        # Connect to the analytics database
        logger.info(f"Connecting to knowledgehub_analytics database...")
        
        try:
            conn = await asyncpg.connect(timescale_url)
            
            # Enable TimescaleDB extension in analytics database
            try:
                await conn.execute("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;")
                logger.info("✅ TimescaleDB extension enabled in analytics database")
            except Exception as e:
                logger.warning(f"Extension in analytics db: {e}")
            
            # Test TimescaleDB functions by creating a test hypertable
            try:
                # Test creating a hypertable (core TimescaleDB functionality)
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS _test_timescale (
                        time TIMESTAMPTZ NOT NULL,
                        value DOUBLE PRECISION
                    );
                """)
                
                result = await conn.fetch("SELECT create_hypertable('_test_timescale', 'time', if_not_exists => TRUE);")
                logger.info(f"✅ TimescaleDB hypertable functions working: {result[0]}")
                
                # Clean up test table
                await conn.execute("DROP TABLE IF EXISTS _test_timescale;")
                
                # Try to get version info from extension
                try:
                    result = await conn.fetch("SELECT extversion FROM pg_extension WHERE extname = 'timescaledb';")
                    if result:
                        version = result[0]['extversion']
                        logger.info(f"✅ TimescaleDB extension version: {version}")
                except Exception as ve:
                    logger.info("ℹ️ Could not get version, but TimescaleDB is functional")
                
            except Exception as e:
                logger.error(f"❌ TimescaleDB hypertable functions not available: {e}")
                return False
            
            await conn.close()
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to analytics database: {e}")
            return False
        
        logger.info("🎉 TimescaleDB initialization completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize TimescaleDB: {e}")
        logger.info("💡 Make sure TimescaleDB container is running and accessible")
        return False


async def test_connection():
    """Test TimescaleDB connection and basic functionality."""
    config = Config()
    
    # Test connection to TimescaleDB analytics database
    timescale_url = config.TIMESCALE_URL
    
    try:
        conn = await asyncpg.connect(timescale_url)
        
        logger.info("🧪 Testing TimescaleDB functionality...")
        
        # Test creating a simple hypertable
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS test_metrics (
                time TIMESTAMPTZ NOT NULL,
                device_id TEXT NOT NULL,
                temperature DOUBLE PRECISION
            );
        """)
        
        # Try to create hypertable
        try:
            result = await conn.fetch("""
                SELECT create_hypertable('test_metrics', 'time', if_not_exists => TRUE);
            """)
            logger.info(f"✅ Hypertable creation successful: {result[0]}")
        except Exception as e:
            logger.warning(f"Hypertable creation: {e}")
        
        # Insert test data
        await conn.execute("""
            INSERT INTO test_metrics (time, device_id, temperature) 
            VALUES (NOW(), 'device_1', 23.5);
        """)
        
        # Query test data
        result = await conn.fetch("SELECT * FROM test_metrics LIMIT 1;")
        if result:
            logger.info("✅ Data insertion and query successful")
        
        # Clean up test table
        await conn.execute("DROP TABLE IF EXISTS test_metrics;")
        
        await conn.close()
        
        logger.info("🎉 TimescaleDB functionality test passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ TimescaleDB functionality test failed: {e}")
        return False


async def main():
    """Main initialization function."""
    logger.info("🚀 Starting TimescaleDB initialization...")
    
    # Step 1: Initialize TimescaleDB extension
    init_success = await init_timescaledb()
    if not init_success:
        logger.error("❌ TimescaleDB initialization failed")
        sys.exit(1)
    
    # Step 2: Test functionality
    test_success = await test_connection()
    if not test_success:
        logger.error("❌ TimescaleDB functionality test failed")
        sys.exit(1)
    
    logger.info("✅ TimescaleDB is ready for time-series analytics!")
    
    # Display the correct URL
    config = Config()
    logger.info(f"📝 TimescaleDB URL configured: {config.TIMESCALE_URL}")


if __name__ == "__main__":
    asyncio.run(main())