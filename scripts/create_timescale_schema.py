#!/usr/bin/env python3
"""
Create TimescaleDB schema for analytics.
"""

import asyncio
import sys
import os
import logging

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from api.services.time_series_analytics import TimeSeriesAnalyticsService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def create_schema():
    """Create TimescaleDB schema with tables and hypertables."""
    logger.info("🚀 Creating TimescaleDB schema...")
    
    service = TimeSeriesAnalyticsService()
    
    try:
        # Initialize the service (this will create tables)
        await service.initialize()
        logger.info("✅ TimescaleDB schema created successfully!")
        
        # Test that tables exist
        from sqlalchemy import text
        async with service.async_engine.begin() as conn:
            # Check tables
            result = await conn.execute(text("""
                SELECT tablename FROM pg_tables 
                WHERE schemaname = 'public' 
                AND tablename LIKE 'ts_%'
                ORDER BY tablename;
            """))
            
            tables = [row[0] for row in result]
            logger.info(f"📊 Created tables: {tables}")
            
            # Check hypertables
            result = await conn.execute(text("""
                SELECT hypertable_name, num_chunks 
                FROM timescaledb_information.hypertables 
                WHERE hypertable_schema = 'public'
                ORDER BY hypertable_name;
            """))
            
            hypertables = [(row[0], row[1]) for row in result]
            logger.info(f"⏰ Created hypertables: {hypertables}")
            
            # Check retention policies  
            try:
                result = await conn.execute(text("""
                    SELECT hypertable_name, drop_after
                    FROM timescaledb_information.policy_stats 
                    WHERE policy_type = 'retention'
                    ORDER BY hypertable_name;
                """))
                
                policies = [(row[0], row[1]) for row in result]
                logger.info(f"🗑️ Retention policies: {policies}")
            except Exception as e:
                logger.info(f"ℹ️ Retention policies info not available: {e}")
            
            # Check continuous aggregates
            try:
                result = await conn.execute(text("""
                    SELECT view_name, materialized_only 
                    FROM timescaledb_information.continuous_aggregates
                    ORDER BY view_name;
                """))
                
                aggregates = [(row[0], row[1]) for row in result]
                logger.info(f"📈 Continuous aggregates: {aggregates}")
            except Exception as e:
                logger.info(f"ℹ️ Continuous aggregates info not available: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to create schema: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        await service.cleanup()


async def main():
    """Main function."""
    success = await create_schema()
    
    if success:
        logger.info("🎉 TimescaleDB schema is ready for analytics!")
        sys.exit(0)
    else:
        logger.error("❌ Failed to create TimescaleDB schema!")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())