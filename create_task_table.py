#!/usr/bin/env python3
"""Create task table in the database"""

import asyncio
import os
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from api.models.task import Task
from api.models.base import Base

# Database configuration
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    f"postgresql+asyncpg://{os.getenv('DATABASE_USER', 'knowledgehub')}:"
    f"{os.getenv('DATABASE_PASSWORD', 'knowledgehub123')}@"
    f"{os.getenv('DATABASE_HOST', 'localhost')}:"
    f"{os.getenv('DATABASE_PORT', '5433')}/"
    f"{os.getenv('DATABASE_NAME', 'knowledgehub')}"
)

async def create_task_table():
    """Create the tasks table"""
    print(f"Connecting to database: {DATABASE_URL}")
    
    # Create async engine
    engine = create_async_engine(DATABASE_URL)
    
    try:
        # Create all tables
        async with engine.begin() as conn:
            print("Creating task table...")
            await conn.run_sync(Base.metadata.create_all, checkfirst=True)
            print("Task table created successfully!")
            
        # Test the table by inserting a sample task
        AsyncSessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        async with AsyncSessionLocal() as session:
            print("Testing table with sample task...")
            
            # Check if we can create a task
            sample_task = Task(
                title="Sample Task",
                description="This is a test task to verify the table works",
                created_by="system"
            )
            
            session.add(sample_task)
            await session.commit()
            await session.refresh(sample_task)
            
            print(f"Sample task created with ID: {sample_task.id}")
            print("Task table is working correctly!")
            
    except Exception as e:
        print(f"Error creating task table: {e}")
        raise
    finally:
        await engine.dispose()

if __name__ == "__main__":
    asyncio.run(create_task_table())