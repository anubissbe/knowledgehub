"""Database connection and session management"""

from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import NullPool
import os
from typing import AsyncGenerator, Generator, Optional

# Database URL from environment
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    f"postgresql://{os.getenv('DATABASE_USER', 'knowledgehub')}:"
    f"{os.getenv('DATABASE_PASSWORD', 'knowledgehub123')}@"
    f"{os.getenv('DATABASE_HOST', 'postgres')}:"
    f"{os.getenv('DATABASE_PORT', '5432')}/"
    f"{os.getenv('DATABASE_NAME', 'knowledgehub')}"
)

# Async database URL
ASYNC_DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")

# Create async engine
async_engine = create_async_engine(
    ASYNC_DATABASE_URL,
    echo=False,
    poolclass=NullPool,
)

# Create async session maker
AsyncSessionLocal = async_sessionmaker(
    async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

# Create sync engine for migrations
sync_engine = create_engine(
    DATABASE_URL,
    echo=False,
    pool_pre_ping=True,
)

# Create sync session maker
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=sync_engine)


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Get async database session"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


def get_db() -> Generator[Session, None, None]:
    """Get sync database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()