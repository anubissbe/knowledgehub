
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
import os

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://knowledgehub:knowledgehub123@192.168.1.25:5433/knowledgehub")

# Optimized database configuration
ENGINE_CONFIG = {
    "pool_size": 20,
    "max_overflow": 40,
    "pool_pre_ping": True,
    "pool_recycle": 3600,
    "echo": False
}

def create_optimized_engine():
    """Create optimized database engine"""
    return create_engine(DATABASE_URL, **ENGINE_CONFIG)
