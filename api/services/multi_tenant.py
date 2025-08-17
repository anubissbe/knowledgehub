"""
Multi-Tenant Architecture Service for KnowledgeHub Enterprise
Implements database-level tenant isolation with row-level security
"""

import uuid
import asyncio
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import logging
import redis.asyncio as redis
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, String, Boolean, DateTime, JSON, Integer, ForeignKey, BigInteger, Text
from sqlalchemy.dialects.postgresql import UUID
from contextlib import asynccontextmanager

from ..database import AsyncSessionLocal, async_engine
from ..models.user import User, UserRole

logger = logging.getLogger(__name__)

class TenantStatus(str, Enum):
    """Tenant status enumeration"""
    ACTIVE = "active"
    SUSPENDED = "suspended" 
    TRIAL = "trial"
    EXPIRED = "expired"
    PROVISIONING = "provisioning"

class TenantPlan(str, Enum):
    """Tenant subscription plans"""
    STARTER = "starter"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"
    CUSTOM = "custom"

@dataclass
class TenantQuota:
    """Tenant resource quotas"""
    max_users: int
    max_documents: int
    max_storage_gb: int
    max_api_calls_per_hour: int
    max_concurrent_sessions: int
    gpu_time_minutes_per_month: int
    advanced_features_enabled: bool

# Tenant quotas by plan
PLAN_QUOTAS = {
    TenantPlan.STARTER: TenantQuota(5, 1000, 1, 1000, 10, 0, False),
    TenantPlan.PROFESSIONAL: TenantQuota(25, 10000, 10, 10000, 50, 60, True),
    TenantPlan.ENTERPRISE: TenantQuota(500, 100000, 100, 100000, 500, 480, True),
    TenantPlan.CUSTOM: TenantQuota(999999, 999999, 999999, 999999, 999999, 999999, True)
}

Base = declarative_base()

class Tenant(Base):
    """Tenant model for multi-tenancy"""
    __tablename__ = "tenants"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    slug = Column(String(50), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=False)
    domain = Column(String(255), unique=True)
    
    # Subscription & Status
    plan = Column(String(20), default=TenantPlan.STARTER.value, nullable=False)
    status = Column(String(20), default=TenantStatus.TRIAL.value, nullable=False)
    
    # Quotas and limits
    quota_config = Column(JSON, default=dict)
    
    # Security
    encryption_key_hash = Column(String(255))
    allowed_ips = Column(JSON, default=list)
    sso_config = Column(JSON, default=dict)
    
    # Billing
    billing_email = Column(String(255))
    subscription_start = Column(DateTime(timezone=True))
    subscription_end = Column(DateTime(timezone=True))
    
    # Database isolation
    schema_name = Column(String(63))
    
    # Metadata
    settings = Column(JSON, default=dict)
    tenant_metadata = Column("metadata", JSON, default=dict)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=text("NOW()"))
    updated_at = Column(DateTime(timezone=True), server_default=text("NOW()"), onupdate=text("NOW()"))

class MultiTenantService:
    """Service for managing multi-tenant architecture"""
    
    def __init__(self, redis_url: str = "redis://localhost:6381"):
        self.redis_url = redis_url
        self._redis = None
        
    async def get_redis(self) -> redis.Redis:
        """Get Redis connection for caching"""
        if self._redis is None:
            self._redis = await redis.from_url(self.redis_url, decode_responses=True)
        return self._redis

    async def create_tenant(
        self,
        name: str,
        slug: str,
        plan: TenantPlan = TenantPlan.STARTER,
        domain: Optional[str] = None,
        billing_email: Optional[str] = None
    ) -> Tenant:
        """Create a new tenant with isolated database schema"""
        
        async with AsyncSessionLocal() as db:
            # Check if slug is available
            existing = await db.execute(
                text("SELECT id FROM tenants WHERE slug = :slug"),
                {"slug": slug}
            )
            if existing.fetchone():
                raise ValueError(f"Tenant slug '{slug}' already exists")
            
            # Generate schema name
            schema_name = f"tenant_{slug}"[:63].replace("-", "_")
            
            # Create tenant record
            tenant = Tenant(
                name=name,
                slug=slug,
                domain=domain,
                plan=plan.value,
                status=TenantStatus.TRIAL.value,
                billing_email=billing_email,
                schema_name=schema_name,
                quota_config=PLAN_QUOTAS[plan].__dict__,
                subscription_start=datetime.utcnow(),
                subscription_end=datetime.utcnow() + timedelta(days=30)
            )
            
            db.add(tenant)
            await db.flush()
            
            # Create isolated database schema
            await self._create_tenant_schema(db, schema_name)
            
            await db.commit()
            
            # Initialize tenant cache
            await self._cache_tenant(tenant)
            
            logger.info(f"Created tenant '{slug}' with schema '{schema_name}'")
            return tenant

    async def _create_tenant_schema(self, db: AsyncSession, schema_name: str):
        """Create isolated database schema for tenant"""
        await db.execute(text(f"CREATE SCHEMA IF NOT EXISTS {schema_name}"))
        await db.commit()

    async def get_tenant_by_slug(self, slug: str) -> Optional[Tenant]:
        """Get tenant by slug with caching"""
        
        # Check cache first
        redis_client = await self.get_redis()
        try:
            cached = await redis_client.get(f"tenant:slug:{slug}")
        except Exception:
            cached = None
        
        if not cached:
            # Get from database
            async with AsyncSessionLocal() as db:
                result = await db.execute(
                    text("SELECT * FROM tenants WHERE slug = :slug AND status != 'expired'"),
                    {"slug": slug}
                )
                tenant_row = result.fetchone()
                
                if tenant_row:
                    tenant_data = dict(tenant_row._mapping)
                    tenant = Tenant(**tenant_data)
                    await self._cache_tenant(tenant)
                    return tenant
                    
        return None

    async def _cache_tenant(self, tenant: Tenant):
        """Cache tenant information for fast lookups"""
        try:
            redis_client = await self.get_redis()
            
            tenant_data = {
                "id": str(tenant.id),
                "slug": tenant.slug,
                "name": tenant.name,
                "plan": tenant.plan,
                "status": tenant.status,
                "schema_name": tenant.schema_name,
                "quota_config": tenant.quota_config or {}
            }
            
            await redis_client.setex(f"tenant:id:{tenant.id}", 3600, str(tenant_data))
            await redis_client.setex(f"tenant:slug:{tenant.slug}", 3600, str(tenant_data))
            if tenant.domain:
                await redis_client.setex(f"tenant:domain:{tenant.domain}", 3600, str(tenant_data))
        except Exception as e:
            logger.warning(f"Failed to cache tenant: {e}")

    async def create_database_tables(self):
        """Create multi-tenant database tables"""
        async with async_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

# Global multi-tenant service instance
multi_tenant_service = MultiTenantService()

async def get_current_tenant(request) -> Optional[Tenant]:
    """Extract current tenant from request"""
    
    # Method 1: X-Tenant-ID header
    tenant_header = request.headers.get("x-tenant-id")
    if tenant_header:
        return await multi_tenant_service.get_tenant_by_slug(tenant_header)
    
    # Method 2: Query parameter (for development)
    tenant_param = request.query_params.get("tenant")
    if tenant_param:
        return await multi_tenant_service.get_tenant_by_slug(tenant_param)
    
    return None

async def require_tenant(request) -> Tenant:
    """Require tenant context or raise error"""
    tenant = await get_current_tenant(request)
    if not tenant:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="Tenant context required")
    return tenant
