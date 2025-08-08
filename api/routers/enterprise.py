"""
Enterprise API Router for KnowledgeHub Phase 4
Multi-tenant, scaling, and security management endpoints
"""

from fastapi import APIRouter, Depends, HTTPException, Request, BackgroundTasks
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime, timedelta

from ..services.multi_tenant import (
    multi_tenant_service, 
    get_current_tenant, 
    require_tenant,
    Tenant,
    TenantPlan,
    TenantStatus
)
from ..services.scaling_manager import (
    scaling_manager,
    ServiceType,
    ServiceNode,
    NodeStatus
)
from ..services.security_compliance import (
    security_compliance_service,
    get_security_context,
    require_permission,
    SecurityContext,
    AuditEventType,
    ResourceType
)

router = APIRouter(prefix="/api/v1/enterprise", tags=["enterprise"])

# Multi-Tenant Management Endpoints

@router.post("/tenants")
async def create_tenant(
    request: Request,
    tenant_data: Dict[str, Any],
    security_context: SecurityContext = Depends(get_security_context)
):
    """Create new tenant (system admin only)"""
    
    if not security_context or not security_context.has_role("system_admin"):
        raise HTTPException(status_code=403, detail="System admin role required")
    
    try:
        tenant = await multi_tenant_service.create_tenant(
            name=tenant_data["name"],
            slug=tenant_data["slug"],
            plan=TenantPlan(tenant_data.get("plan", "starter")),
            domain=tenant_data.get("domain"),
            billing_email=tenant_data.get("billing_email")
        )
        
        await security_compliance_service.log_audit_event(
            security_context,
            AuditEventType.DATA_CREATED,
            ResourceType.TENANT,
            str(tenant.id),
            {"tenant_name": tenant.name, "plan": tenant.plan}
        )
        
        return {
            "id": str(tenant.id),
            "name": tenant.name,
            "slug": tenant.slug,
            "plan": tenant.plan,
            "status": tenant.status,
            "schema_name": tenant.schema_name,
            "created_at": tenant.created_at.isoformat()
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create tenant: {str(e)}")

@router.get("/tenants/{tenant_slug}")
async def get_tenant_info(
    tenant_slug: str,
    security_context: SecurityContext = Depends(get_security_context)
):
    """Get tenant information"""
    
    if not security_context:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    tenant = await multi_tenant_service.get_tenant_by_slug(tenant_slug)
    if not tenant:
        raise HTTPException(status_code=404, detail="Tenant not found")
    
    # Check if user has access to this tenant
    if not security_context.has_role("system_admin") and security_context.tenant_id != tenant.id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    return {
        "id": str(tenant.id),
        "name": tenant.name,
        "slug": tenant.slug,
        "plan": tenant.plan,
        "status": tenant.status,
        "quota_config": tenant.quota_config,
        "created_at": tenant.created_at.isoformat()
    }

# Scaling Management Endpoints

@router.get("/cluster/status")
async def get_cluster_status(
    security_context: SecurityContext = Depends(get_security_context)
):
    """Get cluster status and health metrics"""
    
    if not security_context or not security_context.has_permission("analytics:read"):
        raise HTTPException(status_code=403, detail="Analytics read permission required")
    
    status = await scaling_manager.get_cluster_status()
    
    return {
        "cluster_status": status,
        "timestamp": datetime.utcnow().isoformat()
    }

@router.post("/cluster/nodes/register")
async def register_service_node(
    node_data: Dict[str, Any],
    security_context: SecurityContext = Depends(get_security_context)
):
    """Register new service node in cluster"""
    
    if not security_context or not security_context.has_role("system_admin"):
        raise HTTPException(status_code=403, detail="System admin role required")
    
    try:
        node = ServiceNode(
            id=node_data.get("id", str(uuid.uuid4())),
            service_type=ServiceType(node_data["service_type"]),
            host=node_data["host"],
            port=int(node_data["port"]),
            status=NodeStatus(node_data.get("status", "healthy")),
            metadata=node_data.get("metadata", {})
        )
        
        success = await scaling_manager.register_node(node)
        
        if success:
            await security_compliance_service.log_audit_event(
                security_context,
                AuditEventType.CONFIG_CHANGED,
                event_data={"action": "node_registered", "node_id": node.id, "service_type": node.service_type.value}
            )
            
            return {
                "success": True,
                "node_id": node.id,
                "endpoint": node.endpoint_url
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to register node")
            
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.put("/cluster/nodes/{node_id}/metrics")
async def update_node_metrics(
    node_id: str,
    metrics: Dict[str, Any],
    security_context: SecurityContext = Depends(get_security_context)
):
    """Update node performance metrics"""
    
    # Allow service nodes to update their own metrics
    if not security_context:
        # For now, allow unauthenticated metric updates from service nodes
        # In production, should use service-to-service authentication
        pass
    
    success = await scaling_manager.update_node_metrics(node_id, metrics)
    
    if success:
        return {"success": True}
    else:
        raise HTTPException(status_code=404, detail="Node not found")

@router.get("/cluster/scaling/check")
async def check_scaling_needs(
    security_context: SecurityContext = Depends(get_security_context)
):
    """Check if cluster needs scaling up or down"""
    
    if not security_context or not security_context.has_permission("analytics:read"):
        raise HTTPException(status_code=403, detail="Analytics read permission required")
    
    scaling_actions = await scaling_manager.check_scaling_needs()
    
    return {
        "scaling_actions": scaling_actions,
        "timestamp": datetime.utcnow().isoformat()
    }

@router.get("/cluster/services/{service_type}/endpoint")
async def get_service_endpoint(
    service_type: str,
    security_context: SecurityContext = Depends(get_security_context)
):
    """Get load-balanced endpoint for service type"""
    
    if not security_context:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    try:
        service_enum = ServiceType(service_type)
        endpoint = await scaling_manager.get_service_endpoint(service_enum)
        
        if endpoint:
            return {"endpoint": endpoint, "service_type": service_type}
        else:
            raise HTTPException(status_code=503, detail=f"No healthy nodes for service {service_type}")
            
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid service type: {service_type}")

# Security & Compliance Endpoints

@router.post("/security/roles")
async def create_role(
    role_data: Dict[str, Any],
    security_context: SecurityContext = Depends(get_security_context)
):
    """Create custom role for tenant"""
    
    if not security_context or not security_context.has_permission("user:admin"):
        raise HTTPException(status_code=403, detail="User admin permission required")
    
    # Implementation would create role in database
    # Simplified for demo
    
    await security_compliance_service.log_audit_event(
        security_context,
        AuditEventType.CONFIG_CHANGED,
        event_data={"action": "role_created", "role_name": role_data.get("name")}
    )
    
    return {"success": True, "role_id": str(uuid.uuid4())}

@router.get("/security/audit")
async def get_audit_logs(
    tenant_id: Optional[str] = None,
    event_type: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = 100,
    security_context: SecurityContext = Depends(get_security_context)
):
    """Get audit logs (admin only)"""
    
    if not security_context or not security_context.has_permission("user:admin"):
        raise HTTPException(status_code=403, detail="User admin permission required")
    
    # Implementation would query audit_logs table
    # Simplified for demo
    
    return {
        "audit_logs": [],
        "total_count": 0,
        "filters": {
            "tenant_id": tenant_id,
            "event_type": event_type,
            "start_date": start_date,
            "end_date": end_date
        }
    }

@router.post("/security/encrypt")
async def encrypt_data(
    data: Dict[str, str],
    security_context: SecurityContext = Depends(get_security_context)
):
    """Encrypt sensitive data"""
    
    if not security_context:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    try:
        encrypted = security_compliance_service.encrypt_data(data["plaintext"])
        
        await security_compliance_service.log_audit_event(
            security_context,
            AuditEventType.DATA_CREATED,
            event_data={"action": "data_encrypted", "data_length": len(data["plaintext"])}
        )
        
        return {"encrypted": encrypted}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Encryption failed: {str(e)}")

@router.post("/security/decrypt")
async def decrypt_data(
    data: Dict[str, str],
    security_context: SecurityContext = Depends(get_security_context)
):
    """Decrypt sensitive data"""
    
    if not security_context:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    try:
        decrypted = security_compliance_service.decrypt_data(data["encrypted"])
        
        await security_compliance_service.log_audit_event(
            security_context,
            AuditEventType.ACCESS_GRANTED,
            event_data={"action": "data_decrypted"}
        )
        
        return {"plaintext": decrypted}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Decryption failed: {str(e)}")

# GDPR Compliance Endpoints

@router.post("/gdpr/data-export/{user_id}")
async def export_user_data(
    user_id: str,
    security_context: SecurityContext = Depends(get_security_context),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Export all user data (GDPR Article 20)"""
    
    if not security_context:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    # Check if requesting own data or admin
    if str(security_context.user_id) != user_id and not security_context.has_permission("user:admin"):
        raise HTTPException(status_code=403, detail="Access denied")
    
    # Implementation would collect all user data
    # For demo, return success message
    
    await security_compliance_service.log_audit_event(
        security_context,
        AuditEventType.DATA_EXPORTED,
        ResourceType.USER,
        user_id,
        {"action": "gdpr_export", "requested_by": str(security_context.user_id)}
    )
    
    return {"success": True, "message": "Data export initiated", "export_id": str(uuid.uuid4())}

@router.delete("/gdpr/user-data/{user_id}")
async def delete_user_data(
    user_id: str,
    security_context: SecurityContext = Depends(get_security_context)
):
    """Delete all user data (GDPR Article 17 - Right to be forgotten)"""
    
    if not security_context or not security_context.has_permission("user:admin"):
        raise HTTPException(status_code=403, detail="User admin permission required")
    
    # Implementation would anonymize/delete all user data
    # For demo, return success message
    
    await security_compliance_service.log_audit_event(
        security_context,
        AuditEventType.DATA_DELETED,
        ResourceType.USER,
        user_id,
        {"action": "gdpr_deletion", "requested_by": str(security_context.user_id)}
    )
    
    return {"success": True, "message": "User data deletion initiated"}

# GPU Resource Management (V100-specific)

@router.get("/gpu/status")
async def get_gpu_status(
    security_context: SecurityContext = Depends(get_security_context)
):
    """Get V100 GPU cluster status"""
    
    if not security_context or not security_context.has_permission("gpu:access"):
        raise HTTPException(status_code=403, detail="GPU access permission required")
    
    cluster_status = await scaling_manager.get_cluster_status()
    
    # Extract GPU-specific metrics
    gpu_status = {
        "total_gpu_nodes": cluster_status.get("gpu_nodes", 0),
        "total_vram_free": cluster_status.get("total_vram_free", 0),
        "services": {}
    }
    
    for service_name, service_data in cluster_status.get("services", {}).items():
        if service_data.get("gpu_nodes", 0) > 0:
            gpu_status["services"][service_name] = {
                "gpu_nodes": service_data["gpu_nodes"],
                "vram_free": service_data.get("total_vram_free", 0)
            }
    
    return gpu_status

@router.post("/gpu/allocate")
async def allocate_gpu_resources(
    allocation_request: Dict[str, Any],
    security_context: SecurityContext = Depends(get_security_context)
):
    """Allocate GPU resources for AI workload"""
    
    if not security_context or not security_context.has_permission("gpu:access"):
        raise HTTPException(status_code=403, detail="GPU access permission required")
    
    required_vram = allocation_request.get("required_vram", 1000)  # MB
    
    from ..services.scaling_manager import get_optimal_gpu_allocation
    optimal_node = await get_optimal_gpu_allocation(required_vram, scaling_manager)
    
    if optimal_node:
        await security_compliance_service.log_audit_event(
            security_context,
            AuditEventType.ACCESS_GRANTED,
            event_data={
                "action": "gpu_allocated",
                "node_id": optimal_node.id,
                "vram_requested": required_vram,
                "vram_available": optimal_node.vram_free
            }
        )
        
        return {
            "allocated": True,
            "node_id": optimal_node.id,
            "endpoint": optimal_node.endpoint_url,
            "vram_available": optimal_node.vram_free,
            "gpu_usage": optimal_node.gpu_usage
        }
    else:
        await security_compliance_service.log_audit_event(
            security_context,
            AuditEventType.ACCESS_DENIED,
            event_data={
                "action": "gpu_allocation_failed",
                "vram_requested": required_vram,
                "reason": "insufficient_resources"
            }
        )
        
        raise HTTPException(
            status_code=503, 
            detail=f"No GPU nodes available with {required_vram}MB VRAM"
        )

# Health check endpoint
@router.get("/health")
async def enterprise_health_check():
    """Enterprise features health check"""
    
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "multi_tenant": "healthy",
            "scaling_manager": "healthy", 
            "security_compliance": "healthy"
        }
    }
    
    # Test Redis connectivity
    try:
        await multi_tenant_service.get_redis()
        await scaling_manager.get_redis()
    except Exception:
        health_status["services"]["redis"] = "unhealthy"
        health_status["status"] = "degraded"
    
    return health_status
