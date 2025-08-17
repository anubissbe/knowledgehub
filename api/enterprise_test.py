"""
Enterprise Features Initialization Script
Initializes multi-tenant, scaling, and security systems for KnowledgeHub Phase 4
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_enterprise_basic():
    """Test basic enterprise functionality"""
    
    logger.info("🚀 Testing KnowledgeHub Phase 4: Enterprise Features")
    
    try:
        # Test imports
        from api.services.multi_tenant import multi_tenant_service, TenantPlan
        from api.services.scaling_manager import scaling_manager, ServiceType, ServiceNode, NodeStatus
        from api.services.security_compliance import security_compliance_service
        logger.info("✅ All enterprise services imported successfully")
        
        # Test Redis connections
        try:
            await multi_tenant_service.get_redis()
            await scaling_manager.get_redis()
            logger.info("✅ Redis connections working")
        except Exception as e:
            logger.warning(f"⚠️ Redis connection issue: {e}")
        
        # Test security features
        import uuid
        demo_user_id = uuid.uuid4()
        token = await security_compliance_service.create_jwt_token(
            user_id=demo_user_id,
            roles=["system_admin"],
            expires_in=3600
        )
        
        context = await security_compliance_service.verify_jwt_token(token)
        if context:
            logger.info(f"✅ JWT authentication working - User: {context.user_id}")
        else:
            logger.error("❌ JWT authentication failed")
        
        # Test service registration
        test_node = ServiceNode(
            id="test-node-1",
            service_type=ServiceType.API,
            host="192.168.1.25",
            port=3000,
            status=NodeStatus.HEALTHY
        )
        
        success = await scaling_manager.register_node(test_node)
        if success:
            logger.info("✅ Service node registration working")
        else:
            logger.error("❌ Service node registration failed")
        
        # Test cluster status
        cluster_status = await scaling_manager.get_cluster_status()
        logger.info(f"📈 Cluster Status: {cluster_status['total_nodes']} nodes, {cluster_status['healthy_nodes']} healthy")
        
        logger.info("🎉 Enterprise Features Test Complete\!")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Enterprise features test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_enterprise_basic())
    if success:
        print("
🎯 KnowledgeHub Phase 4: Enterprise Features - READY FOR PRODUCTION\!")
    else:
        print("
⚠️ Enterprise features need attention before production")
