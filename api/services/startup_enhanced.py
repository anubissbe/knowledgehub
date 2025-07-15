"""
Enhanced Startup Service with Security Monitoring

Extends the existing startup service to initialize enhanced security monitoring.
"""

import logging
from typing import Optional

from .startup import initialize_services as base_initialize_services
from .startup import shutdown_services as base_shutdown_services
from ..security.monitoring_enhanced import init_enhanced_monitoring
from ..security.alerting import get_alerting_system

logger = logging.getLogger(__name__)


async def initialize_services():
    """Initialize all services including enhanced security monitoring"""
    # First initialize base services
    await base_initialize_services()
    
    # Initialize enhanced security monitoring
    try:
        logger.info("Initializing enhanced security monitoring...")
        
        # Initialize enhanced monitoring with metrics and alerting
        monitor = init_enhanced_monitoring()
        
        # Configure alerting webhooks if available
        alerting_system = get_alerting_system()
        if alerting_system:
            # Example webhook configuration (can be loaded from environment)
            webhook_config = {
                'webhook_urls': {
                    'default': None,  # Set from environment if available
                    'critical': None
                },
                'slack': {
                    'webhook_url': None  # Set from environment if available
                }
            }
            
            # Update alerting configuration
            alerting_system.config.update(webhook_config)
        
        logger.info("Enhanced security monitoring initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize enhanced security monitoring: {e}")
        # Continue without enhanced monitoring rather than failing startup


async def shutdown_services():
    """Shutdown all services including enhanced monitoring"""
    # Shutdown base services
    await base_shutdown_services()
    
    # Additional cleanup for enhanced monitoring if needed
    logger.info("Enhanced security monitoring shutdown complete")