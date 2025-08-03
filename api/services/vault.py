"""HashiCorp Vault integration for secure credential management"""

import os
import aiohttp
import asyncio
import logging
from typing import Dict, Any, Optional
from functools import lru_cache

logger = logging.getLogger(__name__)

class VaultClient:
    """AsyncIO client for HashiCorp Vault"""
    
    def __init__(self):
        self.vault_addr = os.getenv('VAULT_ADDR', 'http://192.168.1.24:8200')
        self.vault_token = os.getenv('VAULT_TOKEN')
        self.base_path = 'secret/data/knowledgehub'
        
    async def get_secret(self, secret_path: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a secret from Vault
        
        Args:
            secret_path: Path to secret (e.g., 'postgresql', 'minio', 'api')
            
        Returns:
            Dictionary containing secret data or None if error
        """
        if not self.vault_token:
            logger.warning("No Vault token provided, falling back to environment variables")
            return None
            
        full_path = f"{self.base_path}/{secret_path}"
        url = f"{self.vault_addr}/v1/{full_path}"
        
        headers = {
            'X-Vault-Token': self.vault_token,
            'Content-Type': 'application/json'
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get('data', {}).get('data', {})
                    elif response.status == 404:
                        logger.warning(f"Secret not found in Vault: {secret_path}")
                        return None
                    else:
                        logger.error(f"Vault request failed: {response.status} {await response.text()}")
                        return None
                        
        except Exception as e:
            logger.error(f"Error retrieving secret from Vault: {e}")
            return None
    
    async def health_check(self) -> bool:
        """Check if Vault is accessible"""
        try:
            url = f"{self.vault_addr}/v1/sys/health"
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                    return response.status in [200, 429]  # 429 = standby mode, still healthy
        except Exception as e:
            logger.error(f"Vault health check failed: {e}")
            return False


# Global vault client instance
vault_client = VaultClient()


async def get_database_config() -> Dict[str, str]:
    """Get database configuration from Vault or environment"""
    
    # Try to get from Vault first
    vault_config = await vault_client.get_secret('postgresql')
    
    if vault_config:
        logger.info("Using database credentials from Vault")
        username = vault_config.get('username', 'knowledgehub')
        password = vault_config.get('password', 'knowledgehub')
        database = vault_config.get('database', 'knowledgehub')
        host = vault_config.get('host', 'localhost')
        port = vault_config.get('port', '5433')
        
        return {
            'DATABASE_URL': f"postgresql://{username}:{password}@{host}:{port}/{database}"
        }
    else:
        # Fallback to environment variables
        logger.warning("Vault unavailable, using environment variables for database config")
        return {
            'DATABASE_URL': os.getenv('DATABASE_URL', 'postgresql://knowledgehub:knowledgehub123@postgres:5432/knowledgehub')
        }


async def get_minio_config() -> Dict[str, str]:
    """Get MinIO configuration from Vault or environment"""
    
    # Try to get from Vault first
    vault_config = await vault_client.get_secret('minio')
    
    if vault_config:
        logger.info("Using MinIO credentials from Vault")
        return {
            'S3_ACCESS_KEY_ID': vault_config.get('access_key', 'minioadmin'),
            'S3_SECRET_ACCESS_KEY': vault_config.get('secret_key'),
            'S3_ENDPOINT_URL': vault_config.get('endpoint', 'http://localhost:9010'),
            'S3_BUCKET_NAME': vault_config.get('bucket', 'knowledge-hub')
        }
    else:
        # Fallback to environment variables
        logger.warning("Vault unavailable, using environment variables for MinIO config")
        return {
            'S3_ACCESS_KEY_ID': os.getenv('S3_ACCESS_KEY_ID', 'minioadmin'),
            'S3_SECRET_ACCESS_KEY': os.getenv('S3_SECRET_ACCESS_KEY', 'minioadmin'),
            'S3_ENDPOINT_URL': os.getenv('S3_ENDPOINT_URL', 'http://localhost:9010'),
            'S3_BUCKET_NAME': os.getenv('S3_BUCKET_NAME', 'knowledge-hub')
        }


async def get_api_config() -> Dict[str, str]:
    """Get API security configuration from Vault or environment"""
    
    # Try to get from Vault first
    vault_config = await vault_client.get_secret('api')
    
    if vault_config:
        logger.info("Using API credentials from Vault")
        return {
            'SECRET_KEY': vault_config.get('secret_key'),
            'API_KEY': vault_config.get('api_key'),
            'JWT_ALGORITHM': vault_config.get('jwt_algorithm', 'HS256'),
            'JWT_EXPIRY_HOURS': str(vault_config.get('jwt_expiry_hours', '24'))
        }
    else:
        # Fallback to environment variables
        logger.warning("Vault unavailable, using environment variables for API config")
        return {
            'SECRET_KEY': os.getenv('SECRET_KEY', 'change-this-to-a-random-secret-key'),
            'API_KEY': os.getenv('API_KEY', 'dev-api-key-123'),
            'JWT_ALGORITHM': os.getenv('JWT_ALGORITHM', 'HS256'),
            'JWT_EXPIRY_HOURS': os.getenv('JWT_EXPIRY_HOURS', '24')
        }


@lru_cache(maxsize=1)
def get_vault_config_sync() -> Dict[str, str]:
    """Synchronous wrapper for getting all Vault configuration"""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    # Get all configurations
    db_config = loop.run_until_complete(get_database_config())
    minio_config = loop.run_until_complete(get_minio_config())
    api_config = loop.run_until_complete(get_api_config())
    
    # Combine all configurations
    vault_config = {**db_config, **minio_config, **api_config}
    
    # Update environment variables
    for key, value in vault_config.items():
        if value:  # Only set non-empty values
            os.environ[key] = value
    
    logger.info(f"Loaded {len(vault_config)} configuration items from Vault")
    return vault_config


# Initialize Vault configuration on import
try:
    vault_config = get_vault_config_sync()
    logger.info("✅ Vault integration initialized successfully")
except Exception as e:
    logger.error(f"❌ Failed to initialize Vault integration: {e}")
    vault_config = {}