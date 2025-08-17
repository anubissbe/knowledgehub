
import os
import hashlib
import secrets
from typing import Optional
from datetime import datetime, timedelta
import redis
from cryptography.fernet import Fernet

class SecureAPIKeyManager:
    def __init__(self):
        self.redis_client = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            decode_responses=True
        )
        # Generate or load encryption key
        self.cipher_key = os.getenv("API_KEY_ENCRYPTION_KEY", Fernet.generate_key())
        self.cipher = Fernet(self.cipher_key)
    
    def generate_api_key(self, user_id: str, expires_in_days: int = 30) -> str:
        """Generate secure API key"""
        # Create API key
        raw_key = f"{user_id}:{secrets.token_urlsafe(32)}"
        api_key = f"khub_{secrets.token_urlsafe(32)}"
        
        # Hash for storage
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        # Store in Redis with expiration
        self.redis_client.setex(
            f"api_key:{key_hash}",
            timedelta(days=expires_in_days),
            json.dumps({
                "user_id": user_id,
                "created_at": datetime.utcnow().isoformat(),
                "expires_in_days": expires_in_days
            })
        )
        
        return api_key
    
    def validate_api_key(self, api_key: str) -> Optional[dict]:
        """Validate API key and return user data"""
        if not api_key.startswith("khub_"):
            return None
        
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        data = self.redis_client.get(f"api_key:{key_hash}")
        
        if data:
            return json.loads(data)
        return None
    
    def revoke_api_key(self, api_key: str) -> bool:
        """Revoke an API key"""
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        return self.redis_client.delete(f"api_key:{key_hash}") > 0

api_key_manager = SecureAPIKeyManager()
