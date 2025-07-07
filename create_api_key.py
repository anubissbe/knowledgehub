#!/usr/bin/env python3
"""Create a valid API key for the KnowledgeHub system"""

import sys
import secrets
import hashlib
import hmac
from datetime import datetime, timedelta
import asyncio
import os

# Add src to path
sys.path.append('/app/src')

from api.models import get_db
from api.models.auth import APIKey
from api.config import settings

def generate_api_key():
    """Generate a secure random API key"""
    return secrets.token_urlsafe(32)

def hash_api_key(api_key: str, secret_key: str) -> str:
    """Hash API key using HMAC-SHA256"""
    return hmac.new(
        secret_key.encode(),
        api_key.encode(),
        hashlib.sha256
    ).hexdigest()

def create_api_key():
    """Create a new API key and store it in the database"""
    # Generate new API key
    api_key = generate_api_key()
    
    # Hash the key
    key_hash = hash_api_key(api_key, settings.SECRET_KEY)
    
    # Create database entry
    db = next(get_db())
    try:
        # Create API key object
        api_key_obj = APIKey(
            name="Frontend API Key",
            key_hash=key_hash,
            permissions=["read", "write", "admin"],
            expires_at=datetime.utcnow() + timedelta(days=365),
            is_active=True
        )
        
        db.add(api_key_obj)
        db.commit()
        
        print(f"✅ Created API key: {api_key}")
        print(f"📝 Key ID: {api_key_obj.id}")
        print(f"🔐 Hash: {key_hash}")
        
        return api_key
        
    except Exception as e:
        db.rollback()
        print(f"❌ Error creating API key: {e}")
        return None
    finally:
        db.close()

if __name__ == "__main__":
    api_key = create_api_key()
    if api_key:
        print(f"\n🎯 Use this API key for frontend authentication:")
        print(f"API_KEY={api_key}")