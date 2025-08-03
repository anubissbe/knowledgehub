#!/usr/bin/env python3
"""API Key Management Script"""

import os
import sys
import secrets
import hashlib
import hmac
import argparse
from datetime import datetime, timedelta, timezone
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from api.config import settings
from api.models.auth import APIKey
from api.models.base import Base


def generate_api_key():
    """Generate a secure random API key"""
    return f"knhub_{secrets.token_urlsafe(32)}"


def hash_api_key(api_key: str, secret_key: str) -> str:
    """Hash API key using HMAC-SHA256"""
    return hmac.new(
        secret_key.encode(),
        api_key.encode(),
        hashlib.sha256
    ).hexdigest()


def create_api_key(session, name: str, permissions: list = None, expires_days: int = None):
    """Create a new API key"""
    # Generate key
    api_key = generate_api_key()
    key_hash = hash_api_key(api_key, settings.SECRET_KEY)
    
    # Set permissions
    if permissions is None:
        permissions = ["read", "write"]
    
    # Set expiry
    expires_at = None
    if expires_days:
        expires_at = datetime.now(timezone.utc) + timedelta(days=expires_days)
    
    # Create database entry
    db_key = APIKey(
        name=name,
        key_hash=key_hash,
        permissions=permissions,
        expires_at=expires_at,
        is_active=True
    )
    
    session.add(db_key)
    session.commit()
    
    print(f"\n✅ API Key created successfully!")
    print(f"Name: {name}")
    print(f"Permissions: {', '.join(permissions)}")
    print(f"Expires: {expires_at.isoformat() if expires_at else 'Never'}")
    print(f"\n🔑 API Key (save this, it won't be shown again):")
    print(f"{api_key}\n")
    
    return api_key


def list_api_keys(session):
    """List all API keys"""
    keys = session.query(APIKey).all()
    
    if not keys:
        print("No API keys found.")
        return
    
    print("\nAPI Keys:")
    print("-" * 80)
    for key in keys:
        status = "✅ Active" if key.is_active else "❌ Inactive"
        last_used = key.last_used_at.isoformat() if key.last_used_at else "Never"
        expires = key.expires_at.isoformat() if key.expires_at else "Never"
        
        print(f"ID: {key.id}")
        print(f"Name: {key.name}")
        print(f"Status: {status}")
        print(f"Permissions: {', '.join(key.permissions)}")
        print(f"Last Used: {last_used}")
        print(f"Expires: {expires}")
        print(f"Created: {key.created_at.isoformat()}")
        print("-" * 80)


def revoke_api_key(session, key_id: str):
    """Revoke an API key"""
    key = session.query(APIKey).filter(APIKey.id == key_id).first()
    
    if not key:
        print(f"❌ API key with ID {key_id} not found.")
        return
    
    key.is_active = False
    session.commit()
    
    print(f"✅ API key '{key.name}' has been revoked.")


def verify_api_key(session, api_key: str):
    """Verify an API key"""
    key_hash = hash_api_key(api_key, settings.SECRET_KEY)
    
    db_key = session.query(APIKey).filter(APIKey.key_hash == key_hash).first()
    
    if not db_key:
        print("❌ Invalid API key.")
        return
    
    if not db_key.is_valid():
        print(f"❌ API key '{db_key.name}' is not valid (inactive or expired).")
        return
    
    print(f"✅ API key '{db_key.name}' is valid!")
    print(f"Permissions: {', '.join(db_key.permissions)}")
    if db_key.expires_at:
        print(f"Expires: {db_key.expires_at.isoformat()}")


def main():
    parser = argparse.ArgumentParser(description="Manage KnowledgeHub API Keys")
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Create command
    create_parser = subparsers.add_parser("create", help="Create a new API key")
    create_parser.add_argument("name", help="Name for the API key")
    create_parser.add_argument("--permissions", nargs="+", default=["read", "write"],
                             help="Permissions (default: read write)")
    create_parser.add_argument("--expires-days", type=int,
                             help="Days until expiry (default: never)")
    
    # List command
    subparsers.add_parser("list", help="List all API keys")
    
    # Revoke command
    revoke_parser = subparsers.add_parser("revoke", help="Revoke an API key")
    revoke_parser.add_argument("key_id", help="ID of the API key to revoke")
    
    # Verify command
    verify_parser = subparsers.add_parser("verify", help="Verify an API key")
    verify_parser.add_argument("api_key", help="The API key to verify")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Create database session
    engine = create_engine(settings.DATABASE_URL)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        if args.command == "create":
            create_api_key(session, args.name, args.permissions, args.expires_days)
        elif args.command == "list":
            list_api_keys(session)
        elif args.command == "revoke":
            revoke_api_key(session, args.key_id)
        elif args.command == "verify":
            verify_api_key(session, args.api_key)
    finally:
        session.close()


if __name__ == "__main__":
    main()