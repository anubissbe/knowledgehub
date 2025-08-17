#!/usr/bin/env python3
"""
Test database connection script.
"""

import os
import sys
import psycopg2
from sqlalchemy import create_engine

# Test connection parameters
DB_CONFIGS = [
    {
        "name": "localhost_default",
        "host": "localhost",
        "port": 5432,
        "database": "postgres",
        "username": "postgres",
        "password": ""
    },
    {
        "name": "localhost_knowledgehub",
        "host": "localhost", 
        "port": 5432,
        "database": "knowledgehub",
        "username": "knowledgehub",
        "password": "knowledgehub123"
    },
    {
        "name": "docker_setup",
        "host": "localhost",
        "port": 5433,
        "database": "knowledgehub",
        "username": "knowledgehub", 
        "password": "knowledgehub123"
    }
]

def test_psycopg2_connection(config):
    """Test direct psycopg2 connection."""
    try:
        conn = psycopg2.connect(
            host=config["host"],
            port=config["port"],
            database=config["database"],
            user=config["username"],
            password=config["password"]
        )
        cursor = conn.cursor()
        cursor.execute("SELECT version();")
        version = cursor.fetchone()
        cursor.close()
        conn.close()
        return True, version[0] if version else "Connected"
    except Exception as e:
        return False, str(e)

def test_sqlalchemy_connection(config):
    """Test SQLAlchemy connection."""
    try:
        url = f"postgresql://{config['username']}:{config['password']}@{config['host']}:{config['port']}/{config['database']}"
        engine = create_engine(url)
        with engine.connect() as conn:
            result = conn.execute("SELECT version();")
            version = result.fetchone()
            return True, version[0] if version else "Connected"
    except Exception as e:
        return False, str(e)

def main():
    print("Testing database connections...")
    print("=" * 60)
    
    for config in DB_CONFIGS:
        print(f"\nTesting: {config['name']}")
        print(f"  Host: {config['host']}:{config['port']}")
        print(f"  Database: {config['database']}")
        print(f"  User: {config['username']}")
        
        # Test psycopg2
        success, result = test_psycopg2_connection(config)
        print(f"  psycopg2: {'✅ SUCCESS' if success else '❌ FAILED'}")
        if not success:
            print(f"    Error: {result}")
        else:
            print(f"    Version: {result[:80]}...")
        
        # Test SQLAlchemy  
        success, result = test_sqlalchemy_connection(config)
        print(f"  SQLAlchemy: {'✅ SUCCESS' if success else '❌ FAILED'}")
        if not success:
            print(f"    Error: {result}")
        
        print("-" * 40)

if __name__ == "__main__":
    main()