#!/usr/bin/env python3
"""
KnowledgeHub Source Management Tool
Add, list, and manage knowledge sources for scraping
"""

import requests
import json
import sys
from typing import Optional, Dict
from datetime import datetime

API_BASE = "http://192.168.1.25:3000"

class SourceManager:
    def __init__(self, api_base=API_BASE):
        self.api_base = api_base
        
    def add_source(self, url: str, source_type: str, name: str, config: Optional[Dict] = None):
        """Add a new knowledge source"""
        data = {
            "name": name,
            "url": url,
            "type": source_type,
            "config": config or {
                "max_depth": 3,
                "max_pages": 500,
                "crawl_delay": 1.0
            }
        }
        
        response = requests.post(f"{self.api_base}/api/sources/", json=data)
        if response.status_code == 200:
            source = response.json()
            print(f"✅ Added source: {source['name']} (ID: {source['id']})")
            print(f"   Status: {source['status']}")
            return source
        else:
            print(f"❌ Failed to add source: {response.text}")
            return None
    
    def list_sources(self):
        """List all knowledge sources"""
        response = requests.get(f"{self.api_base}/api/sources/")
        if response.status_code == 200:
            sources = response.json()
            print("\n📚 Knowledge Sources:")
            print("-" * 80)
            for source in sources:
                print(f"ID: {source['id']}")
                print(f"Name: {source['name']}")
                print(f"URL: {source['url']}")
                print(f"Type: {source['type']}")
                print(f"Status: {source['status']}")
                print(f"Documents: {source.get('document_count', 0)}")
                print(f"Last Updated: {source.get('last_scraped_at', 'Never')}")
                print("-" * 80)
        else:
            print(f"❌ Failed to list sources: {response.text}")
    
    def refresh_source(self, source_id: str):
        """Trigger re-scraping of a source"""
        response = requests.post(f"{self.api_base}/api/sources/{source_id}/refresh")
        if response.status_code == 200:
            print(f"✅ Refresh triggered for source {source_id}")
        else:
            print(f"❌ Failed to refresh source: {response.text}")
    
    def delete_source(self, source_id: str):
        """Delete a source and all its data"""
        response = requests.delete(f"{self.api_base}/api/sources/{source_id}")
        if response.status_code == 200:
            print(f"✅ Deleted source {source_id}")
        else:
            print(f"❌ Failed to delete source: {response.text}")

def main():
    """CLI interface"""
    manager = SourceManager()
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python manage_sources.py list")
        print("  python manage_sources.py add <url> <type> <name>")
        print("  python manage_sources.py refresh <source_id>")
        print("  python manage_sources.py delete <source_id>")
        print("\nTypes: website, documentation, repository, api, wiki")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "list":
        manager.list_sources()
    
    elif command == "add":
        if len(sys.argv) < 5:
            print("Usage: python manage_sources.py add <url> <type> <name>")
            sys.exit(1)
        url = sys.argv[2]
        source_type = sys.argv[3]
        name = " ".join(sys.argv[4:])
        manager.add_source(url, source_type, name)
    
    elif command == "refresh":
        if len(sys.argv) < 3:
            print("Usage: python manage_sources.py refresh <source_id>")
            sys.exit(1)
        manager.refresh_source(sys.argv[2])
    
    elif command == "delete":
        if len(sys.argv) < 3:
            print("Usage: python manage_sources.py delete <source_id>")
            sys.exit(1)
        manager.delete_source(sys.argv[2])
    
    else:
        print(f"Unknown command: {command}")

if __name__ == "__main__":
    main()