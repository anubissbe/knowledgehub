#!/usr/bin/env python3
"""Test decision recording functionality"""

import requests
import json
from datetime import datetime

BASE_URL = "http://localhost:3000"

def test_decision_recording():
    """Test creating a decision record"""
    print("\n=== Testing Decision Recording ===")
    
    decision_data = {
        "decision": "Use PostgreSQL for persistent storage",
        "category": "database",
        "reasoning": "PostgreSQL provides ACID compliance, JSON support, and excellent performance",
        "alternatives": [
            {
                "option": "MongoDB",
                "pros": ["Flexible schema", "Good for document storage"],
                "cons": ["Eventual consistency", "Less mature ecosystem"],
                "reasoning": "Not suitable for our transactional requirements"
            },
            {
                "option": "Redis",
                "pros": ["Very fast", "Simple"],
                "cons": ["Limited query capabilities", "Memory-based"],
                "reasoning": "Better as a cache than primary storage"
            }
        ],
        "context": {
            "project": "KnowledgeHub",
            "requirements": ["ACID compliance", "Complex queries", "JSON support"]
        },
        "confidence": 0.9,
        "impact": "high",
        "tags": ["database", "architecture", "storage"]
    }
    
    # Convert to query parameters
    params = {
        "decision_title": decision_data["decision"],
        "chosen_solution": decision_data["decision"],
        "reasoning": decision_data["reasoning"],
        "confidence": decision_data["confidence"]
    }
    
    # Send request with query parameters
    response = requests.post(
        f"{BASE_URL}/api/decisions/record",
        params=params,
        json={
            "alternatives": decision_data["alternatives"],
            "context": decision_data["context"],
            "category": decision_data["category"],
            "impact": decision_data["impact"],
            "tags": decision_data["tags"]
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Decision recorded successfully!")
        print(f"   Decision ID: {result.get('decision_id')}")
        print(f"   Title: {result.get('title')}")
        print(f"   Status: {result.get('status')}")
        return result.get('decision_id')
    else:
        print(f"❌ Failed to record decision: {response.status_code}")
        print(f"   Response: {response.text}")
        return None

def test_decision_search():
    """Test searching for decisions"""
    print("\n=== Testing Decision Search ===")
    
    response = requests.get(
        f"{BASE_URL}/api/decisions/search",
        params={"query": "database"}
    )
    
    if response.status_code == 200:
        results = response.json()
        print(f"✅ Search completed successfully!")
        print(f"   Found {len(results)} decisions")
        for decision in results[:3]:  # Show first 3
            print(f"   - {decision.get('title')} (ID: {decision.get('decision_id')})")
    else:
        print(f"❌ Search failed: {response.status_code}")
        print(f"   Response: {response.text}")

def test_decision_explain(decision_id):
    """Test getting decision explanation"""
    print(f"\n=== Testing Decision Explanation ===")
    
    if not decision_id:
        print("⚠️  No decision ID to test with")
        return
    
    response = requests.get(f"{BASE_URL}/api/decisions/explain/{decision_id}")
    
    if response.status_code == 200:
        explanation = response.json()
        print(f"✅ Got decision explanation!")
        print(f"   Decision: {explanation.get('decision')}")
        print(f"   Category: {explanation.get('category')}")
        print(f"   Confidence: {explanation.get('confidence')}")
        print(f"   Alternatives considered: {len(explanation.get('alternatives', []))}")
    else:
        print(f"❌ Failed to get explanation: {response.status_code}")
        print(f"   Response: {response.text}")

def main():
    print("=== Decision Recording Test Suite ===")
    
    # Test recording a decision
    decision_id = test_decision_recording()
    
    # Test searching for decisions
    test_decision_search()
    
    # Test explaining a decision
    test_decision_explain(decision_id)
    
    print("\n=== Tests completed ===")

if __name__ == "__main__":
    main()