#!/usr/bin/env python3
"""Test AI feature endpoints to debug the AI Intelligence page"""

import requests
import json

API_KEY = 'knhub_V05H0-fZ_kUJB93um_pS00Nxv3i60gZogNGhMLtTFbM'
API_URL = 'http://192.168.1.25:3000'
HEADERS = {'X-API-Key': API_KEY}

AI_FEATURES = [
    {'id': 'session-continuity', 'endpoint': '/api/claude-auto/session/current'},
    {'id': 'mistake-learning', 'endpoint': '/api/mistake-learning/patterns'},
    {'id': 'proactive-assistant', 'endpoint': '/api/proactive/health'}, 
    {'id': 'decision-reasoning', 'endpoint': '/api/decisions/search?query=recent'},
    {'id': 'code-evolution', 'endpoint': '/api/code-evolution/history?file_path=test'},
    {'id': 'performance-optimization', 'endpoint': '/api/performance/stats'},
    {'id': 'pattern-recognition', 'endpoint': '/api/patterns/recent'},
    {'id': 'workflow-integration', 'endpoint': '/api/claude-workflow/stats'}
]

print("Testing AI Feature Endpoints...")
print("=" * 60)

working_endpoints = []
broken_endpoints = []

for feature in AI_FEATURES:
    try:
        url = f"{API_URL}{feature['endpoint']}"
        response = requests.get(url, headers=HEADERS, timeout=5)
        
        if response.status_code == 200:
            print(f"✅ {feature['id']}: {response.status_code}")
            working_endpoints.append(feature['id'])
            # Show sample data
            data = response.json()
            if isinstance(data, dict):
                print(f"   Data keys: {list(data.keys())[:5]}")
            elif isinstance(data, list):
                print(f"   Array length: {len(data)}")
        else:
            print(f"❌ {feature['id']}: {response.status_code}")
            broken_endpoints.append(feature['id'])
            
    except Exception as e:
        print(f"❌ {feature['id']}: ERROR - {str(e)}")
        broken_endpoints.append(feature['id'])

print("\n" + "=" * 60)
print(f"Summary: {len(working_endpoints)}/{len(AI_FEATURES)} endpoints working")
print(f"Working: {', '.join(working_endpoints)}")
print(f"Broken: {', '.join(broken_endpoints)}")