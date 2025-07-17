#!/usr/bin/env python3
"""
KnowledgeHub README.md Compliance Checker
Verifies that all claims in README.md are actually implemented
"""

import requests
import sys

def check_endpoint(base_url, endpoint, description, method="GET", data=None):
    try:
        if method == "POST":
            r = requests.post(f"{base_url}{endpoint}", json=data or {}, timeout=2)
        else:
            r = requests.get(f"{base_url}{endpoint}", timeout=2)
        status = "✓" if r.status_code == 200 else f"✗ ({r.status_code})"
        return r.status_code == 200, f"  - {description}: {status}"
    except Exception as e:
        return False, f"  - {description}: ✗ (error: {str(e)[:30]})"

def main():
    base_url = "http://192.168.1.25:3000"
    
    print("🔍 KnowledgeHub README.md Compliance Check")
    print("=" * 50)
    
    all_checks = []
    
    # 1. Core Services
    print("\n📌 Core Services:")
    checks = [
        ("/health", "API Gateway (Port 3000)"),
        ("http://192.168.1.25:3100", "Web UI (Port 3100)")
    ]
    for endpoint, desc in checks:
        if endpoint.startswith("http"):
            try:
                r = requests.get(endpoint, timeout=2)
                status = r.status_code == 200
                result = f"  - {desc}: {'✓' if status else '✗'}"
            except:
                status = False
                result = f"  - {desc}: ✗"
        else:
            status, result = check_endpoint(base_url, endpoint, desc)
        all_checks.append(status)
        print(result)
    
    # 2. AI Intelligence Systems
    print("\n🧠 AI Intelligence Systems (Claimed in README):")
    ai_endpoints = {
        "/api/claude-auto/session/current": "Session Continuity",
        "/api/mistake-learning/stats": "Mistake Learning",
        "/api/decisions/recent": "Decision Recording",
        "/api/performance/report": "Performance Tracking",
        "/api/code-evolution/stats": "Code Evolution",
        "/api/proactive/next-tasks": "Predictive Analytics",
        "/api/claude-workflow/status": "Workflow Automation"
    }
    
    for endpoint, feature in ai_endpoints.items():
        status, result = check_endpoint(base_url, endpoint, feature)
        all_checks.append(status)
        print(result)
    
    # 3. Storage & Search
    print("\n🔍 Advanced Search & Storage:")
    storage_checks = [
        ("/api/memory/search", "Semantic Search (Weaviate)", "POST", {"query": "test"}),
        ("/api/memory/stats", "Multi-Source Integration", "GET"),
        ("/api/knowledge-graph/full", "Knowledge Graph (Neo4j)", "GET")
    ]
    
    for check in storage_checks:
        endpoint, desc = check[0], check[1]
        method = check[2] if len(check) > 2 else "GET"
        data = check[3] if len(check) > 3 else None
        status, result = check_endpoint(base_url, endpoint, desc, method, data)
        all_checks.append(status)
        print(result)
    
    # 4. Integration Features  
    print("\n🤖 AI Tool Integration:")
    integration_checks = [
        ("/api/memory", "Memory API", "POST", {"content": "test", "type": "general"}),
        ("/api/search/text", "Text Search", "POST", {"query": "test"}),
        ("/api/ai-features/summary", "AI Features Summary", "GET")
    ]
    
    for check in integration_checks:
        endpoint, desc = check[0], check[1]
        method = check[2] if len(check) > 2 else "GET"
        data = check[3] if len(check) > 3 else None
        status, result = check_endpoint(base_url, endpoint, desc, method, data)
        all_checks.append(status)
        print(result)
    
    # 5. Helper Scripts
    print("\n📜 Claude Code Integration:")
    try:
        with open("/opt/projects/knowledgehub/claude_code_helpers.sh", "r") as f:
            content = f.read()
            has_helpers = "claude-init" in content and "claude-error" in content
            print(f"  - Helper Scripts: {'✓' if has_helpers else '✗'}")
            all_checks.append(has_helpers)
    except:
        print("  - Helper Scripts: ✗")
        all_checks.append(False)
    
    # 6. MCP Server
    print("\n🔌 MCP Server:")
    try:
        import os
        mcp_exists = os.path.exists("/opt/projects/knowledgehub-mcp-server/index.js")
        print(f"  - MCP Server Implementation: {'✓' if mcp_exists else '✗'}")
        all_checks.append(mcp_exists)
    except:
        print("  - MCP Server Implementation: ✗")
        all_checks.append(False)
    
    # Summary
    print("\n" + "=" * 50)
    total = len(all_checks)
    passed = sum(all_checks)
    percentage = (passed / total) * 100 if total > 0 else 0
    
    print(f"\n📊 Compliance Summary:")
    print(f"  - Total Checks: {total}")
    print(f"  - Passed: {passed}")
    print(f"  - Failed: {total - passed}")
    print(f"  - Compliance Rate: {percentage:.1f}%")
    
    if percentage == 100:
        print("\n✅ FULLY COMPLIANT - All README.md claims are verified!")
    elif percentage >= 90:
        print("\n🟡 MOSTLY COMPLIANT - Minor gaps exist")
    else:
        print("\n⚠️  PARTIAL COMPLIANCE - Significant gaps exist")
    
    return 0 if percentage == 100 else 1

if __name__ == "__main__":
    sys.exit(main())