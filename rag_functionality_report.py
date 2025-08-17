#!/usr/bin/env python3
"""
KnowledgeHub RAG Functionality Assessment Report
Comprehensive evaluation of RAG system status and capabilities
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path

def check_rag_files():
    """Check which RAG implementation files exist"""
    rag_files = {
        "routers": [
            "api/routers/rag.py",
            "api/routers/rag_simple.py", 
            "api/routers/rag_advanced.py",
            "api/routers/graphrag.py",
            "api/routers/llamaindex_rag.py"
        ],
        "services": [
            "api/services/rag/simple_rag_service.py",
            "api/services/rag_pipeline.py",
            "api/services/graphrag_service.py", 
            "api/services/llamaindex_rag_service.py"
        ]
    }
    
    existing_files = {}
    for category, files in rag_files.items():
        existing_files[category] = []
        for file_path in files:
            full_path = Path(file_path)
            if full_path.exists():
                file_size = full_path.stat().st_size
                existing_files[category].append({
                    "file": file_path,
                    "exists": True,
                    "size": file_size,
                    "size_mb": round(file_size / (1024 * 1024), 2)
                })
            else:
                existing_files[category].append({
                    "file": file_path,
                    "exists": False
                })
    
    return existing_files

def check_dependencies():
    """Check RAG-related dependencies"""
    dependencies = {
        "core": {
            "weaviate-client": "Vector database client",
            "sentence-transformers": "Embedding generation", 
            "transformers": "Transformer models",
            "torch": "PyTorch for ML models"
        },
        "optional": {
            "llama-index-core": "LlamaIndex RAG framework",
            "llama-index-vector-stores-weaviate": "Weaviate integration",
            "llama-index-embeddings-huggingface": "HuggingFace embeddings"
        }
    }
    
    installed = {}
    for category, deps in dependencies.items():
        installed[category] = {}
        for dep, description in deps.items():
            try:
                __import__(dep.replace("-", "_"))
                installed[category][dep] = {"available": True, "description": description}
            except ImportError:
                installed[category][dep] = {"available": False, "description": description}
    
    return installed

def analyze_infrastructure():
    """Analyze infrastructure status from previous tests"""
    infrastructure_status = {}
    
    # Check if test results exist
    if Path("rag_test_results.json").exists():
        with open("rag_test_results.json", "r") as f:
            test_results = json.load(f)
            
        infrastructure_status = {
            "weaviate": {
                "status": "operational" if test_results["tests"]["weaviate_connection"]["success"] else "failed",
                "version": test_results["tests"]["weaviate_connection"]["result"].get("version") if test_results["tests"]["weaviate_connection"]["success"] else None,
                "classes": test_results["tests"]["weaviate_connection"]["result"].get("classes") if test_results["tests"]["weaviate_connection"]["success"] else []
            },
            "embeddings": {
                "status": "operational" if test_results["tests"]["embedding_service"]["success"] else "failed",
                "model": test_results["tests"]["embedding_service"]["result"].get("model") if test_results["tests"]["embedding_service"]["success"] else None,
                "dimension": test_results["tests"]["embedding_service"]["result"].get("dimension") if test_results["tests"]["embedding_service"]["success"] else None
            },
            "basic_pipeline": {
                "status": "operational" if test_results["tests"]["complete_rag_pipeline"]["success"] else "failed",
                "test_queries": test_results["tests"]["complete_rag_pipeline"]["result"].get("queries_tested") if test_results["tests"]["complete_rag_pipeline"]["success"] else 0
            }
        }
    else:
        infrastructure_status = {"status": "unknown", "reason": "No test results found"}
    
    return infrastructure_status

def assess_implementation_status():
    """Assess the status of different RAG implementations"""
    implementations = {
        "simple_rag": {
            "description": "Basic RAG using existing infrastructure",
            "files_required": ["api/routers/rag_simple.py", "api/services/rag/simple_rag_service.py"],
            "dependencies": ["weaviate-client", "sentence-transformers"],
            "status": "unknown"
        },
        "advanced_rag": {
            "description": "Advanced RAG with enhanced features",
            "files_required": ["api/routers/rag_advanced.py", "api/services/rag_pipeline.py"],
            "dependencies": ["weaviate-client", "sentence-transformers"],
            "status": "unknown"
        },
        "graphrag": {
            "description": "Graph-based RAG implementation",
            "files_required": ["api/routers/graphrag.py", "api/services/graphrag_service.py"],
            "dependencies": ["neo4j", "networkx"],
            "status": "unknown"
        },
        "llamaindex_rag": {
            "description": "LlamaIndex-based RAG implementation",
            "files_required": ["api/routers/llamaindex_rag.py", "api/services/llamaindex_rag_service.py"],
            "dependencies": ["llama-index-core"],
            "status": "unknown"
        }
    }
    
    files_status = check_rag_files()
    deps_status = check_dependencies()
    
    for impl_name, impl_info in implementations.items():
        # Check if files exist
        files_exist = all(
            any(f["file"] == req_file and f["exists"] 
                for category_files in files_status.values() 
                for f in category_files)
            for req_file in impl_info["files_required"]
        )
        
        # Check dependencies
        deps_available = all(
            any(dep.replace("-", "_") in category_deps and category_deps[dep.replace("-", "_")]["available"]
                for category_deps in deps_status.values())
            for dep in impl_info["dependencies"]
        )
        
        if files_exist and deps_available:
            implementations[impl_name]["status"] = "ready"
        elif files_exist:
            implementations[impl_name]["status"] = "files_present_deps_missing"
        elif deps_available:
            implementations[impl_name]["status"] = "deps_present_files_missing"
        else:
            implementations[impl_name]["status"] = "not_ready"
    
    return implementations

def generate_recommendations():
    """Generate recommendations based on assessment"""
    infrastructure = analyze_infrastructure()
    implementations = assess_implementation_status()
    
    recommendations = []
    
    # Infrastructure recommendations
    if infrastructure.get("weaviate", {}).get("status") == "operational":
        recommendations.append({
            "category": "infrastructure",
            "priority": "low",
            "action": "Weaviate is operational and ready for RAG workloads"
        })
    else:
        recommendations.append({
            "category": "infrastructure", 
            "priority": "high",
            "action": "Fix Weaviate connectivity issues before deploying RAG"
        })
    
    # Implementation recommendations
    ready_implementations = [name for name, impl in implementations.items() if impl["status"] == "ready"]
    
    if ready_implementations:
        recommendations.append({
            "category": "deployment",
            "priority": "medium", 
            "action": f"Ready implementations can be deployed: {', '.join(ready_implementations)}"
        })
    
    if "simple_rag" in ready_implementations:
        recommendations.append({
            "category": "deployment",
            "priority": "high",
            "action": "Start with Simple RAG implementation - it has the fewest dependencies"
        })
    
    # API recommendations
    recommendations.append({
        "category": "api",
        "priority": "high",
        "action": "Fix API startup issues to expose RAG endpoints (API currently not responding)"
    })
    
    # LlamaIndex recommendations
    if not any("llama-index" in impl["dependencies"] and impl["status"] == "ready" 
              for impl in implementations.values()):
        recommendations.append({
            "category": "enhancement",
            "priority": "low",
            "action": "Consider installing LlamaIndex for advanced RAG features (optional)"
        })
    
    return recommendations

def main():
    """Generate comprehensive RAG functionality report"""
    print("📊 KnowledgeHub RAG Systems Assessment Report")
    print("============================================")
    
    # Gather all data
    files_status = check_rag_files()
    deps_status = check_dependencies()
    infrastructure = analyze_infrastructure()
    implementations = assess_implementation_status()
    recommendations = generate_recommendations()
    
    report = {
        "generated_at": datetime.now().isoformat(),
        "summary": {
            "total_implementations": len(implementations),
            "ready_implementations": len([impl for impl in implementations.values() if impl["status"] == "ready"]),
            "infrastructure_operational": infrastructure.get("weaviate", {}).get("status") == "operational"
        },
        "files": files_status,
        "dependencies": deps_status, 
        "infrastructure": infrastructure,
        "implementations": implementations,
        "recommendations": recommendations
    }
    
    # Display summary
    print(f"\n📈 SUMMARY")
    print(f"  Total RAG implementations found: {report['summary']['total_implementations']}")
    print(f"  Ready for deployment: {report['summary']['ready_implementations']}")
    print(f"  Infrastructure status: {'✅ Operational' if report['summary']['infrastructure_operational'] else '❌ Issues detected'}")
    
    # Display implementation status
    print(f"\n🔧 IMPLEMENTATION STATUS")
    for name, impl in implementations.items():
        status_icon = {
            "ready": "✅",
            "files_present_deps_missing": "⚠️",
            "deps_present_files_missing": "⚠️", 
            "not_ready": "❌"
        }.get(impl["status"], "❓")
        print(f"  {name:15} {status_icon} {impl['status']}")
    
    # Display key recommendations
    print(f"\n💡 KEY RECOMMENDATIONS")
    high_priority = [r for r in recommendations if r["priority"] == "high"]
    for rec in high_priority[:3]:  # Show top 3 high priority
        print(f"  🔴 {rec['action']}")
    
    # Save full report
    with open("knowledgehub_rag_assessment.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n📄 Full assessment saved to: knowledgehub_rag_assessment.json")
    
    return report

if __name__ == "__main__":
    main()