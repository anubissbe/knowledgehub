#!/usr/bin/env python3
"""
GPU Dashboard for KnowledgeHub
Quick status display and optimization recommendations
"""

import json
import subprocess
from datetime import datetime
from typing import Dict, List

def get_gpu_status() -> Dict:
    """Get current GPU status"""
    try:
        result = subprocess.run([
            'nvidia-smi', 
            '--query-gpu=index,name,temperature.gpu,power.draw,memory.used,memory.total,utilization.gpu',
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode != 0:
            return {"error": "nvidia-smi failed"}
            
        gpus = []
        for line in result.stdout.strip().split('\n'):
            if not line.strip():
                continue
                
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 7:
                try:
                    gpus.append({
                        "id": int(parts[0]),
                        "name": parts[1],
                        "temp": float(parts[2]) if parts[2] != '[Not Supported]' else 0,
                        "power": float(parts[3]) if parts[3] != '[Not Supported]' else 0,
                        "memory_used": int(parts[4]),
                        "memory_total": int(parts[5]),
                        "utilization": int(parts[6])
                    })
                except (ValueError, IndexError):
                    continue
        
        return {"gpus": gpus, "timestamp": datetime.now().isoformat()}
        
    except Exception as e:
        return {"error": str(e)}

def get_embeddings_service_status() -> Dict:
    """Check embeddings service status"""
    try:
        import requests
        response = requests.get("http://localhost:8100/health", timeout=5)
        if response.status_code == 200:
            return {"status": "active", "data": response.json()}
        else:
            return {"status": "error", "code": response.status_code}
    except Exception as e:
        return {"status": "inactive", "error": str(e)}

def print_dashboard():
    """Print GPU dashboard"""
    print("🎮 KnowledgeHub GPU Dashboard")
    print("=" * 50)
    print(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # GPU Status
    gpu_data = get_gpu_status()
    if "error" in gpu_data:
        print(f"❌ GPU Error: {gpu_data['error']}")
        return
    
    print(f"\n📊 GPU Status ({len(gpu_data['gpus'])} GPUs)")
    print("-" * 40)
    
    total_memory_used = 0
    total_memory_available = 0
    active_gpus = 0
    
    for gpu in gpu_data["gpus"]:
        memory_pct = (gpu["memory_used"] / gpu["memory_total"]) * 100
        
        # Status emoji
        if gpu["utilization"] < 5:
            status = "🟢 IDLE"
        elif gpu["utilization"] < 50:
            status = "🟡 ACTIVE"
        else:
            status = "🔴 BUSY"
            
        if gpu["utilization"] > 5:
            active_gpus += 1
            
        print(f"GPU {gpu['id']} ({gpu['name'][:20]}...): {status}")
        print(f"  🌡️  {gpu['temp']}°C | 🧠 {memory_pct:.1f}% ({gpu['memory_used']}/{gpu['memory_total']} MB)")
        print(f"  ⚡ {gpu['power']}W | 📈 {gpu['utilization']}% utilization")
        
        total_memory_used += gpu["memory_used"]
        total_memory_available += gpu["memory_total"]
    
    # Overall metrics
    overall_memory_pct = (total_memory_used / total_memory_available) * 100
    print(f"\n📈 Overall GPU Utilization:")
    print(f"  Active GPUs: {active_gpus}/{len(gpu_data['gpus'])}")
    print(f"  Total Memory: {total_memory_used}/{total_memory_available} MB ({overall_memory_pct:.1f}%)")
    
    # Embeddings Service Status
    print(f"\n🤖 AI Services Status:")
    embeddings_status = get_embeddings_service_status()
    if embeddings_status["status"] == "active":
        print("  ✅ Embeddings Service: Running")
        data = embeddings_status.get("data", {})
        print(f"     Model: {data.get('model', 'Unknown')}")
        print(f"     Device: {data.get('device', 'Unknown')}")
    elif embeddings_status["status"] == "inactive":
        print("  ❌ Embeddings Service: Not Running")
    else:
        print(f"  ⚠️  Embeddings Service: Error ({embeddings_status.get('code', 'Unknown')})")
    
    # Optimization Recommendations
    print(f"\n💡 Optimization Recommendations:")
    
    idle_gpus = [gpu for gpu in gpu_data["gpus"] if gpu["utilization"] < 10]
    if len(idle_gpus) == len(gpu_data["gpus"]):
        print("  🔧 All GPUs idle - Consider re-enabling GPU embedding generation in RAG processor")
        print("  📝 Action: Uncomment embedding service code in main.py")
    elif len(idle_gpus) > 0:
        idle_ids = [str(gpu["id"]) for gpu in idle_gpus]
        print(f"  ⚖️  GPU(s) {', '.join(idle_ids)} idle - Consider load balancing")
        print("  📝 Action: Deploy additional embedding service instances")
    else:
        print("  ✅ Good GPU utilization across all devices")
    
    if overall_memory_pct < 20:
        print("  📈 Low memory usage - Can increase batch sizes for better performance")
        print("  📝 Action: Increase batch_size in RAG processor configuration")
    
    if embeddings_status["status"] != "active":
        print("  🚨 Embeddings service not active - GPU resources underutilized")
        print("  📝 Action: Check embeddings service container and dependencies")

if __name__ == "__main__":
    try:
        print_dashboard()
    except KeyboardInterrupt:
        print("\n👋 Dashboard closed")
    except Exception as e:
        print(f"❌ Dashboard error: {e}")