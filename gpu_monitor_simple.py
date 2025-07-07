#!/usr/bin/env python3
"""
Simple GPU Monitoring for KnowledgeHub
No external dependencies - uses subprocess and built-in modules only
"""

import os
import time
import json
import subprocess
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('gpu_monitor')

class SimpleGPUMonitor:
    """Simple GPU monitoring without external dependencies"""
    
    def __init__(self, log_file: str = "/tmp/gpu_metrics.jsonl"):
        self.log_file = log_file
        self.alert_thresholds = {
            "temperature": 85,  # Celsius
            "memory_percent": 90,  # Percent
            "power_percent": 95   # Percent of power limit
        }
        
    def get_nvidia_smi_data(self) -> Optional[Dict]:
        """Get comprehensive nvidia-smi data"""
        try:
            # Get GPU info
            result = subprocess.run([
                'nvidia-smi', 
                '--query-gpu=index,name,temperature.gpu,power.draw,power.limit,memory.used,memory.total,utilization.gpu',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode != 0:
                logger.error(f"nvidia-smi failed: {result.stderr}")
                return None
                
            gpu_data = []
            for line in result.stdout.strip().split('\n'):
                if not line.strip():
                    continue
                    
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 8:
                    try:
                        gpu_id = int(parts[0])
                        name = parts[1]
                        temp = float(parts[2]) if parts[2] != '[Not Supported]' else 0
                        power_usage = float(parts[3]) if parts[3] != '[Not Supported]' else 0
                        power_limit = float(parts[4]) if parts[4] != '[Not Supported]' else 250
                        memory_used = int(parts[5])
                        memory_total = int(parts[6])
                        gpu_util = int(parts[7])
                        
                        memory_percent = (memory_used / memory_total * 100) if memory_total > 0 else 0
                        power_percent = (power_usage / power_limit * 100) if power_limit > 0 else 0
                        
                        gpu_data.append({
                            "gpu_id": gpu_id,
                            "name": name,
                            "temperature": temp,
                            "power_usage": power_usage,
                            "power_limit": power_limit,
                            "power_percent": power_percent,
                            "memory_used_mb": memory_used,
                            "memory_total_mb": memory_total,
                            "memory_percent": memory_percent,
                            "gpu_utilization": gpu_util
                        })
                        
                    except (ValueError, IndexError) as e:
                        logger.warning(f"Failed to parse GPU line: {line}, error: {e}")
            
            # Get process info
            processes = self.get_gpu_processes()
            
            return {
                "gpus": gpu_data,
                "processes": processes,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.error(f"nvidia-smi error: {e}")
            return None
    
    def get_gpu_processes(self) -> List[Dict]:
        """Get processes running on GPUs"""
        try:
            result = subprocess.run([
                'nvidia-smi', 
                '--query-compute-apps=gpu_uuid,pid,process_name,used_memory',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode != 0:
                return []
                
            processes = []
            for line in result.stdout.strip().split('\n'):
                if not line.strip():
                    continue
                    
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 4:
                    try:
                        pid = int(parts[1])
                        process_name = parts[2]
                        memory_mb = int(parts[3])
                        
                        # Get process command line
                        try:
                            proc_result = subprocess.run(['ps', '-p', str(pid), '-o', 'cmd', '--no-headers'], 
                                                       capture_output=True, text=True, timeout=5)
                            cmdline = proc_result.stdout.strip() if proc_result.returncode == 0 else process_name
                        except:
                            cmdline = process_name
                            
                        processes.append({
                            "pid": pid,
                            "name": process_name,
                            "memory_mb": memory_mb,
                            "cmdline": cmdline[:100]  # Truncate long command lines
                        })
                        
                    except (ValueError, IndexError):
                        continue
                        
            return processes
            
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return []
    
    def get_system_info(self) -> Dict:
        """Get basic system info without psutil"""
        try:
            # Get CPU info
            with open('/proc/loadavg', 'r') as f:
                load_avg = f.read().strip().split()[:3]
            
            # Get memory info
            memory_info = {}
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    if line.startswith(('MemTotal:', 'MemAvailable:')):
                        key, value = line.split(':')
                        memory_info[key.strip()] = int(value.strip().split()[0])
            
            memory_used = memory_info.get('MemTotal', 0) - memory_info.get('MemAvailable', 0)
            memory_percent = (memory_used / memory_info.get('MemTotal', 1)) * 100
            
            # Get disk info for root
            disk_result = subprocess.run(['df', '/'], capture_output=True, text=True)
            disk_percent = 0
            if disk_result.returncode == 0:
                lines = disk_result.stdout.strip().split('\n')
                if len(lines) > 1:
                    parts = lines[1].split()
                    if len(parts) >= 5:
                        disk_percent = int(parts[4].rstrip('%'))
            
            return {
                "load_avg": [float(x) for x in load_avg],
                "memory_percent": memory_percent,
                "disk_percent": disk_percent,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"System info error: {e}")
            return {"error": str(e)}
    
    def analyze_and_suggest(self, data: Dict) -> Dict:
        """Analyze data and provide alerts/suggestions"""
        alerts = []
        suggestions = []
        
        if not data or "gpus" not in data:
            return {"alerts": ["❌ No GPU data available"], "suggestions": []}
        
        gpus = data["gpus"]
        
        # Check alerts
        for gpu in gpus:
            gpu_id = gpu["gpu_id"]
            
            # Temperature alert
            if gpu["temperature"] > self.alert_thresholds["temperature"]:
                alerts.append(f"🔥 GPU {gpu_id} temperature HIGH: {gpu['temperature']}°C")
            
            # Memory alert
            if gpu["memory_percent"] > self.alert_thresholds["memory_percent"]:
                alerts.append(f"🧠 GPU {gpu_id} memory HIGH: {gpu['memory_percent']:.1f}%")
            
            # Power alert
            if gpu["power_percent"] > self.alert_thresholds["power_percent"]:
                alerts.append(f"⚡ GPU {gpu_id} power HIGH: {gpu['power_percent']:.1f}%")
        
        # Generate suggestions
        idle_gpus = [gpu for gpu in gpus if gpu["gpu_utilization"] < 10 and gpu["memory_percent"] < 5]
        if idle_gpus:
            gpu_ids = [str(gpu["gpu_id"]) for gpu in idle_gpus]
            suggestions.append(f"💡 Idle GPUs available: {', '.join(gpu_ids)} - Consider enabling embedding service")
        
        active_gpus = [gpu for gpu in gpus if gpu["gpu_utilization"] > 5]
        if active_gpus:
            suggestions.append(f"✅ {len(active_gpus)} GPU(s) active - Good utilization")
        
        high_memory_gpus = [gpu for gpu in gpus if gpu["memory_percent"] > 80]
        if high_memory_gpus:
            gpu_ids = [str(gpu["gpu_id"]) for gpu in high_memory_gpus]
            suggestions.append(f"⚠️ High memory usage on GPUs: {', '.join(gpu_ids)}")
        
        # Load balancing suggestion
        if len(gpus) > 1:
            utilizations = [gpu["gpu_utilization"] for gpu in gpus]
            if max(utilizations) - min(utilizations) > 50:
                suggestions.append("⚖️ GPU load imbalance - Consider multi-GPU embedding service")
        
        return {"alerts": alerts, "suggestions": suggestions}
    
    def log_metrics(self, data: Dict):
        """Log metrics to file"""
        try:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(data) + '\n')
        except Exception as e:
            logger.error(f"Failed to log metrics: {e}")
    
    def print_status(self, data: Dict, analysis: Dict):
        """Print current status"""
        print(f"\n🔍 GPU Monitor - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)
        
        if not data or "gpus" not in data:
            print("❌ No GPU data available")
            return
        
        # GPU Status
        for gpu in data["gpus"]:
            gpu_id = gpu["gpu_id"]
            name = gpu["name"]
            temp = gpu["temperature"]
            mem_pct = gpu["memory_percent"]
            util = gpu["gpu_utilization"]
            power = gpu["power_usage"]
            power_pct = gpu["power_percent"]
            
            if util < 10:
                status = "🟢 IDLE"
            elif util < 50:
                status = "🟡 LIGHT"
            elif util < 80:
                status = "🟠 ACTIVE"
            else:
                status = "🔴 BUSY"
            
            print(f"\nGPU {gpu_id} ({name}): {status}")
            print(f"  🌡️  Temperature: {temp}°C")
            print(f"  🧠 Memory: {mem_pct:.1f}% ({gpu['memory_used_mb']}/{gpu['memory_total_mb']} MB)")
            print(f"  ⚡ Power: {power}W ({power_pct:.1f}% of {gpu['power_limit']}W)")
            print(f"  📊 Utilization: {util}%")
        
        # Processes
        if data.get("processes"):
            print(f"\n🏃 GPU Processes ({len(data['processes'])}):")
            for proc in data["processes"]:
                print(f"  PID {proc['pid']}: {proc['name']} ({proc['memory_mb']}MB)")
                if len(proc['cmdline']) > 50:
                    print(f"    {proc['cmdline'][:80]}...")
                else:
                    print(f"    {proc['cmdline']}")
        else:
            print("\n🏃 GPU Processes: None")
        
        # System info
        if "load_avg" in data.get("system", {}):
            sys_data = data["system"]
            print(f"\n💻 System:")
            print(f"  Load Average: {sys_data['load_avg']}")
            print(f"  Memory: {sys_data['memory_percent']:.1f}%")
            print(f"  Disk: {sys_data['disk_percent']}%")
        
        # Alerts
        if analysis["alerts"]:
            print(f"\n⚠️ Alerts:")
            for alert in analysis["alerts"]:
                print(f"  {alert}")
        
        # Suggestions
        if analysis["suggestions"]:
            print(f"\n💡 Suggestions:")
            for suggestion in analysis["suggestions"]:
                print(f"  {suggestion}")
        
        # Summary
        total_memory = sum(gpu["memory_total_mb"] for gpu in data["gpus"])
        used_memory = sum(gpu["memory_used_mb"] for gpu in data["gpus"])
        total_power = sum(gpu["power_usage"] for gpu in data["gpus"])
        
        print(f"\n📈 Summary:")
        print(f"  Total GPUs: {len(data['gpus'])}")
        print(f"  Memory Usage: {used_memory}/{total_memory} MB ({used_memory/total_memory*100:.1f}%)")
        print(f"  Total Power: {total_power:.1f}W")
        print(f"  Active Processes: {len(data.get('processes', []))}")

def main():
    """Main monitoring function"""
    monitor = SimpleGPUMonitor()
    
    print("🚀 Starting Simple GPU Monitor for KnowledgeHub")
    print(f"📊 Logging to: {monitor.log_file}")
    print("Press Ctrl+C to stop\n")
    
    try:
        # Single monitoring pass
        data = monitor.get_nvidia_smi_data()
        if data:
            # Add system info
            data["system"] = monitor.get_system_info()
            
            # Analyze and get suggestions
            analysis = monitor.analyze_and_suggest(data)
            
            # Create full report
            report = {
                **data,
                **analysis,
                "monitoring_timestamp": datetime.utcnow().isoformat()
            }
            
            # Log metrics
            monitor.log_metrics(report)
            
            # Print status
            monitor.print_status(data, analysis)
            
            return report
        else:
            print("❌ Failed to get GPU data")
            return None
            
    except KeyboardInterrupt:
        print("\n👋 Monitor stopped")
        return None
    except Exception as e:
        logger.error(f"Monitor error: {e}")
        return None

if __name__ == "__main__":
    main()