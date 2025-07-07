#!/usr/bin/env python3
"""
GPU Monitoring and Optimization for KnowledgeHub
Monitors GPU usage, provides metrics, and optimizes GPU allocation
"""

import os
import time
import json
import psutil
import subprocess
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import asyncio
import aiofiles
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('gpu_monitor')

@dataclass
class GPUMetrics:
    """GPU metrics data structure"""
    gpu_id: int
    name: str
    temperature: float
    power_usage: float
    power_limit: float
    memory_used: int
    memory_total: int
    memory_percent: float
    gpu_utilization: int
    processes: List[Dict[str, Any]]
    timestamp: str

@dataclass
class SystemMetrics:
    """System-wide metrics"""
    cpu_percent: float
    memory_percent: float
    disk_usage_percent: float
    timestamp: str

class GPUMonitor:
    """Comprehensive GPU monitoring and optimization"""
    
    def __init__(self, log_file: str = "/tmp/gpu_metrics.jsonl"):
        self.log_file = log_file
        self.metrics_history: List[Dict] = []
        self.alert_thresholds = {
            "temperature": 85,  # Celsius
            "memory_percent": 90,  # Percent
            "power_percent": 95   # Percent of power limit
        }
        
    def get_nvidia_smi_output(self) -> Optional[str]:
        """Get nvidia-smi output"""
        try:
            result = subprocess.run([
                'nvidia-smi', 
                '--query-gpu=index,name,temperature.gpu,power.draw,power.limit,memory.used,memory.total,utilization.gpu',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                logger.error(f"nvidia-smi failed: {result.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            logger.error("nvidia-smi timeout")
            return None
        except FileNotFoundError:
            logger.error("nvidia-smi not found")
            return None
    
    def get_gpu_processes(self) -> Dict[int, List[Dict]]:
        """Get processes running on each GPU"""
        try:
            result = subprocess.run([
                'nvidia-smi', 
                '--query-compute-apps=gpu_uuid,pid,process_name,used_memory',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode != 0:
                return {}
                
            # Parse process info
            gpu_processes = {}
            for line in result.stdout.strip().split('\n'):
                if not line.strip():
                    continue
                    
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 4:
                    try:
                        pid = int(parts[1])
                        process_name = parts[2]
                        memory_mb = int(parts[3])
                        
                        # Get additional process info
                        try:
                            proc = psutil.Process(pid)
                            cpu_percent = proc.cpu_percent()
                            cmdline = ' '.join(proc.cmdline()[:3])  # First 3 args
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            cpu_percent = 0
                            cmdline = process_name
                            
                        process_info = {
                            "pid": pid,
                            "name": process_name,
                            "memory_mb": memory_mb,
                            "cpu_percent": cpu_percent,
                            "cmdline": cmdline
                        }
                        
                        # Map to GPU index (simplified - assumes order matches)
                        gpu_id = 0  # Default to GPU 0, can be improved
                        if gpu_id not in gpu_processes:
                            gpu_processes[gpu_id] = []
                        gpu_processes[gpu_id].append(process_info)
                        
                    except (ValueError, IndexError):
                        continue
                        
            return gpu_processes
            
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return {}
    
    def parse_gpu_metrics(self) -> List[GPUMetrics]:
        """Parse GPU metrics from nvidia-smi"""
        output = self.get_nvidia_smi_output()
        if not output:
            return []
        
        gpu_processes = self.get_gpu_processes()
        metrics = []
        
        for line in output.split('\n'):
            if not line.strip():
                continue
                
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 8:
                try:
                    gpu_id = int(parts[0])
                    name = parts[1]
                    temperature = float(parts[2]) if parts[2] != '[Not Supported]' else 0
                    power_usage = float(parts[3]) if parts[3] != '[Not Supported]' else 0
                    power_limit = float(parts[4]) if parts[4] != '[Not Supported]' else 250
                    memory_used = int(parts[5])
                    memory_total = int(parts[6])
                    gpu_util = int(parts[7])
                    
                    memory_percent = (memory_used / memory_total * 100) if memory_total > 0 else 0
                    processes = gpu_processes.get(gpu_id, [])
                    
                    metrics.append(GPUMetrics(
                        gpu_id=gpu_id,
                        name=name,
                        temperature=temperature,
                        power_usage=power_usage,
                        power_limit=power_limit,
                        memory_used=memory_used,
                        memory_total=memory_total,
                        memory_percent=memory_percent,
                        gpu_utilization=gpu_util,
                        processes=processes,
                        timestamp=datetime.utcnow().isoformat()
                    ))
                    
                except (ValueError, IndexError) as e:
                    logger.warning(f"Failed to parse GPU metrics line: {line}, error: {e}")
                    
        return metrics
    
    def get_system_metrics(self) -> SystemMetrics:
        """Get system-wide metrics"""
        return SystemMetrics(
            cpu_percent=psutil.cpu_percent(interval=1),
            memory_percent=psutil.virtual_memory().percent,
            disk_usage_percent=psutil.disk_usage('/').percent,
            timestamp=datetime.utcnow().isoformat()
        )
    
    def check_alerts(self, gpu_metrics: List[GPUMetrics]) -> List[str]:
        """Check for alert conditions"""
        alerts = []
        
        for gpu in gpu_metrics:
            # Temperature alert
            if gpu.temperature > self.alert_thresholds["temperature"]:
                alerts.append(f"🔥 GPU {gpu.gpu_id} temperature HIGH: {gpu.temperature}°C")
            
            # Memory alert
            if gpu.memory_percent > self.alert_thresholds["memory_percent"]:
                alerts.append(f"🧠 GPU {gpu.gpu_id} memory HIGH: {gpu.memory_percent:.1f}%")
            
            # Power alert
            power_percent = (gpu.power_usage / gpu.power_limit * 100) if gpu.power_limit > 0 else 0
            if power_percent > self.alert_thresholds["power_percent"]:
                alerts.append(f"⚡ GPU {gpu.gpu_id} power HIGH: {power_percent:.1f}%")
                
        return alerts
    
    def get_optimization_suggestions(self, gpu_metrics: List[GPUMetrics]) -> List[str]:
        """Provide optimization suggestions"""
        suggestions = []
        
        # Check for idle GPUs
        idle_gpus = [gpu for gpu in gpu_metrics if gpu.gpu_utilization < 10 and gpu.memory_percent < 5]
        if idle_gpus:
            gpu_ids = [str(gpu.gpu_id) for gpu in idle_gpus]
            suggestions.append(f"💡 Idle GPUs available: {', '.join(gpu_ids)} - Consider load balancing")
        
        # Check for memory pressure
        high_memory_gpus = [gpu for gpu in gpu_metrics if gpu.memory_percent > 80]
        if high_memory_gpus:
            gpu_ids = [str(gpu.gpu_id) for gpu in high_memory_gpus]
            suggestions.append(f"⚠️ High memory usage on GPUs: {', '.join(gpu_ids)} - Consider batch size reduction")
        
        # Check for imbalanced load
        if len(gpu_metrics) > 1:
            utilizations = [gpu.gpu_utilization for gpu in gpu_metrics]
            if max(utilizations) - min(utilizations) > 50:
                suggestions.append("⚖️ GPU load imbalance detected - Consider redistributing workload")
        
        return suggestions
    
    async def log_metrics(self, data: Dict[str, Any]):
        """Log metrics to file"""
        try:
            async with aiofiles.open(self.log_file, 'a') as f:
                await f.write(json.dumps(data) + '\n')
        except Exception as e:
            logger.error(f"Failed to log metrics: {e}")
    
    async def collect_and_log_metrics(self):
        """Collect all metrics and log them"""
        gpu_metrics = self.parse_gpu_metrics()
        system_metrics = self.get_system_metrics()
        alerts = self.check_alerts(gpu_metrics)
        suggestions = self.get_optimization_suggestions(gpu_metrics)
        
        data = {
            "timestamp": datetime.utcnow().isoformat(),
            "gpus": [asdict(gpu) for gpu in gpu_metrics],
            "system": asdict(system_metrics),
            "alerts": alerts,
            "suggestions": suggestions,
            "summary": {
                "total_gpus": len(gpu_metrics),
                "active_gpus": len([gpu for gpu in gpu_metrics if gpu.gpu_utilization > 5]),
                "total_memory_used_mb": sum(gpu.memory_used for gpu in gpu_metrics),
                "total_memory_available_mb": sum(gpu.memory_total for gpu in gpu_metrics),
                "avg_temperature": sum(gpu.temperature for gpu in gpu_metrics) / len(gpu_metrics) if gpu_metrics else 0,
                "total_power_usage": sum(gpu.power_usage for gpu in gpu_metrics)
            }
        }
        
        # Log to file
        await self.log_metrics(data)
        
        # Store in memory (keep last 100 entries)
        self.metrics_history.append(data)
        if len(self.metrics_history) > 100:
            self.metrics_history.pop(0)
        
        return data
    
    def print_current_status(self, data: Dict[str, Any]):
        """Print current status to console"""
        print(f"\n🔍 GPU Monitor - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        # GPU Status
        for gpu_data in data["gpus"]:
            gpu_id = gpu_data["gpu_id"]
            name = gpu_data["name"]
            temp = gpu_data["temperature"]
            mem_pct = gpu_data["memory_percent"]
            util = gpu_data["gpu_utilization"]
            power = gpu_data["power_usage"]
            
            status = "🟢 IDLE" if util < 10 else "🟡 ACTIVE" if util < 80 else "🔴 BUSY"
            
            print(f"GPU {gpu_id} ({name}): {status}")
            print(f"  Temperature: {temp}°C | Memory: {mem_pct:.1f}% | Utilization: {util}% | Power: {power}W")
            
            if gpu_data["processes"]:
                print(f"  Processes: {len(gpu_data['processes'])}")
                for proc in gpu_data["processes"][:2]:  # Show first 2
                    print(f"    PID {proc['pid']}: {proc['name']} ({proc['memory_mb']}MB)")
            else:
                print(f"  Processes: None")
        
        # System Status
        sys_data = data["system"]
        print(f"\n💻 System: CPU {sys_data['cpu_percent']:.1f}% | RAM {sys_data['memory_percent']:.1f}% | Disk {sys_data['disk_usage_percent']:.1f}%")
        
        # Alerts
        if data["alerts"]:
            print("\n⚠️ Alerts:")
            for alert in data["alerts"]:
                print(f"  {alert}")
        
        # Suggestions
        if data["suggestions"]:
            print("\n💡 Optimization Suggestions:")
            for suggestion in data["suggestions"]:
                print(f"  {suggestion}")

async def main():
    """Main monitoring loop"""
    monitor = GPUMonitor()
    
    print("🚀 Starting GPU Monitor for KnowledgeHub")
    print(f"📊 Logging metrics to: {monitor.log_file}")
    print("Press Ctrl+C to stop\n")
    
    try:
        while True:
            data = await monitor.collect_and_log_metrics()
            monitor.print_current_status(data)
            
            # Wait before next collection
            await asyncio.sleep(30)  # Collect every 30 seconds
            
    except KeyboardInterrupt:
        print("\n👋 GPU Monitor stopped")
    except Exception as e:
        logger.error(f"Monitor error: {e}")

if __name__ == "__main__":
    asyncio.run(main())