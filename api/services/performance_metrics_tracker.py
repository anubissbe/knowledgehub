"""
Performance & Quality Metrics Tracker - Track command execution, success rates, and optimize performance
"""

import json
import hashlib
import time
import statistics
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, deque
from pathlib import Path
import subprocess
import psutil
import re

from sqlalchemy.orm import Session
from sqlalchemy import cast, String, desc, and_, func, or_

from ..models.memory import MemoryItem


class PerformanceMetricsTracker:
    """Track and analyze performance metrics, command patterns, and system optimization"""
    
    def __init__(self):
        self.metrics_file = Path.home() / ".claude_performance_metrics.json"
        self.patterns_file = Path.home() / ".claude_execution_patterns.json"
        self.optimization_file = Path.home() / ".claude_performance_optimizations.json"
        
        # Command categories
        self.command_categories = {
            "file_operations": ["read", "write", "edit", "multiedit", "glob", "ls"],
            "search_operations": ["grep", "search", "find", "locate"],
            "code_operations": ["refactor", "analyze", "lint", "format"],
            "git_operations": ["commit", "push", "pull", "status", "diff"],
            "build_operations": ["build", "compile", "install", "test"],
            "api_operations": ["request", "fetch", "post", "get"],
            "shell_operations": ["bash", "execute", "run", "command"]
        }
        
        # Performance thresholds
        self.performance_thresholds = {
            "fast": 1.0,          # < 1 second
            "normal": 5.0,        # < 5 seconds
            "slow": 30.0,         # < 30 seconds
            "very_slow": 60.0     # >= 60 seconds
        }
        
        # Success patterns
        self.success_indicators = [
            "successfully", "completed", "done", "finished", "created",
            "updated", "saved", "built", "passed", "✅", "success"
        ]
        
        self.failure_indicators = [
            "error", "failed", "exception", "not found", "denied",
            "timeout", "crashed", "aborted", "❌", "failure"
        ]
        
        # Optimization strategies
        self.optimization_strategies = {
            "batch_operations": {
                "description": "Batch similar operations together",
                "applicable_to": ["file_operations", "api_operations"],
                "expected_improvement": 0.4
            },
            "caching": {
                "description": "Cache frequently accessed data",
                "applicable_to": ["search_operations", "api_operations"],
                "expected_improvement": 0.6
            },
            "parallel_execution": {
                "description": "Execute independent operations in parallel",
                "applicable_to": ["build_operations", "test_operations"],
                "expected_improvement": 0.5
            },
            "early_termination": {
                "description": "Stop early when condition is met",
                "applicable_to": ["search_operations"],
                "expected_improvement": 0.3
            },
            "index_usage": {
                "description": "Use indexes for faster lookups",
                "applicable_to": ["search_operations", "file_operations"],
                "expected_improvement": 0.7
            }
        }
        
        # Performance history cache
        self.performance_cache = deque(maxlen=1000)
        self.pattern_cache = defaultdict(list)
        
        self._load_metrics_history()
        self._load_execution_patterns()
    
    def _load_metrics_history(self):
        """Load performance metrics history"""
        self.metrics_history = {}
        if self.metrics_file.exists():
            with open(self.metrics_file, 'r') as f:
                loaded_data = json.load(f)
                # Convert regular dicts back to defaultdicts
                for category, data in loaded_data.items():
                    self.metrics_history[category] = {
                        "total_executions": data.get("total_executions", 0),
                        "successful_executions": data.get("successful_executions", 0),
                        "total_time": data.get("total_time", 0.0),
                        "patterns_detected": defaultdict(int, data.get("patterns_detected", {})),
                        "hourly_distribution": defaultdict(int, data.get("hourly_distribution", {}))
                    }
    
    def _load_execution_patterns(self):
        """Load learned execution patterns"""
        self.execution_patterns = {}
        if self.patterns_file.exists():
            with open(self.patterns_file, 'r') as f:
                loaded_data = json.load(f)
                # Convert regular dicts back to defaultdicts where needed
                for category, data in loaded_data.items():
                    self.execution_patterns[category] = {
                        "successful_patterns": data.get("successful_patterns", []),
                        "failure_patterns": data.get("failure_patterns", []),
                        "optimization_results": data.get("optimization_results", {}),
                        "common_contexts": defaultdict(int, data.get("common_contexts", {}))
                    }
    
    def track_command_execution(self, db: Session,
                              command_type: str,
                              command_details: Dict[str, Any],
                              execution_time: float,
                              success: bool,
                              output_size: Optional[int] = None,
                              error_message: Optional[str] = None,
                              context: Optional[Dict[str, Any]] = None,
                              project_id: Optional[str] = None,
                              session_id: Optional[str] = None) -> Dict[str, Any]:
        """Track execution of a command with performance metrics"""
        
        # Generate execution ID
        execution_id = hashlib.md5(
            f"{command_type}:{datetime.utcnow().isoformat()}:{execution_time}".encode()
        ).hexdigest()[:12]
        
        # Categorize command
        command_category = self._categorize_command(command_type)
        
        # Analyze performance
        performance_analysis = self._analyze_performance(
            execution_time, success, output_size, command_category
        )
        
        # Detect execution patterns
        detected_patterns = self._detect_execution_patterns(
            command_type, command_details, execution_time, success, context
        )
        
        # Get optimization suggestions
        optimization_suggestions = self._get_optimization_suggestions(
            command_type, execution_time, performance_analysis, detected_patterns
        )
        
        # Calculate quality score
        quality_score = self._calculate_quality_score(
            execution_time, success, performance_analysis
        )
        
        # Create metrics record
        metrics_record = {
            "execution_id": execution_id,
            "timestamp": datetime.utcnow().isoformat(),
            "command_type": command_type,
            "command_category": command_category,
            "command_details": command_details,
            "execution_time": execution_time,
            "success": success,
            "output_size": output_size,
            "error_message": error_message,
            "performance_analysis": performance_analysis,
            "detected_patterns": detected_patterns,
            "optimization_suggestions": optimization_suggestions,
            "quality_score": quality_score,
            "context": context or {},
            "project_id": project_id,
            "session_id": session_id,
            "system_metrics": self._get_system_metrics()
        }
        
        # Store in database
        content = self._format_metrics_content(metrics_record)
        memory = self._store_metrics_memory(db, content, metrics_record, project_id)
        
        # Update metrics history
        self._update_metrics_history(metrics_record)
        
        # Learn from execution
        self._learn_from_execution(metrics_record)
        
        # Update cache
        self.performance_cache.append(metrics_record)
        self.pattern_cache[command_category].append(execution_time)
        
        return {
            "execution_id": execution_id,
            "tracked": True,
            "performance_rating": performance_analysis["performance_rating"],
            "quality_score": quality_score,
            "optimization_available": len(optimization_suggestions) > 0,
            "suggestions_count": len(optimization_suggestions),
            "memory_id": str(memory.id)
        }
    
    def _categorize_command(self, command_type: str) -> str:
        """Categorize command into predefined categories"""
        command_lower = command_type.lower()
        
        for category, keywords in self.command_categories.items():
            if any(keyword in command_lower for keyword in keywords):
                return category
        
        return "general_operation"
    
    def _analyze_performance(self, execution_time: float, success: bool,
                           output_size: Optional[int], category: str) -> Dict[str, Any]:
        """Analyze command performance"""
        
        # Determine performance rating
        if execution_time < self.performance_thresholds["fast"]:
            performance_rating = "fast"
        elif execution_time < self.performance_thresholds["normal"]:
            performance_rating = "normal"
        elif execution_time < self.performance_thresholds["slow"]:
            performance_rating = "slow"
        else:
            performance_rating = "very_slow"
        
        # Calculate efficiency
        if output_size and output_size > 0:
            throughput = output_size / execution_time  # bytes per second
        else:
            throughput = None
        
        # Get historical comparison
        historical_avg = self._get_historical_average(category)
        if historical_avg > 0:
            relative_performance = execution_time / historical_avg
        else:
            relative_performance = 1.0
        
        return {
            "performance_rating": performance_rating,
            "execution_time": execution_time,
            "success": success,
            "throughput": throughput,
            "relative_performance": relative_performance,
            "historical_average": historical_avg,
            "is_outlier": relative_performance > 2.0 or relative_performance < 0.5
        }
    
    def _detect_execution_patterns(self, command_type: str, details: Dict[str, Any],
                                 execution_time: float, success: bool,
                                 context: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect patterns in command execution"""
        patterns = []
        
        # Repeated command pattern
        recent_commands = [r["command_type"] for r in list(self.performance_cache)[-10:]]
        if recent_commands.count(command_type) >= 3:
            patterns.append({
                "pattern": "repeated_command",
                "description": "Same command executed multiple times",
                "frequency": recent_commands.count(command_type),
                "suggestion": "Consider batching or caching results"
            })
        
        # Sequential operation pattern
        if self._detect_sequential_pattern(command_type):
            patterns.append({
                "pattern": "sequential_operations",
                "description": "Operations that could be parallelized",
                "suggestion": "Execute independent operations in parallel"
            })
        
        # Performance degradation pattern
        if self._detect_degradation_pattern(command_type, execution_time):
            patterns.append({
                "pattern": "performance_degradation",
                "description": "Performance getting worse over time",
                "suggestion": "Check for resource leaks or growing datasets"
            })
        
        # Failure pattern
        if not success and self._detect_failure_pattern(command_type):
            patterns.append({
                "pattern": "recurring_failure",
                "description": "Command fails frequently",
                "suggestion": "Review error handling and prerequisites"
            })
        
        # Time-based pattern
        hour = datetime.utcnow().hour
        if self._detect_time_pattern(command_type, hour):
            patterns.append({
                "pattern": "time_based",
                "description": f"Command often runs at this time ({hour}:00)",
                "suggestion": "Consider scheduling or automation"
            })
        
        return patterns
    
    def _detect_sequential_pattern(self, current_command: str) -> bool:
        """Detect if commands are sequential and could be parallelized"""
        if len(self.performance_cache) < 3:
            return False
        
        recent = list(self.performance_cache)[-3:]
        categories = [self._categorize_command(r["command_type"]) for r in recent]
        
        # Check if different categories (could be parallel)
        return len(set(categories)) == len(categories)
    
    def _detect_degradation_pattern(self, command_type: str, current_time: float) -> bool:
        """Detect performance degradation over time"""
        similar_executions = [
            r for r in self.performance_cache
            if r["command_type"] == command_type
        ]
        
        if len(similar_executions) < 5:
            return False
        
        # Get last 5 execution times
        recent_times = [r["execution_time"] for r in similar_executions[-5:]]
        
        # Check if trending upward
        return all(recent_times[i] <= recent_times[i+1] for i in range(len(recent_times)-1))
    
    def _detect_failure_pattern(self, command_type: str) -> bool:
        """Detect recurring failures"""
        similar_executions = [
            r for r in self.performance_cache
            if r["command_type"] == command_type
        ]
        
        if len(similar_executions) < 3:
            return False
        
        # Check failure rate
        failures = sum(1 for r in similar_executions[-5:] if not r["success"])
        return failures >= 2
    
    def _detect_time_pattern(self, command_type: str, current_hour: int) -> bool:
        """Detect time-based execution patterns"""
        category = self._categorize_command(command_type)
        
        if category not in self.execution_patterns:
            return False
        
        hourly_counts = self.execution_patterns[category].get("hourly_distribution", {})
        hour_str = str(current_hour)
        
        if hour_str in hourly_counts:
            # If this hour has > 20% of daily executions
            total = sum(hourly_counts.values())
            return hourly_counts[hour_str] / total > 0.2 if total > 0 else False
        
        return False
    
    def _get_optimization_suggestions(self, command_type: str, execution_time: float,
                                    performance: Dict[str, Any],
                                    patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Get optimization suggestions based on performance and patterns"""
        suggestions = []
        category = self._categorize_command(command_type)
        
        # Check if command is slow
        if performance["performance_rating"] in ["slow", "very_slow"]:
            # Find applicable optimization strategies
            for strategy_name, strategy in self.optimization_strategies.items():
                if category in strategy["applicable_to"]:
                    expected_time = execution_time * (1 - strategy["expected_improvement"])
                    suggestions.append({
                        "strategy": strategy_name,
                        "description": strategy["description"],
                        "expected_improvement": strategy["expected_improvement"],
                        "expected_time": expected_time,
                        "priority": "high" if execution_time > 10 else "medium"
                    })
        
        # Pattern-based suggestions
        for pattern in patterns:
            if pattern["pattern"] == "repeated_command":
                suggestions.append({
                    "strategy": "result_caching",
                    "description": "Cache results to avoid repeated execution",
                    "expected_improvement": 0.8,
                    "expected_time": execution_time * 0.2,
                    "priority": "high"
                })
            elif pattern["pattern"] == "sequential_operations":
                suggestions.append({
                    "strategy": "parallel_execution",
                    "description": "Run independent operations in parallel",
                    "expected_improvement": 0.5,
                    "expected_time": execution_time * 0.5,
                    "priority": "medium"
                })
        
        # Historical performance-based suggestions
        if performance["relative_performance"] > 1.5:
            suggestions.append({
                "strategy": "performance_investigation",
                "description": "Investigate why performance is worse than historical average",
                "expected_improvement": 0.3,
                "expected_time": performance["historical_average"],
                "priority": "medium"
            })
        
        return suggestions
    
    def _calculate_quality_score(self, execution_time: float, success: bool,
                               performance: Dict[str, Any]) -> float:
        """Calculate overall quality score for the execution"""
        score = 0.0
        
        # Success factor (40%)
        if success:
            score += 0.4
        
        # Performance factor (30%)
        if performance["performance_rating"] == "fast":
            score += 0.3
        elif performance["performance_rating"] == "normal":
            score += 0.2
        elif performance["performance_rating"] == "slow":
            score += 0.1
        
        # Relative performance factor (20%)
        if performance["relative_performance"] <= 1.0:
            score += 0.2
        elif performance["relative_performance"] <= 1.5:
            score += 0.1
        
        # Efficiency factor (10%)
        if performance.get("throughput"):
            # Arbitrary threshold for good throughput
            if performance["throughput"] > 1000000:  # 1MB/s
                score += 0.1
            elif performance["throughput"] > 100000:  # 100KB/s
                score += 0.05
        
        return min(1.0, score)
    
    def _get_system_metrics(self) -> Dict[str, Any]:
        """Get current system performance metrics"""
        try:
            return {
                "cpu_percent": psutil.cpu_percent(interval=0.1),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_usage_percent": psutil.disk_usage('/').percent,
                "load_average": psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else None
            }
        except:
            return {}
    
    def _get_historical_average(self, category: str) -> float:
        """Get historical average execution time for category"""
        if category not in self.pattern_cache or not self.pattern_cache[category]:
            return 0.0
        
        recent_times = list(self.pattern_cache[category])[-20:]
        return statistics.mean(recent_times) if recent_times else 0.0
    
    def _format_metrics_content(self, metrics: Dict[str, Any]) -> str:
        """Format metrics record for storage"""
        lines = [
            f"PERFORMANCE: {metrics['command_type']}",
            f"Category: {metrics['command_category']}",
            f"Execution Time: {metrics['execution_time']:.2f}s",
            f"Status: {'✅ Success' if metrics['success'] else '❌ Failed'}",
            f"Performance: {metrics['performance_analysis']['performance_rating']}",
            f"Quality Score: {metrics['quality_score']:.0%}",
            f""
        ]
        
        if metrics['detected_patterns']:
            lines.extend(["Patterns detected:"])
            for pattern in metrics['detected_patterns']:
                lines.append(f"  - {pattern['pattern']}: {pattern['description']}")
        
        if metrics['optimization_suggestions']:
            lines.extend(["", "Optimization suggestions:"])
            for suggestion in metrics['optimization_suggestions'][:3]:
                lines.append(f"  - {suggestion['strategy']}: {suggestion['description']} ({suggestion['expected_improvement']:.0%} improvement)")
        
        if metrics.get('error_message'):
            lines.extend(["", f"Error: {metrics['error_message']}"])
        
        return "\n".join(lines)
    
    def _store_metrics_memory(self, db: Session, content: str,
                            metrics_data: Dict[str, Any],
                            project_id: Optional[str]) -> MemoryItem:
        """Store metrics record in memory"""
        tags = ["performance_metrics", metrics_data["command_category"]]
        if project_id:
            tags.append(f"project:{project_id}")
        
        if not metrics_data["success"]:
            tags.append("failure")
        
        if metrics_data["performance_analysis"]["performance_rating"] == "very_slow":
            tags.append("performance_issue")
        
        # Include execution_id and timestamp in hash to ensure uniqueness
        unique_content = f"{content}\n\nExecution ID: {metrics_data['execution_id']}\nTimestamp: {metrics_data['timestamp']}"
        memory_hash = hashlib.sha256(unique_content.encode()).hexdigest()
        
        # Check if memory with this hash already exists
        existing_memory = db.query(MemoryItem).filter(
            MemoryItem.content_hash == memory_hash
        ).first()
        
        if existing_memory:
            # Update existing memory
            existing_memory.access_count += 1
            existing_memory.accessed_at = datetime.utcnow()
            existing_memory.updated_at = datetime.utcnow()
            # Update metadata with latest execution data
            existing_memory.meta_data = {
                "memory_type": "performance_metrics",
                "importance": existing_memory.meta_data.get("importance", 0.5),
                **metrics_data
            }
            db.commit()
            db.refresh(existing_memory)
            return existing_memory
        
        # Calculate importance based on quality score and optimization potential
        base_importance = 0.5
        quality_factor = (1 - metrics_data["quality_score"]) * 0.3
        optimization_factor = min(0.2, len(metrics_data["optimization_suggestions"]) * 0.05)
        
        importance = base_importance + quality_factor + optimization_factor
        
        # Create new memory
        memory = MemoryItem(
            content=content,
            content_hash=memory_hash,
            tags=tags,
            meta_data={
                "memory_type": "performance_metrics",
                "importance": importance,
                **metrics_data
            },
            access_count=1,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            accessed_at=datetime.utcnow()
        )
        
        try:
            db.add(memory)
            db.commit()
            db.refresh(memory)
            return memory
        except Exception as e:
            db.rollback()
            # If there's still a duplicate key error, find and update the existing record
            if "duplicate key" in str(e).lower():
                existing = db.query(MemoryItem).filter(
                    MemoryItem.content_hash == memory_hash
                ).first()
                if existing:
                    existing.access_count += 1
                    existing.accessed_at = datetime.utcnow()
                    db.commit()
                    db.refresh(existing)
                    return existing
            raise
    
    def _update_metrics_history(self, metrics: Dict[str, Any]):
        """Update local metrics history"""
        category = metrics["command_category"]
        
        if category not in self.metrics_history:
            self.metrics_history[category] = {
                "total_executions": 0,
                "successful_executions": 0,
                "total_time": 0.0,
                "patterns_detected": defaultdict(int),
                "hourly_distribution": defaultdict(int)
            }
        
        # Update statistics
        self.metrics_history[category]["total_executions"] += 1
        if metrics["success"]:
            self.metrics_history[category]["successful_executions"] += 1
        self.metrics_history[category]["total_time"] += metrics["execution_time"]
        
        # Update patterns
        for pattern in metrics["detected_patterns"]:
            self.metrics_history[category]["patterns_detected"][pattern["pattern"]] += 1
        
        # Update hourly distribution
        hour = datetime.fromisoformat(metrics["timestamp"]).hour
        self.metrics_history[category]["hourly_distribution"][str(hour)] += 1
        
        # Save to file
        with open(self.metrics_file, 'w') as f:
            # Convert defaultdict to regular dict for JSON serialization
            serializable_history = {}
            for cat, data in self.metrics_history.items():
                serializable_history[cat] = {
                    "total_executions": data["total_executions"],
                    "successful_executions": data["successful_executions"],
                    "total_time": data["total_time"],
                    "patterns_detected": dict(data["patterns_detected"]),
                    "hourly_distribution": dict(data["hourly_distribution"])
                }
            json.dump(serializable_history, f, indent=2)
    
    def _learn_from_execution(self, metrics: Dict[str, Any]):
        """Learn from command execution to improve future predictions"""
        category = metrics["command_category"]
        
        if category not in self.execution_patterns:
            self.execution_patterns[category] = {
                "successful_patterns": [],
                "failure_patterns": [],
                "optimization_results": {},
                "common_contexts": defaultdict(int)
            }
        
        # Learn from success/failure
        if metrics["success"]:
            pattern = {
                "command_type": metrics["command_type"],
                "execution_time": metrics["execution_time"],
                "quality_score": metrics["quality_score"],
                "context_hash": hashlib.md5(
                    json.dumps(metrics.get("context", {}), sort_keys=True).encode()
                ).hexdigest()[:8]
            }
            self.execution_patterns[category]["successful_patterns"].append(pattern)
        else:
            failure_pattern = {
                "command_type": metrics["command_type"],
                "error_message": metrics.get("error_message", "Unknown error"),
                "context": metrics.get("context", {})
            }
            self.execution_patterns[category]["failure_patterns"].append(failure_pattern)
        
        # Track common contexts
        if metrics.get("context"):
            context_key = json.dumps(metrics["context"], sort_keys=True)
            self.execution_patterns[category]["common_contexts"][context_key] += 1
        
        # Keep only recent patterns
        self.execution_patterns[category]["successful_patterns"] = \
            self.execution_patterns[category]["successful_patterns"][-100:]
        self.execution_patterns[category]["failure_patterns"] = \
            self.execution_patterns[category]["failure_patterns"][-50:]
        
        # Save learned patterns
        with open(self.patterns_file, 'w') as f:
            # Convert defaultdict to regular dict
            serializable_patterns = {}
            for cat, data in self.execution_patterns.items():
                serializable_patterns[cat] = {
                    "successful_patterns": data["successful_patterns"],
                    "failure_patterns": data["failure_patterns"],
                    "optimization_results": data["optimization_results"],
                    "common_contexts": dict(data["common_contexts"])
                }
            json.dump(serializable_patterns, f, indent=2)
    
    def get_performance_report(self, db: Session,
                             category: Optional[str] = None,
                             time_range: Optional[int] = None,
                             project_id: Optional[str] = None) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        
        # Query recent metrics
        query = db.query(MemoryItem).filter(
            cast(MemoryItem.meta_data, String).contains('"memory_type": "performance_metrics"')
        )
        
        if category:
            query = query.filter(
                cast(MemoryItem.meta_data, String).contains(f'"command_category": "{category}"')
            )
        
        if project_id:
            query = query.filter(
                cast(MemoryItem.meta_data, String).contains(f'"project_id": "{project_id}"')
            )
        
        if time_range:
            cutoff = datetime.utcnow() - timedelta(days=time_range)
            query = query.filter(MemoryItem.created_at >= cutoff)
        
        records = query.order_by(desc(MemoryItem.created_at)).limit(500).all()
        
        # Analyze metrics
        report = {
            "summary": {
                "total_commands": len(records),
                "successful_commands": 0,
                "average_execution_time": 0.0,
                "average_quality_score": 0.0,
                "categories": defaultdict(int)
            },
            "performance_breakdown": {},
            "common_patterns": defaultdict(int),
            "optimization_opportunities": [],
            "failure_analysis": {},
            "trends": []
        }
        
        execution_times = []
        quality_scores = []
        
        for record in records:
            meta = record.meta_data
            
            # Update summary
            if meta.get("success"):
                report["summary"]["successful_commands"] += 1
            
            execution_times.append(meta["execution_time"])
            quality_scores.append(meta["quality_score"])
            report["summary"]["categories"][meta["command_category"]] += 1
            
            # Track patterns
            for pattern in meta.get("detected_patterns", []):
                report["common_patterns"][pattern["pattern"]] += 1
            
            # Collect optimization opportunities
            for suggestion in meta.get("optimization_suggestions", []):
                if suggestion not in report["optimization_opportunities"]:
                    report["optimization_opportunities"].append(suggestion)
        
        # Calculate averages
        if execution_times:
            report["summary"]["average_execution_time"] = statistics.mean(execution_times)
            report["summary"]["average_quality_score"] = statistics.mean(quality_scores)
            report["summary"]["success_rate"] = report["summary"]["successful_commands"] / len(records)
        
        # Performance breakdown by category
        for cat, count in report["summary"]["categories"].items():
            cat_records = [r for r in records if r.meta_data["command_category"] == cat]
            cat_times = [r.meta_data["execution_time"] for r in cat_records]
            cat_success = sum(1 for r in cat_records if r.meta_data["success"])
            
            report["performance_breakdown"][cat] = {
                "count": count,
                "average_time": statistics.mean(cat_times) if cat_times else 0,
                "success_rate": cat_success / count if count > 0 else 0,
                "slowest_command": max(cat_times) if cat_times else 0,
                "fastest_command": min(cat_times) if cat_times else 0
            }
        
        # Analyze trends
        if len(records) > 10:
            # Recent vs older performance
            recent_records = records[:len(records)//2]
            older_records = records[len(records)//2:]
            
            recent_avg = statistics.mean([r.meta_data["execution_time"] for r in recent_records])
            older_avg = statistics.mean([r.meta_data["execution_time"] for r in older_records])
            
            trend = "improving" if recent_avg < older_avg else "degrading" if recent_avg > older_avg else "stable"
            
            report["trends"].append({
                "metric": "execution_time",
                "trend": trend,
                "change_percentage": ((recent_avg - older_avg) / older_avg * 100) if older_avg > 0 else 0
            })
        
        # Sort optimization opportunities by expected improvement
        report["optimization_opportunities"].sort(
            key=lambda x: x.get("expected_improvement", 0),
            reverse=True
        )
        
        return report
    
    def predict_performance(self, db: Session, command_type: str,
                          command_details: Dict[str, Any],
                          context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Predict performance for a command before execution"""
        
        category = self._categorize_command(command_type)
        
        # Find similar past executions
        similar_executions = self._find_similar_executions(
            db, command_type, command_details, context
        )
        
        prediction = {
            "command_type": command_type,
            "category": category,
            "predicted_execution_time": 0.0,
            "predicted_success_rate": 0.0,
            "confidence": 0.0,
            "based_on_samples": len(similar_executions),
            "risk_factors": [],
            "optimization_suggestions": []
        }
        
        if not similar_executions:
            # Use category average if no similar executions
            if category in self.metrics_history:
                cat_data = self.metrics_history[category]
                if cat_data["total_executions"] > 0:
                    prediction["predicted_execution_time"] = \
                        cat_data["total_time"] / cat_data["total_executions"]
                    prediction["predicted_success_rate"] = \
                        cat_data["successful_executions"] / cat_data["total_executions"]
                    prediction["confidence"] = 0.3
            else:
                # No data available
                prediction["predicted_execution_time"] = 5.0  # Default guess
                prediction["predicted_success_rate"] = 0.8   # Default guess
                prediction["confidence"] = 0.1
        else:
            # Calculate predictions from similar executions
            exec_times = [e["execution_time"] for e in similar_executions]
            success_count = sum(1 for e in similar_executions if e["success"])
            
            prediction["predicted_execution_time"] = statistics.median(exec_times)
            prediction["predicted_success_rate"] = success_count / len(similar_executions)
            prediction["confidence"] = min(0.9, len(similar_executions) / 10)
            
            # Identify risk factors
            failures = [e for e in similar_executions if not e["success"]]
            if failures:
                common_errors = defaultdict(int)
                for failure in failures:
                    if failure.get("error_message"):
                        common_errors[failure["error_message"]] += 1
                
                for error, count in common_errors.items():
                    if count >= 2:
                        prediction["risk_factors"].append({
                            "risk": "recurring_error",
                            "description": error,
                            "frequency": count / len(similar_executions)
                        })
            
            # Check for performance issues
            slow_executions = [e for e in similar_executions 
                             if e["performance_analysis"]["performance_rating"] in ["slow", "very_slow"]]
            if len(slow_executions) / len(similar_executions) > 0.3:
                prediction["risk_factors"].append({
                    "risk": "performance_issue",
                    "description": "Command often runs slowly",
                    "frequency": len(slow_executions) / len(similar_executions)
                })
        
        # Add optimization suggestions if predicted to be slow
        if prediction["predicted_execution_time"] > self.performance_thresholds["slow"]:
            prediction["optimization_suggestions"] = self._get_optimization_suggestions(
                command_type, prediction["predicted_execution_time"],
                {"performance_rating": "slow"}, []
            )
        
        return prediction
    
    def _find_similar_executions(self, db: Session, command_type: str,
                                command_details: Dict[str, Any],
                                context: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find similar past command executions"""
        
        # Query recent executions of same type
        query = db.query(MemoryItem).filter(
            and_(
                cast(MemoryItem.meta_data, String).contains('"memory_type": "performance_metrics"'),
                cast(MemoryItem.meta_data, String).contains(f'"command_type": "{command_type}"')
            )
        ).order_by(desc(MemoryItem.created_at)).limit(50)
        
        records = query.all()
        
        similar = []
        for record in records:
            meta = record.meta_data
            
            # Calculate similarity score
            similarity = self._calculate_similarity(
                command_details, meta.get("command_details", {}),
                context, meta.get("context", {})
            )
            
            if similarity > 0.7:  # Threshold for similarity
                similar.append(meta)
        
        return similar
    
    def _calculate_similarity(self, details1: Dict[str, Any], details2: Dict[str, Any],
                            context1: Optional[Dict[str, Any]], 
                            context2: Optional[Dict[str, Any]]) -> float:
        """Calculate similarity between two command executions"""
        
        # Simple similarity based on matching keys and values
        score = 0.0
        total_keys = set(details1.keys()) | set(details2.keys())
        
        if not total_keys:
            return 0.5  # No details to compare
        
        # Compare details
        matching_keys = set(details1.keys()) & set(details2.keys())
        score += len(matching_keys) / len(total_keys) * 0.5
        
        # Compare values for matching keys
        value_matches = 0
        for key in matching_keys:
            if details1[key] == details2[key]:
                value_matches += 1
        
        if matching_keys:
            score += (value_matches / len(matching_keys)) * 0.3
        
        # Compare context if available
        if context1 and context2:
            context_keys = set(context1.keys()) | set(context2.keys())
            matching_context = set(context1.keys()) & set(context2.keys())
            if context_keys:
                score += (len(matching_context) / len(context_keys)) * 0.2
        
        return score
    
    def get_optimization_history(self, db: Session,
                               strategy: Optional[str] = None,
                               project_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get history of optimization attempts and their results"""
        
        # This would track actual optimization implementations
        # For now, return theoretical optimizations based on suggestions
        
        query = db.query(MemoryItem).filter(
            and_(
                cast(MemoryItem.meta_data, String).contains('"memory_type": "performance_metrics"'),
                cast(MemoryItem.meta_data, String).contains('"optimization_suggestions"')
            )
        )
        
        if project_id:
            query = query.filter(
                cast(MemoryItem.meta_data, String).contains(f'"project_id": "{project_id}"')
            )
        
        records = query.order_by(desc(MemoryItem.created_at)).limit(100).all()
        
        optimizations = []
        for record in records:
            meta = record.meta_data
            for suggestion in meta.get("optimization_suggestions", []):
                if not strategy or suggestion["strategy"] == strategy:
                    optimizations.append({
                        "execution_id": meta["execution_id"],
                        "command_type": meta["command_type"],
                        "strategy": suggestion["strategy"],
                        "description": suggestion["description"],
                        "expected_improvement": suggestion["expected_improvement"],
                        "current_time": meta["execution_time"],
                        "expected_time": suggestion["expected_time"],
                        "timestamp": meta["timestamp"]
                    })
        
        return optimizations
    
    def analyze_command_patterns(self, db: Session,
                               time_range: int = 7,
                               min_frequency: int = 3) -> Dict[str, Any]:
        """Analyze command execution patterns over time"""
        
        cutoff = datetime.utcnow() - timedelta(days=time_range)
        
        query = db.query(MemoryItem).filter(
            and_(
                cast(MemoryItem.meta_data, String).contains('"memory_type": "performance_metrics"'),
                MemoryItem.created_at >= cutoff
            )
        ).order_by(desc(MemoryItem.created_at)).limit(1000)
        
        records = query.all()
        
        analysis = {
            "command_frequency": defaultdict(int),
            "command_sequences": defaultdict(int),
            "time_patterns": defaultdict(lambda: defaultdict(int)),
            "performance_patterns": defaultdict(list),
            "failure_patterns": defaultdict(int),
            "optimization_candidates": []
        }
        
        # Analyze command frequency
        for i, record in enumerate(records):
            meta = record.meta_data
            cmd_type = meta["command_type"]
            analysis["command_frequency"][cmd_type] += 1
            
            # Track time patterns
            hour = datetime.fromisoformat(meta["timestamp"]).hour
            analysis["time_patterns"][cmd_type][hour] += 1
            
            # Track performance
            analysis["performance_patterns"][cmd_type].append(meta["execution_time"])
            
            # Track failures
            if not meta["success"]:
                analysis["failure_patterns"][cmd_type] += 1
            
            # Analyze sequences (pairs of commands)
            if i < len(records) - 1:
                next_cmd = records[i + 1].meta_data["command_type"]
                sequence = f"{cmd_type} -> {next_cmd}"
                analysis["command_sequences"][sequence] += 1
        
        # Identify optimization candidates
        for cmd_type, frequency in analysis["command_frequency"].items():
            if frequency >= min_frequency:
                avg_time = statistics.mean(analysis["performance_patterns"][cmd_type])
                failure_rate = analysis["failure_patterns"][cmd_type] / frequency
                
                if avg_time > 5.0 or failure_rate > 0.2:
                    analysis["optimization_candidates"].append({
                        "command_type": cmd_type,
                        "frequency": frequency,
                        "average_time": avg_time,
                        "failure_rate": failure_rate,
                        "reason": "high_execution_time" if avg_time > 5.0 else "high_failure_rate"
                    })
        
        # Convert defaultdicts to regular dicts for JSON serialization
        analysis["command_frequency"] = dict(analysis["command_frequency"])
        analysis["command_sequences"] = dict(analysis["command_sequences"])
        analysis["time_patterns"] = {k: dict(v) for k, v in analysis["time_patterns"].items()}
        analysis["performance_patterns"] = {k: v for k, v in analysis["performance_patterns"].items()}
        analysis["failure_patterns"] = dict(analysis["failure_patterns"])
        
        # Sort optimization candidates by frequency
        analysis["optimization_candidates"].sort(key=lambda x: x["frequency"], reverse=True)
        
        return analysis