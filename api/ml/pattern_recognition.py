"""
Advanced Pattern Recognition for Code and Workflow Analysis.

This module implements sophisticated pattern recognition algorithms:
- Code pattern mining and analysis
- Usage pattern detection
- Performance pattern identification
- Error pattern clustering
- Optimization opportunity detection
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import re
import ast
from collections import defaultdict, Counter
from sklearn.cluster import DBSCAN, KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from sqlalchemy.orm import Session as DBSession
from sqlalchemy import func, and_, or_

from ..models.base import get_db_context
from ..models.memory import MemoryItem
from ..models.session import Session
from ..models.error_pattern import ErrorOccurrence
from ..models.enhanced_decision import EnhancedDecision
from shared.logging import setup_logging

logger = setup_logging("pattern_recognition")


class PatternType(str, Enum):
    """Types of patterns that can be recognized."""
    CODE_STRUCTURE = "code_structure"
    USAGE_BEHAVIOR = "usage_behavior"
    PERFORMANCE_BOTTLENECK = "performance_bottleneck"
    ERROR_SEQUENCE = "error_sequence"
    OPTIMIZATION_OPPORTUNITY = "optimization_opportunity"
    WORKFLOW_SEQUENCE = "workflow_sequence"
    DECISION_PATTERN = "decision_pattern"


class PatternCategory(str, Enum):
    """Categories of patterns."""
    FUNCTIONAL = "functional"           # Related to functionality
    STRUCTURAL = "structural"           # Related to code structure
    BEHAVIORAL = "behavioral"           # Related to user behavior
    PERFORMANCE = "performance"         # Related to performance
    QUALITY = "quality"                # Related to code quality


@dataclass
class PatternMatch:
    """A detected pattern match."""
    pattern_type: PatternType
    category: PatternCategory
    confidence: float
    description: str
    examples: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    improvement_suggestion: Optional[str] = None
    impact_level: str = "medium"  # low, medium, high


@dataclass
class CodeMetrics:
    """Code quality and complexity metrics."""
    cyclomatic_complexity: int
    lines_of_code: int
    function_count: int
    class_count: int
    dependency_count: int
    duplicate_lines: int
    test_coverage: float
    maintainability_index: float


class PatternRecognitionEngine:
    """
    Advanced pattern recognition engine for code and workflow analysis.
    
    Features:
    - Multi-dimensional pattern detection
    - Adaptive clustering algorithms
    - Performance bottleneck identification
    - Code quality assessment
    - Optimization opportunity detection
    """
    
    def __init__(self):
        self.vectorizers = {}
        self.clustering_models = {}
        self.pattern_cache = {}
        
        # Pattern detection thresholds
        self.thresholds = {
            "min_pattern_occurrences": 3,
            "confidence_threshold": 0.7,
            "similarity_threshold": 0.8,
            "cluster_min_samples": 5,
            "complexity_threshold": 10
        }
        
        logger.info("Initialized PatternRecognitionEngine")
    
    async def analyze_code_patterns(
        self,
        code_samples: List[Dict[str, Any]],
        user_id: Optional[str] = None,
        project_id: Optional[str] = None
    ) -> List[PatternMatch]:
        """
        Analyze code patterns from samples.
        
        Args:
            code_samples: List of code samples with metadata
            user_id: Optional user filter
            project_id: Optional project filter
            
        Returns:
            List of detected code patterns
        """
        try:
            patterns = []
            
            if not code_samples:
                return patterns
            
            # Extract code content
            code_texts = [sample.get("content", "") for sample in code_samples]
            
            # Detect structural patterns
            structural_patterns = self._detect_structural_patterns(code_texts, code_samples)
            patterns.extend(structural_patterns)
            
            # Detect functional patterns
            functional_patterns = self._detect_functional_patterns(code_texts, code_samples)
            patterns.extend(functional_patterns)
            
            # Detect anti-patterns
            anti_patterns = self._detect_antipatterns(code_texts, code_samples)
            patterns.extend(anti_patterns)
            
            # Calculate code metrics for each sample
            for i, sample in enumerate(code_samples):
                if i < len(code_texts):
                    metrics = self._calculate_code_metrics(code_texts[i])
                    sample["metrics"] = metrics
            
            # Detect complexity patterns
            complexity_patterns = self._detect_complexity_patterns(code_samples)
            patterns.extend(complexity_patterns)
            
            logger.info(f"Detected {len(patterns)} code patterns")
            return patterns
            
        except Exception as e:
            logger.error(f"Failed to analyze code patterns: {e}")
            return []
    
    async def analyze_usage_patterns(
        self,
        activities: List[Dict[str, Any]],
        time_window_hours: int = 168
    ) -> List[PatternMatch]:
        """
        Analyze user behavior and usage patterns.
        
        Args:
            activities: List of user activities
            time_window_hours: Time window for analysis
            
        Returns:
            List of detected usage patterns
        """
        try:
            patterns = []
            
            if not activities:
                return patterns
            
            # Group activities by user
            user_activities = defaultdict(list)
            for activity in activities:
                user_id = activity.get("user_id", "unknown")
                user_activities[user_id].append(activity)
            
            # Detect session patterns
            session_patterns = self._detect_session_patterns(user_activities)
            patterns.extend(session_patterns)
            
            # Detect feature usage patterns
            feature_patterns = self._detect_feature_usage_patterns(user_activities)
            patterns.extend(feature_patterns)
            
            # Detect temporal patterns
            temporal_patterns = self._detect_temporal_patterns(activities)
            patterns.extend(temporal_patterns)
            
            # Detect collaboration patterns
            collaboration_patterns = self._detect_collaboration_patterns(activities)
            patterns.extend(collaboration_patterns)
            
            logger.info(f"Detected {len(patterns)} usage patterns")
            return patterns
            
        except Exception as e:
            logger.error(f"Failed to analyze usage patterns: {e}")
            return []
    
    async def analyze_performance_patterns(
        self,
        performance_data: List[Dict[str, Any]]
    ) -> List[PatternMatch]:
        """
        Analyze performance data to identify bottlenecks and patterns.
        
        Args:
            performance_data: List of performance measurements
            
        Returns:
            List of detected performance patterns
        """
        try:
            patterns = []
            
            if not performance_data:
                return patterns
            
            # Detect slow operations
            slow_patterns = self._detect_slow_operation_patterns(performance_data)
            patterns.extend(slow_patterns)
            
            # Detect resource usage patterns
            resource_patterns = self._detect_resource_usage_patterns(performance_data)
            patterns.extend(resource_patterns)
            
            # Detect performance regressions
            regression_patterns = self._detect_performance_regressions(performance_data)
            patterns.extend(regression_patterns)
            
            # Detect optimization opportunities
            optimization_patterns = self._detect_optimization_opportunities(performance_data)
            patterns.extend(optimization_patterns)
            
            logger.info(f"Detected {len(patterns)} performance patterns")
            return patterns
            
        except Exception as e:
            logger.error(f"Failed to analyze performance patterns: {e}")
            return []
    
    async def analyze_error_patterns(
        self,
        errors: List[Dict[str, Any]]
    ) -> List[PatternMatch]:
        """
        Analyze error patterns and sequences.
        
        Args:
            errors: List of error occurrences
            
        Returns:
            List of detected error patterns
        """
        try:
            patterns = []
            
            if not errors:
                return patterns
            
            # Group errors by type and context
            error_groups = self._group_errors_by_similarity(errors)
            
            # Detect recurring error patterns
            recurring_patterns = self._detect_recurring_error_patterns(error_groups)
            patterns.extend(recurring_patterns)
            
            # Detect error cascades
            cascade_patterns = self._detect_error_cascade_patterns(errors)
            patterns.extend(cascade_patterns)
            
            # Detect error hotspots
            hotspot_patterns = self._detect_error_hotspots(errors)
            patterns.extend(hotspot_patterns)
            
            logger.info(f"Detected {len(patterns)} error patterns")
            return patterns
            
        except Exception as e:
            logger.error(f"Failed to analyze error patterns: {e}")
            return []
    
    async def suggest_optimizations(
        self,
        patterns: List[PatternMatch]
    ) -> List[Dict[str, Any]]:
        """
        Generate optimization suggestions based on detected patterns.
        
        Args:
            patterns: List of detected patterns
            
        Returns:
            List of optimization suggestions
        """
        try:
            suggestions = []
            
            # Group patterns by category
            pattern_groups = defaultdict(list)
            for pattern in patterns:
                pattern_groups[pattern.category].append(pattern)
            
            # Generate category-specific suggestions
            for category, category_patterns in pattern_groups.items():
                category_suggestions = self._generate_category_suggestions(
                    category, category_patterns
                )
                suggestions.extend(category_suggestions)
            
            # Prioritize suggestions by impact
            suggestions.sort(key=lambda x: x.get("impact_score", 0), reverse=True)
            
            logger.info(f"Generated {len(suggestions)} optimization suggestions")
            return suggestions
            
        except Exception as e:
            logger.error(f"Failed to suggest optimizations: {e}")
            return []
    
    # Code pattern detection methods
    
    def _detect_structural_patterns(
        self,
        code_texts: List[str],
        code_samples: List[Dict[str, Any]]
    ) -> List[PatternMatch]:
        """Detect structural code patterns."""
        patterns = []
        
        try:
            # Detect class hierarchies
            class_patterns = self._detect_class_hierarchy_patterns(code_texts)
            patterns.extend(class_patterns)
            
            # Detect function patterns
            function_patterns = self._detect_function_patterns(code_texts)
            patterns.extend(function_patterns)
            
            # Detect import patterns
            import_patterns = self._detect_import_patterns(code_texts)
            patterns.extend(import_patterns)
            
            # Detect design patterns
            design_patterns = self._detect_design_patterns(code_texts)
            patterns.extend(design_patterns)
            
            return patterns
            
        except Exception as e:
            logger.error(f"Failed to detect structural patterns: {e}")
            return []
    
    def _detect_functional_patterns(
        self,
        code_texts: List[str],
        code_samples: List[Dict[str, Any]]
    ) -> List[PatternMatch]:
        """Detect functional code patterns."""
        patterns = []
        
        try:
            # Use TF-IDF to find similar code blocks
            if len(code_texts) < 2:
                return patterns
            
            # Vectorize code texts
            vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 3)
            )
            
            tfidf_matrix = vectorizer.fit_transform(code_texts)
            
            # Cluster similar code
            clustering = DBSCAN(eps=0.3, min_samples=2)
            clusters = clustering.fit_predict(tfidf_matrix.toarray())
            
            # Analyze clusters for patterns
            cluster_groups = defaultdict(list)
            for i, cluster_id in enumerate(clusters):
                if cluster_id != -1:  # Not noise
                    cluster_groups[cluster_id].append(i)
            
            for cluster_id, indices in cluster_groups.items():
                if len(indices) >= self.thresholds["min_pattern_occurrences"]:
                    # Extract common patterns from cluster
                    cluster_samples = [code_samples[i] for i in indices]
                    cluster_texts = [code_texts[i] for i in indices]
                    
                    pattern = self._analyze_code_cluster(
                        cluster_texts, cluster_samples
                    )
                    if pattern:
                        patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            logger.error(f"Failed to detect functional patterns: {e}")
            return []
    
    def _detect_antipatterns(
        self,
        code_texts: List[str],
        code_samples: List[Dict[str, Any]]
    ) -> List[PatternMatch]:
        """Detect code anti-patterns."""
        patterns = []
        
        try:
            for i, code_text in enumerate(code_texts):
                sample = code_samples[i] if i < len(code_samples) else {}
                
                # Detect long functions
                if self._has_long_functions(code_text):
                    patterns.append(PatternMatch(
                        pattern_type=PatternType.CODE_STRUCTURE,
                        category=PatternCategory.QUALITY,
                        confidence=0.9,
                        description="Long function detected",
                        examples=[{"code_sample": sample, "issue": "Function too long"}],
                        metadata={"file": sample.get("file", "unknown")},
                        improvement_suggestion="Break function into smaller, focused functions",
                        impact_level="medium"
                    ))
                
                # Detect deep nesting
                if self._has_deep_nesting(code_text):
                    patterns.append(PatternMatch(
                        pattern_type=PatternType.CODE_STRUCTURE,
                        category=PatternCategory.QUALITY,
                        confidence=0.8,
                        description="Deep nesting detected",
                        examples=[{"code_sample": sample, "issue": "Excessive nesting"}],
                        metadata={"file": sample.get("file", "unknown")},
                        improvement_suggestion="Refactor to reduce nesting levels",
                        impact_level="medium"
                    ))
                
                # Detect code duplication
                duplicates = self._find_code_duplicates(code_text, code_texts)
                if duplicates:
                    patterns.append(PatternMatch(
                        pattern_type=PatternType.CODE_STRUCTURE,
                        category=PatternCategory.QUALITY,
                        confidence=0.85,
                        description="Code duplication detected",
                        examples=[{"code_sample": sample, "duplicates": duplicates}],
                        metadata={"file": sample.get("file", "unknown")},
                        improvement_suggestion="Extract common code into reusable functions",
                        impact_level="high"
                    ))
            
            return patterns
            
        except Exception as e:
            logger.error(f"Failed to detect anti-patterns: {e}")
            return []
    
    def _detect_complexity_patterns(
        self,
        code_samples: List[Dict[str, Any]]
    ) -> List[PatternMatch]:
        """Detect complexity-related patterns."""
        patterns = []
        
        try:
            high_complexity_samples = []
            
            for sample in code_samples:
                metrics = sample.get("metrics")
                if not metrics:
                    continue
                
                # Check for high complexity
                if metrics.cyclomatic_complexity > self.thresholds["complexity_threshold"]:
                    high_complexity_samples.append(sample)
            
            if len(high_complexity_samples) >= self.thresholds["min_pattern_occurrences"]:
                patterns.append(PatternMatch(
                    pattern_type=PatternType.CODE_STRUCTURE,
                    category=PatternCategory.QUALITY,
                    confidence=0.9,
                    description="High complexity pattern detected",
                    examples=high_complexity_samples,
                    metadata={
                        "avg_complexity": np.mean([
                            s["metrics"].cyclomatic_complexity 
                            for s in high_complexity_samples
                        ]),
                        "count": len(high_complexity_samples)
                    },
                    improvement_suggestion="Refactor complex functions to improve maintainability",
                    impact_level="high"
                ))
            
            return patterns
            
        except Exception as e:
            logger.error(f"Failed to detect complexity patterns: {e}")
            return []
    
    # Usage pattern detection methods
    
    def _detect_session_patterns(
        self,
        user_activities: Dict[str, List[Dict[str, Any]]]
    ) -> List[PatternMatch]:
        """Detect user session patterns."""
        patterns = []
        
        try:
            session_durations = []
            session_frequencies = defaultdict(int)
            
            for user_id, activities in user_activities.items():
                # Group activities by session
                sessions = self._group_activities_by_session(activities)
                
                for session in sessions:
                    if len(session) > 1:
                        duration = (
                            session[-1]["timestamp"] - session[0]["timestamp"]
                        ).total_seconds() / 3600  # hours
                        session_durations.append(duration)
                        
                        # Count session frequency by hour
                        hour = session[0]["timestamp"].hour
                        session_frequencies[hour] += 1
            
            if session_durations:
                avg_duration = np.mean(session_durations)
                
                # Detect long sessions pattern
                if avg_duration > 4:  # More than 4 hours
                    patterns.append(PatternMatch(
                        pattern_type=PatternType.USAGE_BEHAVIOR,
                        category=PatternCategory.BEHAVIORAL,
                        confidence=0.8,
                        description="Long session pattern detected",
                        examples=[{"avg_duration_hours": avg_duration}],
                        metadata={"session_count": len(session_durations)},
                        improvement_suggestion="Consider implementing session breaks or reminders",
                        impact_level="medium"
                    ))
                
                # Detect peak usage hours
                if session_frequencies:
                    peak_hour = max(session_frequencies, key=session_frequencies.get)
                    peak_count = session_frequencies[peak_hour]
                    
                    if peak_count >= 5:  # Significant peak
                        patterns.append(PatternMatch(
                            pattern_type=PatternType.USAGE_BEHAVIOR,
                            category=PatternCategory.BEHAVIORAL,
                            confidence=0.7,
                            description="Peak usage time pattern detected",
                            examples=[{"peak_hour": peak_hour, "session_count": peak_count}],
                            metadata={"usage_distribution": dict(session_frequencies)},
                            improvement_suggestion="Optimize system resources for peak hours",
                            impact_level="low"
                        ))
            
            return patterns
            
        except Exception as e:
            logger.error(f"Failed to detect session patterns: {e}")
            return []
    
    def _detect_feature_usage_patterns(
        self,
        user_activities: Dict[str, List[Dict[str, Any]]]
    ) -> List[PatternMatch]:
        """Detect feature usage patterns."""
        patterns = []
        
        try:
            feature_usage = Counter()
            feature_sequences = []
            
            for user_id, activities in user_activities.items():
                # Count feature usage
                for activity in activities:
                    feature = activity.get("feature", "unknown")
                    feature_usage[feature] += 1
                
                # Extract feature sequences
                sequence = [a.get("feature", "unknown") for a in activities]
                if len(sequence) > 2:
                    feature_sequences.append(sequence)
            
            # Detect underused features
            if feature_usage:
                total_usage = sum(feature_usage.values())
                underused_features = [
                    feature for feature, count in feature_usage.items()
                    if count / total_usage < 0.05  # Less than 5% usage
                ]
                
                if underused_features:
                    patterns.append(PatternMatch(
                        pattern_type=PatternType.USAGE_BEHAVIOR,
                        category=PatternCategory.BEHAVIORAL,
                        confidence=0.8,
                        description="Underused features detected",
                        examples=[{"features": underused_features}],
                        metadata={"feature_usage": dict(feature_usage)},
                        improvement_suggestion="Consider improving discoverability of underused features",
                        impact_level="medium"
                    ))
                
                # Detect most popular features
                top_features = feature_usage.most_common(3)
                if top_features:
                    patterns.append(PatternMatch(
                        pattern_type=PatternType.USAGE_BEHAVIOR,
                        category=PatternCategory.BEHAVIORAL,
                        confidence=0.9,
                        description="Popular features pattern detected",
                        examples=[{"top_features": top_features}],
                        metadata={"feature_usage": dict(feature_usage)},
                        improvement_suggestion="Focus optimization efforts on popular features",
                        impact_level="high"
                    ))
            
            return patterns
            
        except Exception as e:
            logger.error(f"Failed to detect feature usage patterns: {e}")
            return []
    
    def _detect_temporal_patterns(
        self,
        activities: List[Dict[str, Any]]
    ) -> List[PatternMatch]:
        """Detect temporal usage patterns."""
        patterns = []
        
        try:
            if not activities:
                return patterns
            
            # Extract timestamps
            timestamps = [a.get("timestamp") for a in activities if a.get("timestamp")]
            
            if len(timestamps) < 10:
                return patterns
            
            # Analyze usage by hour
            hourly_usage = defaultdict(int)
            daily_usage = defaultdict(int)
            
            for timestamp in timestamps:
                hourly_usage[timestamp.hour] += 1
                daily_usage[timestamp.weekday()] += 1
            
            # Detect night usage pattern
            night_hours = [22, 23, 0, 1, 2, 3, 4, 5]
            night_usage = sum(hourly_usage[hour] for hour in night_hours)
            total_usage = sum(hourly_usage.values())
            
            if night_usage / total_usage > 0.3:  # More than 30% night usage
                patterns.append(PatternMatch(
                    pattern_type=PatternType.USAGE_BEHAVIOR,
                    category=PatternCategory.BEHAVIORAL,
                    confidence=0.8,
                    description="Night usage pattern detected",
                    examples=[{"night_usage_percentage": night_usage / total_usage}],
                    metadata={"hourly_distribution": dict(hourly_usage)},
                    improvement_suggestion="Consider work-life balance implications",
                    impact_level="medium"
                ))
            
            # Detect weekend usage pattern
            weekend_usage = daily_usage[5] + daily_usage[6]  # Saturday + Sunday
            weekday_usage = sum(daily_usage[i] for i in range(5))
            
            if weekend_usage > 0 and weekend_usage / (weekend_usage + weekday_usage) > 0.25:
                patterns.append(PatternMatch(
                    pattern_type=PatternType.USAGE_BEHAVIOR,
                    category=PatternCategory.BEHAVIORAL,
                    confidence=0.7,
                    description="Weekend usage pattern detected",
                    examples=[{"weekend_usage_percentage": weekend_usage / total_usage}],
                    metadata={"daily_distribution": dict(daily_usage)},
                    improvement_suggestion="Monitor weekend work patterns",
                    impact_level="low"
                ))
            
            return patterns
            
        except Exception as e:
            logger.error(f"Failed to detect temporal patterns: {e}")
            return []
    
    def _detect_collaboration_patterns(
        self,
        activities: List[Dict[str, Any]]
    ) -> List[PatternMatch]:
        """Detect collaboration patterns."""
        patterns = []
        
        try:
            # Group activities by project
            project_users = defaultdict(set)
            project_activities = defaultdict(list)
            
            for activity in activities:
                project_id = activity.get("project_id")
                user_id = activity.get("user_id")
                
                if project_id and user_id:
                    project_users[project_id].add(user_id)
                    project_activities[project_id].append(activity)
            
            # Detect multi-user projects
            collaborative_projects = {
                project_id: users for project_id, users in project_users.items()
                if len(users) > 1
            }
            
            if collaborative_projects:
                patterns.append(PatternMatch(
                    pattern_type=PatternType.USAGE_BEHAVIOR,
                    category=PatternCategory.BEHAVIORAL,
                    confidence=0.9,
                    description="Collaboration pattern detected",
                    examples=[{
                        "collaborative_projects": len(collaborative_projects),
                        "avg_users_per_project": np.mean([
                            len(users) for users in collaborative_projects.values()
                        ])
                    }],
                    metadata={"project_details": {
                        pid: len(users) for pid, users in collaborative_projects.items()
                    }},
                    improvement_suggestion="Consider collaboration tools and features",
                    impact_level="medium"
                ))
            
            return patterns
            
        except Exception as e:
            logger.error(f"Failed to detect collaboration patterns: {e}")
            return []
    
    # Performance pattern detection methods
    
    def _detect_slow_operation_patterns(
        self,
        performance_data: List[Dict[str, Any]]
    ) -> List[PatternMatch]:
        """Detect slow operation patterns."""
        patterns = []
        
        try:
            # Group by operation type
            operation_times = defaultdict(list)
            
            for data_point in performance_data:
                operation = data_point.get("operation", "unknown")
                duration = data_point.get("duration_ms", 0)
                operation_times[operation].append(duration)
            
            # Detect consistently slow operations
            slow_operations = []
            for operation, durations in operation_times.items():
                if len(durations) >= 5:  # Need enough samples
                    avg_duration = np.mean(durations)
                    p95_duration = np.percentile(durations, 95)
                    
                    if avg_duration > 1000:  # More than 1 second average
                        slow_operations.append({
                            "operation": operation,
                            "avg_duration_ms": avg_duration,
                            "p95_duration_ms": p95_duration,
                            "sample_count": len(durations)
                        })
            
            if slow_operations:
                patterns.append(PatternMatch(
                    pattern_type=PatternType.PERFORMANCE_BOTTLENECK,
                    category=PatternCategory.PERFORMANCE,
                    confidence=0.9,
                    description="Slow operation pattern detected",
                    examples=slow_operations,
                    metadata={"operation_count": len(slow_operations)},
                    improvement_suggestion="Optimize slow operations through caching or algorithmic improvements",
                    impact_level="high"
                ))
            
            return patterns
            
        except Exception as e:
            logger.error(f"Failed to detect slow operation patterns: {e}")
            return []
    
    def _detect_resource_usage_patterns(
        self,
        performance_data: List[Dict[str, Any]]
    ) -> List[PatternMatch]:
        """Detect resource usage patterns."""
        patterns = []
        
        try:
            cpu_usage = [d.get("cpu_usage", 0) for d in performance_data if "cpu_usage" in d]
            memory_usage = [d.get("memory_usage", 0) for d in performance_data if "memory_usage" in d]
            
            # Detect high CPU usage pattern
            if cpu_usage and np.mean(cpu_usage) > 0.8:  # More than 80%
                patterns.append(PatternMatch(
                    pattern_type=PatternType.PERFORMANCE_BOTTLENECK,
                    category=PatternCategory.PERFORMANCE,
                    confidence=0.8,
                    description="High CPU usage pattern detected",
                    examples=[{
                        "avg_cpu_usage": np.mean(cpu_usage),
                        "max_cpu_usage": np.max(cpu_usage),
                        "sample_count": len(cpu_usage)
                    }],
                    metadata={"cpu_stats": {"mean": np.mean(cpu_usage), "std": np.std(cpu_usage)}},
                    improvement_suggestion="Investigate CPU-intensive operations and optimize algorithms",
                    impact_level="high"
                ))
            
            # Detect high memory usage pattern
            if memory_usage and np.mean(memory_usage) > 0.8:  # More than 80%
                patterns.append(PatternMatch(
                    pattern_type=PatternType.PERFORMANCE_BOTTLENECK,
                    category=PatternCategory.PERFORMANCE,
                    confidence=0.8,
                    description="High memory usage pattern detected",
                    examples=[{
                        "avg_memory_usage": np.mean(memory_usage),
                        "max_memory_usage": np.max(memory_usage),
                        "sample_count": len(memory_usage)
                    }],
                    metadata={"memory_stats": {"mean": np.mean(memory_usage), "std": np.std(memory_usage)}},
                    improvement_suggestion="Review memory usage and implement memory optimization strategies",
                    impact_level="high"
                ))
            
            return patterns
            
        except Exception as e:
            logger.error(f"Failed to detect resource usage patterns: {e}")
            return []
    
    def _detect_performance_regressions(
        self,
        performance_data: List[Dict[str, Any]]
    ) -> List[PatternMatch]:
        """Detect performance regression patterns."""
        patterns = []
        
        try:
            # Sort by timestamp
            sorted_data = sorted(
                [d for d in performance_data if "timestamp" in d and "duration_ms" in d],
                key=lambda x: x["timestamp"]
            )
            
            if len(sorted_data) < 10:
                return patterns
            
            # Split into recent and historical data
            split_point = len(sorted_data) // 2
            historical = sorted_data[:split_point]
            recent = sorted_data[split_point:]
            
            historical_durations = [d["duration_ms"] for d in historical]
            recent_durations = [d["duration_ms"] for d in recent]
            
            historical_avg = np.mean(historical_durations)
            recent_avg = np.mean(recent_durations)
            
            # Detect significant regression (>20% increase)
            if recent_avg > historical_avg * 1.2:
                regression_percentage = ((recent_avg - historical_avg) / historical_avg) * 100
                
                patterns.append(PatternMatch(
                    pattern_type=PatternType.PERFORMANCE_BOTTLENECK,
                    category=PatternCategory.PERFORMANCE,
                    confidence=0.8,
                    description="Performance regression detected",
                    examples=[{
                        "historical_avg_ms": historical_avg,
                        "recent_avg_ms": recent_avg,
                        "regression_percentage": regression_percentage
                    }],
                    metadata={
                        "historical_samples": len(historical),
                        "recent_samples": len(recent)
                    },
                    improvement_suggestion="Investigate recent changes that may have caused performance regression",
                    impact_level="high"
                ))
            
            return patterns
            
        except Exception as e:
            logger.error(f"Failed to detect performance regressions: {e}")
            return []
    
    def _detect_optimization_opportunities(
        self,
        performance_data: List[Dict[str, Any]]
    ) -> List[PatternMatch]:
        """Detect optimization opportunities."""
        patterns = []
        
        try:
            # Group by endpoint/operation
            operation_groups = defaultdict(list)
            
            for data_point in performance_data:
                operation = data_point.get("operation", "unknown")
                operation_groups[operation].append(data_point)
            
            # Analyze each operation group
            optimization_candidates = []
            
            for operation, data_points in operation_groups.items():
                if len(data_points) < 5:
                    continue
                
                durations = [d.get("duration_ms", 0) for d in data_points]
                call_count = len(data_points)
                avg_duration = np.mean(durations)
                
                # Calculate optimization potential
                total_time = sum(durations)
                potential_savings = total_time * 0.3  # Assume 30% improvement possible
                
                if avg_duration > 500 and call_count > 10:  # Frequent, slow operation
                    optimization_candidates.append({
                        "operation": operation,
                        "avg_duration_ms": avg_duration,
                        "call_count": call_count,
                        "total_time_ms": total_time,
                        "potential_savings_ms": potential_savings
                    })
            
            if optimization_candidates:
                # Sort by potential impact
                optimization_candidates.sort(
                    key=lambda x: x["potential_savings_ms"], 
                    reverse=True
                )
                
                patterns.append(PatternMatch(
                    pattern_type=PatternType.OPTIMIZATION_OPPORTUNITY,
                    category=PatternCategory.PERFORMANCE,
                    confidence=0.7,
                    description="Optimization opportunities detected",
                    examples=optimization_candidates[:5],  # Top 5 candidates
                    metadata={"total_candidates": len(optimization_candidates)},
                    improvement_suggestion="Focus optimization efforts on high-impact operations",
                    impact_level="high"
                ))
            
            return patterns
            
        except Exception as e:
            logger.error(f"Failed to detect optimization opportunities: {e}")
            return []
    
    # Error pattern detection methods
    
    def _group_errors_by_similarity(
        self,
        errors: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Group errors by similarity."""
        try:
            # Extract error messages
            error_messages = [e.get("message", "") for e in errors]
            
            if len(error_messages) < 2:
                return {"default": errors}
            
            # Vectorize error messages
            vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(error_messages)
            
            # Cluster similar errors
            clustering = DBSCAN(eps=0.5, min_samples=2)
            clusters = clustering.fit_predict(tfidf_matrix.toarray())
            
            # Group errors by cluster
            error_groups = defaultdict(list)
            for i, cluster_id in enumerate(clusters):
                if cluster_id == -1:
                    cluster_id = f"noise_{i}"
                error_groups[str(cluster_id)].append(errors[i])
            
            return dict(error_groups)
            
        except Exception as e:
            logger.error(f"Failed to group errors by similarity: {e}")
            return {"default": errors}
    
    def _detect_recurring_error_patterns(
        self,
        error_groups: Dict[str, List[Dict[str, Any]]]
    ) -> List[PatternMatch]:
        """Detect recurring error patterns."""
        patterns = []
        
        try:
            for group_id, group_errors in error_groups.items():
                if len(group_errors) >= self.thresholds["min_pattern_occurrences"]:
                    # Analyze error frequency and timing
                    timestamps = [
                        e.get("timestamp") for e in group_errors 
                        if e.get("timestamp")
                    ]
                    
                    if timestamps:
                        # Check if errors are clustered in time
                        timestamps.sort()
                        time_diffs = [
                            (timestamps[i] - timestamps[i-1]).total_seconds()
                            for i in range(1, len(timestamps))
                        ]
                        
                        avg_interval = np.mean(time_diffs) if time_diffs else 0
                        
                        patterns.append(PatternMatch(
                            pattern_type=PatternType.ERROR_SEQUENCE,
                            category=PatternCategory.QUALITY,
                            confidence=0.8,
                            description=f"Recurring error pattern detected",
                            examples=group_errors[:3],  # Show first 3 examples
                            metadata={
                                "error_count": len(group_errors),
                                "avg_interval_seconds": avg_interval,
                                "group_id": group_id
                            },
                            improvement_suggestion="Investigate root cause of recurring errors",
                            impact_level="high"
                        ))
            
            return patterns
            
        except Exception as e:
            logger.error(f"Failed to detect recurring error patterns: {e}")
            return []
    
    def _detect_error_cascade_patterns(
        self,
        errors: List[Dict[str, Any]]
    ) -> List[PatternMatch]:
        """Detect error cascade patterns."""
        patterns = []
        
        try:
            # Sort errors by timestamp
            sorted_errors = sorted(
                [e for e in errors if e.get("timestamp")],
                key=lambda x: x["timestamp"]
            )
            
            if len(sorted_errors) < 5:
                return patterns
            
            # Look for error cascades (multiple errors in short time)
            cascade_windows = []
            current_cascade = []
            
            for i, error in enumerate(sorted_errors):
                if not current_cascade:
                    current_cascade = [error]
                    continue
                
                # Check if error is within cascade window (5 minutes)
                time_diff = (error["timestamp"] - current_cascade[-1]["timestamp"]).total_seconds()
                
                if time_diff <= 300:  # 5 minutes
                    current_cascade.append(error)
                else:
                    # End current cascade if it has enough errors
                    if len(current_cascade) >= 3:
                        cascade_windows.append(current_cascade)
                    current_cascade = [error]
            
            # Don't forget the last cascade
            if len(current_cascade) >= 3:
                cascade_windows.append(current_cascade)
            
            if cascade_windows:
                patterns.append(PatternMatch(
                    pattern_type=PatternType.ERROR_SEQUENCE,
                    category=PatternCategory.QUALITY,
                    confidence=0.8,
                    description="Error cascade pattern detected",
                    examples=[{
                        "cascade_count": len(cascade_windows),
                        "largest_cascade": max(len(cascade) for cascade in cascade_windows),
                        "sample_cascade": cascade_windows[0][:3]
                    }],
                    metadata={"total_cascades": len(cascade_windows)},
                    improvement_suggestion="Implement circuit breaker patterns to prevent error cascades",
                    impact_level="high"
                ))
            
            return patterns
            
        except Exception as e:
            logger.error(f"Failed to detect error cascade patterns: {e}")
            return []
    
    def _detect_error_hotspots(
        self,
        errors: List[Dict[str, Any]]
    ) -> List[PatternMatch]:
        """Detect error hotspots (areas with high error density)."""
        patterns = []
        
        try:
            # Group errors by location (file, function, etc.)
            location_counts = defaultdict(int)
            
            for error in errors:
                # Try to extract location from error context
                location = self._extract_error_location(error)
                if location:
                    location_counts[location] += 1
            
            if location_counts:
                # Find hotspots (locations with many errors)
                total_errors = sum(location_counts.values())
                hotspots = [
                    (location, count) for location, count in location_counts.items()
                    if count >= 5 and count / total_errors >= 0.1  # At least 5 errors and 10% of total
                ]
                
                if hotspots:
                    patterns.append(PatternMatch(
                        pattern_type=PatternType.ERROR_SEQUENCE,
                        category=PatternCategory.QUALITY,
                        confidence=0.9,
                        description="Error hotspot detected",
                        examples=[{
                            "hotspots": hotspots,
                            "total_errors": total_errors
                        }],
                        metadata={"hotspot_count": len(hotspots)},
                        improvement_suggestion="Focus debugging efforts on error hotspots",
                        impact_level="high"
                    ))
            
            return patterns
            
        except Exception as e:
            logger.error(f"Failed to detect error hotspots: {e}")
            return []
    
    # Helper methods
    
    def _calculate_code_metrics(self, code_text: str) -> CodeMetrics:
        """Calculate code quality metrics."""
        try:
            # Basic metrics calculation
            lines = code_text.split('\n')
            lines_of_code = len([line for line in lines if line.strip()])
            
            # Count functions and classes (simplified)
            function_count = code_text.count('def ')
            class_count = code_text.count('class ')
            
            # Estimate cyclomatic complexity (simplified)
            complexity_keywords = ['if', 'elif', 'for', 'while', 'try', 'except', 'and', 'or']
            cyclomatic_complexity = 1  # Base complexity
            for keyword in complexity_keywords:
                cyclomatic_complexity += code_text.count(f' {keyword} ')
            
            # Estimate dependency count
            import_lines = [line for line in lines if line.strip().startswith(('import ', 'from '))]
            dependency_count = len(import_lines)
            
            # Simplified duplicate detection
            line_counts = Counter(line.strip() for line in lines if line.strip())
            duplicate_lines = sum(count - 1 for count in line_counts.values() if count > 1)
            
            # Placeholder values for complex metrics
            test_coverage = 0.7  # Would need actual test analysis
            maintainability_index = max(0, 100 - (cyclomatic_complexity * 2) - (lines_of_code / 10))
            
            return CodeMetrics(
                cyclomatic_complexity=cyclomatic_complexity,
                lines_of_code=lines_of_code,
                function_count=function_count,
                class_count=class_count,
                dependency_count=dependency_count,
                duplicate_lines=duplicate_lines,
                test_coverage=test_coverage,
                maintainability_index=maintainability_index
            )
            
        except Exception as e:
            logger.error(f"Failed to calculate code metrics: {e}")
            return CodeMetrics(0, 0, 0, 0, 0, 0, 0.0, 0.0)
    
    def _has_long_functions(self, code_text: str) -> bool:
        """Check if code has long functions."""
        try:
            # Simple heuristic: look for functions with many lines
            lines = code_text.split('\n')
            in_function = False
            current_function_lines = 0
            
            for line in lines:
                if line.strip().startswith('def '):
                    if in_function and current_function_lines > 50:  # Function with >50 lines
                        return True
                    in_function = True
                    current_function_lines = 0
                elif in_function:
                    if line.strip() and not line.startswith(' '):
                        # End of function
                        if current_function_lines > 50:
                            return True
                        in_function = False
                    else:
                        current_function_lines += 1
            
            # Check last function
            return in_function and current_function_lines > 50
            
        except Exception:
            return False
    
    def _has_deep_nesting(self, code_text: str) -> bool:
        """Check if code has deep nesting."""
        try:
            lines = code_text.split('\n')
            max_indent = 0
            
            for line in lines:
                if line.strip():
                    indent_level = (len(line) - len(line.lstrip())) // 4  # Assuming 4-space indents
                    max_indent = max(max_indent, indent_level)
            
            return max_indent > 4  # More than 4 levels of nesting
            
        except Exception:
            return False
    
    def _find_code_duplicates(self, code_text: str, all_code_texts: List[str]) -> List[str]:
        """Find code duplicates."""
        try:
            lines = code_text.split('\n')
            duplicates = []
            
            # Look for duplicate blocks of 3+ lines
            for i in range(len(lines) - 2):
                block = '\n'.join(lines[i:i+3])
                if len(block.strip()) < 20:  # Skip very short blocks
                    continue
                
                # Check in other code texts
                for j, other_code in enumerate(all_code_texts):
                    if other_code != code_text and block in other_code:
                        duplicates.append(f"Duplicate block found in text {j}")
                        break
            
            return duplicates
            
        except Exception:
            return []
    
    def _analyze_code_cluster(
        self,
        cluster_texts: List[str],
        cluster_samples: List[Dict[str, Any]]
    ) -> Optional[PatternMatch]:
        """Analyze a cluster of similar code."""
        try:
            if len(cluster_texts) < 2:
                return None
            
            # Find common patterns in the cluster
            common_keywords = self._find_common_keywords(cluster_texts)
            common_structure = self._find_common_structure(cluster_texts)
            
            return PatternMatch(
                pattern_type=PatternType.CODE_STRUCTURE,
                category=PatternCategory.FUNCTIONAL,
                confidence=0.8,
                description=f"Common code pattern detected ({len(cluster_texts)} instances)",
                examples=cluster_samples,
                metadata={
                    "common_keywords": common_keywords,
                    "common_structure": common_structure,
                    "instance_count": len(cluster_texts)
                },
                improvement_suggestion="Consider extracting common pattern into reusable component",
                impact_level="medium"
            )
            
        except Exception as e:
            logger.error(f"Failed to analyze code cluster: {e}")
            return None
    
    def _find_common_keywords(self, code_texts: List[str]) -> List[str]:
        """Find common keywords in code texts."""
        try:
            all_words = []
            for text in code_texts:
                words = re.findall(r'\b\w+\b', text.lower())
                all_words.extend(words)
            
            word_counts = Counter(all_words)
            
            # Return words that appear in most texts
            common_words = [
                word for word, count in word_counts.most_common(10)
                if count >= len(code_texts) * 0.5  # Appears in at least half the texts
            ]
            
            return common_words
            
        except Exception:
            return []
    
    def _find_common_structure(self, code_texts: List[str]) -> Dict[str, Any]:
        """Find common structural patterns."""
        try:
            structures = []
            
            for text in code_texts:
                structure = {
                    "has_class": "class " in text,
                    "has_function": "def " in text,
                    "has_loops": any(keyword in text for keyword in ["for ", "while "]),
                    "has_conditionals": any(keyword in text for keyword in ["if ", "elif ", "else:"]),
                    "has_try_catch": "try:" in text
                }
                structures.append(structure)
            
            # Find common structural elements
            common_structure = {}
            for key in structures[0].keys():
                values = [s[key] for s in structures]
                if sum(values) >= len(values) * 0.7:  # Present in 70% of samples
                    common_structure[key] = True
            
            return common_structure
            
        except Exception:
            return {}
    
    def _group_activities_by_session(
        self,
        activities: List[Dict[str, Any]]
    ) -> List[List[Dict[str, Any]]]:
        """Group activities by session."""
        try:
            sessions = []
            current_session = []
            last_timestamp = None
            
            for activity in sorted(activities, key=lambda x: x.get("timestamp", datetime.min)):
                timestamp = activity.get("timestamp")
                if not timestamp:
                    continue
                
                # Start new session if gap > 30 minutes
                if (last_timestamp and 
                    (timestamp - last_timestamp).total_seconds() > 1800):
                    if current_session:
                        sessions.append(current_session)
                    current_session = []
                
                current_session.append(activity)
                last_timestamp = timestamp
            
            # Add final session
            if current_session:
                sessions.append(current_session)
            
            return sessions
            
        except Exception:
            return []
    
    def _extract_error_location(self, error: Dict[str, Any]) -> Optional[str]:
        """Extract location information from error."""
        try:
            # Try to extract from various fields
            context = error.get("execution_context", {})
            
            # Look for file information
            if "file" in context:
                return context["file"]
            
            # Look for function information
            if "function" in context:
                return f"function:{context['function']}"
            
            # Look for class information
            if "class" in context:
                return f"class:{context['class']}"
            
            # Extract from stack trace if available
            stack_trace = error.get("stack_trace", "")
            if stack_trace:
                # Simple extraction from stack trace
                lines = stack_trace.split('\n')
                for line in lines:
                    if 'File "' in line:
                        match = re.search(r'File "([^"]+)"', line)
                        if match:
                            return match.group(1)
            
            return None
            
        except Exception:
            return None
    
    def _generate_category_suggestions(
        self,
        category: PatternCategory,
        patterns: List[PatternMatch]
    ) -> List[Dict[str, Any]]:
        """Generate optimization suggestions for a pattern category."""
        suggestions = []
        
        try:
            if category == PatternCategory.PERFORMANCE:
                suggestions.extend(self._generate_performance_suggestions(patterns))
            elif category == PatternCategory.QUALITY:
                suggestions.extend(self._generate_quality_suggestions(patterns))
            elif category == PatternCategory.BEHAVIORAL:
                suggestions.extend(self._generate_behavioral_suggestions(patterns))
            elif category == PatternCategory.STRUCTURAL:
                suggestions.extend(self._generate_structural_suggestions(patterns))
            elif category == PatternCategory.FUNCTIONAL:
                suggestions.extend(self._generate_functional_suggestions(patterns))
            
            return suggestions
            
        except Exception as e:
            logger.error(f"Failed to generate category suggestions: {e}")
            return []
    
    def _generate_performance_suggestions(
        self,
        patterns: List[PatternMatch]
    ) -> List[Dict[str, Any]]:
        """Generate performance optimization suggestions."""
        suggestions = []
        
        # Count different types of performance issues
        slow_operations = [p for p in patterns if "slow operation" in p.description.lower()]
        high_resource_usage = [p for p in patterns if "high" in p.description.lower() and "usage" in p.description.lower()]
        regressions = [p for p in patterns if "regression" in p.description.lower()]
        
        if slow_operations:
            suggestions.append({
                "title": "Optimize Slow Operations",
                "description": f"Detected {len(slow_operations)} slow operation patterns",
                "impact_score": 90,
                "effort_estimate": "medium",
                "recommendations": [
                    "Implement caching for frequently accessed data",
                    "Optimize database queries with proper indexing",
                    "Consider asynchronous processing for heavy operations",
                    "Profile code to identify specific bottlenecks"
                ]
            })
        
        if high_resource_usage:
            suggestions.append({
                "title": "Reduce Resource Usage",
                "description": f"Detected {len(high_resource_usage)} high resource usage patterns",
                "impact_score": 80,
                "effort_estimate": "high",
                "recommendations": [
                    "Implement memory pooling and object reuse",
                    "Optimize algorithms to reduce computational complexity",
                    "Add resource monitoring and alerts",
                    "Consider horizontal scaling for CPU-intensive tasks"
                ]
            })
        
        if regressions:
            suggestions.append({
                "title": "Address Performance Regressions",
                "description": f"Detected {len(regressions)} performance regression patterns",
                "impact_score": 95,
                "effort_estimate": "medium",
                "recommendations": [
                    "Review recent code changes for performance impact",
                    "Implement performance regression testing",
                    "Add performance benchmarks to CI/CD pipeline",
                    "Consider reverting problematic changes"
                ]
            })
        
        return suggestions
    
    def _generate_quality_suggestions(
        self,
        patterns: List[PatternMatch]
    ) -> List[Dict[str, Any]]:
        """Generate code quality suggestions."""
        suggestions = []
        
        complexity_issues = [p for p in patterns if "complexity" in p.description.lower()]
        error_patterns = [p for p in patterns if "error" in p.description.lower()]
        duplication_issues = [p for p in patterns if "duplication" in p.description.lower()]
        
        if complexity_issues:
            suggestions.append({
                "title": "Reduce Code Complexity",
                "description": f"Detected {len(complexity_issues)} complexity-related patterns",
                "impact_score": 70,
                "effort_estimate": "medium",
                "recommendations": [
                    "Break down complex functions into smaller ones",
                    "Extract common logic into utility functions",
                    "Implement design patterns to improve structure",
                    "Add comprehensive unit tests for complex code"
                ]
            })
        
        if error_patterns:
            suggestions.append({
                "title": "Improve Error Handling",
                "description": f"Detected {len(error_patterns)} error patterns",
                "impact_score": 85,
                "effort_estimate": "medium",
                "recommendations": [
                    "Implement proper exception handling",
                    "Add input validation and sanitization",
                    "Create error recovery mechanisms",
                    "Improve error logging and monitoring"
                ]
            })
        
        if duplication_issues:
            suggestions.append({
                "title": "Eliminate Code Duplication",
                "description": f"Detected {len(duplication_issues)} duplication patterns",
                "impact_score": 60,
                "effort_estimate": "low",
                "recommendations": [
                    "Extract common code into shared functions",
                    "Create reusable components and libraries",
                    "Implement inheritance where appropriate",
                    "Use configuration files for repeated values"
                ]
            })
        
        return suggestions
    
    def _generate_behavioral_suggestions(
        self,
        patterns: List[PatternMatch]
    ) -> List[Dict[str, Any]]:
        """Generate behavioral optimization suggestions."""
        suggestions = []
        
        usage_patterns = [p for p in patterns if "usage" in p.description.lower()]
        collaboration_patterns = [p for p in patterns if "collaboration" in p.description.lower()]
        
        if usage_patterns:
            suggestions.append({
                "title": "Optimize User Experience",
                "description": f"Detected {len(usage_patterns)} usage patterns",
                "impact_score": 65,
                "effort_estimate": "medium",
                "recommendations": [
                    "Improve discoverability of underused features",
                    "Optimize workflows for common use cases",
                    "Add user guidance and onboarding",
                    "Implement usage analytics for better insights"
                ]
            })
        
        if collaboration_patterns:
            suggestions.append({
                "title": "Enhance Collaboration Features",
                "description": f"Detected {len(collaboration_patterns)} collaboration patterns",
                "impact_score": 55,
                "effort_estimate": "high",
                "recommendations": [
                    "Implement real-time collaboration features",
                    "Add team management and permissions",
                    "Create shared workspaces and resources",
                    "Improve communication and notification systems"
                ]
            })
        
        return suggestions
    
    def _generate_structural_suggestions(
        self,
        patterns: List[PatternMatch]
    ) -> List[Dict[str, Any]]:
        """Generate structural improvement suggestions."""
        return [{
            "title": "Improve Code Structure",
            "description": f"Detected {len(patterns)} structural patterns",
            "impact_score": 60,
            "effort_estimate": "medium",
            "recommendations": [
                "Apply consistent coding standards",
                "Implement proper separation of concerns",
                "Use appropriate design patterns",
                "Improve module organization and dependencies"
            ]
        }]
    
    def _generate_functional_suggestions(
        self,
        patterns: List[PatternMatch]
    ) -> List[Dict[str, Any]]:
        """Generate functional improvement suggestions."""
        return [{
            "title": "Enhance Functionality",
            "description": f"Detected {len(patterns)} functional patterns",
            "impact_score": 65,
            "effort_estimate": "medium",
            "recommendations": [
                "Extract common functionality into reusable components",
                "Improve function interfaces and documentation",
                "Add comprehensive error handling",
                "Implement proper testing for all functions"
            ]
        }]


# Global pattern recognition engine
pattern_recognition_engine = PatternRecognitionEngine()