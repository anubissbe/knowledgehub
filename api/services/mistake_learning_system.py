"""
Mistake Learning System - Track, analyze, and prevent repeated errors
"""

import re
import json
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from pathlib import Path

from sqlalchemy.orm import Session
from sqlalchemy import cast, String, desc, and_, func

from ..models.memory import MemoryItem


class MistakeLearningSystem:
    """Learn from mistakes to prevent repetition"""
    
    def __init__(self):
        self.patterns_file = Path.home() / ".claude_error_patterns.json"
        self.lessons_file = Path.home() / ".claude_lessons_learned.json"
        self.prevention_rules_file = Path.home() / ".claude_prevention_rules.json"
        
        # Common error patterns
        self.error_patterns = {
            "import_error": {
                "regex": r"(ImportError|ModuleNotFoundError).*['\"](\w+)['\"]",
                "category": "dependency",
                "severity": "medium"
            },
            "attribute_error": {
                "regex": r"AttributeError.*'(\w+)'.*has no attribute '(\w+)'",
                "category": "api_misuse",
                "severity": "high"
            },
            "type_error": {
                "regex": r"TypeError.*expected (\w+).*got (\w+)",
                "category": "type_mismatch",
                "severity": "high"
            },
            "syntax_error": {
                "regex": r"SyntaxError.*invalid syntax|unexpected",
                "category": "syntax",
                "severity": "critical"
            },
            "key_error": {
                "regex": r"KeyError.*['\"](\w+)['\"]",
                "category": "data_access",
                "severity": "medium"
            },
            "timeout_error": {
                "regex": r"(TimeoutError|ReadTimeout|ConnectTimeout)",
                "category": "performance",
                "severity": "high"
            },
            "permission_error": {
                "regex": r"(PermissionError|403|Forbidden)",
                "category": "security",
                "severity": "critical"
            }
        }
        
        self._load_learned_patterns()
    
    def _load_learned_patterns(self):
        """Load previously learned error patterns"""
        if self.patterns_file.exists():
            with open(self.patterns_file, 'r') as f:
                learned = json.load(f)
                self.error_patterns.update(learned)
    
    def track_mistake(self, db: Session, error_type: str, error_message: str, 
                     context: Dict[str, Any], attempted_solution: Optional[str] = None,
                     successful_solution: Optional[str] = None, 
                     project_id: Optional[str] = None) -> Dict[str, Any]:
        """Track a mistake with full context"""
        
        # Generate mistake ID
        mistake_id = hashlib.md5(f"{error_type}:{error_message}".encode()).hexdigest()[:12]
        
        # Analyze error pattern
        pattern_info = self._analyze_error_pattern(error_type, error_message)
        
        # Check if this is a repeated mistake
        similar_mistakes = self._find_similar_mistakes(db, error_type, error_message, project_id)
        is_repeated = len(similar_mistakes) > 0
        repetition_count = len(similar_mistakes) + 1
        
        # Extract lesson if solution provided
        lesson = None
        if successful_solution:
            lesson = self._extract_lesson(
                error_type, error_message, 
                attempted_solution, successful_solution,
                pattern_info
            )
        
        # Create comprehensive mistake record
        mistake_record = {
            "mistake_id": mistake_id,
            "timestamp": datetime.utcnow().isoformat(),
            "error_type": error_type,
            "error_message": error_message,
            "pattern": pattern_info,
            "context": context,
            "attempted_solution": attempted_solution,
            "successful_solution": successful_solution,
            "lesson": lesson,
            "is_repeated": is_repeated,
            "repetition_count": repetition_count,
            "project_id": project_id,
            "prevented": False
        }
        
        # Store in database
        content = f"MISTAKE: {error_type} - {error_message[:100]}..."
        if lesson:
            content += f"\nLESSON: {lesson['summary']}"
        
        memory = self._store_mistake_memory(
            db, content, mistake_record, project_id
        )
        
        # Update pattern statistics
        self._update_pattern_stats(pattern_info, successful_solution is not None)
        
        # Generate prevention rule if applicable
        if successful_solution and repetition_count > 1:
            prevention_rule = self._generate_prevention_rule(
                error_type, error_message, successful_solution, pattern_info
            )
            if prevention_rule:
                self._save_prevention_rule(prevention_rule)
        
        return {
            "mistake_id": mistake_id,
            "tracked": True,
            "is_repeated": is_repeated,
            "repetition_count": repetition_count,
            "pattern": pattern_info,
            "lesson": lesson,
            "similar_count": len(similar_mistakes)
        }
    
    def _analyze_error_pattern(self, error_type: str, error_message: str) -> Dict[str, Any]:
        """Analyze error to identify patterns"""
        pattern_info = {
            "matched_pattern": None,
            "category": "unknown",
            "severity": "medium",
            "extracted_values": {}
        }
        
        # Check against known patterns
        for pattern_name, pattern_config in self.error_patterns.items():
            regex = pattern_config.get("regex")
            if regex:
                match = re.search(regex, f"{error_type}: {error_message}", re.IGNORECASE)
                if match:
                    pattern_info["matched_pattern"] = pattern_name
                    pattern_info["category"] = pattern_config.get("category", "unknown")
                    pattern_info["severity"] = pattern_config.get("severity", "medium")
                    pattern_info["extracted_values"] = {
                        f"group_{i}": g for i, g in enumerate(match.groups())
                    }
                    break
        
        return pattern_info
    
    def _find_similar_mistakes(self, db: Session, error_type: str, 
                              error_message: str, project_id: Optional[str]) -> List[MemoryItem]:
        """Find similar mistakes in history"""
        # Search for similar errors
        query = db.query(MemoryItem).filter(
            and_(
                cast(MemoryItem.meta_data, String).contains('"mistake_id"'),
                cast(MemoryItem.meta_data, String).contains(f'"error_type": "{error_type}"')
            )
        )
        
        if project_id:
            query = query.filter(
                cast(MemoryItem.meta_data, String).contains(f'"project_id": "{project_id}"')
            )
        
        similar = query.order_by(desc(MemoryItem.created_at)).limit(10).all()
        
        # Filter by similarity
        filtered = []
        for item in similar:
            if self._calculate_error_similarity(
                error_message, 
                item.meta_data.get("error_message", "")
            ) > 0.7:
                filtered.append(item)
        
        return filtered
    
    def _calculate_error_similarity(self, error1: str, error2: str) -> float:
        """Calculate similarity between two error messages"""
        # Simple word-based similarity
        words1 = set(error1.lower().split())
        words2 = set(error2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _extract_lesson(self, error_type: str, error_message: str,
                       attempted: Optional[str], successful: str,
                       pattern: Dict[str, Any]) -> Dict[str, Any]:
        """Extract lesson learned from mistake resolution"""
        lesson = {
            "summary": f"When encountering {error_type}, use: {successful}",
            "mistake_pattern": pattern.get("matched_pattern"),
            "category": pattern.get("category"),
            "what_failed": attempted,
            "what_worked": successful,
            "key_insight": None,
            "prevention_tip": None
        }
        
        # Extract key insights based on category
        if pattern["category"] == "dependency":
            lesson["key_insight"] = "Missing dependency - check imports"
            lesson["prevention_tip"] = "Verify all imports before running"
        elif pattern["category"] == "api_misuse":
            lesson["key_insight"] = "API usage error - check documentation"
            lesson["prevention_tip"] = "Review API docs for correct usage"
        elif pattern["category"] == "type_mismatch":
            lesson["key_insight"] = "Type mismatch - verify data types"
            lesson["prevention_tip"] = "Use type hints and validation"
        elif pattern["category"] == "performance":
            lesson["key_insight"] = "Performance issue - optimize or increase timeout"
            lesson["prevention_tip"] = "Monitor performance metrics"
        
        return lesson
    
    def _store_mistake_memory(self, db: Session, content: str, 
                             mistake_data: Dict[str, Any], 
                             project_id: Optional[str]) -> MemoryItem:
        """Store mistake in memory with metadata"""
        tags = ["mistake", mistake_data["pattern"]["category"]]
        if project_id:
            tags.append(f"project:{project_id}")
        
        memory_hash = hashlib.sha256(content.encode()).hexdigest()
        
        memory = MemoryItem(
            content=content,
            content_hash=memory_hash,
            tags=tags,
            meta_data={
                "memory_type": "mistake",
                "importance": 0.9 if mistake_data["is_repeated"] else 0.7,
                **mistake_data
            },
            access_count=1,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            accessed_at=datetime.utcnow()
        )
        db.add(memory)
        db.commit()
        db.refresh(memory)
        
        return memory
    
    def _update_pattern_stats(self, pattern: Dict[str, Any], was_solved: bool):
        """Update statistics for error patterns"""
        stats_file = Path.home() / ".claude_pattern_stats.json"
        
        stats = {}
        if stats_file.exists():
            with open(stats_file, 'r') as f:
                stats = json.load(f)
        
        pattern_key = pattern.get("matched_pattern", "unknown")
        if pattern_key not in stats:
            stats[pattern_key] = {
                "occurrences": 0,
                "solved": 0,
                "category": pattern.get("category"),
                "last_seen": None
            }
        
        stats[pattern_key]["occurrences"] += 1
        if was_solved:
            stats[pattern_key]["solved"] += 1
        stats[pattern_key]["last_seen"] = datetime.utcnow().isoformat()
        
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
    
    def _generate_prevention_rule(self, error_type: str, error_message: str,
                                 solution: str, pattern: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate rule to prevent future occurrences"""
        if not pattern.get("matched_pattern"):
            return None
        
        rule = {
            "rule_id": hashlib.md5(f"{pattern['matched_pattern']}:{solution}".encode()).hexdigest()[:8],
            "pattern": pattern["matched_pattern"],
            "error_type": error_type,
            "prevention_action": solution,
            "triggers": {
                "keywords": self._extract_keywords(error_message),
                "category": pattern["category"]
            },
            "confidence": 0.8,
            "created": datetime.utcnow().isoformat()
        }
        
        return rule
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract relevant keywords from error message"""
        # Remove common words and extract meaningful terms
        stop_words = {"the", "a", "an", "is", "at", "in", "on", "to", "for", "of", "with"}
        words = re.findall(r'\b\w+\b', text.lower())
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        return list(set(keywords))[:5]
    
    def _save_prevention_rule(self, rule: Dict[str, Any]):
        """Save prevention rule to file"""
        rules = {}
        if self.prevention_rules_file.exists():
            with open(self.prevention_rules_file, 'r') as f:
                rules = json.load(f)
        
        rules[rule["rule_id"]] = rule
        
        with open(self.prevention_rules_file, 'w') as f:
            json.dump(rules, f, indent=2)
    
    def check_for_prevention(self, action: str, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Check if an action might trigger a known mistake"""
        if not self.prevention_rules_file.exists():
            return None
        
        with open(self.prevention_rules_file, 'r') as f:
            rules = json.load(f)
        
        # Check each rule
        for rule_id, rule in rules.items():
            triggers = rule.get("triggers", {})
            keywords = triggers.get("keywords", [])
            
            # Check if action contains trigger keywords
            action_lower = action.lower()
            if any(keyword in action_lower for keyword in keywords):
                return {
                    "warning": True,
                    "rule_id": rule_id,
                    "pattern": rule["pattern"],
                    "suggestion": rule["prevention_action"],
                    "confidence": rule["confidence"]
                }
        
        return None
    
    def get_lessons_learned(self, db: Session, category: Optional[str] = None,
                           project_id: Optional[str] = None, 
                           days: int = 30) -> List[Dict[str, Any]]:
        """Get lessons learned from past mistakes"""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        query = db.query(MemoryItem).filter(
            and_(
                cast(MemoryItem.meta_data, String).contains('"lesson"'),
                MemoryItem.created_at > cutoff_date
            )
        )
        
        if category:
            query = query.filter(
                cast(MemoryItem.meta_data, String).contains(f'"category": "{category}"')
            )
        
        if project_id:
            query = query.filter(
                cast(MemoryItem.meta_data, String).contains(f'"project_id": "{project_id}"')
            )
        
        mistakes = query.order_by(desc(MemoryItem.created_at)).limit(50).all()
        
        lessons = []
        for mistake in mistakes:
            if mistake.meta_data.get("lesson"):
                lessons.append({
                    "id": str(mistake.id),
                    "lesson": mistake.meta_data["lesson"],
                    "error_type": mistake.meta_data.get("error_type"),
                    "repetitions": mistake.meta_data.get("repetition_count", 1),
                    "created": mistake.created_at.isoformat()
                })
        
        return lessons
    
    def get_mistake_patterns(self, db: Session, min_occurrences: int = 2) -> Dict[str, Any]:
        """Analyze mistake patterns across all projects"""
        stats_file = Path.home() / ".claude_pattern_stats.json"
        
        patterns = {}
        if stats_file.exists():
            with open(stats_file, 'r') as f:
                stats = json.load(f)
            
            # Filter by minimum occurrences
            patterns = {
                k: v for k, v in stats.items() 
                if v.get("occurrences", 0) >= min_occurrences
            }
        
        # Sort by frequency
        sorted_patterns = sorted(
            patterns.items(), 
            key=lambda x: x[1].get("occurrences", 0), 
            reverse=True
        )
        
        return {
            "patterns": dict(sorted_patterns),
            "total_patterns": len(sorted_patterns),
            "most_common": sorted_patterns[0] if sorted_patterns else None
        }
    
    def generate_mistake_report(self, db: Session, days: int = 7) -> Dict[str, Any]:
        """Generate comprehensive mistake analysis report"""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        # Get all mistakes in timeframe
        mistakes = db.query(MemoryItem).filter(
            and_(
                cast(MemoryItem.meta_data, String).contains('"memory_type": "mistake"'),
                MemoryItem.created_at > cutoff_date
            )
        ).all()
        
        # Analyze mistakes
        total_mistakes = len(mistakes)
        repeated_mistakes = sum(1 for m in mistakes if m.meta_data.get("is_repeated", False))
        solved_mistakes = sum(1 for m in mistakes if m.meta_data.get("successful_solution"))
        
        # Category breakdown
        categories = defaultdict(int)
        for mistake in mistakes:
            category = mistake.meta_data.get("pattern", {}).get("category", "unknown")
            categories[category] += 1
        
        # Prevention effectiveness
        prevented = 0
        if self.prevention_rules_file.exists():
            with open(self.prevention_rules_file, 'r') as f:
                rules = json.load(f)
                prevented = len(rules)
        
        return {
            "period_days": days,
            "total_mistakes": total_mistakes,
            "repeated_mistakes": repeated_mistakes,
            "repetition_rate": repeated_mistakes / total_mistakes if total_mistakes > 0 else 0,
            "solved_mistakes": solved_mistakes,
            "solution_rate": solved_mistakes / total_mistakes if total_mistakes > 0 else 0,
            "categories": dict(categories),
            "prevention_rules": prevented,
            "top_lessons": self.get_lessons_learned(db, days=days)[:5]
        }