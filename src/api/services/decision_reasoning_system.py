"""
Decision Reasoning System - Track decisions, alternatives, and reasoning with confidence scores
"""

import json
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
from pathlib import Path
import re

from sqlalchemy.orm import Session
from sqlalchemy import cast, String, desc, and_, func, or_

from ..models.memory import MemoryItem


class DecisionReasoningSystem:
    """Track and explain decision-making with full reasoning history"""
    
    def __init__(self):
        self.decisions_file = Path.home() / ".claude_decisions.json"
        self.reasoning_patterns_file = Path.home() / ".claude_reasoning_patterns.json"
        self.confidence_history_file = Path.home() / ".claude_confidence_history.json"
        
        # Decision categories
        self.decision_categories = {
            "architecture": ["design", "structure", "pattern", "framework"],
            "implementation": ["code", "algorithm", "approach", "method"],
            "debugging": ["fix", "solution", "workaround", "patch"],
            "optimization": ["performance", "efficiency", "speed", "memory"],
            "security": ["auth", "permission", "encryption", "validation"],
            "tooling": ["library", "package", "tool", "dependency"]
        }
        
        # Confidence factors
        self.confidence_factors = {
            "evidence_based": 0.9,      # Based on concrete evidence
            "experience_based": 0.8,    # Based on past experience
            "documented": 0.85,         # Has official documentation
            "tested": 0.9,              # Has been tested
            "community_validated": 0.8, # Validated by community
            "theoretical": 0.6,         # Based on theory only
            "experimental": 0.5         # Experimental approach
        }
        
        self._load_reasoning_patterns()
    
    def _load_reasoning_patterns(self):
        """Load learned reasoning patterns"""
        self.reasoning_patterns = {
            "performance": {
                "caching": "Cache frequently accessed data to reduce computation",
                "async": "Use asynchronous operations for I/O-bound tasks",
                "batching": "Batch operations to reduce overhead"
            },
            "error_handling": {
                "try_except": "Wrap risky operations in try-except blocks",
                "validation": "Validate inputs before processing",
                "graceful_degradation": "Provide fallbacks for failures"
            },
            "architecture": {
                "separation_of_concerns": "Separate business logic from presentation",
                "dependency_injection": "Inject dependencies for better testing",
                "interface_design": "Design clear interfaces between components"
            }
        }
        
        if self.reasoning_patterns_file.exists():
            with open(self.reasoning_patterns_file, 'r') as f:
                learned = json.load(f)
                self.reasoning_patterns.update(learned)
    
    def record_decision(self, db: Session,
                       decision_title: str,
                       chosen_solution: str,
                       reasoning: str,
                       alternatives: List[Dict[str, Any]],
                       context: Dict[str, Any],
                       confidence: float,
                       evidence: Optional[List[str]] = None,
                       trade_offs: Optional[Dict[str, Any]] = None,
                       project_id: Optional[str] = None,
                       session_id: Optional[str] = None) -> Dict[str, Any]:
        """Record a decision with full reasoning and alternatives"""
        
        # Generate decision ID
        decision_id = hashlib.md5(
            f"{decision_title}:{chosen_solution}:{datetime.utcnow().isoformat()}".encode()
        ).hexdigest()[:12]
        
        # Determine category
        category = self._categorize_decision(decision_title, reasoning)
        
        # Calculate adjusted confidence
        adjusted_confidence = self._adjust_confidence(
            confidence, evidence, len(alternatives), trade_offs
        )
        
        # Create decision record
        decision_record = {
            "decision_id": decision_id,
            "timestamp": datetime.utcnow().isoformat(),
            "title": decision_title,
            "category": category,
            "chosen_solution": chosen_solution,
            "reasoning": reasoning,
            "alternatives": alternatives,
            "alternative_count": len(alternatives),
            "context": context,
            "confidence": {
                "initial": confidence,
                "adjusted": adjusted_confidence,
                "factors": self._get_confidence_factors(evidence, alternatives, trade_offs)
            },
            "evidence": evidence or [],
            "trade_offs": trade_offs or {},
            "project_id": project_id,
            "session_id": session_id,
            "outcome": None,  # To be updated later
            "impact": None    # To be measured later
        }
        
        # Store in database
        content = self._format_decision_content(decision_record)
        memory = self._store_decision_memory(db, content, decision_record, project_id)
        
        # Update decision history
        self._update_decision_history(decision_record)
        
        # Learn from reasoning pattern
        self._learn_reasoning_pattern(category, reasoning, confidence)
        
        return {
            "decision_id": decision_id,
            "recorded": True,
            "category": category,
            "confidence": adjusted_confidence,
            "alternatives_considered": len(alternatives),
            "memory_id": str(memory.id)
        }
    
    def _categorize_decision(self, title: str, reasoning: str) -> str:
        """Categorize decision based on content"""
        combined_text = f"{title} {reasoning}".lower()
        
        scores = {}
        for category, keywords in self.decision_categories.items():
            score = sum(1 for keyword in keywords if keyword in combined_text)
            if score > 0:
                scores[category] = score
        
        if scores:
            return max(scores.items(), key=lambda x: x[1])[0]
        return "general"
    
    def _adjust_confidence(self, base_confidence: float, evidence: Optional[List[str]],
                          alternatives_count: int, trade_offs: Optional[Dict[str, Any]]) -> float:
        """Adjust confidence based on various factors"""
        adjusted = base_confidence
        
        # More evidence increases confidence
        if evidence:
            adjusted += min(0.1, len(evidence) * 0.02)
        
        # More alternatives considered increases confidence
        if alternatives_count > 2:
            adjusted += min(0.1, alternatives_count * 0.02)
        
        # Clear trade-offs increase confidence
        if trade_offs and len(trade_offs) > 1:
            adjusted += 0.05
        
        # Cap at 0.95 (never 100% certain)
        return min(0.95, adjusted)
    
    def _get_confidence_factors(self, evidence: Optional[List[str]],
                               alternatives: List[Dict[str, Any]],
                               trade_offs: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze factors affecting confidence"""
        factors = {
            "has_evidence": bool(evidence),
            "evidence_count": len(evidence) if evidence else 0,
            "alternatives_analyzed": len(alternatives),
            "has_trade_offs": bool(trade_offs),
            "factors_considered": []
        }
        
        # Check which confidence factors apply
        if evidence:
            for ev in evidence:
                ev_lower = ev.lower()
                if "test" in ev_lower:
                    factors["factors_considered"].append("tested")
                if "document" in ev_lower or "doc" in ev_lower:
                    factors["factors_considered"].append("documented")
                if "community" in ev_lower or "stackoverflow" in ev_lower:
                    factors["factors_considered"].append("community_validated")
        
        return factors
    
    def _format_decision_content(self, decision: Dict[str, Any]) -> str:
        """Format decision for storage"""
        lines = [
            f"DECISION: {decision['title']}",
            f"Category: {decision['category']}",
            f"Chosen: {decision['chosen_solution']}",
            f"Confidence: {decision['confidence']['adjusted']:.0%}",
            f"",
            f"Reasoning: {decision['reasoning']}",
            f"",
            f"Alternatives considered: {decision['alternative_count']}"
        ]
        
        for i, alt in enumerate(decision['alternatives'][:3], 1):
            lines.append(f"  {i}. {alt.get('solution', 'Unknown')}")
            if alt.get('reason_rejected'):
                lines.append(f"     Rejected: {alt['reason_rejected']}")
        
        if decision['evidence']:
            lines.extend(["", "Evidence:"])
            for ev in decision['evidence'][:3]:
                lines.append(f"  - {ev}")
        
        return "\n".join(lines)
    
    def _store_decision_memory(self, db: Session, content: str,
                              decision_data: Dict[str, Any],
                              project_id: Optional[str]) -> MemoryItem:
        """Store decision in memory with metadata"""
        tags = ["decision", decision_data["category"]]
        if project_id:
            tags.append(f"project:{project_id}")
        
        memory_hash = hashlib.sha256(content.encode()).hexdigest()
        
        memory = MemoryItem(
            content=content,
            content_hash=memory_hash,
            tags=tags,
            meta_data={
                "memory_type": "decision",
                "importance": 0.8 + (decision_data["confidence"]["adjusted"] * 0.2),
                **decision_data
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
    
    def _update_decision_history(self, decision: Dict[str, Any]):
        """Update local decision history"""
        history = {}
        if self.decisions_file.exists():
            with open(self.decisions_file, 'r') as f:
                history = json.load(f)
        
        # Store by ID and category
        decision_id = decision["decision_id"]
        category = decision["category"]
        
        if category not in history:
            history[category] = {}
        
        history[category][decision_id] = {
            "title": decision["title"],
            "timestamp": decision["timestamp"],
            "confidence": decision["confidence"]["adjusted"],
            "alternatives": len(decision["alternatives"])
        }
        
        with open(self.decisions_file, 'w') as f:
            json.dump(history, f, indent=2)
    
    def _learn_reasoning_pattern(self, category: str, reasoning: str, confidence: float):
        """Learn from successful reasoning patterns"""
        if confidence < 0.7:  # Only learn from confident decisions
            return
        
        # Extract key phrases from reasoning
        key_phrases = re.findall(r'(?:because|since|therefore|thus)\s+(.+?)(?:\.|,|$)', 
                                reasoning, re.IGNORECASE)
        
        if key_phrases and category not in self.reasoning_patterns:
            self.reasoning_patterns[category] = {}
        
        for phrase in key_phrases:
            # Create a pattern key
            pattern_key = self._extract_pattern_key(phrase)
            if pattern_key:
                self.reasoning_patterns[category][pattern_key] = phrase.strip()
        
        # Save learned patterns
        with open(self.reasoning_patterns_file, 'w') as f:
            json.dump(self.reasoning_patterns, f, indent=2)
    
    def _extract_pattern_key(self, phrase: str) -> Optional[str]:
        """Extract a key from reasoning phrase"""
        # Simple keyword extraction
        keywords = ["performance", "security", "efficiency", "clarity", 
                   "maintainability", "scalability", "reliability"]
        
        phrase_lower = phrase.lower()
        for keyword in keywords:
            if keyword in phrase_lower:
                return keyword
        
        # Use first significant word as key
        words = phrase.split()
        for word in words:
            if len(word) > 4 and word.isalpha():
                return word.lower()
        
        return None
    
    def update_decision_outcome(self, db: Session, decision_id: str,
                               outcome: str, impact: Dict[str, Any],
                               lessons_learned: Optional[str] = None) -> Dict[str, Any]:
        """Update a decision with its actual outcome"""
        # Find decision in database
        decision_memory = db.query(MemoryItem).filter(
            cast(MemoryItem.meta_data, String).contains(f'"decision_id": "{decision_id}"')
        ).first()
        
        if not decision_memory:
            return {"updated": False, "error": "Decision not found"}
        
        # Update metadata
        meta_data = decision_memory.meta_data
        meta_data["outcome"] = outcome
        meta_data["impact"] = impact
        if lessons_learned:
            meta_data["lessons_learned"] = lessons_learned
        
        # Update confidence history based on outcome
        self._update_confidence_history(
            meta_data["category"],
            meta_data["confidence"]["adjusted"],
            outcome
        )
        
        decision_memory.meta_data = meta_data
        decision_memory.updated_at = datetime.utcnow()
        db.commit()
        
        return {
            "updated": True,
            "decision_id": decision_id,
            "outcome": outcome,
            "confidence_was": meta_data["confidence"]["adjusted"]
        }
    
    def _update_confidence_history(self, category: str, confidence: float, outcome: str):
        """Track confidence accuracy over time"""
        history = {}
        if self.confidence_history_file.exists():
            with open(self.confidence_history_file, 'r') as f:
                history = json.load(f)
        
        if category not in history:
            history[category] = {"predictions": []}
        
        history[category]["predictions"].append({
            "confidence": confidence,
            "outcome": outcome,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Calculate accuracy metrics
        predictions = history[category]["predictions"][-50:]  # Last 50
        successful = sum(1 for p in predictions if p["outcome"] == "successful")
        history[category]["accuracy"] = successful / len(predictions) if predictions else 0
        
        with open(self.confidence_history_file, 'w') as f:
            json.dump(history, f, indent=2)
    
    def explain_decision(self, db: Session, decision_id: str) -> Optional[Dict[str, Any]]:
        """Explain the reasoning behind a past decision"""
        # Find decision
        decision_memory = db.query(MemoryItem).filter(
            cast(MemoryItem.meta_data, String).contains(f'"decision_id": "{decision_id}"')
        ).first()
        
        if not decision_memory:
            return None
        
        meta = decision_memory.meta_data
        
        explanation = {
            "decision": meta["title"],
            "made_on": meta["timestamp"],
            "category": meta["category"],
            "what_was_chosen": meta["chosen_solution"],
            "why": meta["reasoning"],
            "confidence_level": f"{meta['confidence']['adjusted']:.0%}",
            "confidence_factors": meta["confidence"]["factors"],
            "alternatives_considered": [],
            "evidence_used": meta.get("evidence", []),
            "trade_offs": meta.get("trade_offs", {}),
            "outcome": meta.get("outcome", "Not yet measured"),
            "impact": meta.get("impact", {}),
            "lessons_learned": meta.get("lessons_learned", "None recorded")
        }
        
        # Format alternatives
        for alt in meta["alternatives"]:
            explanation["alternatives_considered"].append({
                "option": alt.get("solution", "Unknown"),
                "pros": alt.get("pros", []),
                "cons": alt.get("cons", []),
                "rejected_because": alt.get("reason_rejected", "Not specified")
            })
        
        return explanation
    
    def find_similar_decisions(self, db: Session, category: str,
                              keywords: List[str], limit: int = 5) -> List[Dict[str, Any]]:
        """Find similar past decisions"""
        query = db.query(MemoryItem).filter(
            and_(
                cast(MemoryItem.meta_data, String).contains('"memory_type": "decision"'),
                cast(MemoryItem.meta_data, String).contains(f'"category": "{category}"')
            )
        )
        
        # Filter by keywords
        for keyword in keywords:
            query = query.filter(
                or_(
                    MemoryItem.content.ilike(f"%{keyword}%"),
                    cast(MemoryItem.meta_data, String).contains(keyword)
                )
            )
        
        decisions = query.order_by(desc(MemoryItem.created_at)).limit(limit * 2).all()
        
        # Format results
        results = []
        for dec in decisions[:limit]:
            meta = dec.meta_data
            results.append({
                "decision_id": meta.get("decision_id"),
                "title": meta.get("title"),
                "category": meta.get("category"),
                "chosen": meta.get("chosen_solution"),
                "confidence": meta["confidence"]["adjusted"],
                "outcome": meta.get("outcome", "Unknown"),
                "created": dec.created_at.isoformat()
            })
        
        return results
    
    def get_confidence_report(self, category: Optional[str] = None) -> Dict[str, Any]:
        """Get report on confidence accuracy"""
        if not self.confidence_history_file.exists():
            return {"status": "No confidence history available"}
        
        with open(self.confidence_history_file, 'r') as f:
            history = json.load(f)
        
        report = {
            "overall_accuracy": 0,
            "categories": {},
            "recommendations": []
        }
        
        total_predictions = 0
        total_successful = 0
        
        for cat, data in history.items():
            if category and cat != category:
                continue
            
            predictions = data.get("predictions", [])
            if predictions:
                successful = sum(1 for p in predictions if p["outcome"] == "successful")
                accuracy = successful / len(predictions)
                
                report["categories"][cat] = {
                    "accuracy": accuracy,
                    "total_decisions": len(predictions),
                    "successful": successful,
                    "average_confidence": sum(p["confidence"] for p in predictions) / len(predictions)
                }
                
                total_predictions += len(predictions)
                total_successful += successful
                
                # Generate recommendations
                if accuracy < 0.6:
                    report["recommendations"].append(
                        f"Low accuracy in {cat} ({accuracy:.0%}) - consider more careful analysis"
                    )
                elif accuracy > 0.9:
                    report["recommendations"].append(
                        f"High accuracy in {cat} ({accuracy:.0%}) - confidence is well-calibrated"
                    )
        
        if total_predictions > 0:
            report["overall_accuracy"] = total_successful / total_predictions
        
        return report
    
    def suggest_decision(self, db: Session, problem: str, 
                        context: Dict[str, Any]) -> Dict[str, Any]:
        """Suggest decision based on past experience"""
        # Categorize problem
        category = self._categorize_decision(problem, "")
        
        # Find similar past decisions
        keywords = problem.lower().split()
        similar = self.find_similar_decisions(db, category, keywords[:3])
        
        # Get relevant reasoning patterns
        patterns = self.reasoning_patterns.get(category, {})
        
        # Build suggestion
        suggestion = {
            "problem": problem,
            "category": category,
            "suggested_approach": None,
            "confidence": 0.5,
            "based_on": [],
            "reasoning_patterns": [],
            "alternatives_to_consider": []
        }
        
        # If we have similar successful decisions
        successful_similar = [d for d in similar if d.get("outcome") == "successful"]
        if successful_similar:
            best = max(successful_similar, key=lambda x: x["confidence"])
            suggestion["suggested_approach"] = best["chosen"]
            suggestion["confidence"] = best["confidence"] * 0.8  # Slightly lower for new context
            suggestion["based_on"].append({
                "decision": best["title"],
                "confidence": best["confidence"],
                "outcome": best["outcome"]
            })
        
        # Add reasoning patterns
        for pattern_key, pattern_text in patterns.items():
            if any(word in problem.lower() for word in pattern_key.split('_')):
                suggestion["reasoning_patterns"].append(pattern_text)
        
        # Suggest alternatives to consider
        all_alternatives = []
        for decision in similar[:3]:
            all_alternatives.append(decision["chosen"])
        suggestion["alternatives_to_consider"] = list(set(all_alternatives))
        
        return suggestion