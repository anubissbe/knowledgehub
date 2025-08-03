"""
Error Learning Service for Claude Code
Tracks errors, solutions, and patterns to avoid repeating mistakes
"""

import re
import json
from typing import Optio, TYPE_CHECKINGnal, List, Dict, Any, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
from sqlalchemy.orm import Session
from sqlalchemy import and_, desc, or_, func

from ..models.memory import Memory
from .memory_service import MemoryService


class ErrorLearningService:
    """Learns from errors to help Claude Code avoid repeating mistakes"""
    
    def __init__(self, db: Session, memory_service: MemoryService):
        self.db = db
        self.memory_service = memory_service
        
        # Common error patterns
        self.error_patterns = {
            "import_error": r"(ModuleNotFoundError|ImportError|No module named)",
            "syntax_error": r"(SyntaxError|IndentationError|TabError)",
            "type_error": r"(TypeError|AttributeError)",
            "key_error": r"(KeyError|IndexError)",
            "file_error": r"(FileNotFoundError|PermissionError|OSError)",
            "value_error": r"(ValueError|ZeroDivisionError)",
            "connection_error": r"(ConnectionError|TimeoutError|RequestException)",
            "undefined_var": r"(NameError|UnboundLocalError)",
            "memory_error": r"(MemoryError|RecursionError)",
            "auth_error": r"(AuthenticationError|401|403|Unauthorized)"
        }
    
    async def record_error(
        self,
        error_type: str,
        error_message: str,
        context: str,
        solution_applied: Optional[str] = None,
        success: bool = False,
        session_id: Optional[str] = None,
        project_id: Optional[str] = None
    ) -> Memory:
        """
        Record an error and its solution
        
        Args:
            error_type: Type of error (e.g., ImportError, SyntaxError)
            error_message: Full error message
            context: Context where error occurred (file, function, etc.)
            solution_applied: What solution was attempted
            success: Whether the solution worked
            session_id: Current session ID
            project_id: Project where error occurred
        """
        # Categorize error
        category = self._categorize_error(error_type, error_message)
        
        # Create error signature for matching
        signature = self._create_error_signature(error_type, error_message, context)
        
        # Check if we've seen this error before
        existing = await self._find_existing_error(signature)
        
        if existing:
            # Update existing error record
            occurrences = existing.metadata.get("occurrences", 1) + 1
            solutions = existing.metadata.get("solutions", [])
            
            if solution_applied:
                # Add new solution attempt
                solutions.append({
                    "solution": solution_applied,
                    "success": success,
                    "applied_at": datetime.utcnow().isoformat(),
                    "session_id": session_id
                })
            
            # Update success rate
            successful_solutions = [s for s in solutions if s.get("success", False)]
            success_rate = len(successful_solutions) / len(solutions) if solutions else 0
            
            existing.metadata.update({
                "occurrences": occurrences,
                "solutions": solutions,
                "success_rate": success_rate,
                "last_seen": datetime.utcnow().isoformat()
            })
            
            existing.access_count += 1
            existing.updated_at = datetime.utcnow()
            self.db.commit()
            
            return existing
        
        # Create new error record
        content = f"ERROR: [{error_type}] {error_message}\nContext: {context}"
        if solution_applied:
            content += f"\nSolution: {solution_applied} ({'✓' if success else '✗'})"
        
        metadata = {
            "error_type": error_type,
            "error_category": category,
            "error_signature": signature,
            "context": context,
            "occurrences": 1,
            "solutions": [{
                "solution": solution_applied,
                "success": success,
                "applied_at": datetime.utcnow().isoformat(),
                "session_id": session_id
            }] if solution_applied else [],
            "success_rate": 1.0 if success else 0.0,
            "first_seen": datetime.utcnow().isoformat(),
            "last_seen": datetime.utcnow().isoformat()
        }
        
        if project_id:
            metadata["project_id"] = project_id
        
        return await self.memory_service.create_memory(
            session_id=session_id or "error-learning",
            content=content,
            memory_type="error",
            importance=0.8 if not success else 0.9,
            metadata=metadata
        )
    
    async def find_similar_errors(
        self,
        error_type: str,
        error_message: str,
        context: Optional[str] = None,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Find similar errors and their solutions
        
        Returns:
            List of similar errors with successful solutions
        """
        # Create signature for matching
        signature = self._create_error_signature(error_type, error_message, context or "")
        category = self._categorize_error(error_type, error_message)
        
        # Find errors with same category
        similar_errors = self.db.query(Memory).filter(
            and_(
                Memory.memory_type == "error",
                Memory.metadata.contains({"error_category": category})
            )
        ).order_by(desc(Memory.access_count)).limit(limit * 2).all()
        
        # Score and rank by similarity
        scored_errors = []
        for error in similar_errors:
            similarity = self._calculate_similarity(
                signature,
                error.metadata.get("error_signature", "")
            )
            
            # Only include if has successful solutions
            solutions = error.metadata.get("solutions", [])
            successful = [s for s in solutions if s.get("success", False)]
            
            if successful and similarity > 0.3:
                scored_errors.append({
                    "error": error,
                    "similarity": similarity,
                    "successful_solutions": successful
                })
        
        # Sort by similarity and return top matches
        scored_errors.sort(key=lambda x: x["similarity"], reverse=True)
        
        results = []
        for item in scored_errors[:limit]:
            error = item["error"]
            results.append({
                "id": str(error.id),
                "error_type": error.metadata.get("error_type"),
                "error_message": error.content.split("\n")[0],
                "context": error.metadata.get("context"),
                "similarity": item["similarity"],
                "occurrences": error.metadata.get("occurrences", 1),
                "success_rate": error.metadata.get("success_rate", 0),
                "successful_solutions": item["successful_solutions"],
                "all_solutions": error.metadata.get("solutions", [])
            })
        
        return results
    
    async def get_error_patterns(
        self,
        project_id: Optional[str] = None,
        time_range_days: int = 30
    ) -> Dict[str, Any]:
        """
        Get common error patterns and statistics
        
        Returns:
            Error patterns, frequencies, and success rates
        """
        # Build query
        query = self.db.query(Memory).filter(
            and_(
                Memory.memory_type == "error",
                Memory.created_at > datetime.utcnow() - timedelta(days=time_range_days)
            )
        )
        
        if project_id:
            query = query.filter(Memory.metadata.contains({"project_id": project_id}))
        
        errors = query.all()
        
        # Analyze patterns
        patterns = {
            "by_category": defaultdict(int),
            "by_type": defaultdict(int),
            "by_success_rate": defaultdict(list),
            "most_common": [],
            "hardest_to_solve": [],
            "recently_solved": []
        }
        
        for error in errors:
            category = error.metadata.get("error_category", "unknown")
            error_type = error.metadata.get("error_type", "unknown")
            occurrences = error.metadata.get("occurrences", 1)
            success_rate = error.metadata.get("success_rate", 0)
            
            patterns["by_category"][category] += occurrences
            patterns["by_type"][error_type] += occurrences
            patterns["by_success_rate"][self._bucket_success_rate(success_rate)].append(error)
        
        # Find most common errors
        common_errors = sorted(errors, key=lambda e: e.metadata.get("occurrences", 1), reverse=True)[:5]
        patterns["most_common"] = [
            {
                "error_type": e.metadata.get("error_type"),
                "occurrences": e.metadata.get("occurrences", 1),
                "first_seen": e.metadata.get("first_seen"),
                "last_seen": e.metadata.get("last_seen")
            }
            for e in common_errors
        ]
        
        # Find hardest to solve (low success rate, multiple attempts)
        hard_errors = [e for e in errors if len(e.metadata.get("solutions", [])) > 2]
        hard_errors.sort(key=lambda e: e.metadata.get("success_rate", 0))
        patterns["hardest_to_solve"] = [
            {
                "error_type": e.metadata.get("error_type"),
                "attempts": len(e.metadata.get("solutions", [])),
                "success_rate": e.metadata.get("success_rate", 0),
                "context": e.metadata.get("context")
            }
            for e in hard_errors[:5]
        ]
        
        # Recently solved errors
        solved = [e for e in errors if e.metadata.get("success_rate", 0) > 0]
        solved.sort(key=lambda e: e.updated_at, reverse=True)
        patterns["recently_solved"] = [
            {
                "error_type": e.metadata.get("error_type"),
                "solution": next((s["solution"] for s in e.metadata.get("solutions", []) if s.get("success")), None),
                "solved_at": e.updated_at.isoformat()
            }
            for e in solved[:5]
        ]
        
        return patterns
    
    async def suggest_solution(
        self,
        error_type: str,
        error_message: str,
        context: str
    ) -> Optional[Dict[str, Any]]:
        """
        Suggest a solution based on past experience
        
        Returns:
            Suggested solution with confidence score
        """
        # Find similar errors
        similar = await self.find_similar_errors(error_type, error_message, context)
        
        if not similar:
            return None
        
        # Collect all successful solutions
        all_solutions = []
        for error_data in similar:
            for solution in error_data["successful_solutions"]:
                all_solutions.append({
                    "solution": solution["solution"],
                    "similarity": error_data["similarity"],
                    "success_rate": error_data["success_rate"],
                    "from_error": error_data["id"]
                })
        
        if not all_solutions:
            return None
        
        # Rank solutions by similarity and success rate
        all_solutions.sort(
            key=lambda s: s["similarity"] * s["success_rate"],
            reverse=True
        )
        
        best_solution = all_solutions[0]
        
        # Calculate confidence
        confidence = best_solution["similarity"] * best_solution["success_rate"]
        
        return {
            "suggested_solution": best_solution["solution"],
            "confidence": confidence,
            "based_on_errors": [s["from_error"] for s in all_solutions[:3]],
            "alternative_solutions": [s["solution"] for s in all_solutions[1:4]]
        }
    
    def _categorize_error(self, error_type: str, error_message: str) -> str:
        """Categorize error based on type and message"""
        full_text = f"{error_type} {error_message}"
        
        for category, pattern in self.error_patterns.items():
            if re.search(pattern, full_text, re.IGNORECASE):
                return category
        
        return "other"
    
    def _create_error_signature(self, error_type: str, error_message: str, context: str) -> str:
        """Create a signature for error matching"""
        # Remove variable parts like file paths, line numbers
        cleaned_message = re.sub(r'["\'].*?["\']', '<STRING>', error_message)
        cleaned_message = re.sub(r'\d+', '<NUM>', cleaned_message)
        cleaned_message = re.sub(r'/[\w/.-]+', '<PATH>', cleaned_message)
        
        # Extract key context (file extension, function name)
        context_key = ""
        if context:
            if "." in context:
                ext = context.split(".")[-1]
                context_key = f"ext:{ext}"
            if "function" in context.lower():
                func_match = re.search(r'function[:\s]+(\w+)', context, re.IGNORECASE)
                if func_match:
                    context_key += f" func:{func_match.group(1)}"
        
        return f"{error_type}|{cleaned_message}|{context_key}".lower()
    
    def _calculate_similarity(self, sig1: str, sig2: str) -> float:
        """Calculate similarity between two error signatures"""
        if sig1 == sig2:
            return 1.0
        
        # Split into components
        parts1 = set(sig1.split("|"))
        parts2 = set(sig2.split("|"))
        
        # Jaccard similarity
        intersection = len(parts1.intersection(parts2))
        union = len(parts1.union(parts2))
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    async def _find_existing_error(self, signature: str) -> Optional[Memory]:
        """Find existing error by signature"""
        return self.db.query(Memory).filter(
            and_(
                Memory.memory_type == "error",
                Memory.metadata.contains({"error_signature": signature})
            )
        ).first()
    
    def _bucket_success_rate(self, rate: float) -> str:
        """Bucket success rate for analysis"""
        if rate == 0:
            return "unsolved"
        elif rate < 0.3:
            return "difficult"
        elif rate < 0.7:
            return "moderate"
        else:
            return "solved"