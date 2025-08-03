"""
Code Evolution Tracker - Track code changes, refactoring patterns, and improvements over time
"""

import json
import hashlib
import difflib
import re
import ast
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
from pathlib import Path
import subprocess

from sqlalchemy.orm import Session
from sqlalchemy import cast, String, desc, and_, func, or_

from ..models.memory import MemoryItem


class CodeEvolutionTracker:
    """Track and analyze code evolution patterns, refactoring, and improvements"""
    
    def __init__(self):
        self.evolution_file = Path.home() / ".claude_code_evolution.json"
        self.patterns_file = Path.home() / ".claude_refactoring_patterns.json"
        self.improvements_file = Path.home() / ".claude_code_improvements.json"
        
        # Refactoring pattern signatures
        self.refactoring_patterns = {
            "extract_method": {
                "description": "Extract method from large function",
                "indicators": ["new function definition", "function call replacement"],
                "benefits": ["improved readability", "code reuse", "easier testing"]
            },
            "rename_variable": {
                "description": "Rename variable for clarity",
                "indicators": ["variable name change", "same type", "same scope"],
                "benefits": ["improved readability", "better semantics"]
            },
            "remove_duplication": {
                "description": "Remove code duplication",
                "indicators": ["similar code blocks", "new shared function"],
                "benefits": ["DRY principle", "easier maintenance"]
            },
            "add_error_handling": {
                "description": "Add error handling to code",
                "indicators": ["try/catch addition", "error checking"],
                "benefits": ["improved robustness", "better user experience"]
            },
            "optimize_performance": {
                "description": "Optimize code for performance",
                "indicators": ["algorithm change", "data structure change"],
                "benefits": ["faster execution", "lower resource usage"]
            },
            "improve_typing": {
                "description": "Add or improve type annotations",
                "indicators": ["type hint addition", "typing import"],
                "benefits": ["better IDE support", "early error detection"]
            },
            "modernize_syntax": {
                "description": "Update to modern language features",
                "indicators": ["f-strings", "list comprehensions", "context managers"],
                "benefits": ["cleaner code", "better performance"]
            },
            "extract_constant": {
                "description": "Extract magic numbers/strings to constants",
                "indicators": ["constant definition", "literal replacement"],
                "benefits": ["easier maintenance", "clearer intent"]
            }
        }
        
        # Code quality metrics
        self.quality_metrics = {
            "complexity": ["cyclomatic complexity", "nesting depth", "function length"],
            "readability": ["variable names", "function names", "comments"],
            "maintainability": ["duplication", "coupling", "cohesion"],
            "testability": ["function size", "dependencies", "side effects"]
        }
        
        self._load_evolution_history()
        self._load_learned_patterns()
    
    def _load_evolution_history(self):
        """Load code evolution history"""
        self.evolution_history = {}
        if self.evolution_file.exists():
            with open(self.evolution_file, 'r') as f:
                self.evolution_history = json.load(f)
    
    def _load_learned_patterns(self):
        """Load learned refactoring patterns"""
        self.learned_patterns = {}
        if self.patterns_file.exists():
            with open(self.patterns_file, 'r') as f:
                self.learned_patterns = json.load(f)
    
    def track_code_change(self, db: Session,
                         file_path: str,
                         before_code: str,
                         after_code: str,
                         change_description: str,
                         change_reason: str,
                         project_id: Optional[str] = None,
                         session_id: Optional[str] = None) -> Dict[str, Any]:
        """Track a code change with before/after comparison"""
        
        # Generate change ID
        change_id = hashlib.md5(
            f"{file_path}:{change_description}:{datetime.utcnow().isoformat()}".encode()
        ).hexdigest()[:12]
        
        # Calculate diff
        diff_lines = list(difflib.unified_diff(
            before_code.splitlines(keepends=True),
            after_code.splitlines(keepends=True),
            fromfile=f"{file_path} (before)",
            tofile=f"{file_path} (after)",
            lineterm=""
        ))
        diff_text = ''.join(diff_lines)
        
        # Analyze the change
        change_analysis = self._analyze_code_change(
            before_code, after_code, change_description, change_reason
        )
        
        # Detect refactoring patterns
        detected_patterns = self._detect_refactoring_patterns(
            before_code, after_code, diff_text
        )
        
        # Calculate quality metrics
        quality_before = self._calculate_quality_metrics(before_code)
        quality_after = self._calculate_quality_metrics(after_code)
        quality_improvement = self._calculate_quality_improvement(
            quality_before, quality_after
        )
        
        # Create evolution record
        evolution_record = {
            "change_id": change_id,
            "timestamp": datetime.utcnow().isoformat(),
            "file_path": file_path,
            "change_description": change_description,
            "change_reason": change_reason,
            "before_code": before_code,
            "after_code": after_code,
            "diff": diff_text,
            "lines_added": len([l for l in diff_lines if l.startswith('+') and not l.startswith('+++')]),
            "lines_removed": len([l for l in diff_lines if l.startswith('-') and not l.startswith('---')]),
            "change_analysis": change_analysis,
            "detected_patterns": detected_patterns,
            "quality_before": quality_before,
            "quality_after": quality_after,
            "quality_improvement": quality_improvement,
            "project_id": project_id,
            "session_id": session_id,
            "impact_measured": False,
            "success_rating": None
        }
        
        # Store in database
        content = self._format_evolution_content(evolution_record)
        memory = self._store_evolution_memory(db, content, evolution_record, project_id)
        
        # Update evolution history
        self._update_evolution_history(evolution_record)
        
        # Learn from patterns if improvement detected
        if quality_improvement["overall_improvement"] > 0:
            self._learn_from_successful_change(evolution_record)
        
        return {
            "change_id": change_id,
            "tracked": True,
            "patterns_detected": len(detected_patterns),
            "quality_improvement": quality_improvement["overall_improvement"],
            "lines_changed": evolution_record["lines_added"] + evolution_record["lines_removed"],
            "memory_id": str(memory.id)
        }
    
    def _analyze_code_change(self, before: str, after: str, 
                           description: str, reason: str) -> Dict[str, Any]:
        """Analyze the nature of a code change"""
        
        # Basic change metrics
        before_lines = before.count('\n')
        after_lines = after.count('\n')
        
        # AST analysis for structural changes
        structural_changes = self._analyze_structural_changes(before, after)
        
        # Categorize change type
        change_type = self._categorize_change_type(description, reason, structural_changes)
        
        # Assess change scope
        change_scope = self._assess_change_scope(before, after)
        
        return {
            "change_type": change_type,
            "change_scope": change_scope,
            "line_delta": after_lines - before_lines,
            "structural_changes": structural_changes,
            "complexity_change": "unknown",  # Will be calculated later
            "risk_level": self._assess_change_risk(change_scope, structural_changes)
        }
    
    def _analyze_structural_changes(self, before: str, after: str) -> Dict[str, Any]:
        """Analyze structural changes using AST parsing"""
        changes = {
            "functions_added": [],
            "functions_removed": [],
            "functions_modified": [],
            "classes_added": [],
            "classes_removed": [],
            "imports_added": [],
            "imports_removed": []
        }
        
        try:
            # Parse AST for both versions
            before_ast = ast.parse(before)
            after_ast = ast.parse(after)
            
            # Extract functions and classes
            before_funcs = self._extract_functions(before_ast)
            after_funcs = self._extract_functions(after_ast)
            
            before_classes = self._extract_classes(before_ast)
            after_classes = self._extract_classes(after_ast)
            
            before_imports = self._extract_imports(before_ast)
            after_imports = self._extract_imports(after_ast)
            
            # Compare functions
            changes["functions_added"] = list(set(after_funcs) - set(before_funcs))
            changes["functions_removed"] = list(set(before_funcs) - set(after_funcs))
            
            # Compare classes
            changes["classes_added"] = list(set(after_classes) - set(before_classes))
            changes["classes_removed"] = list(set(before_classes) - set(after_classes))
            
            # Compare imports
            changes["imports_added"] = list(set(after_imports) - set(before_imports))
            changes["imports_removed"] = list(set(before_imports) - set(after_imports))
            
        except SyntaxError:
            # If AST parsing fails, fall back to simple analysis
            changes["parse_error"] = True
        
        return changes
    
    def _extract_functions(self, tree: ast.AST) -> List[str]:
        """Extract function names from AST"""
        functions = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.append(node.name)
        return functions
    
    def _extract_classes(self, tree: ast.AST) -> List[str]:
        """Extract class names from AST"""
        classes = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes.append(node.name)
        return classes
    
    def _extract_imports(self, tree: ast.AST) -> List[str]:
        """Extract import statements from AST"""
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    imports.append(f"{module}.{alias.name}")
        return imports
    
    def _categorize_change_type(self, description: str, reason: str, 
                               structural: Dict[str, Any]) -> str:
        """Categorize the type of change"""
        desc_lower = description.lower()
        reason_lower = reason.lower()
        
        # Check for specific change types
        if any(word in desc_lower for word in ["refactor", "restructure", "reorganize"]):
            return "refactoring"
        elif any(word in desc_lower for word in ["fix", "bug", "error", "issue"]):
            return "bug_fix"
        elif any(word in desc_lower for word in ["add", "new", "implement", "feature"]):
            return "feature_addition"
        elif any(word in desc_lower for word in ["optimize", "performance", "speed"]):
            return "optimization"
        elif any(word in desc_lower for word in ["test", "testing", "spec"]):
            return "testing"
        elif any(word in desc_lower for word in ["doc", "comment", "readme"]):
            return "documentation"
        elif any(word in desc_lower for word in ["style", "format", "lint"]):
            return "style_improvement"
        elif structural.get("functions_added") or structural.get("classes_added"):
            return "feature_addition"
        elif structural.get("functions_removed") or structural.get("classes_removed"):
            return "code_removal"
        else:
            return "general_improvement"
    
    def _assess_change_scope(self, before: str, after: str) -> str:
        """Assess the scope/magnitude of changes"""
        before_lines = len(before.splitlines())
        after_lines = len(after.splitlines())
        
        # Calculate percentage change
        max_lines = max(before_lines, after_lines, 1)
        change_percentage = abs(after_lines - before_lines) / max_lines
        
        if change_percentage < 0.1:
            return "minor"
        elif change_percentage < 0.3:
            return "moderate" 
        elif change_percentage < 0.6:
            return "major"
        else:
            return "extensive"
    
    def _assess_change_risk(self, scope: str, structural: Dict[str, Any]) -> str:
        """Assess the risk level of the change"""
        risk_factors = 0
        
        # Scope-based risk
        scope_risk = {"minor": 0, "moderate": 1, "major": 2, "extensive": 3}
        risk_factors += scope_risk.get(scope, 1)
        
        # Structural change risk
        if structural.get("functions_removed") or structural.get("classes_removed"):
            risk_factors += 2
        if structural.get("imports_removed"):
            risk_factors += 1
        if len(structural.get("functions_added", [])) > 3:
            risk_factors += 1
        
        if risk_factors <= 1:
            return "low"
        elif risk_factors <= 3:
            return "medium"
        else:
            return "high"
    
    def _detect_refactoring_patterns(self, before: str, after: str, 
                                   diff: str) -> List[Dict[str, Any]]:
        """Detect common refactoring patterns"""
        detected = []
        
        # Extract method pattern
        if self._detect_extract_method(before, after):
            detected.append({
                "pattern": "extract_method",
                "confidence": 0.8,
                "description": "Method extraction detected"
            })
        
        # Remove duplication pattern
        if self._detect_remove_duplication(before, after):
            detected.append({
                "pattern": "remove_duplication",
                "confidence": 0.7,
                "description": "Code duplication removal detected"
            })
        
        # Add error handling pattern
        if self._detect_add_error_handling(before, after):
            detected.append({
                "pattern": "add_error_handling",
                "confidence": 0.9,
                "description": "Error handling addition detected"
            })
        
        # Improve typing pattern
        if self._detect_improve_typing(before, after):
            detected.append({
                "pattern": "improve_typing",
                "confidence": 0.85,
                "description": "Type annotation improvement detected"
            })
        
        # Modernize syntax pattern
        if self._detect_modernize_syntax(before, after):
            detected.append({
                "pattern": "modernize_syntax", 
                "confidence": 0.75,
                "description": "Syntax modernization detected"
            })
        
        return detected
    
    def _detect_extract_method(self, before: str, after: str) -> bool:
        """Detect extract method refactoring"""
        # Look for new function definitions and corresponding calls
        before_funcs = len(re.findall(r'def\s+\w+\s*\(', before))
        after_funcs = len(re.findall(r'def\s+\w+\s*\(', after))
        
        # More functions in after, and similar line count suggests extraction
        return after_funcs > before_funcs and abs(len(after.splitlines()) - len(before.splitlines())) < 10
    
    def _detect_remove_duplication(self, before: str, after: str) -> bool:
        """Detect code duplication removal"""
        # Simple heuristic: significant line reduction with new function
        line_reduction = len(before.splitlines()) - len(after.splitlines())
        new_functions = len(re.findall(r'def\s+\w+\s*\(', after)) > len(re.findall(r'def\s+\w+\s*\(', before))
        
        return line_reduction > 5 and new_functions
    
    def _detect_add_error_handling(self, before: str, after: str) -> bool:
        """Detect error handling addition"""
        before_try = before.count('try:')
        after_try = after.count('try:')
        
        before_except = before.count('except')
        after_except = after.count('except')
        
        return (after_try > before_try) or (after_except > before_except)
    
    def _detect_improve_typing(self, before: str, after: str) -> bool:
        """Detect type annotation improvements"""
        before_typing = before.count('->') + before.count(': ')
        after_typing = after.count('->') + after.count(': ')
        
        typing_imports = 'from typing import' in after and 'from typing import' not in before
        
        return after_typing > before_typing or typing_imports
    
    def _detect_modernize_syntax(self, before: str, after: str) -> bool:
        """Detect syntax modernization"""
        # Look for f-strings, list comprehensions, etc.
        modern_features = [
            (r'f["\']', "f-strings"),
            (r'\[.*for.*in.*\]', "list comprehensions"),
            (r'with\s+open\(', "context managers")
        ]
        
        improvements = 0
        for pattern, _ in modern_features:
            before_count = len(re.findall(pattern, before))
            after_count = len(re.findall(pattern, after))
            if after_count > before_count:
                improvements += 1
        
        return improvements > 0
    
    def _calculate_quality_metrics(self, code: str) -> Dict[str, Any]:
        """Calculate code quality metrics"""
        lines = code.splitlines()
        
        metrics = {
            "total_lines": len(lines),
            "non_empty_lines": len([l for l in lines if l.strip()]),
            "comment_lines": len([l for l in lines if l.strip().startswith('#')]),
            "function_count": len(re.findall(r'def\s+\w+\s*\(', code)),
            "class_count": len(re.findall(r'class\s+\w+\s*[:(]', code)),
            "complexity_estimate": self._estimate_complexity(code),
            "avg_line_length": sum(len(l) for l in lines) / max(len(lines), 1),
            "max_line_length": max((len(l) for l in lines), default=0),
            "has_docstrings": '"""' in code or "'''" in code,
            "has_type_hints": '->' in code or ': ' in code
        }
        
        return metrics
    
    def _estimate_complexity(self, code: str) -> int:
        """Estimate cyclomatic complexity"""
        # Simple complexity estimation
        complexity_keywords = ['if', 'elif', 'else', 'for', 'while', 'try', 'except', 'with']
        complexity = 1  # Base complexity
        
        for keyword in complexity_keywords:
            complexity += len(re.findall(rf'\b{keyword}\b', code))
        
        return complexity
    
    def _calculate_quality_improvement(self, before: Dict[str, Any], 
                                     after: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate quality improvement metrics"""
        
        improvements = {}
        
        # Calculate percentage changes
        for metric in ["complexity_estimate", "avg_line_length", "max_line_length"]:
            before_val = before.get(metric, 0)
            after_val = after.get(metric, 0)
            
            if before_val > 0:
                change = (after_val - before_val) / before_val
                improvements[f"{metric}_change"] = change
        
        # Boolean improvements
        improvements["docstring_added"] = after["has_docstrings"] and not before["has_docstrings"]
        improvements["type_hints_added"] = after["has_type_hints"] and not before["has_type_hints"]
        
        # Calculate overall improvement score
        overall_score = 0
        
        # Lower complexity is better
        if improvements.get("complexity_estimate_change", 0) < 0:
            overall_score += 0.3
        
        # Shorter lines are generally better
        if improvements.get("avg_line_length_change", 0) < 0:
            overall_score += 0.1
        
        # Added documentation
        if improvements["docstring_added"]:
            overall_score += 0.2
        
        # Added type hints
        if improvements["type_hints_added"]:
            overall_score += 0.2
        
        # More functions (modularity)
        func_change = after["function_count"] - before["function_count"]
        if func_change > 0:
            overall_score += min(0.2, func_change * 0.05)
        
        improvements["overall_improvement"] = overall_score
        
        return improvements
    
    def _format_evolution_content(self, evolution: Dict[str, Any]) -> str:
        """Format evolution record for storage"""
        lines = [
            f"CODE EVOLUTION: {evolution['change_description']}",
            f"File: {evolution['file_path']}",
            f"Type: {evolution['change_analysis']['change_type']}",
            f"Scope: {evolution['change_analysis']['change_scope']}",
            f"Risk: {evolution['change_analysis']['risk_level']}",
            f"",
            f"Reason: {evolution['change_reason']}",
            f"",
            f"Changes: +{evolution['lines_added']} -{evolution['lines_removed']} lines"
        ]
        
        if evolution['detected_patterns']:
            lines.extend(["", "Patterns detected:"])
            for pattern in evolution['detected_patterns']:
                lines.append(f"  - {pattern['pattern']}: {pattern['description']} ({pattern['confidence']:.0%} confidence)")
        
        quality_imp = evolution['quality_improvement']
        if quality_imp['overall_improvement'] > 0:
            lines.extend(["", f"Quality improvement: {quality_imp['overall_improvement']:.0%}"])
            if quality_imp.get('docstring_added'):
                lines.append("  + Documentation added")
            if quality_imp.get('type_hints_added'):
                lines.append("  + Type hints added")
        
        return "\n".join(lines)
    
    def _store_evolution_memory(self, db: Session, content: str,
                              evolution_data: Dict[str, Any],
                              project_id: Optional[str]) -> MemoryItem:
        """Store evolution record in memory"""
        tags = ["code_evolution", evolution_data["change_analysis"]["change_type"]]
        if project_id:
            tags.append(f"project:{project_id}")
        
        # Add pattern tags
        for pattern in evolution_data["detected_patterns"]:
            tags.append(f"pattern:{pattern['pattern']}")
        
        memory_hash = hashlib.sha256(content.encode()).hexdigest()
        
        # Check if memory with this hash already exists
        existing_memory = db.query(MemoryItem).filter(
            MemoryItem.content_hash == memory_hash
        ).first()
        
        # Calculate importance based on quality improvement and patterns
        base_importance = 0.6
        quality_bonus = min(0.3, evolution_data["quality_improvement"]["overall_improvement"])
        pattern_bonus = min(0.1, len(evolution_data["detected_patterns"]) * 0.05)
        
        importance = base_importance + quality_bonus + pattern_bonus
        
        if existing_memory:
            # Update existing memory
            existing_memory.access_count += 1
            existing_memory.accessed_at = datetime.utcnow()
            existing_memory.updated_at = datetime.utcnow()
            # Update metadata with latest evolution data
            existing_memory.meta_data = {
                "memory_type": "code_evolution",
                "importance": importance,
                **evolution_data
            }
            # Merge tags
            existing_tags = set(existing_memory.tags or [])
            existing_memory.tags = list(existing_tags.union(set(tags)))
            db.commit()
            db.refresh(existing_memory)
            memory = existing_memory
        else:
            # Create new memory
            memory = MemoryItem(
                content=content,
                content_hash=memory_hash,
                tags=tags,
                meta_data={
                    "memory_type": "code_evolution",
                    "importance": importance,
                    **evolution_data
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
                        memory = existing
                else:
                    raise
        
        return memory
    
    def _update_evolution_history(self, evolution: Dict[str, Any]):
        """Update local evolution history"""
        file_path = evolution["file_path"]
        
        if file_path not in self.evolution_history:
            self.evolution_history[file_path] = {"changes": []}
        
        # Store summary of change
        change_summary = {
            "change_id": evolution["change_id"],
            "timestamp": evolution["timestamp"],
            "description": evolution["change_description"],
            "type": evolution["change_analysis"]["change_type"],
            "quality_improvement": evolution["quality_improvement"]["overall_improvement"],
            "patterns": [p["pattern"] for p in evolution["detected_patterns"]]
        }
        
        self.evolution_history[file_path]["changes"].append(change_summary)
        
        # Keep only last 50 changes per file
        self.evolution_history[file_path]["changes"] = \
            self.evolution_history[file_path]["changes"][-50:]
        
        # Save to file
        with open(self.evolution_file, 'w') as f:
            json.dump(self.evolution_history, f, indent=2)
    
    def _learn_from_successful_change(self, evolution: Dict[str, Any]):
        """Learn patterns from successful code changes"""
        change_type = evolution["change_analysis"]["change_type"]
        patterns = evolution["detected_patterns"]
        quality_imp = evolution["quality_improvement"]["overall_improvement"]
        
        if change_type not in self.learned_patterns:
            self.learned_patterns[change_type] = {
                "successful_patterns": defaultdict(int),
                "quality_improvements": [],
                "common_reasons": defaultdict(int)
            }
        
        # Track successful patterns
        for pattern in patterns:
            self.learned_patterns[change_type]["successful_patterns"][pattern["pattern"]] += 1
        
        # Track quality improvements
        self.learned_patterns[change_type]["quality_improvements"].append(quality_imp)
        
        # Track common reasons
        reason_words = evolution["change_reason"].lower().split()
        for word in reason_words:
            if len(word) > 3:  # Skip short words
                self.learned_patterns[change_type]["common_reasons"][word] += 1
        
        # Save learned patterns
        with open(self.patterns_file, 'w') as f:
            # Convert defaultdict to regular dict for JSON serialization
            serializable_patterns = {}
            for change_type, data in self.learned_patterns.items():
                serializable_patterns[change_type] = {
                    "successful_patterns": dict(data["successful_patterns"]),
                    "quality_improvements": data["quality_improvements"],
                    "common_reasons": dict(data["common_reasons"])
                }
            json.dump(serializable_patterns, f, indent=2)
    
    def get_evolution_history(self, db: Session, file_path: Optional[str] = None,
                            project_id: Optional[str] = None,
                            change_type: Optional[str] = None,
                            limit: int = 10) -> List[Dict[str, Any]]:
        """Get code evolution history"""
        query = db.query(MemoryItem).filter(
            cast(MemoryItem.meta_data, String).contains('"memory_type": "code_evolution"')
        )
        
        if file_path:
            query = query.filter(
                cast(MemoryItem.meta_data, String).contains(f'"file_path": "{file_path}"')
            )
        
        if project_id:
            query = query.filter(
                cast(MemoryItem.meta_data, String).contains(f'"project_id": "{project_id}"')
            )
        
        if change_type:
            query = query.filter(
                cast(MemoryItem.meta_data, String).contains(f'"change_type": "{change_type}"')
            )
        
        records = query.order_by(desc(MemoryItem.created_at)).limit(limit).all()
        
        history = []
        for record in records:
            meta = record.meta_data
            history.append({
                "change_id": meta.get("change_id"),
                "timestamp": meta.get("timestamp"),
                "file_path": meta.get("file_path"),
                "description": meta.get("change_description"),
                "type": meta.get("change_analysis", {}).get("change_type"),
                "scope": meta.get("change_analysis", {}).get("change_scope"),
                "patterns": [p["pattern"] for p in meta.get("detected_patterns", [])],
                "quality_improvement": meta.get("quality_improvement", {}).get("overall_improvement", 0),
                "lines_changed": meta.get("lines_added", 0) + meta.get("lines_removed", 0)
            })
        
        return history
    
    def get_refactoring_suggestions(self, db: Session, code: str,
                                  file_path: str,
                                  project_id: Optional[str] = None) -> Dict[str, Any]:
        """Get refactoring suggestions based on learned patterns"""
        
        # Analyze current code quality
        current_metrics = self._calculate_quality_metrics(code)
        
        # Find similar successful changes
        similar_changes = self._find_similar_successful_changes(
            db, file_path, current_metrics, project_id
        )
        
        suggestions = {
            "current_metrics": current_metrics,
            "improvement_opportunities": [],
            "based_on_history": similar_changes[:3],
            "pattern_recommendations": []
        }
        
        # Analyze improvement opportunities
        if current_metrics["complexity_estimate"] > 10:
            suggestions["improvement_opportunities"].append({
                "type": "complexity",
                "description": "High complexity detected - consider extracting methods",
                "suggested_pattern": "extract_method",
                "confidence": 0.7
            })
        
        if current_metrics["max_line_length"] > 100:
            suggestions["improvement_opportunities"].append({
                "type": "readability",
                "description": "Long lines detected - consider breaking them up",
                "suggested_pattern": "improve_formatting",
                "confidence": 0.6
            })
        
        if not current_metrics["has_type_hints"]:
            suggestions["improvement_opportunities"].append({
                "type": "typing",
                "description": "No type hints detected - consider adding them",
                "suggested_pattern": "improve_typing",
                "confidence": 0.8
            })
        
        if not current_metrics["has_docstrings"]:
            suggestions["improvement_opportunities"].append({
                "type": "documentation",
                "description": "No docstrings detected - consider adding documentation",
                "suggested_pattern": "add_documentation",
                "confidence": 0.9
            })
        
        # Get pattern recommendations from learned patterns
        for change_type, data in self.learned_patterns.items():
            if data["successful_patterns"]:
                most_successful = max(data["successful_patterns"].items(), 
                                    key=lambda x: x[1])
                avg_improvement = sum(data["quality_improvements"]) / len(data["quality_improvements"])
                
                if avg_improvement > 0.1:  # Only recommend if historically successful
                    suggestions["pattern_recommendations"].append({
                        "pattern": most_successful[0],
                        "change_type": change_type,
                        "success_count": most_successful[1],
                        "avg_improvement": avg_improvement,
                        "description": self.refactoring_patterns.get(
                            most_successful[0], {}
                        ).get("description", "Unknown pattern")
                    })
        
        return suggestions
    
    def _find_similar_successful_changes(self, db: Session, file_path: str,
                                       current_metrics: Dict[str, Any],
                                       project_id: Optional[str]) -> List[Dict[str, Any]]:
        """Find similar successful changes for reference"""
        
        # Query for successful changes (quality improvement > 0)
        query = db.query(MemoryItem).filter(
            and_(
                cast(MemoryItem.meta_data, String).contains('"memory_type": "code_evolution"'),
                cast(MemoryItem.meta_data, String).contains('"overall_improvement"')
            )
        )
        
        if project_id:
            query = query.filter(
                cast(MemoryItem.meta_data, String).contains(f'"project_id": "{project_id}"')
            )
        
        records = query.order_by(desc(MemoryItem.created_at)).limit(20).all()
        
        similar_changes = []
        for record in records:
            meta = record.meta_data
            quality_imp = meta.get("quality_improvement", {}).get("overall_improvement", 0)
            
            if quality_imp > 0.1:  # Only include significant improvements
                similar_changes.append({
                    "change_id": meta.get("change_id"),
                    "file_path": meta.get("file_path"),
                    "description": meta.get("change_description"),
                    "type": meta.get("change_analysis", {}).get("change_type"),
                    "patterns": [p["pattern"] for p in meta.get("detected_patterns", [])],
                    "quality_improvement": quality_imp,
                    "reason": meta.get("change_reason")
                })
        
        return sorted(similar_changes, key=lambda x: x["quality_improvement"], reverse=True)
    
    def compare_code_versions(self, db: Session, change_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed before/after comparison for a specific change"""
        
        # Find the change record
        change_memory = db.query(MemoryItem).filter(
            cast(MemoryItem.meta_data, String).contains(f'"change_id": "{change_id}"')
        ).first()
        
        if not change_memory:
            return None
        
        meta = change_memory.meta_data
        
        return {
            "change_id": change_id,
            "file_path": meta.get("file_path"),
            "description": meta.get("change_description"),
            "reason": meta.get("change_reason"),
            "timestamp": meta.get("timestamp"),
            "before_code": meta.get("before_code"),
            "after_code": meta.get("after_code"),
            "diff": meta.get("diff"),
            "analysis": meta.get("change_analysis"),
            "patterns": meta.get("detected_patterns"),
            "quality_before": meta.get("quality_before"),
            "quality_after": meta.get("quality_after"),
            "quality_improvement": meta.get("quality_improvement")
        }
    
    def get_pattern_analytics(self, project_id: Optional[str] = None) -> Dict[str, Any]:
        """Get analytics on refactoring patterns and code evolution"""
        
        analytics = {
            "learned_patterns": dict(self.learned_patterns),
            "pattern_success_rates": {},
            "common_improvement_types": defaultdict(int),
            "quality_trends": []
        }
        
        # Calculate pattern success rates
        for change_type, data in self.learned_patterns.items():
            total_attempts = sum(data["successful_patterns"].values())
            if total_attempts > 0:
                avg_improvement = sum(data["quality_improvements"]) / len(data["quality_improvements"])
                analytics["pattern_success_rates"][change_type] = {
                    "total_attempts": total_attempts,
                    "avg_improvement": avg_improvement,
                    "most_common_pattern": max(data["successful_patterns"].items(), 
                                             key=lambda x: x[1])[0] if data["successful_patterns"] else None
                }
        
        # Analyze common improvement types
        for change_type, data in self.learned_patterns.items():
            analytics["common_improvement_types"][change_type] = len(data["quality_improvements"])
        
        return analytics
    
    def update_change_impact(self, db: Session, change_id: str,
                           success_rating: float,
                           impact_notes: str,
                           performance_impact: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Update a change record with measured impact"""
        
        # Find and update the change record
        change_memory = db.query(MemoryItem).filter(
            cast(MemoryItem.meta_data, String).contains(f'"change_id": "{change_id}"')
        ).first()
        
        if not change_memory:
            return {"updated": False, "error": "Change not found"}
        
        # Update metadata
        meta_data = change_memory.meta_data
        meta_data["impact_measured"] = True
        meta_data["success_rating"] = success_rating
        meta_data["impact_notes"] = impact_notes
        if performance_impact:
            meta_data["performance_impact"] = performance_impact
        
        change_memory.meta_data = meta_data
        change_memory.updated_at = datetime.utcnow()
        db.commit()
        
        # Learn from the impact measurement
        self._learn_from_impact_measurement(meta_data, success_rating)
        
        return {
            "updated": True,
            "change_id": change_id,
            "success_rating": success_rating,
            "impact_measured": True
        }
    
    def _learn_from_impact_measurement(self, meta_data: Dict[str, Any], 
                                     success_rating: float):
        """Learn from measured impact of changes"""
        change_type = meta_data.get("change_analysis", {}).get("change_type")
        patterns = meta_data.get("detected_patterns", [])
        
        if change_type and success_rating >= 0.7:  # Consider successful if rating >= 0.7
            # Update learned patterns with success confirmation
            if change_type not in self.learned_patterns:
                self.learned_patterns[change_type] = {
                    "successful_patterns": defaultdict(int),
                    "quality_improvements": [],
                    "common_reasons": defaultdict(int)
                }
            
            # Boost pattern confidence
            for pattern in patterns:
                pattern_name = pattern["pattern"]
                # Increase count to reflect confirmed success
                self.learned_patterns[change_type]["successful_patterns"][pattern_name] += 1
            
            # Save updated patterns
            with open(self.patterns_file, 'w') as f:
                serializable_patterns = {}
                for ct, data in self.learned_patterns.items():
                    serializable_patterns[ct] = {
                        "successful_patterns": dict(data["successful_patterns"]),
                        "quality_improvements": data["quality_improvements"],
                        "common_reasons": dict(data["common_reasons"])
                    }
                json.dump(serializable_patterns, f, indent=2)
    
    def track_simple_change(self, db: Session, file_path: str, 
                           change_type: str, description: str,
                           user_id: str) -> Dict[str, Any]:
        """Track a simple code change without before/after comparison"""
        
        # Create a memory for the change
        content = f"Code change in {file_path}: {description}"
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        
        # Check if memory with this hash already exists
        existing_memory = db.query(MemoryItem).filter(
            MemoryItem.content_hash == content_hash
        ).first()
        
        if existing_memory:
            # Update existing memory
            existing_memory.access_count += 1
            existing_memory.accessed_at = datetime.utcnow()
            existing_memory.updated_at = datetime.utcnow()
            # Update metadata
            existing_memory.meta_data = {
                "memory_type": "code_evolution",
                "file_path": file_path,
                "change_type": change_type,
                "change_description": description,
                "user_id": user_id,
                "source": "code_evolution",
                "importance": 0.6,
                "simple_tracking": True
            }
            db.commit()
            memory = existing_memory
        else:
            # Create new memory
            memory = MemoryItem(
                content=content,
                content_hash=content_hash,
                tags=["code_evolution", f"change_type:{change_type}", f"file:{file_path}"],
                meta_data={
                    "memory_type": "code_evolution",
                    "file_path": file_path,
                    "change_type": change_type,
                    "change_description": description,
                    "user_id": user_id,
                    "source": "code_evolution",
                    "importance": 0.6,
                    "simple_tracking": True
                }
            )
            
            try:
                db.add(memory)
                db.commit()
            except Exception as e:
                db.rollback()
                # If there's still a duplicate key error, find and update the existing record
                if "duplicate key" in str(e).lower():
                    existing = db.query(MemoryItem).filter(
                        MemoryItem.content_hash == content_hash
                    ).first()
                    if existing:
                        existing.access_count += 1
                        existing.accessed_at = datetime.utcnow()
                        db.commit()
                        memory = existing
                else:
                    raise
        
        return {
            "tracked": True,
            "change_id": str(memory.id),
            "file_path": file_path,
            "change_type": change_type
        }