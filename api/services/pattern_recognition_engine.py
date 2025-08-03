"""
Advanced Pattern Recognition Engine
Uses ML models to identify coding patterns, anti-patterns, and best practices
"""

import asyncio
import logging
import json
import ast
import re
from typing import Dict, Any, List, Optional, Tuple, Set
from datetime import datetime
from collections import defaultdict, Counter
import hashlib

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class CodePattern(BaseModel):
    """Represents a detected code pattern"""
    pattern_id: str = Field(default_factory=lambda: hashlib.md5(str(datetime.utcnow()).encode()).hexdigest())
    pattern_type: str
    name: str
    description: str
    examples: List[Dict[str, Any]]
    confidence: float = Field(ge=0.0, le=1.0)
    frequency: int
    metadata: Dict[str, Any] = Field(default_factory=dict)
    detected_at: datetime = Field(default_factory=datetime.utcnow)


class PatternCategory(str):
    """Categories of patterns we can detect"""
    DESIGN_PATTERN = "design_pattern"
    ANTI_PATTERN = "anti_pattern"
    CODE_SMELL = "code_smell"
    BEST_PRACTICE = "best_practice"
    SECURITY_PATTERN = "security_pattern"
    PERFORMANCE_PATTERN = "performance_pattern"
    ERROR_PATTERN = "error_pattern"
    REFACTORING_OPPORTUNITY = "refactoring_opportunity"


class PatternRecognitionEngine:
    """
    Advanced pattern recognition engine that:
    - Analyzes code structure using AST
    - Identifies common patterns and anti-patterns
    - Learns from code evolution
    - Suggests improvements based on patterns
    """
    
    def __init__(self):
        self.patterns_db: Dict[str, CodePattern] = {}
        self.vectorizer = TfidfVectorizer(
            analyzer='word',
            token_pattern=r'\b\w+\b',
            max_features=1000,
            ngram_range=(1, 3)
        )
        self.pattern_embeddings = {}
        self.learned_patterns = defaultdict(list)
        
        # Initialize with known patterns
        self._initialize_known_patterns()
        
    def _initialize_known_patterns(self):
        """Initialize with common design patterns and anti-patterns"""
        known_patterns = [
            {
                "type": PatternCategory.DESIGN_PATTERN,
                "name": "Singleton",
                "description": "Ensures a class has only one instance",
                "indicators": ["__instance", "_instance", "__new__", "instance is None"]
            },
            {
                "type": PatternCategory.DESIGN_PATTERN,
                "name": "Factory",
                "description": "Creates objects without specifying exact classes",
                "indicators": ["create_", "factory", "build_", "make_"]
            },
            {
                "type": PatternCategory.ANTI_PATTERN,
                "name": "God Class",
                "description": "Class with too many responsibilities",
                "indicators": ["class with >20 methods", "class with >500 lines"]
            },
            {
                "type": PatternCategory.CODE_SMELL,
                "name": "Long Method",
                "description": "Method that is too long and complex",
                "indicators": ["function with >50 lines", "cyclomatic complexity >10"]
            },
            {
                "type": PatternCategory.SECURITY_PATTERN,
                "name": "SQL Injection Risk",
                "description": "Direct string concatenation in SQL queries",
                "indicators": ["execute(.*\\+.*)", "execute(.*%.*)", "execute(.*format.*)"]
            },
            {
                "type": PatternCategory.BEST_PRACTICE,
                "name": "Context Manager",
                "description": "Using with statement for resource management",
                "indicators": ["with open", "__enter__", "__exit__", "contextmanager"]
            }
        ]
        
        for pattern_def in known_patterns:
            pattern = CodePattern(
                pattern_type=pattern_def["type"],
                name=pattern_def["name"],
                description=pattern_def["description"],
                examples=[],
                confidence=1.0,
                frequency=0,
                metadata={"indicators": pattern_def["indicators"]}
            )
            self.patterns_db[pattern.pattern_id] = pattern
            
    async def analyze_code(self, code: str, language: str = "python") -> List[CodePattern]:
        """Analyze code to detect patterns"""
        detected_patterns = []
        
        if language == "python":
            detected_patterns.extend(await self._analyze_python_code(code))
        else:
            # Generic pattern detection for other languages
            detected_patterns.extend(await self._analyze_generic_code(code))
            
        # Use ML to find similar patterns
        if detected_patterns:
            similar_patterns = await self._find_similar_patterns(code)
            detected_patterns.extend(similar_patterns)
            
        return detected_patterns
        
    async def _analyze_python_code(self, code: str) -> List[CodePattern]:
        """Analyze Python code using AST"""
        detected = []
        
        try:
            tree = ast.parse(code)
            
            # Analyze AST for patterns
            analyzer = PythonPatternAnalyzer()
            ast_patterns = analyzer.analyze(tree, code)
            
            # Convert AST patterns to CodePattern objects
            for pattern_info in ast_patterns:
                if pattern_info["name"] == "God Class":
                    description = f"Class '{pattern_info['class']}' has too many responsibilities: "
                    description += f"{pattern_info['method_count']} methods"
                    if 'attribute_count' in pattern_info:
                        description += f", {pattern_info['attribute_count']} attributes"
                    pattern = CodePattern(
                        pattern_type=PatternCategory.ANTI_PATTERN,
                        name="God Class",
                        description=description,
                        examples=[{"class": pattern_info["class"]}],
                        confidence=pattern_info["confidence"],
                        frequency=1,
                        metadata=pattern_info
                    )
                    detected.append(pattern)
                elif pattern_info["name"] == "Long Method":
                    pattern = CodePattern(
                        pattern_type=PatternCategory.CODE_SMELL,
                        name="Long Method",
                        description=f"Method with {pattern_info['lines']} lines (>50)",
                        examples=[{"function": pattern_info["function"]}],
                        confidence=pattern_info["confidence"],
                        frequency=1,
                        metadata=pattern_info
                    )
                    detected.append(pattern)
            
            # Also check for known patterns
            for line_num, line in enumerate(code.split('\n'), 1):
                # Context manager pattern
                if 'with open' in line or '__enter__' in line or '__exit__' in line:
                    pattern = CodePattern(
                        pattern_type=PatternCategory.BEST_PRACTICE,
                        name="Context Manager",
                        description="Using context manager for resource management",
                        examples=[{"line": line_num, "code": line.strip()}],
                        confidence=0.9,
                        frequency=1
                    )
                    detected.append(pattern)
                    
        except SyntaxError as e:
            logger.warning(f"Syntax error in code analysis: {e}")
            
        return detected
        
    async def _analyze_generic_code(self, code: str) -> List[CodePattern]:
        """Generic pattern detection for any language"""
        detected = []
        lines = code.split('\n')
        
        # Simple heuristics
        patterns_found = []
        
        # Long function detection
        function_lines = 0
        in_function = False
        
        for i, line in enumerate(lines):
            # Detect function start (generic)
            if re.match(r'^\s*(def|function|func|void|int|public|private)\s+\w+\s*\(', line):
                in_function = True
                function_lines = 0
            elif in_function:
                function_lines += 1
                if re.match(r'^\s*}?\s*$', line) or function_lines > 50:
                    if function_lines > 50:
                        patterns_found.append({
                            "type": PatternCategory.CODE_SMELL,
                            "name": "Long Method",
                            "line": i - function_lines,
                            "confidence": 0.9
                        })
                    in_function = False
                    
        # Convert found patterns to CodePattern objects
        for p in patterns_found:
            pattern = CodePattern(
                pattern_type=p["type"],
                name=p["name"],
                description=f"Detected {p['name']} pattern",
                examples=[{"line": p["line"]}],
                confidence=p["confidence"],
                frequency=1
            )
            detected.append(pattern)
            
        return detected
        
    def _matches_pattern(self, pattern_info: Dict, known_pattern: CodePattern) -> bool:
        """Check if detected pattern matches a known pattern"""
        indicators = known_pattern.metadata.get("indicators", [])
        
        for indicator in indicators:
            if isinstance(indicator, str):
                # Simple string matching
                if indicator in str(pattern_info.get("code_snippet", "")):
                    return True
                # Regex matching
                if re.search(indicator, str(pattern_info.get("code_snippet", ""))):
                    return True
                    
        return False
        
    async def _find_similar_patterns(self, code: str) -> List[CodePattern]:
        """Use ML to find patterns similar to ones we've seen before"""
        similar = []
        
        try:
            # Extract features from code
            if not hasattr(self.vectorizer, 'vocabulary_'):
                # Fit vectorizer if not already fitted
                if self.learned_patterns:
                    all_code = [p["code"] for patterns in self.learned_patterns.values() for p in patterns]
                    self.vectorizer.fit(all_code)
                else:
                    return similar
                    
            code_vector = self.vectorizer.transform([code])
            
            # Compare with learned patterns
            for pattern_type, examples in self.learned_patterns.items():
                if examples:
                    example_vectors = self.vectorizer.transform([e["code"] for e in examples])
                    similarities = cosine_similarity(code_vector, example_vectors)[0]
                    
                    # If highly similar to known examples
                    if max(similarities) > 0.8:
                        pattern = CodePattern(
                            pattern_type=pattern_type,
                            name=f"Learned {pattern_type}",
                            description=f"Similar to previously seen {pattern_type}",
                            examples=[{"similarity": float(max(similarities))}],
                            confidence=float(max(similarities)),
                            frequency=len(examples)
                        )
                        similar.append(pattern)
                        
        except Exception as e:
            logger.error(f"Error in ML pattern matching: {e}")
            
        return similar
        
    async def learn_pattern(self, code: str, pattern_type: str, metadata: Dict[str, Any] = None):
        """Learn a new pattern from code"""
        self.learned_patterns[pattern_type].append({
            "code": code,
            "metadata": metadata or {},
            "learned_at": datetime.utcnow().isoformat()
        })
        
        # Keep only recent patterns
        if len(self.learned_patterns[pattern_type]) > 100:
            self.learned_patterns[pattern_type] = self.learned_patterns[pattern_type][-100:]
            
        logger.info(f"Learned new {pattern_type} pattern")
        
    async def suggest_improvements(self, patterns: List[CodePattern]) -> List[Dict[str, Any]]:
        """Suggest improvements based on detected patterns"""
        suggestions = []
        
        for pattern in patterns:
            if pattern.pattern_type == PatternCategory.ANTI_PATTERN:
                suggestions.append({
                    "pattern": pattern.name,
                    "severity": "high",
                    "suggestion": f"Refactor to eliminate {pattern.name}",
                    "confidence": pattern.confidence
                })
            elif pattern.pattern_type == PatternCategory.CODE_SMELL:
                suggestions.append({
                    "pattern": pattern.name,
                    "severity": "medium",
                    "suggestion": f"Consider refactoring {pattern.name}",
                    "confidence": pattern.confidence
                })
            elif pattern.pattern_type == PatternCategory.SECURITY_PATTERN:
                suggestions.append({
                    "pattern": pattern.name,
                    "severity": "critical",
                    "suggestion": f"Security issue: {pattern.description}",
                    "confidence": pattern.confidence
                })
                
        return suggestions
        
    async def evolve_patterns(self, feedback: Dict[str, Any]):
        """Evolve pattern detection based on feedback"""
        pattern_id = feedback.get("pattern_id")
        was_correct = feedback.get("correct", True)
        
        if pattern_id in self.patterns_db:
            pattern = self.patterns_db[pattern_id]
            
            # Adjust confidence based on feedback
            if was_correct:
                pattern.confidence = min(1.0, pattern.confidence + 0.05)
            else:
                pattern.confidence = max(0.0, pattern.confidence - 0.1)
                
            # Remove pattern if confidence too low
            if pattern.confidence < 0.3:
                del self.patterns_db[pattern_id]
                logger.info(f"Removed low-confidence pattern: {pattern.name}")


class PythonPatternAnalyzer(ast.NodeVisitor):
    """AST visitor for Python pattern analysis"""
    
    def __init__(self):
        self.patterns = []
        self.current_class = None
        self.class_methods = defaultdict(list)
        self.class_attributes = defaultdict(list)
        self.function_lengths = {}
        self.imports = []
        
    def analyze(self, tree: ast.AST, source_code: str) -> List[Dict[str, Any]]:
        """Analyze AST and return detected patterns"""
        self.source_lines = source_code.split('\n')
        self.visit(tree)
        
        # Analyze collected data for patterns
        patterns = []
        
        # Check for God Class (too many methods OR too many attributes)
        for class_name, methods in self.class_methods.items():
            attributes = self.class_attributes.get(class_name, [])
            total_members = len(methods) + len(attributes)
            
            if len(methods) > 20 or len(attributes) > 10 or total_members > 15:
                patterns.append({
                    "type": "anti_pattern",
                    "name": "God Class",
                    "class": class_name,
                    "method_count": len(methods),
                    "attribute_count": len(attributes),
                    "total_members": total_members,
                    "confidence": 0.9
                })
                
        # Check for Long Methods
        for func_name, length in self.function_lengths.items():
            if length > 50:
                patterns.append({
                    "type": "code_smell",
                    "name": "Long Method",
                    "function": func_name,
                    "lines": length,
                    "confidence": 0.95
                })
                
        return patterns
        
    def visit_ClassDef(self, node):
        """Visit class definitions"""
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = None
        
    def visit_FunctionDef(self, node):
        """Visit function definitions"""
        if self.current_class:
            self.class_methods[self.current_class].append(node.name)
            
            # Track attributes in __init__
            if node.name == "__init__":
                for stmt in ast.walk(node):
                    if isinstance(stmt, ast.Assign):
                        for target in stmt.targets:
                            if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "self":
                                self.class_attributes[self.current_class].append(target.attr)
            
        # Calculate function length
        if hasattr(node, 'end_lineno') and hasattr(node, 'lineno'):
            length = node.end_lineno - node.lineno
            self.function_lengths[node.name] = length
            
        self.generic_visit(node)
        
    def visit_Import(self, node):
        """Track imports"""
        for alias in node.names:
            self.imports.append(alias.name)
        self.generic_visit(node)
        
    def visit_ImportFrom(self, node):
        """Track from imports"""
        module = node.module or ""
        for alias in node.names:
            self.imports.append(f"{module}.{alias.name}")
        self.generic_visit(node)


# Singleton instance
_pattern_engine: Optional[PatternRecognitionEngine] = None


async def get_pattern_engine() -> PatternRecognitionEngine:
    """Get or create the pattern recognition engine"""
    global _pattern_engine
    
    if _pattern_engine is None:
        _pattern_engine = PatternRecognitionEngine()
        
    return _pattern_engine