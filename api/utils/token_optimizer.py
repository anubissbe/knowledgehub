#!/usr/bin/env python3
"""
Token Optimization Module - Reduce token usage while preserving meaning
"""

import re
import json
import hashlib
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import tiktoken
from datetime import datetime


class TokenOptimizer:
    """
    Optimize content to reduce token usage while maintaining semantic meaning
    """
    
    def __init__(self, model: str = "gpt-4"):
        self.encoder = tiktoken.encoding_for_model(model)
        self.reference_cache = {}
        self.abbreviations = self._load_abbreviations()
        
    def optimize(self, content: str, context: Optional[Dict] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Optimize content and return optimized version with metadata
        
        Returns:
            (optimized_content, optimization_metadata)
        """
        original_tokens = self.count_tokens(content)
        
        # Apply optimization strategies
        optimized = content
        metadata = {
            "original_tokens": original_tokens,
            "strategies_applied": []
        }
        
        # 1. Remove redundant whitespace
        optimized = self._normalize_whitespace(optimized)
        if optimized != content:
            metadata["strategies_applied"].append("whitespace_normalization")
        
        # 2. Extract and reference repeated content
        optimized, refs = self._extract_references(optimized)
        if refs:
            metadata["strategies_applied"].append("reference_extraction")
            metadata["references"] = refs
        
        # 3. Compress verbose patterns
        optimized = self._compress_patterns(optimized)
        if optimized != content:
            metadata["strategies_applied"].append("pattern_compression")
        
        # 4. Use abbreviations for common terms
        optimized, abbrevs = self._apply_abbreviations(optimized, context)
        if abbrevs:
            metadata["strategies_applied"].append("abbreviations")
            metadata["abbreviations"] = abbrevs
        
        # 5. Remove redundant context
        if context:
            optimized = self._remove_redundant_context(optimized, context)
            metadata["strategies_applied"].append("context_deduplication")
        
        # Calculate savings
        optimized_tokens = self.count_tokens(optimized)
        metadata["optimized_tokens"] = optimized_tokens
        metadata["token_savings"] = original_tokens - optimized_tokens
        metadata["savings_percentage"] = (
            (original_tokens - optimized_tokens) / original_tokens * 100
            if original_tokens > 0 else 0
        )
        
        return optimized, metadata
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.encoder.encode(text))
    
    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace while preserving structure"""
        # Collapse multiple spaces
        text = re.sub(r' +', ' ', text)
        
        # Normalize line breaks
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        # Remove trailing whitespace
        text = '\n'.join(line.rstrip() for line in text.split('\n'))
        
        return text.strip()
    
    def _extract_references(self, text: str) -> Tuple[str, Dict[str, str]]:
        """Extract repeated content and replace with references"""
        references = {}
        
        # Find repeated long strings (>50 chars, appears 3+ times)
        pattern = re.compile(r'(.{50,}?)(?=.*\1.*\1)', re.DOTALL)
        
        for match in pattern.finditer(text):
            repeated_text = match.group(1)
            if text.count(repeated_text) >= 3:
                # Generate reference ID
                ref_id = f"REF_{hashlib.md5(repeated_text.encode()).hexdigest()[:8]}"
                
                # Store reference
                references[ref_id] = repeated_text
                
                # Replace in text
                text = text.replace(repeated_text, f"[{ref_id}]")
        
        return text, references
    
    def _compress_patterns(self, text: str) -> str:
        """Compress common verbose patterns"""
        compressions = [
            # Verbose -> Concise
            (r'in order to', 'to'),
            (r'due to the fact that', 'because'),
            (r'at this point in time', 'now'),
            (r'in the event that', 'if'),
            (r'for the purpose of', 'for'),
            (r'with regard to', 'about'),
            (r'in spite of the fact that', 'although'),
            (r'as a result of', 'from'),
            (r'is able to', 'can'),
            (r'has the ability to', 'can'),
            
            # Code patterns
            (r'function\s+(\w+)\s*\([^)]*\)\s*{', r'fn \1{'),
            (r'public\s+static\s+void', 'psv'),
            (r'private\s+final', 'pf'),
            (r'return\s+', 'ret '),
            
            # Common programming terms
            (r'implementation', 'impl'),
            (r'configuration', 'config'),
            (r'initialization', 'init'),
            (r'documentation', 'docs'),
            (r'repository', 'repo'),
            (r'dependency', 'dep'),
            (r'parameter', 'param'),
            (r'argument', 'arg'),
        ]
        
        for verbose, concise in compressions:
            text = re.sub(verbose, concise, text, flags=re.IGNORECASE)
        
        return text
    
    def _apply_abbreviations(self, text: str, context: Optional[Dict]) -> Tuple[str, Dict[str, str]]:
        """Apply context-aware abbreviations"""
        used_abbreviations = {}
        
        if not context:
            return text, used_abbreviations
        
        # Get project-specific terms
        project_terms = context.get('project_terms', {})
        
        for full_term, abbreviation in project_terms.items():
            if full_term in text and len(full_term) > len(abbreviation) + 3:
                text = text.replace(full_term, abbreviation)
                used_abbreviations[abbreviation] = full_term
        
        return text, used_abbreviations
    
    def _remove_redundant_context(self, text: str, context: Dict) -> str:
        """Remove information already present in context"""
        # If context contains project info, remove from text
        if 'project_name' in context:
            text = text.replace(f"In the {context['project_name']} project,", "")
            text = text.replace(f"The {context['project_name']}", "It")
        
        # Remove repeated user info
        if 'user_id' in context:
            text = re.sub(f"User {context['user_id']}'s", "Your", text)
        
        return text
    
    def _load_abbreviations(self) -> Dict[str, str]:
        """Load common abbreviations"""
        return {
            # Technical terms
            "application": "app",
            "database": "db",
            "development": "dev",
            "production": "prod",
            "environment": "env",
            "configuration": "config",
            "authentication": "auth",
            "authorization": "authz",
            "administrator": "admin",
            "management": "mgmt",
            "service": "svc",
            "controller": "ctrl",
            "repository": "repo",
            "implementation": "impl",
            "specification": "spec",
            "documentation": "docs",
            "organization": "org",
            "information": "info",
            "message": "msg",
            "response": "resp",
            "request": "req",
            "parameter": "param",
            "argument": "arg",
            "function": "fn",
            "variable": "var",
            "constant": "const",
            "temporary": "tmp",
            "directory": "dir",
            "source": "src",
            "destination": "dst",
            "reference": "ref",
            "pointer": "ptr",
            "iterator": "iter",
            "generator": "gen",
            "exception": "exc",
            "error": "err",
            "warning": "warn",
            "debug": "dbg",
            "utility": "util",
            "helper": "hlpr",
            "handler": "hdlr",
            "manager": "mgr",
            "factory": "fctry",
            "builder": "bldr",
            "validator": "vldr",
            "converter": "cnvtr",
            "formatter": "fmtr",
            "serializer": "srlzr",
            "deserializer": "dsrlzr",
            "encoder": "enc",
            "decoder": "dec",
            "compressor": "cmprs",
            "decompressor": "dcmprs",
        }
    
    def expand_optimized(self, optimized_text: str, metadata: Dict[str, Any]) -> str:
        """Expand optimized text back to full form"""
        expanded = optimized_text
        
        # Restore references
        if "references" in metadata:
            for ref_id, full_text in metadata["references"].items():
                expanded = expanded.replace(f"[{ref_id}]", full_text)
        
        # Restore abbreviations
        if "abbreviations" in metadata:
            for abbrev, full_term in metadata["abbreviations"].items():
                expanded = expanded.replace(abbrev, full_term)
        
        return expanded
    
    def estimate_savings(self, texts: List[str]) -> Dict[str, float]:
        """Estimate token savings for a list of texts"""
        total_original = 0
        total_optimized = 0
        
        for text in texts:
            optimized, metadata = self.optimize(text)
            total_original += metadata["original_tokens"]
            total_optimized += metadata["optimized_tokens"]
        
        return {
            "total_original_tokens": total_original,
            "total_optimized_tokens": total_optimized,
            "total_savings": total_original - total_optimized,
            "average_savings_percentage": (
                (total_original - total_optimized) / total_original * 100
                if total_original > 0 else 0
            )
        }


# Utility functions
def count_tokens(text: str, model: str = "gpt-4") -> int:
    """Quick token count function"""
    encoder = tiktoken.encoding_for_model(model)
    return len(encoder.encode(text))