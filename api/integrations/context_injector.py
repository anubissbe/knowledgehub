"""
Context Injection System for AI Tools.

Provides intelligent context injection for various AI tools including GitHub Copilot,
Claude, and other AI assistants, enhancing their capabilities with KnowledgeHub
intelligence and project-specific context.
"""

import json
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum

from ..services.memory_service import MemoryService
from ..services.session_service import SessionService
from ..services.pattern_service import PatternService
from ..services.ai_service import AIService

logger = logging.getLogger(__name__)


class ContextType(Enum):
    """Types of context that can be injected."""
    PROJECT = "project"
    SESSION = "session"
    MEMORY = "memory"
    PATTERN = "pattern"
    DECISION = "decision"
    ERROR = "error"
    PERFORMANCE = "performance"
    GIT = "git"
    FILE = "file"
    CODE = "code"


class InjectionMode(Enum):
    """Modes of context injection."""
    PREPEND = "prepend"  # Add context before original content
    APPEND = "append"    # Add context after original content
    INLINE = "inline"    # Inject context inline with markers
    METADATA = "metadata"  # Add as metadata/headers
    EMBED = "embed"      # Embed within existing structure


@dataclass
class ContextElement:
    """Individual context element to be injected."""
    type: ContextType
    content: str
    relevance: float
    source: str
    metadata: Dict[str, Any]
    priority: int = 1  # 1=highest, 5=lowest


@dataclass
class InjectionRule:
    """Rule for how to inject context."""
    context_types: List[ContextType]
    mode: InjectionMode
    max_length: Optional[int] = None
    priority_threshold: int = 3
    relevance_threshold: float = 0.5
    template: Optional[str] = None


class ContextInjector:
    """
    Main context injection system.
    
    Intelligently injects KnowledgeHub context into AI tool requests
    to enhance their capabilities with project-specific knowledge.
    """
    
    def __init__(self):
        self.memory_service = MemoryService()
        self.session_service = SessionService()
        self.pattern_service = PatternService()
        self.ai_service = AIService()
        
        # Default injection rules
        self.default_rules = {
            "code_completion": InjectionRule(
                context_types=[ContextType.PATTERN, ContextType.PROJECT, ContextType.FILE],
                mode=InjectionMode.PREPEND,
                max_length=1000,
                template="/* KnowledgeHub Context:\n{context}\n*/\n"
            ),
            "code_explanation": InjectionRule(
                context_types=[ContextType.MEMORY, ContextType.DECISION, ContextType.PATTERN],
                mode=InjectionMode.APPEND,
                max_length=1500,
                template="\n\n# Relevant Context:\n{context}"
            ),
            "bug_fix": InjectionRule(
                context_types=[ContextType.ERROR, ContextType.MEMORY, ContextType.PATTERN],
                mode=InjectionMode.PREPEND,
                max_length=800,
                template="# Error History & Patterns:\n{context}\n\n"
            ),
            "refactoring": InjectionRule(
                context_types=[ContextType.PATTERN, ContextType.DECISION, ContextType.PROJECT],
                mode=InjectionMode.INLINE,
                max_length=1200,
                template="// Context: {context}\n"
            ),
            "documentation": InjectionRule(
                context_types=[ContextType.DECISION, ContextType.PROJECT, ContextType.MEMORY],
                mode=InjectionMode.APPEND,
                max_length=2000,
                template="\n\n## Background Information:\n{context}"
            )
        }
    
    async def inject_context(
        self,
        request: Dict[str, Any],
        context_hint: str = "general",
        user_id: Optional[str] = None,
        project_id: Optional[str] = None,
        custom_rules: Optional[Dict[str, InjectionRule]] = None
    ) -> Dict[str, Any]:
        """
        Inject context into an AI tool request.
        
        Args:
            request: Original AI tool request
            context_hint: Hint about the type of request (code_completion, bug_fix, etc.)
            user_id: User ID for personalized context
            project_id: Project ID for project-specific context
            custom_rules: Custom injection rules to override defaults
            
        Returns:
            Enhanced request with injected context
        """
        try:
            logger.info(f"Injecting context for request type: {context_hint}")
            
            # Get injection rules
            rules = custom_rules or self.default_rules
            rule = rules.get(context_hint, rules.get("general", self._get_default_rule()))
            
            # Extract relevant information from request
            request_info = self._extract_request_info(request)
            
            # Gather context elements
            context_elements = await self._gather_context_elements(
                request_info, rule, user_id, project_id
            )
            
            # Filter and prioritize context
            filtered_context = self._filter_context(context_elements, rule)
            
            # Format context for injection
            formatted_context = self._format_context(filtered_context, rule)
            
            # Inject context into request
            enhanced_request = self._inject_into_request(
                request, formatted_context, rule, request_info
            )
            
            # Add metadata about injection
            enhanced_request["_knowledgehub_injection"] = {
                "context_hint": context_hint,
                "elements_used": len(filtered_context),
                "total_elements": len(context_elements),
                "injection_mode": rule.mode.value,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            logger.info(f"Context injection completed: {len(filtered_context)} elements injected")
            return enhanced_request
            
        except Exception as e:
            logger.error(f"Error injecting context: {e}")
            # Return original request if injection fails
            return request
    
    async def inject_smart_context(
        self,
        request: Dict[str, Any],
        user_id: Optional[str] = None,
        project_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Inject context with AI-powered smart detection of context needs.
        
        Analyzes the request to determine what type of context would be most helpful.
        """
        try:
            # Extract request content for analysis
            request_content = self._extract_request_content(request)
            
            # Analyze request to determine context needs
            context_analysis = await self._analyze_context_needs(
                request_content, user_id, project_id
            )
            
            # Generate dynamic injection rules based on analysis
            dynamic_rules = self._generate_dynamic_rules(context_analysis)
            
            # Inject context using dynamic rules
            return await self.inject_context(
                request,
                context_hint=context_analysis.get("primary_type", "general"),
                user_id=user_id,
                project_id=project_id,
                custom_rules=dynamic_rules
            )
            
        except Exception as e:
            logger.error(f"Error in smart context injection: {e}")
            return request
    
    async def _gather_context_elements(
        self,
        request_info: Dict[str, Any],
        rule: InjectionRule,
        user_id: Optional[str],
        project_id: Optional[str]
    ) -> List[ContextElement]:
        """Gather relevant context elements based on injection rules."""
        elements = []
        
        for context_type in rule.context_types:
            try:
                if context_type == ContextType.PROJECT:
                    project_elements = await self._get_project_context(project_id, request_info)
                    elements.extend(project_elements)
                
                elif context_type == ContextType.SESSION:
                    session_elements = await self._get_session_context(user_id, request_info)
                    elements.extend(session_elements)
                
                elif context_type == ContextType.MEMORY:
                    memory_elements = await self._get_memory_context(
                        request_info, user_id, project_id
                    )
                    elements.extend(memory_elements)
                
                elif context_type == ContextType.PATTERN:
                    pattern_elements = await self._get_pattern_context(
                        request_info, user_id, project_id
                    )
                    elements.extend(pattern_elements)
                
                elif context_type == ContextType.DECISION:
                    decision_elements = await self._get_decision_context(
                        request_info, user_id, project_id
                    )
                    elements.extend(decision_elements)
                
                elif context_type == ContextType.ERROR:
                    error_elements = await self._get_error_context(
                        request_info, user_id, project_id
                    )
                    elements.extend(error_elements)
                
                elif context_type == ContextType.GIT:
                    git_elements = await self._get_git_context(request_info, project_id)
                    elements.extend(git_elements)
                
                elif context_type == ContextType.FILE:
                    file_elements = await self._get_file_context(request_info, project_id)
                    elements.extend(file_elements)
                
                elif context_type == ContextType.CODE:
                    code_elements = await self._get_code_context(
                        request_info, user_id, project_id
                    )
                    elements.extend(code_elements)
                
            except Exception as e:
                logger.error(f"Error gathering {context_type.value} context: {e}")
                continue
        
        return elements
    
    async def _get_project_context(
        self,
        project_id: Optional[str],
        request_info: Dict[str, Any]
    ) -> List[ContextElement]:
        """Get project-specific context."""
        if not project_id:
            return []
        
        elements = []
        
        # Add project information
        elements.append(ContextElement(
            type=ContextType.PROJECT,
            content=f"Project: {project_id}",
            relevance=0.8,
            source="project_info",
            metadata={"project_id": project_id},
            priority=2
        ))
        
        return elements
    
    async def _get_session_context(
        self,
        user_id: Optional[str],
        request_info: Dict[str, Any]
    ) -> List[ContextElement]:
        """Get current session context."""
        if not user_id:
            return []
        
        elements = []
        
        try:
            session = await self.session_service.get_active_session(user_id)
            if session:
                elements.append(ContextElement(
                    type=ContextType.SESSION,
                    content=f"Current focus: {session.context_data.get('focus', 'Development')}",
                    relevance=0.7,
                    source="active_session",
                    metadata={"session_id": session.session_id},
                    priority=2
                ))
        except Exception as e:
            logger.error(f"Error getting session context: {e}")
        
        return elements
    
    async def _get_memory_context(
        self,
        request_info: Dict[str, Any],
        user_id: Optional[str],
        project_id: Optional[str]
    ) -> List[ContextElement]:
        """Get relevant memories for context."""
        elements = []
        
        try:
            # Search for relevant memories based on request content
            query = request_info.get("content", "")[:200]  # Limit query size
            
            memories = await self.memory_service.search_memories(
                query=query,
                user_id=user_id,
                project_id=project_id,
                limit=5
            )
            
            for memory in memories:
                elements.append(ContextElement(
                    type=ContextType.MEMORY,
                    content=memory.content[:200],  # Limit content size
                    relevance=getattr(memory, "similarity_score", 0.7),
                    source="memory_search",
                    metadata={"memory_id": memory.id, "memory_type": memory.memory_type},
                    priority=1
                ))
                
        except Exception as e:
            logger.error(f"Error getting memory context: {e}")
        
        return elements
    
    async def _get_pattern_context(
        self,
        request_info: Dict[str, Any],
        user_id: Optional[str],
        project_id: Optional[str]
    ) -> List[ContextElement]:
        """Get relevant code patterns for context."""
        elements = []
        
        try:
            # Analyze patterns if code is present
            if "code" in request_info:
                patterns = await self.pattern_service.analyze_code_patterns(
                    user_id=user_id,
                    project_id=project_id
                )
                
                for pattern in patterns.get("patterns", [])[:3]:  # Top 3 patterns
                    elements.append(ContextElement(
                        type=ContextType.PATTERN,
                        content=f"Pattern: {pattern.get('description', 'Unknown')}",
                        relevance=pattern.get("confidence", 0.6),
                        source="pattern_analysis",
                        metadata={"pattern_type": pattern.get("pattern_type")},
                        priority=2
                    ))
                    
        except Exception as e:
            logger.error(f"Error getting pattern context: {e}")
        
        return elements
    
    async def _get_decision_context(
        self,
        request_info: Dict[str, Any],
        user_id: Optional[str],
        project_id: Optional[str]
    ) -> List[ContextElement]:
        """Get relevant technical decisions for context."""
        elements = []
        
        try:
            # Search for relevant decisions
            query = request_info.get("content", "")[:200]
            
            decisions = await self.memory_service.search_memories(
                query=query,
                user_id=user_id,
                project_id=project_id,
                memory_type="decision",
                limit=3
            )
            
            for decision in decisions:
                elements.append(ContextElement(
                    type=ContextType.DECISION,
                    content=f"Decision: {decision.content[:150]}",
                    relevance=getattr(decision, "similarity_score", 0.7),
                    source="decision_search",
                    metadata={"decision_id": decision.id},
                    priority=1
                ))
                
        except Exception as e:
            logger.error(f"Error getting decision context: {e}")
        
        return elements
    
    async def _get_error_context(
        self,
        request_info: Dict[str, Any],
        user_id: Optional[str],
        project_id: Optional[str]
    ) -> List[ContextElement]:
        """Get relevant error history for context."""
        elements = []
        
        try:
            # Search for error-related memories
            query = request_info.get("content", "")[:200]
            
            errors = await self.memory_service.search_memories(
                query=query,
                user_id=user_id,
                project_id=project_id,
                memory_type="error",
                limit=3
            )
            
            for error in errors:
                elements.append(ContextElement(
                    type=ContextType.ERROR,
                    content=f"Similar error: {error.content[:150]}",
                    relevance=getattr(error, "similarity_score", 0.7),
                    source="error_history",
                    metadata={"error_id": error.id},
                    priority=1
                ))
                
        except Exception as e:
            logger.error(f"Error getting error context: {e}")
        
        return elements
    
    async def _get_git_context(
        self,
        request_info: Dict[str, Any],
        project_id: Optional[str]
    ) -> List[ContextElement]:
        """Get Git-related context."""
        elements = []
        
        # Placeholder for Git context - would integrate with Git service
        elements.append(ContextElement(
            type=ContextType.GIT,
            content="Branch: main",
            relevance=0.5,
            source="git_info",
            metadata={"branch": "main"},
            priority=3
        ))
        
        return elements
    
    async def _get_file_context(
        self,
        request_info: Dict[str, Any],
        project_id: Optional[str]
    ) -> List[ContextElement]:
        """Get file-specific context."""
        elements = []
        
        file_path = request_info.get("file_path")
        if file_path:
            elements.append(ContextElement(
                type=ContextType.FILE,
                content=f"Working on: {file_path}",
                relevance=0.8,
                source="file_info",
                metadata={"file_path": file_path},
                priority=2
            ))
        
        return elements
    
    async def _get_code_context(
        self,
        request_info: Dict[str, Any],
        user_id: Optional[str],
        project_id: Optional[str]
    ) -> List[ContextElement]:
        """Get code-specific context."""
        elements = []
        
        language = request_info.get("language")
        if language:
            elements.append(ContextElement(
                type=ContextType.CODE,
                content=f"Language: {language}",
                relevance=0.6,
                source="code_info",
                metadata={"language": language},
                priority=3
            ))
        
        return elements
    
    def _filter_context(
        self,
        elements: List[ContextElement],
        rule: InjectionRule
    ) -> List[ContextElement]:
        """Filter and prioritize context elements based on rules."""
        # Filter by relevance threshold
        filtered = [
            elem for elem in elements
            if elem.relevance >= rule.relevance_threshold
        ]
        
        # Filter by priority threshold
        filtered = [
            elem for elem in filtered
            if elem.priority <= rule.priority_threshold
        ]
        
        # Sort by priority (lower is higher) then relevance (higher is better)
        filtered.sort(key=lambda x: (x.priority, -x.relevance))
        
        # Limit by max length if specified
        if rule.max_length:
            total_length = 0
            result = []
            for elem in filtered:
                if total_length + len(elem.content) <= rule.max_length:
                    result.append(elem)
                    total_length += len(elem.content)
                else:
                    break
            return result
        
        return filtered
    
    def _format_context(
        self,
        elements: List[ContextElement],
        rule: InjectionRule
    ) -> str:
        """Format context elements according to injection rule."""
        if not elements:
            return ""
        
        # Group by type for better organization
        by_type = {}
        for elem in elements:
            if elem.type not in by_type:
                by_type[elem.type] = []
            by_type[elem.type].append(elem)
        
        # Format by type
        formatted_parts = []
        for context_type, type_elements in by_type.items():
            type_content = []
            for elem in type_elements:
                type_content.append(elem.content)
            
            if type_content:
                formatted_parts.append(f"{context_type.value.title()}: {'; '.join(type_content)}")
        
        context_text = "\n".join(formatted_parts)
        
        # Apply template if provided
        if rule.template:
            return rule.template.format(context=context_text)
        
        return context_text
    
    def _inject_into_request(
        self,
        request: Dict[str, Any],
        formatted_context: str,
        rule: InjectionRule,
        request_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Inject formatted context into the request."""
        if not formatted_context:
            return request
        
        enhanced_request = request.copy()
        
        # Find the main content field to modify
        content_field = request_info.get("content_field", "prompt")
        original_content = request.get(content_field, "")
        
        if rule.mode == InjectionMode.PREPEND:
            enhanced_request[content_field] = formatted_context + "\n" + original_content
        elif rule.mode == InjectionMode.APPEND:
            enhanced_request[content_field] = original_content + "\n" + formatted_context
        elif rule.mode == InjectionMode.INLINE:
            # Insert context at natural break points
            enhanced_content = self._inject_inline(original_content, formatted_context)
            enhanced_request[content_field] = enhanced_content
        elif rule.mode == InjectionMode.METADATA:
            # Add as separate metadata field
            enhanced_request["_context"] = formatted_context
        elif rule.mode == InjectionMode.EMBED:
            # Embed within existing structure
            enhanced_request[content_field] = self._embed_context(original_content, formatted_context)
        
        return enhanced_request
    
    def _inject_inline(self, content: str, context: str) -> str:
        """Inject context inline at natural break points."""
        # Find good insertion points (line breaks, code blocks, etc.)
        lines = content.split('\n')
        if len(lines) > 3:
            # Insert context after first few lines
            insert_point = min(3, len(lines) // 2)
            lines.insert(insert_point, f"\n// Context: {context}\n")
            return '\n'.join(lines)
        else:
            # Prepend if content is short
            return f"// Context: {context}\n{content}"
    
    def _embed_context(self, content: str, context: str) -> str:
        """Embed context within existing content structure."""
        # Simple embedding - could be made more sophisticated
        return f"Given context: {context}\n\n{content}"
    
    def _extract_request_info(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Extract relevant information from the request."""
        info = {}
        
        # Common content fields
        for field in ["prompt", "query", "content", "message", "text"]:
            if field in request:
                info["content"] = request[field]
                info["content_field"] = field
                break
        
        # Extract other useful fields
        info["language"] = request.get("language")
        info["file_path"] = request.get("file_path")
        info["context"] = request.get("context", {})
        
        return info
    
    def _extract_request_content(self, request: Dict[str, Any]) -> str:
        """Extract the main content from a request for analysis."""
        for field in ["prompt", "query", "content", "message", "text"]:
            if field in request:
                return str(request[field])
        return ""
    
    async def _analyze_context_needs(
        self,
        content: str,
        user_id: Optional[str],
        project_id: Optional[str]
    ) -> Dict[str, Any]:
        """Analyze what type of context would be most helpful for this request."""
        try:
            # Use AI service to analyze context needs
            analysis = await self.ai_service.generate_ai_insights(
                context="context_needs_analysis",
                data={
                    "request_content": content[:500],  # Limit size
                    "user_id": user_id,
                    "project_id": project_id
                }
            )
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing context needs: {e}")
            # Fallback to simple keyword-based analysis
            return self._simple_context_analysis(content)
    
    def _simple_context_analysis(self, content: str) -> Dict[str, Any]:
        """Simple keyword-based context analysis as fallback."""
        content_lower = content.lower()
        
        if any(word in content_lower for word in ["error", "bug", "fix", "issue"]):
            return {"primary_type": "bug_fix", "confidence": 0.7}
        elif any(word in content_lower for word in ["refactor", "improve", "optimize"]):
            return {"primary_type": "refactoring", "confidence": 0.7}
        elif any(word in content_lower for word in ["explain", "document", "what", "how"]):
            return {"primary_type": "code_explanation", "confidence": 0.6}
        elif any(word in content_lower for word in ["complete", "implement", "add"]):
            return {"primary_type": "code_completion", "confidence": 0.6}
        else:
            return {"primary_type": "general", "confidence": 0.5}
    
    def _generate_dynamic_rules(self, analysis: Dict[str, Any]) -> Dict[str, InjectionRule]:
        """Generate dynamic injection rules based on context analysis."""
        primary_type = analysis.get("primary_type", "general")
        confidence = analysis.get("confidence", 0.5)
        
        # Adjust rules based on confidence
        max_length = int(1000 * confidence) if confidence > 0.6 else 500
        priority_threshold = 2 if confidence > 0.8 else 3
        
        if primary_type in self.default_rules:
            # Use default rule as base and adjust
            base_rule = self.default_rules[primary_type]
            return {
                primary_type: InjectionRule(
                    context_types=base_rule.context_types,
                    mode=base_rule.mode,
                    max_length=max_length,
                    priority_threshold=priority_threshold,
                    relevance_threshold=base_rule.relevance_threshold,
                    template=base_rule.template
                )
            }
        
        # Generate new rule for unknown types
        return {
            primary_type: InjectionRule(
                context_types=[ContextType.MEMORY, ContextType.PROJECT],
                mode=InjectionMode.PREPEND,
                max_length=max_length,
                priority_threshold=priority_threshold,
                template="# Context:\n{context}\n\n"
            )
        }
    
    def _get_default_rule(self) -> InjectionRule:
        """Get default injection rule."""
        return InjectionRule(
            context_types=[ContextType.PROJECT, ContextType.SESSION],
            mode=InjectionMode.PREPEND,
            max_length=500,
            template="# Context:\n{context}\n\n"
        )