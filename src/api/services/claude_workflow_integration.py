"""
Claude Code Workflow Integration System

Automatically captures memories from Claude Code conversations,
extracts context from terminal output, integrates with tool usage,
and saves important discoveries.
"""

import json
import re
import hashlib
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import cast, String

from ..models import MemoryItem, get_db
from ..config import settings
from .claude_session_manager import ClaudeSessionManager
from .mistake_learning_system import MistakeLearningSystem
from .performance_metrics_tracker import PerformanceMetricsTracker
from .code_evolution_tracker import CodeEvolutionTracker
from .decision_reasoning_system import DecisionReasoningSystem


class ClaudeWorkflowIntegration:
    """Integrates KnowledgeHub with Claude Code's workflow"""
    
    def __init__(self):
        self.session_manager = ClaudeSessionManager()
        self.mistake_tracker = MistakeLearningSystem()
        self.performance_tracker = PerformanceMetricsTracker()
        self.code_tracker = CodeEvolutionTracker()
        self.decision_tracker = DecisionReasoningSystem()
        
        # Patterns for extracting information from conversations
        self.patterns = {
            'error': re.compile(r'(?:Error|Exception|Failed|Error:)\s*([^\n]+)', re.IGNORECASE),
            'solution': re.compile(r'(?:Fixed|Resolved|Solution|Fixed by|Solved with):\s*([^\n]+)', re.IGNORECASE),
            'command': re.compile(r'(?:^\$\s|>>>|>\s)([^\n]+)', re.MULTILINE),
            'file_path': re.compile(r'(?:File|Created|Modified|Updated):\s*([\/\w\-\.]+\.\w+)', re.IGNORECASE),
            'discovery': re.compile(r'(?:Found|Discovered|Learned|Note|Important|TIL):\s*([^\n]+)', re.IGNORECASE),
            'todo': re.compile(r'(?:TODO|FIXME|NOTE|HACK):\s*([^\n]+)', re.IGNORECASE),
            'decision': re.compile(r'(?:Decided to|Choosing|Selected|Will use):\s*([^\n]+)', re.IGNORECASE),
            'performance': re.compile(r'(?:Took|Completed in|Duration|Time:)\s*([\d\.]+)\s*(?:ms|s|seconds|minutes)', re.IGNORECASE)
        }
        
        # Tool usage patterns
        self.tool_patterns = {
            'Read': {'action': 'file_read', 'extract': ['file_path']},
            'Write': {'action': 'file_write', 'extract': ['file_path', 'content']},
            'Edit': {'action': 'file_edit', 'extract': ['file_path', 'old_string', 'new_string']},
            'Bash': {'action': 'command_execute', 'extract': ['command', 'output']},
            'Search': {'action': 'search', 'extract': ['query', 'results']},
            'Task': {'action': 'task_execute', 'extract': ['description', 'result']}
        }
    
    def capture_conversation_memory(self, db: Session, conversation_text: str,
                                  session_id: Optional[str] = None,
                                  project_id: Optional[str] = None) -> Dict[str, Any]:
        """Automatically capture memories from conversation text"""
        
        memories_created = []
        patterns_found = {}
        
        # Extract errors and solutions
        errors = self.patterns['error'].findall(conversation_text)
        solutions = self.patterns['solution'].findall(conversation_text)
        
        if errors:
            patterns_found['errors'] = errors
            for i, error in enumerate(errors):
                solution = solutions[i] if i < len(solutions) else None
                if solution:
                    # Track the mistake with its solution
                    result = self.mistake_tracker.track_mistake(
                        db, "ConversationError", error,
                        {"conversation": True, "session_id": session_id},
                        successful_solution=solution
                    )
                    memories_created.append({
                        'type': 'error_solution',
                        'error': error,
                        'solution': solution,
                        'memory_id': result['memory_id']
                    })
        
        # Extract commands
        commands = self.patterns['command'].findall(conversation_text)
        if commands:
            patterns_found['commands'] = commands
            for command in commands:
                memory = self._create_memory(
                    db, f"Command used: {command}",
                    'code', 'high',
                    {'command': command, 'session_id': session_id}
                )
                memories_created.append({
                    'type': 'command',
                    'command': command,
                    'memory_id': str(memory.id)
                })
        
        # Extract discoveries
        discoveries = self.patterns['discovery'].findall(conversation_text)
        if discoveries:
            patterns_found['discoveries'] = discoveries
            for discovery in discoveries:
                memory = self._create_memory(
                    db, f"Discovery: {discovery}",
                    'fact', 'high',
                    {'discovery': True, 'session_id': session_id}
                )
                memories_created.append({
                    'type': 'discovery',
                    'content': discovery,
                    'memory_id': str(memory.id)
                })
        
        # Extract TODOs
        todos = self.patterns['todo'].findall(conversation_text)
        if todos:
            patterns_found['todos'] = todos
            for todo in todos:
                memory = self._create_memory(
                    db, f"TODO: {todo}",
                    'pattern', 'medium',
                    {'todo': True, 'session_id': session_id}
                )
                memories_created.append({
                    'type': 'todo',
                    'content': todo,
                    'memory_id': str(memory.id)
                })
        
        # Extract decisions
        decisions = self.patterns['decision'].findall(conversation_text)
        if decisions:
            patterns_found['decisions'] = decisions
            for decision in decisions:
                memory = self._create_memory(
                    db, f"Decision: {decision}",
                    'decision', 'high',
                    {'conversation_decision': True, 'session_id': session_id}
                )
                memories_created.append({
                    'type': 'decision',
                    'content': decision,
                    'memory_id': str(memory.id)
                })
        
        return {
            'memories_created': len(memories_created),
            'patterns_found': patterns_found,
            'memories': memories_created,
            'session_id': session_id,
            'project_id': project_id
        }
    
    def extract_terminal_context(self, db: Session, terminal_output: str,
                               command: str, exit_code: int = 0,
                               execution_time: Optional[float] = None) -> Dict[str, Any]:
        """Extract context and insights from terminal output"""
        
        insights = []
        
        # Track command performance
        if execution_time:
            perf_result = self.performance_tracker.track_command_execution(
                db, 'terminal_command',
                {'command': command, 'exit_code': exit_code},
                execution_time,
                exit_code == 0
            )
            insights.append({
                'type': 'performance',
                'execution_time': execution_time,
                'success': exit_code == 0,
                'performance_id': perf_result['execution_id']
            })
        
        # Extract errors from output
        if exit_code != 0 or 'error' in terminal_output.lower():
            error_lines = [line for line in terminal_output.split('\n') 
                          if 'error' in line.lower() or 'exception' in line.lower()]
            
            for error_line in error_lines[:5]:  # Limit to first 5 errors
                memory = self._create_memory(
                    db, f"Terminal error: {error_line}",
                    'error', 'high',
                    {'command': command, 'exit_code': exit_code}
                )
                insights.append({
                    'type': 'error',
                    'content': error_line,
                    'memory_id': str(memory.id)
                })
        
        # Extract file paths mentioned
        file_paths = self.patterns['file_path'].findall(terminal_output)
        for file_path in file_paths[:10]:  # Limit to first 10 files
            insights.append({
                'type': 'file_mention',
                'file_path': file_path
            })
        
        # Extract performance metrics
        perf_matches = self.patterns['performance'].findall(terminal_output)
        for duration in perf_matches:
            insights.append({
                'type': 'performance_metric',
                'duration': duration
            })
        
        # Save important output patterns
        important_patterns = [
            (r'Successfully\s+([^\n]+)', 'success'),
            (r'Warning:\s+([^\n]+)', 'warning'),
            (r'Created\s+([^\n]+)', 'creation'),
            (r'Deleted\s+([^\n]+)', 'deletion'),
            (r'Updated\s+([^\n]+)', 'update')
        ]
        
        for pattern, pattern_type in important_patterns:
            matches = re.findall(pattern, terminal_output, re.IGNORECASE)
            for match in matches[:3]:  # Limit each type
                memory = self._create_memory(
                    db, f"{pattern_type.title()}: {match}",
                    'fact', 'medium',
                    {'terminal_output': True, 'command': command}
                )
                insights.append({
                    'type': pattern_type,
                    'content': match,
                    'memory_id': str(memory.id)
                })
        
        return {
            'command': command,
            'exit_code': exit_code,
            'insights_extracted': len(insights),
            'insights': insights,
            'execution_time': execution_time
        }
    
    def capture_tool_usage(self, db: Session, tool_name: str, 
                          tool_params: Dict[str, Any],
                          tool_result: Any,
                          execution_time: float,
                          session_id: Optional[str] = None) -> Dict[str, Any]:
        """Capture memories from Claude Code tool usage"""
        
        memories = []
        
        # Get tool pattern
        tool_pattern = self.tool_patterns.get(tool_name, {})
        action = tool_pattern.get('action', tool_name.lower())
        
        # Track performance
        perf_result = self.performance_tracker.track_command_execution(
            db, f"tool_{action}",
            {'tool': tool_name, 'params': tool_params},
            execution_time,
            tool_result is not None
        )
        
        # Handle specific tools
        if tool_name == 'Read':
            file_path = tool_params.get('file_path', '')
            if file_path:
                memory = self._create_memory(
                    db, f"Read file: {file_path}",
                    'fact', 'medium',
                    {'tool': 'Read', 'file_path': file_path, 'session_id': session_id}
                )
                memories.append({
                    'type': 'file_read',
                    'file_path': file_path,
                    'memory_id': str(memory.id)
                })
        
        elif tool_name == 'Write':
            file_path = tool_params.get('file_path', '')
            if file_path:
                memory = self._create_memory(
                    db, f"Created/Updated file: {file_path}",
                    'fact', 'high',
                    {'tool': 'Write', 'file_path': file_path, 'session_id': session_id}
                )
                memories.append({
                    'type': 'file_write',
                    'file_path': file_path,
                    'memory_id': str(memory.id)
                })
        
        elif tool_name == 'Edit':
            file_path = tool_params.get('file_path', '')
            old_string = tool_params.get('old_string', '')[:50]  # First 50 chars
            new_string = tool_params.get('new_string', '')[:50]
            
            if file_path:
                # Track code evolution
                self.code_tracker.track_code_change(
                    db, file_path,
                    old_string, new_string,
                    "Tool-based edit",
                    "Claude Code modification"
                )
                
                memory = self._create_memory(
                    db, f"Edited {file_path}: {old_string} → {new_string}",
                    'code', 'high',
                    {'tool': 'Edit', 'file_path': file_path, 'session_id': session_id}
                )
                memories.append({
                    'type': 'file_edit',
                    'file_path': file_path,
                    'memory_id': str(memory.id)
                })
        
        elif tool_name == 'Bash':
            command = tool_params.get('command', '')
            if command:
                memory = self._create_memory(
                    db, f"Executed: {command}",
                    'code', 'medium',
                    {'tool': 'Bash', 'command': command, 'session_id': session_id}
                )
                memories.append({
                    'type': 'command_execution',
                    'command': command,
                    'memory_id': str(memory.id)
                })
        
        elif tool_name == 'Search':
            query = tool_params.get('query', '')
            if query:
                memory = self._create_memory(
                    db, f"Searched for: {query}",
                    'pattern', 'low',
                    {'tool': 'Search', 'query': query, 'session_id': session_id}
                )
                memories.append({
                    'type': 'search',
                    'query': query,
                    'memory_id': str(memory.id)
                })
        
        return {
            'tool': tool_name,
            'action': action,
            'memories_created': len(memories),
            'memories': memories,
            'performance_id': perf_result['execution_id'],
            'execution_time': execution_time
        }
    
    def save_discovery(self, db: Session, discovery_type: str,
                      content: str, context: Dict[str, Any],
                      importance: str = 'high',
                      tags: Optional[List[str]] = None) -> Dict[str, Any]:
        """Save an important discovery or insight"""
        
        # Determine memory type based on discovery type
        type_mapping = {
            'pattern': 'pattern',
            'solution': 'fact',
            'bug_fix': 'error',
            'optimization': 'preference',
            'architecture': 'decision',
            'algorithm': 'code',
            'configuration': 'preference'
        }
        
        memory_type = type_mapping.get(discovery_type, 'fact')
        
        # Create discovery memory
        meta_data = {
            'discovery_type': discovery_type,
            'discovered_at': datetime.utcnow().isoformat(),
            **context
        }
        
        if tags:
            meta_data['tags'] = tags
        
        memory = self._create_memory(
            db, content, memory_type, importance, meta_data
        )
        
        # If it's a solution, track it in mistake learning
        if discovery_type == 'solution' and 'error_type' in context:
            self.mistake_tracker.track_mistake(
                db, context['error_type'],
                context.get('error_message', 'Unknown error'),
                context,
                successful_solution=content
            )
        
        return {
            'discovery_id': str(memory.id),
            'type': discovery_type,
            'memory_type': memory_type,
            'importance': importance,
            'tags': tags or [],
            'saved': True
        }
    
    def auto_extract_insights(self, db: Session, message: str,
                            message_type: str = 'assistant',
                            session_id: Optional[str] = None) -> Dict[str, Any]:
        """Automatically extract insights from Claude's messages"""
        
        insights = []
        
        # Patterns specific to assistant messages
        if message_type == 'assistant':
            # Extract implementation decisions
            impl_pattern = re.compile(r"(?:I'll|I will|Let me|I'm going to)\s+([^\n]+)", re.IGNORECASE)
            implementations = impl_pattern.findall(message)[:5]
            
            for impl in implementations:
                if any(keyword in impl.lower() for keyword in ['create', 'implement', 'add', 'fix', 'update']):
                    insights.append({
                        'type': 'implementation',
                        'content': impl,
                        'confidence': 0.8
                    })
            
            # Extract findings
            finding_pattern = re.compile(r"(?:I found|I see|I notice|It seems|This shows)\s+([^\n]+)", re.IGNORECASE)
            findings = finding_pattern.findall(message)[:5]
            
            for finding in findings:
                memory = self._create_memory(
                    db, f"Finding: {finding}",
                    'fact', 'medium',
                    {'auto_extracted': True, 'session_id': session_id}
                )
                insights.append({
                    'type': 'finding',
                    'content': finding,
                    'memory_id': str(memory.id)
                })
        
        # Extract code blocks
        code_blocks = re.findall(r'```(\w+)?\n(.*?)```', message, re.DOTALL)
        for language, code in code_blocks[:3]:  # Limit to first 3 code blocks
            if len(code) > 50:  # Only save substantial code
                memory = self._create_memory(
                    db, f"Code snippet ({language or 'unknown'}): {code[:100]}...",
                    'code', 'medium',
                    {'language': language, 'full_code': code, 'session_id': session_id}
                )
                insights.append({
                    'type': 'code_snippet',
                    'language': language,
                    'memory_id': str(memory.id)
                })
        
        return {
            'message_type': message_type,
            'insights_found': len(insights),
            'insights': insights,
            'session_id': session_id
        }
    
    def _create_memory(self, db: Session, content: str, memory_type: str,
                      priority: str, meta_data: Dict[str, Any]) -> MemoryItem:
        """Helper to create a memory item"""
        
        # Add type and priority to metadata since MemoryItem doesn't have these fields
        enhanced_meta = {
            **meta_data,
            'memory_type': memory_type,
            'priority': priority
        }
        
        # Generate content hash
        import hashlib
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        
        # Create tags based on type and priority
        tags = [memory_type, f"priority_{priority}"]
        if 'tags' in meta_data:
            tags.extend(meta_data['tags'])
        
        memory = MemoryItem(
            content=content,
            content_hash=content_hash,
            tags=tags,
            meta_data=enhanced_meta
        )
        
        db.add(memory)
        db.commit()
        db.refresh(memory)
        
        return memory
    
    def get_workflow_stats(self, db: Session, session_id: Optional[str] = None,
                          time_range: int = 7) -> Dict[str, Any]:
        """Get statistics about workflow integration"""
        
        query = db.query(MemoryItem).filter(
            MemoryItem.created_at >= datetime.utcnow() - timedelta(days=time_range)
        )
        
        if session_id:
            query = query.filter(
                cast(MemoryItem.meta_data, String).contains(f'"session_id": "{session_id}"')
            )
        
        # Filter for auto-captured memories
        auto_captured = query.filter(
            cast(MemoryItem.meta_data, String).contains('"auto_extracted": true')
        ).all()
        
        tool_captured = query.filter(
            cast(MemoryItem.meta_data, String).contains('"tool":')
        ).all()
        
        discoveries = query.filter(
            cast(MemoryItem.meta_data, String).contains('"discovery_type":')
        ).all()
        
        return {
            'time_range_days': time_range,
            'session_id': session_id,
            'stats': {
                'auto_captured_insights': len(auto_captured),
                'tool_usage_memories': len(tool_captured),
                'discoveries_saved': len(discoveries),
                'total_workflow_memories': len(auto_captured) + len(tool_captured) + len(discoveries)
            },
            'breakdown': {
                'by_type': self._count_by_field(auto_captured + tool_captured + discoveries, 'type'),
                'by_priority': self._count_by_field(auto_captured + tool_captured + discoveries, 'priority')
            }
        }
    
    def _count_by_field(self, memories: List[MemoryItem], field: str) -> Dict[str, int]:
        """Count memories by a specific field"""
        counts = {}
        for memory in memories:
            # Since type and priority are in metadata, extract from there
            if field in ['memory_type', 'priority']:
                value = memory.meta_data.get(field, 'unknown')
            else:
                value = getattr(memory, field, 'unknown')
            counts[value] = counts.get(value, 0) + 1
        return counts


from datetime import timedelta