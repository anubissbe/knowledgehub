"""Pattern Learning Service

This service handles pattern extraction, storage, and management for the learning system.
It identifies patterns in user interactions, code structures, and decision-making processes.
"""

import logging
import hashlib
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timezone
from uuid import UUID, uuid4

from sqlalchemy.orm import Session
from sqlalchemy import select, and_, or_, func
from sqlalchemy.exc import IntegrityError

from ..models.learning_pattern import LearningPattern, PatternType
from ..models.user_feedback import UserFeedback
from ...memory_system.models.memory import Memory, MemoryType

logger = logging.getLogger(__name__)


class PatternLearningService:
    """Service for learning and managing patterns from interactions"""
    
    def __init__(self, db: Session):
        """Initialize the pattern learning service"""
        self.db = db
        
        # Pattern extraction configuration
        self.min_pattern_length = 3
        self.max_pattern_length = 100
        self.similarity_threshold = 0.85
        
        # Pattern type mappings
        self.pattern_extractors = {
            'code': self._extract_code_patterns,
            'interaction': self._extract_interaction_patterns,
            'decision': self._extract_decision_patterns,
            'error': self._extract_error_patterns,
            'workflow': self._extract_workflow_patterns
        }
    
    async def extract_patterns(
        self,
        interaction_data: Dict[str, Any]
    ) -> List[LearningPattern]:
        """Extract patterns from interaction data
        
        Args:
            interaction_data: Data containing user input, system response, context, etc.
            
        Returns:
            List of extracted patterns
        """
        patterns = []
        
        try:
            # Determine interaction type
            interaction_type = self._determine_interaction_type(interaction_data)
            
            # Extract patterns based on type
            if interaction_type in self.pattern_extractors:
                extractor = self.pattern_extractors[interaction_type]
                extracted = await extractor(interaction_data)
                patterns.extend(extracted)
            
            # Extract cross-cutting patterns
            common_patterns = await self._extract_common_patterns(interaction_data)
            patterns.extend(common_patterns)
            
            # Deduplicate patterns
            patterns = self._deduplicate_patterns(patterns)
            
            logger.info(f"Extracted {len(patterns)} patterns from interaction")
            return patterns
            
        except Exception as e:
            logger.error(f"Error extracting patterns: {e}")
            return []
    
    async def store_pattern(self, pattern: LearningPattern) -> LearningPattern:
        """Store a pattern in the database
        
        Args:
            pattern: Pattern to store
            
        Returns:
            Stored pattern
        """
        try:
            # Check for existing pattern
            existing = await self._find_existing_pattern(pattern.pattern_hash)
            
            if existing:
                # Update existing pattern
                return await self._update_existing_pattern(existing, pattern)
            else:
                # Store new pattern
                pattern.id = uuid4()
                pattern.created_at = datetime.now(timezone.utc)
                pattern.updated_at = datetime.now(timezone.utc)
                pattern.usage_count = 1
                pattern.last_used = datetime.now(timezone.utc)
                
                self.db.add(pattern)
                self.db.commit()
                self.db.refresh(pattern)
                
                return pattern
                
        except IntegrityError as e:
            logger.error(f"Integrity error storing pattern: {e}")
            self.db.rollback()
            # Try to find the existing pattern
            existing = await self._find_existing_pattern(pattern.pattern_hash)
            if existing:
                return existing
            raise
        except Exception as e:
            logger.error(f"Error storing pattern: {e}")
            self.db.rollback()
            raise
    
    async def update_pattern_from_feedback(
        self,
        pattern: LearningPattern,
        feedback: UserFeedback
    ) -> LearningPattern:
        """Update a pattern based on user feedback
        
        Args:
            pattern: Pattern to update
            feedback: User feedback
            
        Returns:
            Updated pattern
        """
        try:
            # Adjust confidence based on feedback
            if feedback.feedback_type == 'correction':
                # Corrections reduce confidence
                pattern.confidence_score *= 0.8
            elif feedback.feedback_type == 'rating':
                # Ratings adjust confidence proportionally
                rating = feedback.feedback_data.get('rating', 3) / 5.0
                pattern.confidence_score = (
                    pattern.confidence_score * 0.7 + rating * 0.3
                )
            elif feedback.feedback_type == 'confirmation':
                # Confirmations increase confidence
                pattern.confidence_score = min(
                    1.0,
                    pattern.confidence_score * 1.1
                )
            
            # Update pattern data with feedback insights
            if feedback.corrected_content:
                pattern.pattern_data['corrections'] = pattern.pattern_data.get(
                    'corrections', []
                )
                pattern.pattern_data['corrections'].append({
                    'original': feedback.original_content,
                    'corrected': feedback.corrected_content,
                    'timestamp': datetime.now(timezone.utc).isoformat()
                })
            
            # Update metadata
            pattern.updated_at = datetime.now(timezone.utc)
            pattern.pattern_data['feedback_count'] = pattern.pattern_data.get(
                'feedback_count', 0
            ) + 1
            
            self.db.commit()
            return pattern
            
        except Exception as e:
            logger.error(f"Error updating pattern from feedback: {e}")
            self.db.rollback()
            raise
    
    async def find_similar_patterns(
        self,
        pattern_data: Dict[str, Any],
        threshold: float = 0.85
    ) -> List[LearningPattern]:
        """Find patterns similar to the given pattern data
        
        Args:
            pattern_data: Pattern data to compare
            threshold: Similarity threshold (0.0 to 1.0)
            
        Returns:
            List of similar patterns
        """
        try:
            # For now, use pattern type and key matching
            # In production, this would use vector similarity
            pattern_type = pattern_data.get('type')
            keywords = pattern_data.get('keywords', [])
            
            query = select(LearningPattern)
            
            if pattern_type:
                query = query.where(
                    LearningPattern.pattern_data['type'].astext == pattern_type
                )
            
            if keywords:
                # Match patterns with similar keywords
                keyword_conditions = []
                for keyword in keywords:
                    keyword_conditions.append(
                        LearningPattern.pattern_data['keywords'].contains([keyword])
                    )
                if keyword_conditions:
                    query = query.where(or_(*keyword_conditions))
            
            result = await self.db.execute(query)
            patterns = result.scalars().all()
            
            # Calculate similarity scores
            similar_patterns = []
            for pattern in patterns:
                similarity = self._calculate_similarity(pattern_data, pattern.pattern_data)
                if similarity >= threshold:
                    similar_patterns.append(pattern)
            
            return similar_patterns
            
        except Exception as e:
            logger.error(f"Error finding similar patterns: {e}")
            return []
    
    # Private methods
    
    def _determine_interaction_type(self, interaction_data: Dict[str, Any]) -> str:
        """Determine the type of interaction"""
        user_input = interaction_data.get('user_input', '').lower()
        context = interaction_data.get('context', {})
        
        # Code-related interactions
        if any(keyword in user_input for keyword in ['code', 'function', 'class', 'implement', 'fix', 'bug']):
            return 'code'
        
        # Decision-making interactions
        if any(keyword in user_input for keyword in ['should', 'which', 'choose', 'decide', 'recommend']):
            return 'decision'
        
        # Error-related interactions
        if any(keyword in user_input for keyword in ['error', 'exception', 'fail', 'wrong', 'issue']):
            return 'error'
        
        # Workflow interactions
        if any(keyword in user_input for keyword in ['workflow', 'process', 'steps', 'how to']):
            return 'workflow'
        
        return 'interaction'
    
    async def _extract_code_patterns(
        self,
        interaction_data: Dict[str, Any]
    ) -> List[LearningPattern]:
        """Extract patterns from code-related interactions"""
        patterns = []
        
        # Extract language patterns
        languages = self._extract_languages(interaction_data)
        if languages:
            pattern = LearningPattern(
                pattern_type=PatternType.CODE,
                pattern_data={
                    'type': 'language_preference',
                    'languages': languages,
                    'context': interaction_data.get('context', {})
                },
                pattern_hash=self._hash_pattern({
                    'type': 'language_preference',
                    'languages': sorted(languages)
                }),
                confidence_score=0.7,
                source='code_interaction'
            )
            patterns.append(pattern)
        
        # Extract framework patterns
        frameworks = self._extract_frameworks(interaction_data)
        if frameworks:
            pattern = LearningPattern(
                pattern_type=PatternType.CODE,
                pattern_data={
                    'type': 'framework_usage',
                    'frameworks': frameworks,
                    'context': interaction_data.get('context', {})
                },
                pattern_hash=self._hash_pattern({
                    'type': 'framework_usage',
                    'frameworks': sorted(frameworks)
                }),
                confidence_score=0.7,
                source='code_interaction'
            )
            patterns.append(pattern)
        
        # Extract coding style patterns
        style_indicators = self._extract_coding_style(interaction_data)
        if style_indicators:
            pattern = LearningPattern(
                pattern_type=PatternType.CODE,
                pattern_data={
                    'type': 'coding_style',
                    'style_indicators': style_indicators,
                    'context': interaction_data.get('context', {})
                },
                pattern_hash=self._hash_pattern({
                    'type': 'coding_style',
                    'indicators': style_indicators
                }),
                confidence_score=0.6,
                source='code_interaction'
            )
            patterns.append(pattern)
        
        return patterns
    
    async def _extract_interaction_patterns(
        self,
        interaction_data: Dict[str, Any]
    ) -> List[LearningPattern]:
        """Extract patterns from general interactions"""
        patterns = []
        
        # Extract communication style
        comm_style = self._analyze_communication_style(interaction_data)
        if comm_style:
            pattern = LearningPattern(
                pattern_type=PatternType.PREFERENCE,
                pattern_data={
                    'type': 'communication_style',
                    'style': comm_style,
                    'indicators': self._extract_style_indicators(interaction_data)
                },
                pattern_hash=self._hash_pattern({
                    'type': 'communication_style',
                    'style': comm_style
                }),
                confidence_score=0.6,
                source='user_interaction'
            )
            patterns.append(pattern)
        
        # Extract task patterns
        task_type = self._identify_task_type(interaction_data)
        if task_type:
            pattern = LearningPattern(
                pattern_type=PatternType.WORKFLOW,
                pattern_data={
                    'type': 'task_preference',
                    'task_type': task_type,
                    'approach': self._extract_task_approach(interaction_data)
                },
                pattern_hash=self._hash_pattern({
                    'type': 'task_preference',
                    'task': task_type
                }),
                confidence_score=0.7,
                source='user_interaction'
            )
            patterns.append(pattern)
        
        return patterns
    
    async def _extract_decision_patterns(
        self,
        interaction_data: Dict[str, Any]
    ) -> List[LearningPattern]:
        """Extract patterns from decision-making interactions"""
        patterns = []
        
        # Extract decision criteria
        criteria = self._extract_decision_criteria(interaction_data)
        if criteria:
            pattern = LearningPattern(
                pattern_type=PatternType.DECISION,
                pattern_data={
                    'type': 'decision_criteria',
                    'criteria': criteria,
                    'context': interaction_data.get('context', {})
                },
                pattern_hash=self._hash_pattern({
                    'type': 'decision_criteria',
                    'criteria': sorted(criteria)
                }),
                confidence_score=0.75,
                source='decision_interaction'
            )
            patterns.append(pattern)
        
        # Extract risk tolerance
        risk_level = self._assess_risk_tolerance(interaction_data)
        if risk_level:
            pattern = LearningPattern(
                pattern_type=PatternType.PREFERENCE,
                pattern_data={
                    'type': 'risk_tolerance',
                    'level': risk_level,
                    'indicators': self._extract_risk_indicators(interaction_data)
                },
                pattern_hash=self._hash_pattern({
                    'type': 'risk_tolerance',
                    'level': risk_level
                }),
                confidence_score=0.65,
                source='decision_interaction'
            )
            patterns.append(pattern)
        
        return patterns
    
    async def _extract_error_patterns(
        self,
        interaction_data: Dict[str, Any]
    ) -> List[LearningPattern]:
        """Extract patterns from error-related interactions"""
        patterns = []
        
        # Extract error type patterns
        error_info = self._extract_error_information(interaction_data)
        if error_info:
            pattern = LearningPattern(
                pattern_type=PatternType.ERROR,
                pattern_data={
                    'type': 'error_pattern',
                    'error_type': error_info.get('type'),
                    'context': error_info.get('context'),
                    'resolution': error_info.get('resolution')
                },
                pattern_hash=self._hash_pattern({
                    'type': 'error_pattern',
                    'error': error_info.get('type')
                }),
                confidence_score=0.8,
                source='error_interaction'
            )
            patterns.append(pattern)
        
        # Extract debugging approach patterns
        debug_approach = self._extract_debugging_approach(interaction_data)
        if debug_approach:
            pattern = LearningPattern(
                pattern_type=PatternType.WORKFLOW,
                pattern_data={
                    'type': 'debugging_approach',
                    'approach': debug_approach,
                    'tools': self._extract_debugging_tools(interaction_data)
                },
                pattern_hash=self._hash_pattern({
                    'type': 'debugging_approach',
                    'approach': debug_approach
                }),
                confidence_score=0.7,
                source='error_interaction'
            )
            patterns.append(pattern)
        
        return patterns
    
    async def _extract_workflow_patterns(
        self,
        interaction_data: Dict[str, Any]
    ) -> List[LearningPattern]:
        """Extract patterns from workflow interactions"""
        patterns = []
        
        # Extract workflow steps
        workflow_steps = self._extract_workflow_steps(interaction_data)
        if workflow_steps:
            pattern = LearningPattern(
                pattern_type=PatternType.WORKFLOW,
                pattern_data={
                    'type': 'workflow_sequence',
                    'steps': workflow_steps,
                    'context': interaction_data.get('context', {})
                },
                pattern_hash=self._hash_pattern({
                    'type': 'workflow_sequence',
                    'steps': workflow_steps
                }),
                confidence_score=0.75,
                source='workflow_interaction'
            )
            patterns.append(pattern)
        
        # Extract tool preferences
        tool_prefs = self._extract_tool_preferences(interaction_data)
        if tool_prefs:
            pattern = LearningPattern(
                pattern_type=PatternType.PREFERENCE,
                pattern_data={
                    'type': 'tool_preference',
                    'tools': tool_prefs,
                    'context': interaction_data.get('context', {})
                },
                pattern_hash=self._hash_pattern({
                    'type': 'tool_preference',
                    'tools': sorted(tool_prefs)
                }),
                confidence_score=0.7,
                source='workflow_interaction'
            )
            patterns.append(pattern)
        
        return patterns
    
    async def _extract_common_patterns(
        self,
        interaction_data: Dict[str, Any]
    ) -> List[LearningPattern]:
        """Extract patterns common to all interaction types"""
        patterns = []
        
        # Extract timing patterns
        timing_pattern = self._extract_timing_pattern(interaction_data)
        if timing_pattern:
            pattern = LearningPattern(
                pattern_type=PatternType.PREFERENCE,
                pattern_data={
                    'type': 'timing_preference',
                    'pattern': timing_pattern,
                    'indicators': self._extract_timing_indicators(interaction_data)
                },
                pattern_hash=self._hash_pattern({
                    'type': 'timing_preference',
                    'pattern': timing_pattern
                }),
                confidence_score=0.5,
                source='common_interaction'
            )
            patterns.append(pattern)
        
        # Extract success patterns
        if interaction_data.get('outcome', {}).get('success'):
            success_factors = self._extract_success_factors(interaction_data)
            if success_factors:
                pattern = LearningPattern(
                    pattern_type=PatternType.SUCCESS,
                    pattern_data={
                        'type': 'success_pattern',
                        'factors': success_factors,
                        'context': interaction_data.get('context', {})
                    },
                    pattern_hash=self._hash_pattern({
                        'type': 'success_pattern',
                        'factors': sorted(success_factors)
                    }),
                    confidence_score=0.8,
                    source='success_outcome'
                )
                patterns.append(pattern)
        
        return patterns
    
    def _deduplicate_patterns(
        self,
        patterns: List[LearningPattern]
    ) -> List[LearningPattern]:
        """Remove duplicate patterns from the list"""
        seen_hashes = set()
        unique_patterns = []
        
        for pattern in patterns:
            if pattern.pattern_hash not in seen_hashes:
                seen_hashes.add(pattern.pattern_hash)
                unique_patterns.append(pattern)
        
        return unique_patterns
    
    async def _find_existing_pattern(self, pattern_hash: str) -> Optional[LearningPattern]:
        """Find an existing pattern by hash"""
        result = await self.db.execute(
            select(LearningPattern).where(
                LearningPattern.pattern_hash == pattern_hash
            )
        )
        return result.scalar_one_or_none()
    
    async def _update_existing_pattern(
        self,
        existing: LearningPattern,
        new: LearningPattern
    ) -> LearningPattern:
        """Update an existing pattern with new information"""
        # Increment usage count
        existing.usage_count += 1
        existing.last_used = datetime.now(timezone.utc)
        
        # Update confidence (weighted average)
        total_weight = existing.usage_count
        existing.confidence_score = (
            (existing.confidence_score * (total_weight - 1) + new.confidence_score) /
            total_weight
        )
        
        # Merge pattern data
        existing.pattern_data = self._merge_pattern_data(
            existing.pattern_data,
            new.pattern_data
        )
        
        existing.updated_at = datetime.now(timezone.utc)
        
        self.db.commit()
        return existing
    
    def _merge_pattern_data(
        self,
        existing: Dict[str, Any],
        new: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge pattern data intelligently"""
        merged = existing.copy()
        
        for key, value in new.items():
            if key not in merged:
                merged[key] = value
            elif isinstance(value, list) and isinstance(merged[key], list):
                # Merge lists without duplicates
                merged[key] = list(set(merged[key] + value))
            elif isinstance(value, dict) and isinstance(merged[key], dict):
                # Recursively merge dictionaries
                merged[key] = self._merge_pattern_data(merged[key], value)
            elif key == 'confidence' and isinstance(value, (int, float)):
                # Average confidence values
                merged[key] = (merged[key] + value) / 2
        
        return merged
    
    def _calculate_similarity(
        self,
        pattern1: Dict[str, Any],
        pattern2: Dict[str, Any]
    ) -> float:
        """Calculate similarity between two patterns"""
        # Simple similarity based on common keys and values
        # In production, this would use more sophisticated methods
        
        if pattern1.get('type') != pattern2.get('type'):
            return 0.0
        
        # Calculate Jaccard similarity for lists
        similarity_scores = []
        
        for key in ['keywords', 'languages', 'frameworks', 'tools']:
            if key in pattern1 and key in pattern2:
                set1 = set(pattern1[key]) if isinstance(pattern1[key], list) else {pattern1[key]}
                set2 = set(pattern2[key]) if isinstance(pattern2[key], list) else {pattern2[key]}
                
                if set1 or set2:
                    jaccard = len(set1 & set2) / len(set1 | set2)
                    similarity_scores.append(jaccard)
        
        # Calculate string similarity for other fields
        for key in ['style', 'approach', 'level']:
            if key in pattern1 and key in pattern2:
                if pattern1[key] == pattern2[key]:
                    similarity_scores.append(1.0)
                else:
                    similarity_scores.append(0.0)
        
        if not similarity_scores:
            return 0.0
        
        return sum(similarity_scores) / len(similarity_scores)
    
    def _hash_pattern(self, pattern_data: Dict[str, Any]) -> str:
        """Generate a hash for pattern data"""
        # Sort keys for consistent hashing
        sorted_data = json.dumps(pattern_data, sort_keys=True)
        return hashlib.sha256(sorted_data.encode()).hexdigest()
    
    # Helper methods for pattern extraction
    
    def _extract_languages(self, interaction_data: Dict[str, Any]) -> List[str]:
        """Extract programming languages from interaction"""
        languages = []
        text = interaction_data.get('user_input', '') + ' ' + interaction_data.get('system_response', '')
        
        # Common programming languages
        lang_keywords = {
            'python': ['python', 'py', 'django', 'flask'],
            'javascript': ['javascript', 'js', 'node', 'react', 'vue'],
            'typescript': ['typescript', 'ts', 'angular'],
            'java': ['java', 'spring', 'maven'],
            'csharp': ['c#', 'csharp', '.net', 'dotnet'],
            'go': ['golang', 'go '],
            'rust': ['rust', 'cargo'],
            'cpp': ['c++', 'cpp'],
            'sql': ['sql', 'query', 'database']
        }
        
        text_lower = text.lower()
        for lang, keywords in lang_keywords.items():
            if any(kw in text_lower for kw in keywords):
                languages.append(lang)
        
        return languages
    
    def _extract_frameworks(self, interaction_data: Dict[str, Any]) -> List[str]:
        """Extract frameworks from interaction"""
        frameworks = []
        text = interaction_data.get('user_input', '') + ' ' + interaction_data.get('system_response', '')
        
        # Common frameworks
        framework_keywords = {
            'react': ['react', 'jsx', 'useState', 'useEffect'],
            'vue': ['vue', 'vuex', 'vue.js'],
            'angular': ['angular', '@angular'],
            'django': ['django', 'models.py', 'views.py'],
            'flask': ['flask', 'app.route'],
            'express': ['express', 'app.get', 'app.post'],
            'spring': ['spring', '@Controller', '@Service'],
            'fastapi': ['fastapi', 'uvicorn', 'pydantic'],
            'nextjs': ['next.js', 'nextjs', 'getServerSideProps']
        }
        
        text_lower = text.lower()
        for framework, keywords in framework_keywords.items():
            if any(kw in text_lower for kw in keywords):
                frameworks.append(framework)
        
        return frameworks
    
    def _extract_coding_style(self, interaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract coding style indicators"""
        style = {}
        code = interaction_data.get('system_response', '')
        
        # Check for async/await usage
        if 'async ' in code or 'await ' in code:
            style['async'] = True
        
        # Check for type hints (Python)
        if '->' in code or ': ' in code:
            style['type_hints'] = True
        
        # Check for documentation
        if '"""' in code or "'''" in code or '/**' in code:
            style['documented'] = True
        
        # Check for testing
        if 'test_' in code or 'Test' in code or 'describe(' in code:
            style['testing'] = True
        
        return style
    
    def _analyze_communication_style(self, interaction_data: Dict[str, Any]) -> str:
        """Analyze user's communication style"""
        user_input = interaction_data.get('user_input', '')
        
        # Length-based analysis
        if len(user_input) < 50:
            return 'concise'
        elif len(user_input) > 200:
            return 'detailed'
        
        # Formality analysis
        formal_indicators = ['please', 'could you', 'would you', 'kindly']
        if any(indicator in user_input.lower() for indicator in formal_indicators):
            return 'formal'
        
        return 'casual'
    
    def _extract_style_indicators(self, interaction_data: Dict[str, Any]) -> List[str]:
        """Extract specific style indicators"""
        indicators = []
        user_input = interaction_data.get('user_input', '').lower()
        
        if '?' in user_input:
            indicators.append('questioning')
        if '!' in user_input:
            indicators.append('emphatic')
        if 'asap' in user_input or 'urgent' in user_input:
            indicators.append('urgent')
        if 'please' in user_input:
            indicators.append('polite')
        
        return indicators
    
    def _identify_task_type(self, interaction_data: Dict[str, Any]) -> str:
        """Identify the type of task requested"""
        user_input = interaction_data.get('user_input', '').lower()
        
        task_keywords = {
            'implementation': ['implement', 'create', 'build', 'develop'],
            'debugging': ['fix', 'debug', 'error', 'issue', 'problem'],
            'optimization': ['optimize', 'improve', 'faster', 'performance'],
            'refactoring': ['refactor', 'clean', 'reorganize', 'restructure'],
            'documentation': ['document', 'explain', 'describe', 'comment'],
            'testing': ['test', 'verify', 'validate', 'check']
        }
        
        for task_type, keywords in task_keywords.items():
            if any(kw in user_input for kw in keywords):
                return task_type
        
        return 'general'
    
    def _extract_task_approach(self, interaction_data: Dict[str, Any]) -> str:
        """Extract the preferred approach to tasks"""
        user_input = interaction_data.get('user_input', '').lower()
        
        if 'step by step' in user_input or 'gradually' in user_input:
            return 'incremental'
        elif 'quick' in user_input or 'fast' in user_input:
            return 'rapid'
        elif 'thorough' in user_input or 'comprehensive' in user_input:
            return 'comprehensive'
        
        return 'balanced'
    
    def _extract_decision_criteria(self, interaction_data: Dict[str, Any]) -> List[str]:
        """Extract decision-making criteria"""
        criteria = []
        user_input = interaction_data.get('user_input', '').lower()
        
        criteria_keywords = {
            'performance': ['performance', 'speed', 'fast', 'efficient'],
            'maintainability': ['maintainable', 'clean', 'readable', 'simple'],
            'security': ['secure', 'security', 'safe', 'vulnerability'],
            'scalability': ['scalable', 'scale', 'growth', 'expand'],
            'cost': ['cost', 'cheap', 'expensive', 'budget'],
            'compatibility': ['compatible', 'work with', 'integrate']
        }
        
        for criterion, keywords in criteria_keywords.items():
            if any(kw in user_input for kw in keywords):
                criteria.append(criterion)
        
        return criteria
    
    def _assess_risk_tolerance(self, interaction_data: Dict[str, Any]) -> str:
        """Assess user's risk tolerance"""
        user_input = interaction_data.get('user_input', '').lower()
        
        conservative_indicators = ['safe', 'stable', 'proven', 'reliable', 'conservative']
        aggressive_indicators = ['latest', 'cutting-edge', 'experimental', 'new', 'innovative']
        
        conservative_count = sum(1 for ind in conservative_indicators if ind in user_input)
        aggressive_count = sum(1 for ind in aggressive_indicators if ind in user_input)
        
        if conservative_count > aggressive_count:
            return 'conservative'
        elif aggressive_count > conservative_count:
            return 'aggressive'
        
        return 'moderate'
    
    def _extract_risk_indicators(self, interaction_data: Dict[str, Any]) -> List[str]:
        """Extract specific risk indicators"""
        indicators = []
        user_input = interaction_data.get('user_input', '').lower()
        
        if 'production' in user_input:
            indicators.append('production_aware')
        if 'test' in user_input or 'sandbox' in user_input:
            indicators.append('testing_focused')
        if 'backup' in user_input:
            indicators.append('backup_conscious')
        
        return indicators
    
    def _extract_error_information(self, interaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract error information from interaction"""
        error_info = {}
        text = interaction_data.get('user_input', '') + ' ' + interaction_data.get('system_response', '')
        
        # Common error types
        error_types = {
            'syntax': ['SyntaxError', 'syntax error', 'unexpected token'],
            'type': ['TypeError', 'type error', 'wrong type'],
            'value': ['ValueError', 'value error', 'invalid value'],
            'import': ['ImportError', 'ModuleNotFoundError', 'cannot import'],
            'attribute': ['AttributeError', 'has no attribute'],
            'key': ['KeyError', 'key error'],
            'index': ['IndexError', 'index out of range']
        }
        
        for error_type, indicators in error_types.items():
            if any(ind in text for ind in indicators):
                error_info['type'] = error_type
                break
        
        # Extract context
        if 'line' in text.lower():
            error_info['context'] = 'line_specific'
        elif 'file' in text.lower():
            error_info['context'] = 'file_specific'
        
        # Check if resolution was provided
        if 'fixed' in text.lower() or 'solved' in text.lower():
            error_info['resolution'] = 'provided'
        
        return error_info
    
    def _extract_debugging_approach(self, interaction_data: Dict[str, Any]) -> str:
        """Extract debugging approach from interaction"""
        text = interaction_data.get('user_input', '') + ' ' + interaction_data.get('system_response', '')
        
        if 'print' in text or 'console.log' in text:
            return 'print_debugging'
        elif 'debugger' in text or 'breakpoint' in text:
            return 'interactive_debugging'
        elif 'log' in text:
            return 'logging'
        elif 'test' in text:
            return 'test_driven'
        
        return 'systematic'
    
    def _extract_debugging_tools(self, interaction_data: Dict[str, Any]) -> List[str]:
        """Extract debugging tools mentioned"""
        tools = []
        text = interaction_data.get('user_input', '') + ' ' + interaction_data.get('system_response', '')
        
        tool_keywords = {
            'pdb': ['pdb', 'python debugger'],
            'chrome_devtools': ['chrome devtools', 'developer tools'],
            'vscode_debugger': ['vscode debugger', 'vs code debugger'],
            'pytest': ['pytest', 'py.test'],
            'jest': ['jest', 'jest test'],
            'postman': ['postman', 'api testing']
        }
        
        text_lower = text.lower()
        for tool, keywords in tool_keywords.items():
            if any(kw in text_lower for kw in keywords):
                tools.append(tool)
        
        return tools
    
    def _extract_workflow_steps(self, interaction_data: Dict[str, Any]) -> List[str]:
        """Extract workflow steps from interaction"""
        steps = []
        text = interaction_data.get('system_response', '')
        
        # Look for numbered lists
        import re
        numbered_pattern = r'^\s*\d+\.\s+(.+)$'
        for line in text.split('\n'):
            match = re.match(numbered_pattern, line)
            if match:
                steps.append(match.group(1).strip())
        
        # Look for bullet points
        if not steps:
            bullet_pattern = r'^\s*[-*]\s+(.+)$'
            for line in text.split('\n'):
                match = re.match(bullet_pattern, line)
                if match:
                    steps.append(match.group(1).strip())
        
        return steps[:10]  # Limit to 10 steps
    
    def _extract_tool_preferences(self, interaction_data: Dict[str, Any]) -> List[str]:
        """Extract tool preferences from interaction"""
        tools = []
        text = interaction_data.get('user_input', '') + ' ' + interaction_data.get('system_response', '')
        
        # Development tools
        dev_tools = {
            'vscode': ['vscode', 'visual studio code'],
            'pycharm': ['pycharm'],
            'intellij': ['intellij'],
            'vim': ['vim', 'neovim'],
            'git': ['git', 'github', 'gitlab'],
            'docker': ['docker', 'container'],
            'kubernetes': ['kubernetes', 'k8s'],
            'terraform': ['terraform'],
            'ansible': ['ansible']
        }
        
        text_lower = text.lower()
        for tool, keywords in dev_tools.items():
            if any(kw in text_lower for kw in keywords):
                tools.append(tool)
        
        return tools
    
    def _extract_timing_pattern(self, interaction_data: Dict[str, Any]) -> str:
        """Extract timing patterns from interaction"""
        timestamp = interaction_data.get('timestamp')
        if not timestamp:
            return None
        
        # Extract hour of day
        hour = datetime.fromisoformat(timestamp).hour
        
        if 6 <= hour < 12:
            return 'morning'
        elif 12 <= hour < 17:
            return 'afternoon'
        elif 17 <= hour < 22:
            return 'evening'
        else:
            return 'night'
    
    def _extract_timing_indicators(self, interaction_data: Dict[str, Any]) -> List[str]:
        """Extract timing-related indicators"""
        indicators = []
        user_input = interaction_data.get('user_input', '').lower()
        
        if 'asap' in user_input or 'urgent' in user_input:
            indicators.append('urgent')
        if 'deadline' in user_input:
            indicators.append('deadline_driven')
        if 'when you can' in user_input or 'no rush' in user_input:
            indicators.append('flexible')
        
        return indicators
    
    def _extract_success_factors(self, interaction_data: Dict[str, Any]) -> List[str]:
        """Extract factors that contributed to success"""
        factors = []
        
        # Check if user was satisfied
        if interaction_data.get('user_feedback', {}).get('positive'):
            factors.append('user_satisfied')
        
        # Check if task was completed
        if interaction_data.get('outcome', {}).get('task_completed'):
            factors.append('task_completed')
        
        # Check response time
        if interaction_data.get('response_time', 0) < 2:
            factors.append('quick_response')
        
        # Check if solution worked first time
        if interaction_data.get('outcome', {}).get('first_try_success'):
            factors.append('first_try_success')
        
        return factors