"""Adaptation Engine Service

This service handles behavioral adaptation based on learned patterns,
adjusting system responses and approaches based on user preferences and success patterns.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime, timezone, timedelta
from uuid import UUID, uuid4
import json

from sqlalchemy.orm import Session
from sqlalchemy import select, and_, or_, func

from ..models.learning_pattern import LearningPattern, PatternType
from ...memory_system.models.memory import Memory, MemoryType
from ...services.cache import redis_client

logger = logging.getLogger(__name__)


class AdaptationEngine:
    """Service for applying behavioral adaptations based on learned patterns"""
    
    def __init__(self, db: Session):
        """Initialize the adaptation engine"""
        self.db = db
        
        # Adaptation configuration
        self.min_pattern_confidence = 0.7
        self.adaptation_threshold = 0.75
        self.max_adaptations_per_request = 5
        
        # Adaptation strategies
        self.adaptation_strategies = {
            PatternType.CODE: self._adapt_code_generation,
            PatternType.PREFERENCE: self._adapt_user_preferences,
            PatternType.WORKFLOW: self._adapt_workflow_approach,
            PatternType.SUCCESS: self._adapt_based_on_success,
            PatternType.ERROR: self._adapt_to_avoid_errors,
            PatternType.DECISION: self._adapt_decision_making,
            PatternType.CORRECTION: self._adapt_from_corrections
        }
        
        # Cache for active adaptations
        self._active_adaptations = {}
        self._adaptation_cache_ttl = 300  # 5 minutes
    
    async def apply_adaptations(
        self,
        patterns: List[LearningPattern],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply adaptations based on learned patterns
        
        Args:
            patterns: List of applicable patterns
            context: Current context for adaptation
            
        Returns:
            Dictionary with applied adaptations
        """
        try:
            # Filter patterns by confidence
            high_confidence_patterns = [
                p for p in patterns
                if p.confidence_score >= self.min_pattern_confidence
            ]
            
            if not high_confidence_patterns:
                return {
                    'adaptations_applied': 0,
                    'message': 'No high-confidence patterns available'
                }
            
            # Sort by confidence and limit
            sorted_patterns = sorted(
                high_confidence_patterns,
                key=lambda p: p.confidence_score,
                reverse=True
            )[:self.max_adaptations_per_request]
            
            # Apply adaptations by pattern type
            adaptations = []
            for pattern in sorted_patterns:
                pattern_type = PatternType(pattern.pattern_type)
                if pattern_type in self.adaptation_strategies:
                    strategy = self.adaptation_strategies[pattern_type]
                    adaptation = await strategy(pattern, context)
                    if adaptation:
                        adaptations.append(adaptation)
            
            # Merge and prioritize adaptations
            merged_adaptations = await self._merge_adaptations(adaptations)
            
            # Cache active adaptations
            await self._cache_adaptations(context.get('session_id'), merged_adaptations)
            
            return {
                'adaptations_applied': len(merged_adaptations),
                'adaptations': merged_adaptations,
                'confidence': self._calculate_overall_confidence(sorted_patterns),
                'adapted_context': await self._apply_to_context(context, merged_adaptations)
            }
            
        except Exception as e:
            logger.error(f"Error applying adaptations: {e}")
            return {
                'adaptations_applied': 0,
                'error': str(e)
            }
    
    async def trigger_adaptation(
        self,
        patterns: List[LearningPattern]
    ) -> Dict[str, Any]:
        """Trigger behavioral adaptation based on high-confidence patterns
        
        Args:
            patterns: Patterns that triggered adaptation
            
        Returns:
            Dictionary with adaptation trigger results
        """
        try:
            # Analyze patterns to determine adaptation type
            adaptation_type = await self._determine_adaptation_type(patterns)
            
            # Create adaptation record
            adaptation_record = {
                'id': str(uuid4()),
                'triggered_at': datetime.now(timezone.utc).isoformat(),
                'pattern_count': len(patterns),
                'adaptation_type': adaptation_type,
                'patterns': [
                    {
                        'id': str(p.id),
                        'type': p.pattern_type,
                        'confidence': p.confidence_score
                    }
                    for p in patterns
                ]
            }
            
            # Store adaptation trigger
            await self._store_adaptation_trigger(adaptation_record)
            
            # Schedule adaptation application
            await self._schedule_adaptation(adaptation_record)
            
            return {
                'triggered': True,
                'adaptation_id': adaptation_record['id'],
                'type': adaptation_type,
                'pattern_count': len(patterns)
            }
            
        except Exception as e:
            logger.error(f"Error triggering adaptation: {e}")
            return {
                'triggered': False,
                'error': str(e)
            }
    
    async def get_active_adaptations(
        self,
        session_id: Optional[UUID] = None
    ) -> List[Dict[str, Any]]:
        """Get currently active adaptations
        
        Args:
            session_id: Optional session ID to filter by
            
        Returns:
            List of active adaptations
        """
        if session_id:
            cached = self._active_adaptations.get(str(session_id))
            if cached and cached['expires'] > datetime.now(timezone.utc):
                return cached['adaptations']
        
        # Return all active adaptations if no session specified
        active = []
        for session_key, data in self._active_adaptations.items():
            if data['expires'] > datetime.now(timezone.utc):
                active.extend(data['adaptations'])
        
        return active
    
    async def evaluate_adaptation_effectiveness(
        self,
        adaptation_id: str,
        outcome_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evaluate how effective an adaptation was
        
        Args:
            adaptation_id: ID of the adaptation to evaluate
            outcome_data: Data about the outcome after adaptation
            
        Returns:
            Dictionary with effectiveness evaluation
        """
        try:
            # Find adaptation record
            adaptation = await self._get_adaptation_record(adaptation_id)
            if not adaptation:
                return {'error': 'Adaptation not found'}
            
            # Calculate effectiveness score
            effectiveness_score = await self._calculate_effectiveness_score(
                adaptation,
                outcome_data
            )
            
            # Update adaptation patterns based on effectiveness
            if effectiveness_score > 0.8:
                await self._reinforce_adaptation_patterns(adaptation)
            elif effectiveness_score < 0.4:
                await self._weaken_adaptation_patterns(adaptation)
            
            return {
                'adaptation_id': adaptation_id,
                'effectiveness_score': effectiveness_score,
                'outcome': 'positive' if effectiveness_score > 0.7 else 'negative',
                'patterns_updated': True
            }
            
        except Exception as e:
            logger.error(f"Error evaluating adaptation: {e}")
            return {'error': str(e)}
    
    # Private adaptation strategies
    
    async def _adapt_code_generation(
        self,
        pattern: LearningPattern,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Adapt code generation based on learned patterns"""
        adaptation = {
            'type': 'code_generation',
            'pattern_id': str(pattern.id),
            'confidence': pattern.confidence_score
        }
        
        pattern_data = pattern.pattern_data
        
        # Language preferences
        if 'languages' in pattern_data:
            adaptation['preferred_languages'] = pattern_data['languages']
        
        # Framework preferences
        if 'frameworks' in pattern_data:
            adaptation['preferred_frameworks'] = pattern_data['frameworks']
        
        # Coding style preferences
        if 'style_indicators' in pattern_data:
            style = pattern_data['style_indicators']
            adaptation['style_preferences'] = {
                'use_async': style.get('async', False),
                'use_type_hints': style.get('type_hints', False),
                'include_docs': style.get('documented', False),
                'include_tests': style.get('testing', False)
            }
        
        # Apply to context
        adaptation['instructions'] = self._generate_code_instructions(adaptation)
        
        return adaptation
    
    async def _adapt_user_preferences(
        self,
        pattern: LearningPattern,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Adapt based on user preferences"""
        adaptation = {
            'type': 'user_preference',
            'pattern_id': str(pattern.id),
            'confidence': pattern.confidence_score
        }
        
        pattern_data = pattern.pattern_data
        preference_type = pattern_data.get('type')
        
        if preference_type == 'communication_style':
            style = pattern_data.get('style', 'balanced')
            adaptation['communication'] = {
                'style': style,
                'verbosity': 'concise' if style == 'concise' else 'detailed',
                'formality': 'formal' if style == 'formal' else 'casual'
            }
        
        elif preference_type == 'timing_preference':
            timing = pattern_data.get('pattern', 'flexible')
            adaptation['timing'] = {
                'preference': timing,
                'urgency_aware': 'urgent' in pattern_data.get('indicators', [])
            }
        
        elif preference_type == 'tool_preference':
            tools = pattern_data.get('tools', [])
            adaptation['tools'] = {
                'preferred': tools,
                'avoid': []  # Could be populated from negative patterns
            }
        
        return adaptation
    
    async def _adapt_workflow_approach(
        self,
        pattern: LearningPattern,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Adapt workflow approach based on patterns"""
        adaptation = {
            'type': 'workflow',
            'pattern_id': str(pattern.id),
            'confidence': pattern.confidence_score
        }
        
        pattern_data = pattern.pattern_data
        workflow_type = pattern_data.get('type')
        
        if workflow_type == 'workflow_sequence':
            steps = pattern_data.get('steps', [])
            adaptation['workflow'] = {
                'suggested_steps': steps,
                'approach': 'sequential'
            }
        
        elif workflow_type == 'task_preference':
            task_type = pattern_data.get('task_type')
            approach = pattern_data.get('approach', 'balanced')
            adaptation['task_handling'] = {
                'type': task_type,
                'approach': approach,
                'methodology': self._get_methodology_for_approach(approach)
            }
        
        elif workflow_type == 'debugging_approach':
            adaptation['debugging'] = {
                'approach': pattern_data.get('approach', 'systematic'),
                'tools': pattern_data.get('tools', [])
            }
        
        return adaptation
    
    async def _adapt_based_on_success(
        self,
        pattern: LearningPattern,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Adapt based on success patterns"""
        adaptation = {
            'type': 'success_based',
            'pattern_id': str(pattern.id),
            'confidence': pattern.confidence_score
        }
        
        pattern_data = pattern.pattern_data
        
        # Extract success factors
        success_factors = pattern_data.get('success_factors', [])
        if success_factors:
            adaptation['emphasize'] = success_factors
            adaptation['priority'] = 'high'
        
        # Apply decision criteria that led to success
        if 'decision_type' in pattern_data:
            adaptation['decision_guidance'] = {
                'type': pattern_data['decision_type'],
                'proven_approach': True,
                'confidence': pattern.confidence_score
            }
        
        return adaptation
    
    async def _adapt_to_avoid_errors(
        self,
        pattern: LearningPattern,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Adapt to avoid known error patterns"""
        adaptation = {
            'type': 'error_avoidance',
            'pattern_id': str(pattern.id),
            'confidence': pattern.confidence_score
        }
        
        pattern_data = pattern.pattern_data
        
        # Error type to avoid
        if 'error_type' in pattern_data:
            adaptation['avoid_errors'] = {
                'type': pattern_data['error_type'],
                'context': pattern_data.get('context', 'general')
            }
        
        # Failure reasons to avoid
        if 'failure_reasons' in pattern_data:
            adaptation['avoid_failures'] = pattern_data['failure_reasons']
            adaptation['preventive_measures'] = self._get_preventive_measures(
                pattern_data['failure_reasons']
            )
        
        return adaptation
    
    async def _adapt_decision_making(
        self,
        pattern: LearningPattern,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Adapt decision-making approach"""
        adaptation = {
            'type': 'decision_making',
            'pattern_id': str(pattern.id),
            'confidence': pattern.confidence_score
        }
        
        pattern_data = pattern.pattern_data
        
        # Decision criteria
        if 'criteria' in pattern_data:
            adaptation['criteria'] = {
                'primary': pattern_data['criteria'],
                'weights': self._get_criteria_weights(pattern_data['criteria'])
            }
        
        # Risk tolerance
        if 'risk_tolerance' in pattern_data:
            adaptation['risk_approach'] = {
                'level': pattern_data['risk_tolerance'],
                'considerations': self._get_risk_considerations(
                    pattern_data['risk_tolerance']
                )
            }
        
        return adaptation
    
    async def _adapt_from_corrections(
        self,
        pattern: LearningPattern,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Adapt based on user corrections"""
        adaptation = {
            'type': 'correction_based',
            'pattern_id': str(pattern.id),
            'confidence': pattern.confidence_score
        }
        
        pattern_data = pattern.pattern_data
        
        # Correction examples
        if 'examples' in pattern_data:
            adaptation['corrections'] = pattern_data['examples']
            adaptation['correction_type'] = pattern_data.get('correction_type', 'general')
        
        # Apply correction rules
        adaptation['rules'] = self._extract_correction_rules(pattern_data)
        
        return adaptation
    
    # Helper methods
    
    async def _merge_adaptations(
        self,
        adaptations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Merge and prioritize multiple adaptations"""
        # Group by type
        grouped = {}
        for adaptation in adaptations:
            adapt_type = adaptation['type']
            if adapt_type not in grouped:
                grouped[adapt_type] = []
            grouped[adapt_type].append(adaptation)
        
        # Merge within each type
        merged = []
        for adapt_type, group in grouped.items():
            if len(group) == 1:
                merged.append(group[0])
            else:
                # Merge multiple adaptations of same type
                merged_adaptation = {
                    'type': adapt_type,
                    'confidence': max(a['confidence'] for a in group),
                    'merged_from': len(group)
                }
                
                # Type-specific merging
                if adapt_type == 'code_generation':
                    merged_adaptation.update(self._merge_code_adaptations(group))
                elif adapt_type == 'user_preference':
                    merged_adaptation.update(self._merge_preference_adaptations(group))
                # Add more type-specific merging as needed
                
                merged.append(merged_adaptation)
        
        # Sort by confidence
        merged.sort(key=lambda a: a['confidence'], reverse=True)
        
        return merged
    
    def _merge_code_adaptations(
        self,
        adaptations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Merge multiple code generation adaptations"""
        merged = {
            'preferred_languages': [],
            'preferred_frameworks': [],
            'style_preferences': {}
        }
        
        for adapt in adaptations:
            if 'preferred_languages' in adapt:
                merged['preferred_languages'].extend(adapt['preferred_languages'])
            if 'preferred_frameworks' in adapt:
                merged['preferred_frameworks'].extend(adapt['preferred_frameworks'])
            if 'style_preferences' in adapt:
                merged['style_preferences'].update(adapt['style_preferences'])
        
        # Remove duplicates
        merged['preferred_languages'] = list(set(merged['preferred_languages']))
        merged['preferred_frameworks'] = list(set(merged['preferred_frameworks']))
        
        return merged
    
    def _merge_preference_adaptations(
        self,
        adaptations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Merge multiple preference adaptations"""
        merged = {}
        
        for adapt in adaptations:
            # Merge all preference data
            for key, value in adapt.items():
                if key not in ['type', 'pattern_id', 'confidence']:
                    if key not in merged:
                        merged[key] = value
                    elif isinstance(value, dict) and isinstance(merged[key], dict):
                        merged[key].update(value)
        
        return merged
    
    async def _apply_to_context(
        self,
        context: Dict[str, Any],
        adaptations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Apply adaptations to the current context"""
        adapted_context = context.copy()
        
        # Add adaptations to context
        adapted_context['adaptations'] = adaptations
        
        # Apply specific adaptations
        for adaptation in adaptations:
            if adaptation['type'] == 'code_generation':
                if 'code_preferences' not in adapted_context:
                    adapted_context['code_preferences'] = {}
                adapted_context['code_preferences'].update(
                    adaptation.get('style_preferences', {})
                )
            
            elif adaptation['type'] == 'user_preference':
                if 'user_preferences' not in adapted_context:
                    adapted_context['user_preferences'] = {}
                adapted_context['user_preferences'].update(adaptation)
            
            elif adaptation['type'] == 'workflow':
                adapted_context['workflow_guidance'] = adaptation.get('workflow', {})
        
        return adapted_context
    
    def _calculate_overall_confidence(
        self,
        patterns: List[LearningPattern]
    ) -> float:
        """Calculate overall confidence for adaptations"""
        if not patterns:
            return 0.0
        
        # Weighted average based on usage count
        total_weight = sum(p.usage_count for p in patterns)
        if total_weight == 0:
            return sum(p.confidence_score for p in patterns) / len(patterns)
        
        weighted_sum = sum(
            p.confidence_score * p.usage_count for p in patterns
        )
        
        return weighted_sum / total_weight
    
    async def _cache_adaptations(
        self,
        session_id: Optional[UUID],
        adaptations: List[Dict[str, Any]]
    ):
        """Cache active adaptations"""
        if session_id:
            self._active_adaptations[str(session_id)] = {
                'adaptations': adaptations,
                'expires': datetime.now(timezone.utc) + timedelta(
                    seconds=self._adaptation_cache_ttl
                )
            }
            
            # Also cache in Redis if available
            if redis_client.client:
                try:
                    key = f"adaptations:{session_id}"
                    await redis_client.set(
                        key,
                        {'adaptations': adaptations},
                        expiry=self._adaptation_cache_ttl
                    )
                except Exception as e:
                    logger.warning(f"Failed to cache adaptations in Redis: {e}")
    
    async def _determine_adaptation_type(
        self,
        patterns: List[LearningPattern]
    ) -> str:
        """Determine the primary type of adaptation needed"""
        # Count pattern types
        type_counts = {}
        for pattern in patterns:
            pattern_type = pattern.pattern_type
            type_counts[pattern_type] = type_counts.get(pattern_type, 0) + 1
        
        # Return most common type
        if type_counts:
            return max(type_counts, key=type_counts.get)
        
        return 'general'
    
    async def _store_adaptation_trigger(self, adaptation_record: Dict[str, Any]):
        """Store record of adaptation trigger"""
        # This would store in a dedicated adaptations table
        # For now, log it
        logger.info(f"Adaptation triggered: {adaptation_record['id']}")
    
    async def _schedule_adaptation(self, adaptation_record: Dict[str, Any]):
        """Schedule adaptation for application"""
        # This would use a task queue in production
        # For now, mark as ready
        adaptation_record['status'] = 'ready'
    
    async def _get_adaptation_record(self, adaptation_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve adaptation record"""
        # This would query the adaptations table
        # For now, return None
        return None
    
    async def _calculate_effectiveness_score(
        self,
        adaptation: Dict[str, Any],
        outcome_data: Dict[str, Any]
    ) -> float:
        """Calculate effectiveness score for an adaptation"""
        score = 0.5  # Base score
        
        # Positive indicators
        if outcome_data.get('user_satisfied'):
            score += 0.2
        if outcome_data.get('task_completed'):
            score += 0.2
        if outcome_data.get('performance_improved'):
            score += 0.1
        
        # Negative indicators
        if outcome_data.get('errors_encountered'):
            score -= 0.2
        if outcome_data.get('user_rejected'):
            score -= 0.3
        
        return max(0.0, min(1.0, score))
    
    async def _reinforce_adaptation_patterns(self, adaptation: Dict[str, Any]):
        """Reinforce patterns that led to successful adaptation"""
        pattern_ids = [p['id'] for p in adaptation.get('patterns', [])]
        
        for pattern_id in pattern_ids:
            # This would update pattern confidence
            logger.info(f"Reinforcing pattern {pattern_id}")
    
    async def _weaken_adaptation_patterns(self, adaptation: Dict[str, Any]):
        """Weaken patterns that led to unsuccessful adaptation"""
        pattern_ids = [p['id'] for p in adaptation.get('patterns', [])]
        
        for pattern_id in pattern_ids:
            # This would reduce pattern confidence
            logger.info(f"Weakening pattern {pattern_id}")
    
    def _generate_code_instructions(self, adaptation: Dict[str, Any]) -> str:
        """Generate instructions for code generation based on adaptation"""
        instructions = []
        
        if 'preferred_languages' in adaptation:
            langs = adaptation['preferred_languages']
            instructions.append(f"Prefer using: {', '.join(langs)}")
        
        if 'style_preferences' in adaptation:
            style = adaptation['style_preferences']
            if style.get('use_async'):
                instructions.append("Use async/await patterns")
            if style.get('use_type_hints'):
                instructions.append("Include type hints")
            if style.get('include_docs'):
                instructions.append("Include documentation")
            if style.get('include_tests'):
                instructions.append("Include test cases")
        
        return "; ".join(instructions)
    
    def _get_methodology_for_approach(self, approach: str) -> str:
        """Get methodology based on approach preference"""
        methodologies = {
            'incremental': 'Start simple and build up gradually',
            'rapid': 'Focus on quick implementation',
            'comprehensive': 'Cover all aspects thoroughly',
            'balanced': 'Balance speed and completeness'
        }
        return methodologies.get(approach, 'Use best judgment')
    
    def _get_preventive_measures(self, failure_reasons: List[str]) -> List[str]:
        """Get preventive measures for known failure reasons"""
        measures = []
        
        for reason in failure_reasons:
            if 'performance' in reason.lower():
                measures.append('Add performance monitoring')
            elif 'error' in reason.lower():
                measures.append('Add comprehensive error handling')
            elif 'validation' in reason.lower():
                measures.append('Add input validation')
            elif 'security' in reason.lower():
                measures.append('Apply security best practices')
        
        return measures
    
    def _get_criteria_weights(self, criteria: List[str]) -> Dict[str, float]:
        """Get weights for decision criteria"""
        # Default weights
        weights = {
            'performance': 0.3,
            'maintainability': 0.25,
            'security': 0.2,
            'scalability': 0.15,
            'cost': 0.1
        }
        
        # Adjust weights based on criteria list order (higher position = higher weight)
        for i, criterion in enumerate(criteria[:5]):
            if criterion in weights:
                weights[criterion] += (5 - i) * 0.05
        
        # Normalize weights
        total = sum(weights.values())
        return {k: v/total for k, v in weights.items()}
    
    def _get_risk_considerations(self, risk_level: str) -> List[str]:
        """Get considerations based on risk tolerance"""
        considerations = {
            'conservative': [
                'Prioritize stability',
                'Use proven solutions',
                'Include fallback options',
                'Extensive testing required'
            ],
            'moderate': [
                'Balance innovation and stability',
                'Test thoroughly',
                'Have rollback plan'
            ],
            'aggressive': [
                'Can use latest technologies',
                'Experimentation encouraged',
                'Fast iteration acceptable'
            ]
        }
        return considerations.get(risk_level, considerations['moderate'])
    
    def _extract_correction_rules(self, pattern_data: Dict[str, Any]) -> List[str]:
        """Extract rules from correction patterns"""
        rules = []
        
        correction_type = pattern_data.get('correction_type')
        if correction_type == 'case_sensitivity':
            rules.append('Pay attention to case sensitivity')
        elif correction_type == 'typo':
            rules.append('Double-check for typos')
        elif correction_type == 'structure':
            rules.append('Maintain consistent structure and formatting')
        
        # Extract specific rules from examples
        examples = pattern_data.get('examples', [])
        if examples:
            # This would analyze examples to extract specific rules
            rules.append('Apply learned corrections')
        
        return rules