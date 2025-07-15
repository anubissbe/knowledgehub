"""Cross-Session Pattern Recognition Service

Identifies and analyzes patterns that emerge across multiple learning sessions,
enabling better understanding of user behavior and learning effectiveness.
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, List, Tuple, Set
from uuid import UUID
from collections import defaultdict, Counter

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, desc, func, text
from sqlalchemy.orm import selectinload

from ..models.learning_session import LearningSession
from ..models.learning_pattern import LearningPattern
from ..models.pattern_evolution import PatternEvolution, EvolutionType
from ..models.knowledge_transfer import KnowledgeTransfer
from ..models.user_learning_profile import UserLearningProfile
from ..models.decision_outcome import DecisionOutcome

logger = logging.getLogger(__name__)


class PatternSimilarity:
    """Helper class for pattern similarity calculations"""
    
    @staticmethod
    def calculate_pattern_similarity(pattern1: Dict[str, Any], pattern2: Dict[str, Any]) -> float:
        """Calculate similarity between two patterns"""
        
        if not pattern1 or not pattern2:
            return 0.0
        
        # Compare pattern structure
        structure_similarity = PatternSimilarity._compare_structure(pattern1, pattern2)
        
        # Compare pattern content
        content_similarity = PatternSimilarity._compare_content(pattern1, pattern2)
        
        # Weighted combination
        return 0.6 * structure_similarity + 0.4 * content_similarity
    
    @staticmethod
    def _compare_structure(pattern1: Dict[str, Any], pattern2: Dict[str, Any]) -> float:
        """Compare structural similarity of patterns"""
        
        keys1 = set(pattern1.keys())
        keys2 = set(pattern2.keys())
        
        if not keys1 and not keys2:
            return 1.0
        
        intersection = keys1 & keys2
        union = keys1 | keys2
        
        return len(intersection) / len(union) if union else 0.0
    
    @staticmethod
    def _compare_content(pattern1: Dict[str, Any], pattern2: Dict[str, Any]) -> float:
        """Compare content similarity of patterns"""
        
        common_keys = set(pattern1.keys()) & set(pattern2.keys())
        if not common_keys:
            return 0.0
        
        similarity_scores = []
        for key in common_keys:
            val1, val2 = pattern1[key], pattern2[key]
            
            if val1 == val2:
                similarity_scores.append(1.0)
            elif isinstance(val1, str) and isinstance(val2, str):
                # String similarity
                words1 = set(val1.lower().split())
                words2 = set(val2.lower().split())
                if words1 or words2:
                    similarity_scores.append(len(words1 & words2) / len(words1 | words2))
            elif isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                # Numeric similarity
                if val1 == 0 and val2 == 0:
                    similarity_scores.append(1.0)
                else:
                    max_val = max(abs(val1), abs(val2))
                    if max_val > 0:
                        similarity_scores.append(1.0 - abs(val1 - val2) / max_val)
        
        return sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0.0


class CrossSessionPatternRecognition:
    """Service for recognizing patterns across learning sessions"""
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self.similarity_threshold = 0.7  # Threshold for considering patterns similar
    
    async def analyze_cross_session_patterns(
        self,
        user_id: str,
        time_window_days: int = 30,
        min_pattern_count: int = 3
    ) -> Dict[str, Any]:
        """Analyze patterns that appear across multiple sessions"""
        
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=time_window_days)
        
        # Get learning sessions in time window
        sessions_query = select(LearningSession).where(
            and_(
                LearningSession.user_id == user_id,
                LearningSession.last_activity_at >= cutoff_date
            )
        ).order_by(desc(LearningSession.last_activity_at))
        
        sessions_result = await self.db.execute(sessions_query)
        sessions = sessions_result.scalars().all()
        
        if len(sessions) < 2:
            return {
                'cross_session_patterns': [],
                'pattern_evolution_trends': [],
                'learning_consistency_score': 0.0,
                'sessions_analyzed': len(sessions)
            }
        
        # Get patterns associated with these sessions
        session_ids = [session.id for session in sessions]
        patterns_query = select(LearningPattern).join(
            PatternEvolution, LearningPattern.id == PatternEvolution.pattern_id
        ).where(
            PatternEvolution.learning_session_id.in_(session_ids)
        ).distinct()
        
        patterns_result = await self.db.execute(patterns_query)
        patterns = patterns_result.scalars().all()
        
        # Analyze cross-session patterns
        cross_session_patterns = await self._identify_cross_session_patterns(
            patterns, session_ids, min_pattern_count
        )
        
        # Analyze pattern evolution trends
        evolution_trends = await self._analyze_pattern_evolution_trends(
            patterns, session_ids
        )
        
        # Calculate learning consistency
        consistency_score = await self._calculate_learning_consistency(
            user_id, sessions
        )
        
        return {
            'cross_session_patterns': cross_session_patterns,
            'pattern_evolution_trends': evolution_trends,
            'learning_consistency_score': consistency_score,
            'sessions_analyzed': len(sessions),
            'patterns_analyzed': len(patterns),
            'analysis_time_window_days': time_window_days
        }
    
    async def detect_recurring_patterns(
        self,
        user_id: str,
        pattern_type: Optional[str] = None,
        recurrence_threshold: int = 3
    ) -> List[Dict[str, Any]]:
        """Detect patterns that recur across sessions"""
        
        # Get all patterns for user
        query = select(LearningPattern).where(
            LearningPattern.usage_count >= recurrence_threshold
        )
        
        # Join with pattern evolutions to get session information
        evolutions_subquery = select(
            PatternEvolution.pattern_id,
            func.count(func.distinct(PatternEvolution.learning_session_id)).label('session_count'),
            func.array_agg(func.distinct(PatternEvolution.learning_session_id)).label('session_ids')
        ).where(
            PatternEvolution.user_id == user_id
        ).group_by(PatternEvolution.pattern_id).subquery()
        
        query = query.join(
            evolutions_subquery,
            LearningPattern.id == evolutions_subquery.c.pattern_id
        ).where(
            evolutions_subquery.c.session_count >= recurrence_threshold
        )
        
        if pattern_type:
            query = query.where(LearningPattern.pattern_type == pattern_type)
        
        result = await self.db.execute(query)
        patterns = result.all()
        
        recurring_patterns = []
        for pattern, session_count, session_ids in patterns:
            # Get detailed evolution information
            evolutions_query = select(PatternEvolution).where(
                and_(
                    PatternEvolution.pattern_id == pattern.id,
                    PatternEvolution.user_id == user_id
                )
            ).order_by(PatternEvolution.evolved_at)
            
            evolutions_result = await self.db.execute(evolutions_query)
            evolutions = evolutions_result.scalars().all()
            
            pattern_info = {
                'pattern_id': str(pattern.id),
                'pattern_type': pattern.pattern_type,
                'pattern_data': pattern.pattern_data,
                'confidence_score': pattern.confidence_score,
                'usage_count': pattern.usage_count,
                'session_count': session_count,
                'session_ids': [str(sid) for sid in session_ids if sid],
                'evolution_count': len(evolutions),
                'first_occurrence': evolutions[0].evolved_at.isoformat() if evolutions else None,
                'last_occurrence': evolutions[-1].evolved_at.isoformat() if evolutions else None,
                'evolution_types': [e.evolution_type for e in evolutions],
                'average_confidence_change': sum(
                    e.confidence_change for e in evolutions if e.confidence_change
                ) / len([e for e in evolutions if e.confidence_change]) if evolutions else 0.0
            }
            
            recurring_patterns.append(pattern_info)
        
        # Sort by confidence and usage count
        recurring_patterns.sort(
            key=lambda x: (x['confidence_score'], x['usage_count']),
            reverse=True
        )
        
        return recurring_patterns
    
    async def identify_pattern_clusters(
        self,
        user_id: str,
        similarity_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Identify clusters of similar patterns across sessions"""
        
        threshold = similarity_threshold or self.similarity_threshold
        
        # Get all patterns for user with evolutions
        patterns_query = select(LearningPattern).join(
            PatternEvolution, LearningPattern.id == PatternEvolution.pattern_id
        ).where(
            PatternEvolution.user_id == user_id
        ).distinct()
        
        patterns_result = await self.db.execute(patterns_query)
        patterns = patterns_result.scalars().all()
        
        if len(patterns) < 2:
            return []
        
        # Calculate similarity matrix
        pattern_list = list(patterns)
        similarity_matrix = {}
        
        for i, pattern1 in enumerate(pattern_list):
            for j, pattern2 in enumerate(pattern_list[i+1:], i+1):
                similarity = PatternSimilarity.calculate_pattern_similarity(
                    pattern1.pattern_data, pattern2.pattern_data
                )
                similarity_matrix[(i, j)] = similarity
        
        # Find clusters using simple threshold-based clustering
        clusters = []
        used_patterns = set()
        
        for i, pattern1 in enumerate(pattern_list):
            if i in used_patterns:
                continue
            
            cluster = [pattern1]
            used_patterns.add(i)
            
            for j, pattern2 in enumerate(pattern_list[i+1:], i+1):
                if j in used_patterns:
                    continue
                
                similarity = similarity_matrix.get((i, j), 0.0)
                if similarity >= threshold:
                    cluster.append(pattern2)
                    used_patterns.add(j)
            
            if len(cluster) > 1:  # Only include multi-pattern clusters
                cluster_info = await self._create_cluster_info(cluster, user_id)
                clusters.append(cluster_info)
        
        # Sort clusters by size and average confidence
        clusters.sort(
            key=lambda x: (x['pattern_count'], x['average_confidence']),
            reverse=True
        )
        
        return clusters
    
    async def analyze_learning_progression(
        self,
        user_id: str,
        session_sequence: Optional[List[UUID]] = None
    ) -> Dict[str, Any]:
        """Analyze how learning progresses across sessions"""
        
        if session_sequence:
            sessions_query = select(LearningSession).where(
                LearningSession.id.in_(session_sequence)
            ).order_by(LearningSession.started_at)
        else:
            sessions_query = select(LearningSession).where(
                LearningSession.user_id == user_id
            ).order_by(LearningSession.started_at)
        
        sessions_result = await self.db.execute(sessions_query)
        sessions = sessions_result.scalars().all()
        
        if len(sessions) < 2:
            return {
                'progression_score': 0.0,
                'learning_velocity': 0.0,
                'knowledge_accumulation': [],
                'effectiveness_trend': []
            }
        
        # Analyze progression metrics
        progression_data = []
        knowledge_accumulation = []
        effectiveness_trend = []
        
        total_patterns = 0
        total_knowledge_units = 0
        
        for session in sessions:
            total_patterns += session.patterns_learned
            total_knowledge_units += session.knowledge_units_created
            
            session_data = {
                'session_id': str(session.id),
                'session_type': session.session_type,
                'started_at': session.started_at.isoformat(),
                'patterns_learned': session.patterns_learned,
                'patterns_reinforced': session.patterns_reinforced,
                'knowledge_units_created': session.knowledge_units_created,
                'learning_effectiveness': session.learning_effectiveness,
                'success_rate': session.success_rate,
                'cumulative_patterns': total_patterns,
                'cumulative_knowledge_units': total_knowledge_units
            }
            
            progression_data.append(session_data)
            knowledge_accumulation.append(total_knowledge_units)
            
            if session.learning_effectiveness is not None:
                effectiveness_trend.append(session.learning_effectiveness)
        
        # Calculate progression score
        progression_score = self._calculate_progression_score(progression_data)
        
        # Calculate learning velocity (knowledge units per day)
        if len(sessions) > 1:
            time_span = (sessions[-1].started_at - sessions[0].started_at).days
            learning_velocity = total_knowledge_units / max(time_span, 1)
        else:
            learning_velocity = 0.0
        
        return {
            'progression_score': progression_score,
            'learning_velocity': learning_velocity,
            'knowledge_accumulation': knowledge_accumulation,
            'effectiveness_trend': effectiveness_trend,
            'session_progression': progression_data,
            'total_sessions': len(sessions),
            'total_patterns_learned': total_patterns,
            'total_knowledge_units': total_knowledge_units
        }
    
    async def predict_learning_outcomes(
        self,
        user_id: str,
        current_session_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Predict learning outcomes based on historical patterns"""
        
        # Get user learning profile
        profile_query = select(UserLearningProfile).where(
            UserLearningProfile.user_id == user_id
        )
        profile_result = await self.db.execute(profile_query)
        profile = profile_result.scalar_one_or_none()
        
        if not profile:
            return {
                'predicted_effectiveness': 0.5,
                'predicted_patterns_learned': 1,
                'confidence': 0.0,
                'recommendations': []
            }
        
        # Get similar historical sessions
        similar_sessions = await self._find_similar_sessions(
            user_id, current_session_context
        )
        
        if not similar_sessions:
            return {
                'predicted_effectiveness': profile.average_learning_effectiveness or 0.5,
                'predicted_patterns_learned': 1,
                'confidence': 0.3,
                'recommendations': profile.get_learning_recommendations()
            }
        
        # Calculate predictions based on similar sessions
        effectiveness_scores = [s.learning_effectiveness for s in similar_sessions if s.learning_effectiveness]
        patterns_learned = [s.patterns_learned for s in similar_sessions]
        
        predicted_effectiveness = sum(effectiveness_scores) / len(effectiveness_scores) if effectiveness_scores else 0.5
        predicted_patterns = sum(patterns_learned) / len(patterns_learned) if patterns_learned else 1
        
        # Calculate confidence based on number of similar sessions and variance
        confidence = min(0.9, len(similar_sessions) / 10)
        if len(effectiveness_scores) > 1:
            variance = sum((x - predicted_effectiveness) ** 2 for x in effectiveness_scores) / len(effectiveness_scores)
            confidence *= max(0.1, 1.0 - variance)
        
        # Generate recommendations
        recommendations = profile.get_learning_recommendations()
        
        return {
            'predicted_effectiveness': predicted_effectiveness,
            'predicted_patterns_learned': int(predicted_patterns),
            'confidence': confidence,
            'recommendations': recommendations,
            'similar_sessions_count': len(similar_sessions),
            'historical_average_effectiveness': profile.average_learning_effectiveness
        }
    
    async def _identify_cross_session_patterns(
        self,
        patterns: List[LearningPattern],
        session_ids: List[UUID],
        min_count: int
    ) -> List[Dict[str, Any]]:
        """Identify patterns that appear across multiple sessions"""
        
        cross_session = []
        
        for pattern in patterns:
            # Get evolutions for this pattern across sessions
            evolutions_query = select(PatternEvolution).where(
                and_(
                    PatternEvolution.pattern_id == pattern.id,
                    PatternEvolution.learning_session_id.in_(session_ids)
                )
            )
            
            evolutions_result = await self.db.execute(evolutions_query)
            evolutions = evolutions_result.scalars().all()
            
            # Count unique sessions
            unique_sessions = set(e.learning_session_id for e in evolutions if e.learning_session_id)
            
            if len(unique_sessions) >= min_count:
                pattern_info = {
                    'pattern_id': str(pattern.id),
                    'pattern_type': pattern.pattern_type,
                    'confidence_score': pattern.confidence_score,
                    'usage_count': pattern.usage_count,
                    'sessions_count': len(unique_sessions),
                    'evolution_count': len(evolutions),
                    'pattern_data_summary': self._summarize_pattern_data(pattern.pattern_data)
                }
                cross_session.append(pattern_info)
        
        return cross_session
    
    async def _analyze_pattern_evolution_trends(
        self,
        patterns: List[LearningPattern],
        session_ids: List[UUID]
    ) -> List[Dict[str, Any]]:
        """Analyze how patterns evolve across sessions"""
        
        trends = []
        
        for pattern in patterns:
            evolutions_query = select(PatternEvolution).where(
                and_(
                    PatternEvolution.pattern_id == pattern.id,
                    PatternEvolution.learning_session_id.in_(session_ids)
                )
            ).order_by(PatternEvolution.evolved_at)
            
            evolutions_result = await self.db.execute(evolutions_query)
            evolutions = evolutions_result.scalars().all()
            
            if len(evolutions) > 1:
                # Analyze confidence trend
                confidence_changes = [e.confidence_change for e in evolutions if e.confidence_change]
                avg_confidence_change = sum(confidence_changes) / len(confidence_changes) if confidence_changes else 0.0
                
                # Analyze evolution types
                evolution_types = Counter(e.evolution_type for e in evolutions)
                
                trend_info = {
                    'pattern_id': str(pattern.id),
                    'pattern_type': pattern.pattern_type,
                    'evolution_count': len(evolutions),
                    'average_confidence_change': avg_confidence_change,
                    'evolution_types': dict(evolution_types),
                    'trend_direction': 'improving' if avg_confidence_change > 0.1 else 'declining' if avg_confidence_change < -0.1 else 'stable'
                }
                trends.append(trend_info)
        
        return trends
    
    async def _calculate_learning_consistency(
        self,
        user_id: str,
        sessions: List[LearningSession]
    ) -> float:
        """Calculate consistency of learning across sessions"""
        
        if len(sessions) < 2:
            return 0.0
        
        # Calculate consistency based on effectiveness scores
        effectiveness_scores = [s.learning_effectiveness for s in sessions if s.learning_effectiveness]
        
        if len(effectiveness_scores) < 2:
            return 0.5
        
        # Calculate coefficient of variation (lower = more consistent)
        mean_effectiveness = sum(effectiveness_scores) / len(effectiveness_scores)
        variance = sum((x - mean_effectiveness) ** 2 for x in effectiveness_scores) / len(effectiveness_scores)
        std_dev = variance ** 0.5
        
        if mean_effectiveness == 0:
            return 0.0
        
        cv = std_dev / mean_effectiveness
        
        # Convert to consistency score (0.0 to 1.0, higher = more consistent)
        consistency = max(0.0, 1.0 - cv)
        
        return consistency
    
    def _summarize_pattern_data(self, pattern_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary of pattern data for analysis"""
        
        summary = {
            'keys_count': len(pattern_data.keys()),
            'has_code': 'code' in pattern_data or 'implementation' in pattern_data,
            'has_context': 'context' in pattern_data or 'situation' in pattern_data,
            'has_outcome': 'outcome' in pattern_data or 'result' in pattern_data
        }
        
        # Add key insights
        if 'category' in pattern_data:
            summary['category'] = pattern_data['category']
        if 'importance' in pattern_data:
            summary['importance'] = pattern_data['importance']
        
        return summary
    
    def _calculate_progression_score(self, progression_data: List[Dict[str, Any]]) -> float:
        """Calculate overall progression score"""
        
        if len(progression_data) < 2:
            return 0.0
        
        # Factors for progression score
        factors = []
        
        # Knowledge accumulation trend
        knowledge_values = [p['cumulative_knowledge_units'] for p in progression_data]
        if len(knowledge_values) > 1:
            knowledge_trend = (knowledge_values[-1] - knowledge_values[0]) / len(knowledge_values)
            factors.append(min(1.0, knowledge_trend / 10))  # Normalize to 0-1
        
        # Effectiveness improvement
        effectiveness_values = [p['learning_effectiveness'] for p in progression_data if p['learning_effectiveness']]
        if len(effectiveness_values) > 1:
            effectiveness_trend = effectiveness_values[-1] - effectiveness_values[0]
            factors.append(max(0.0, min(1.0, effectiveness_trend + 0.5)))  # Normalize to 0-1
        
        # Pattern learning consistency
        pattern_values = [p['patterns_learned'] for p in progression_data]
        if pattern_values:
            avg_patterns = sum(pattern_values) / len(pattern_values)
            consistency = 1.0 - (max(pattern_values) - min(pattern_values)) / max(1, max(pattern_values))
            factors.append(consistency * avg_patterns / 5)  # Normalize
        
        return sum(factors) / len(factors) if factors else 0.0
    
    async def _find_similar_sessions(
        self,
        user_id: str,
        current_context: Dict[str, Any],
        max_results: int = 10
    ) -> List[LearningSession]:
        """Find sessions with similar context"""
        
        sessions_query = select(LearningSession).where(
            and_(
                LearningSession.user_id == user_id,
                LearningSession.status == 'completed'
            )
        ).order_by(desc(LearningSession.started_at)).limit(50)
        
        sessions_result = await self.db.execute(sessions_query)
        all_sessions = sessions_result.scalars().all()
        
        # Score sessions by context similarity
        scored_sessions = []
        for session in all_sessions:
            similarity = PatternSimilarity.calculate_pattern_similarity(
                session.learning_context, current_context
            )
            scored_sessions.append((session, similarity))
        
        # Sort by similarity and return top results
        scored_sessions.sort(key=lambda x: x[1], reverse=True)
        return [session for session, _ in scored_sessions[:max_results] if _ > 0.3]
    
    async def _create_cluster_info(
        self,
        cluster_patterns: List[LearningPattern],
        user_id: str
    ) -> Dict[str, Any]:
        """Create cluster information"""
        
        pattern_ids = [p.id for p in cluster_patterns]
        
        # Get evolution information for cluster patterns
        evolutions_query = select(PatternEvolution).where(
            and_(
                PatternEvolution.pattern_id.in_(pattern_ids),
                PatternEvolution.user_id == user_id
            )
        )
        
        evolutions_result = await self.db.execute(evolutions_query)
        evolutions = evolutions_result.scalars().all()
        
        # Calculate cluster metrics
        avg_confidence = sum(p.confidence_score for p in cluster_patterns) / len(cluster_patterns)
        total_usage = sum(p.usage_count for p in cluster_patterns)
        unique_sessions = set(e.learning_session_id for e in evolutions if e.learning_session_id)
        
        return {
            'cluster_id': f"cluster_{hash(tuple(sorted(str(p.id) for p in cluster_patterns))) & 0xffffffff}",
            'pattern_count': len(cluster_patterns),
            'pattern_ids': [str(p.id) for p in cluster_patterns],
            'average_confidence': avg_confidence,
            'total_usage_count': total_usage,
            'sessions_count': len(unique_sessions),
            'evolution_count': len(evolutions),
            'pattern_types': list(set(p.pattern_type for p in cluster_patterns)),
            'cluster_summary': f"Cluster of {len(cluster_patterns)} similar patterns with {avg_confidence:.2f} avg confidence"
        }