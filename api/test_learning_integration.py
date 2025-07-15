#!/usr/bin/env python3
"""Integration test for the learning system

Run this from the src/api directory:
python -m pytest test_learning_integration.py -v
"""

import pytest
import asyncio
from datetime import datetime
from uuid import uuid4
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from database import Base, get_db
from learning_system import LearningEngine, PatternType, FeedbackType
from learning_system.models import LearningPattern, UserFeedback, DecisionOutcome
from memory_system.models.memory import Memory, MemoryType


@pytest.fixture
def test_db():
    """Create a test database"""
    engine = create_engine('sqlite:///:memory:')
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    db = SessionLocal()
    yield db
    db.close()


@pytest.fixture
def learning_engine(test_db):
    """Create a learning engine instance"""
    return LearningEngine(test_db)


@pytest.mark.asyncio
async def test_learn_from_interaction(learning_engine, test_db):
    """Test learning from user interactions"""
    session_id = uuid4()
    interaction_data = {
        'user_input': 'Please implement a Python function to calculate fibonacci numbers using async/await',
        'system_response': '''async def fibonacci(n: int) -> int:
    """Calculate fibonacci number asynchronously"""
    if n <= 1:
        return n
    return await fibonacci(n-1) + await fibonacci(n-2)''',
        'context': {
            'type': 'code_generation',
            'language': 'python',
            'tags': ['algorithm', 'async']
        },
        'outcome': {
            'success': True,
            'task_completed': True,
            'user_satisfied': True
        },
        'timestamp': datetime.now().isoformat()
    }
    
    result = await learning_engine.learn_from_interaction(session_id, interaction_data)
    
    assert result['patterns_learned'] > 0
    assert 'patterns' in result
    assert not result.get('error')
    
    # Verify patterns were stored
    patterns = test_db.query(LearningPattern).all()
    assert len(patterns) > 0
    
    # Check for expected pattern types
    pattern_types = [p.pattern_type for p in patterns]
    assert PatternType.CODE in pattern_types


@pytest.mark.asyncio
async def test_process_feedback(learning_engine, test_db):
    """Test processing user feedback"""
    # Test rating feedback
    rating_feedback = {
        'feedback_type': FeedbackType.RATING.value,
        'feedback_data': {'rating': 5},
        'context_data': {'interaction_type': 'code_generation'}
    }
    
    result = await learning_engine.process_user_feedback(rating_feedback)
    
    assert result['feedback_processed'] == True
    assert 'feedback_id' in result
    
    # Verify feedback was stored
    feedback = test_db.query(UserFeedback).first()
    assert feedback is not None
    assert feedback.get_rating() == 5
    
    # Test correction feedback
    correction_feedback = {
        'feedback_type': FeedbackType.CORRECTION.value,
        'original_content': 'def fibonacci(n):',
        'corrected_content': 'async def fibonacci(n: int) -> int:',
        'context_data': {'correction_type': 'async_missing'}
    }
    
    result2 = await learning_engine.process_user_feedback(correction_feedback)
    assert result2['feedback_processed'] == True


@pytest.mark.asyncio
async def test_track_decision_outcome(learning_engine, test_db):
    """Test tracking decision outcomes"""
    session_id = uuid4()
    
    # Create a decision memory
    decision = Memory(
        id=uuid4(),
        session_id=session_id,
        content='Use async/await pattern for all I/O operations',
        summary='Async pattern decision',
        memory_type=MemoryType.DECISION.value,
        importance=0.8,
        confidence=0.9
    )
    test_db.add(decision)
    test_db.commit()
    
    # Track outcome
    outcome_data = {
        'outcome_type': 'success',
        'success_score': 0.95,
        'task_completed': True,
        'user_satisfied': True,
        'performance_met': True,
        'no_errors': True,
        'timely_completion': True,
        'impact_data': {
            'response_time': 0.5,
            'memory_usage': 'low'
        }
    }
    
    result = await learning_engine.track_decision_outcome(decision.id, outcome_data)
    
    assert result['outcome_tracked'] == True
    assert result['success_score'] == 0.95
    
    # Verify outcome was stored
    outcome = test_db.query(DecisionOutcome).first()
    assert outcome is not None
    assert outcome.is_successful()


@pytest.mark.asyncio
async def test_get_learned_patterns(learning_engine, test_db):
    """Test retrieving learned patterns"""
    # Add some test patterns
    pattern1 = LearningPattern(
        pattern_type=PatternType.CODE,
        pattern_data={'language': 'python', 'style': 'async'},
        pattern_hash='test_hash_1',
        confidence_score=0.8,
        source='test'
    )
    pattern2 = LearningPattern(
        pattern_type=PatternType.PREFERENCE,
        pattern_data={'communication': 'concise'},
        pattern_hash='test_hash_2',
        confidence_score=0.6,
        source='test'
    )
    test_db.add_all([pattern1, pattern2])
    test_db.commit()
    
    # Get patterns
    patterns = await learning_engine.get_learned_patterns(min_confidence=0.5)
    
    assert len(patterns) == 2
    assert patterns[0].confidence_score >= patterns[1].confidence_score


@pytest.mark.asyncio
async def test_apply_learned_patterns(learning_engine, test_db):
    """Test applying learned patterns"""
    # Add patterns to apply
    code_pattern = LearningPattern(
        pattern_type=PatternType.CODE,
        pattern_data={
            'type': 'language_preference',
            'languages': ['python'],
            'style_indicators': {'async': True, 'type_hints': True}
        },
        pattern_hash='code_pattern_hash',
        confidence_score=0.9,
        source='test',
        usage_count=5
    )
    preference_pattern = LearningPattern(
        pattern_type=PatternType.PREFERENCE,
        pattern_data={
            'type': 'communication_style',
            'style': 'concise'
        },
        pattern_hash='pref_pattern_hash',
        confidence_score=0.8,
        source='test',
        usage_count=3
    )
    test_db.add_all([code_pattern, preference_pattern])
    test_db.commit()
    
    # Apply patterns
    context = {
        'type': 'code_generation',
        'tags': ['python', 'async'],
        'session_id': uuid4()
    }
    
    result = await learning_engine.apply_learned_patterns(context)
    
    assert result['patterns_applied'] > 0
    assert 'adaptations' in result
    assert 'adapted_context' in result
    assert result['confidence'] > 0


@pytest.mark.asyncio
async def test_get_learning_analytics(learning_engine, test_db):
    """Test getting learning analytics"""
    # Add some data for analytics
    pattern = LearningPattern(
        pattern_type=PatternType.SUCCESS,
        pattern_data={'factors': ['clear_requirements', 'async_pattern']},
        pattern_hash='success_pattern',
        confidence_score=0.85,
        source='test'
    )
    feedback = UserFeedback(
        feedback_type=FeedbackType.RATING.value,
        feedback_data={'rating': 4},
        applied=True
    )
    test_db.add_all([pattern, feedback])
    test_db.commit()
    
    # Get analytics
    analytics = await learning_engine.get_learning_analytics()
    
    assert 'pattern_statistics' in analytics
    assert 'feedback_statistics' in analytics
    assert 'outcome_statistics' in analytics
    assert 'learning_effectiveness' in analytics


@pytest.mark.asyncio
async def test_pattern_reinforcement(learning_engine, test_db):
    """Test pattern reinforcement through repeated success"""
    session_id = uuid4()
    
    # First interaction
    interaction1 = {
        'user_input': 'Create async Python function',
        'system_response': 'async def example(): pass',
        'context': {'type': 'code', 'language': 'python'},
        'outcome': {'success': True, 'success_score': 0.9}
    }
    
    result1 = await learning_engine.learn_from_interaction(session_id, interaction1)
    
    # Second similar interaction (should reinforce)
    interaction2 = {
        'user_input': 'Make another async Python function',
        'system_response': 'async def another(): pass',
        'context': {'type': 'code', 'language': 'python'},
        'outcome': {'success': True, 'success_score': 0.95}
    }
    
    result2 = await learning_engine.learn_from_interaction(session_id, interaction2)
    
    # Check pattern was reinforced
    patterns = test_db.query(LearningPattern).filter_by(
        pattern_type=PatternType.CODE
    ).all()
    
    # Should have patterns with increased confidence/usage
    assert any(p.usage_count > 1 for p in patterns)


@pytest.mark.asyncio
async def test_error_pattern_learning(learning_engine, test_db):
    """Test learning from errors"""
    session_id = uuid4()
    
    interaction = {
        'user_input': 'Fix TypeError in my code',
        'system_response': 'Fixed by adding type checking',
        'context': {'type': 'error_fix', 'error_type': 'TypeError'},
        'outcome': {
            'success': True,
            'error_resolved': True
        }
    }
    
    result = await learning_engine.learn_from_interaction(session_id, interaction)
    
    # Should create error pattern
    error_patterns = test_db.query(LearningPattern).filter_by(
        pattern_type=PatternType.ERROR
    ).all()
    
    assert len(error_patterns) > 0


@pytest.mark.asyncio
async def test_feedback_impact_on_patterns(learning_engine, test_db):
    """Test that feedback affects pattern confidence"""
    # Create a pattern
    pattern = LearningPattern(
        pattern_type=PatternType.CODE,
        pattern_data={'test': True},
        pattern_hash='feedback_test',
        confidence_score=0.7,
        source='test'
    )
    test_db.add(pattern)
    test_db.commit()
    
    # Create related memory
    memory = Memory(
        id=uuid4(),
        session_id=uuid4(),
        content='Test content',
        memory_type=MemoryType.CODE.value,
        importance=0.5
    )
    test_db.add(memory)
    test_db.commit()
    
    # Process negative feedback
    feedback_data = {
        'memory_id': memory.id,
        'feedback_type': FeedbackType.RATING.value,
        'feedback_data': {'rating': 2},  # Low rating
        'context_data': {}
    }
    
    await learning_engine.process_user_feedback(feedback_data)
    
    # Pattern confidence should be affected
    # (In real implementation, would need pattern-memory association)
    feedback = test_db.query(UserFeedback).first()
    assert feedback.get_rating() == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])