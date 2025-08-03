#!/usr/bin/env python3
"""
Test Session Management System.

This script tests the session management functionality including:
- Session creation and management
- Context window operations
- Session handoffs
- Checkpoint creation
- Session analytics
"""

import asyncio
import sys
import os
import json
from datetime import datetime

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from api.services.session_service import session_service
from api.models.session import (
    SessionCreate, SessionUpdate, SessionHandoffCreate, SessionCheckpointCreate,
    SessionType, SessionState, HandoffReason
)

async def test_session_management():
    """Test the complete session management system."""
    
    print("🧠 Testing AI Session Management System...")
    
    try:
        # Initialize the service
        await session_service.initialize()
        print("✅ Session service initialized")
        
        # Test 1: Create a new session
        print("\n📝 Test 1: Creating a new session...")
        session_data = SessionCreate(
            user_id="test_user",
            project_id="test_project",
            session_type=SessionType.INTERACTIVE,
            title="Test Session",
            description="Testing session management functionality",
            preferences={"auto_save": True, "context_size": 50},
            max_context_size=50
        )
        
        session = await session_service.create_session(session_data)
        print(f"✅ Created session: {session.id}")
        print(f"   - User: {session.user_id}")
        print(f"   - Type: {session.session_type}")
        print(f"   - State: {session.state}")
        print(f"   - Context size: {session.context_size}/{session.max_context_size}")
        
        session_id = session.id
        
        # Test 2: Add items to context
        print("\n📝 Test 2: Adding items to context...")
        for i in range(5):
            memory_id = f"test_memory_{i}"
            context_response = await session_service.add_to_context(
                session_id=session_id,
                memory_id=memory_id,
                auto_optimize=True
            )
            print(f"✅ Added memory {memory_id} to context")
        
        print(f"   - Context size: {context_response.context_size}")
        print(f"   - Context window: {context_response.context_window}")
        
        # Test 3: Update session
        print("\n📝 Test 3: Updating session...")
        update_data = SessionUpdate(
            title="Updated Test Session",
            description="Updated description",
            user_satisfaction=0.9
        )
        
        updated_session = await session_service.update_session(session_id, update_data)
        print(f"✅ Updated session: {updated_session.title}")
        print(f"   - Satisfaction: {updated_session.user_satisfaction}")
        
        # Test 4: Create checkpoint
        print("\n📝 Test 4: Creating checkpoint...")
        checkpoint_data = SessionCheckpointCreate(
            session_id=session_id,
            checkpoint_name="Test Checkpoint",
            description="Manual test checkpoint",
            checkpoint_type="manual",
            is_recovery_point=True,
            recovery_priority=5
        )
        
        checkpoint_id = await session_service.create_checkpoint(checkpoint_data)
        print(f"✅ Created checkpoint: {checkpoint_id}")
        
        # Test 5: Create handoff
        print("\n📝 Test 5: Creating session handoff...")
        handoff_data = SessionHandoffCreate(
            source_session_id=session_id,
            reason=HandoffReason.MANUAL,
            handoff_message="Test handoff for session continuity",
            continuation_instructions="Continue with the test session",
            context_data={"test": True, "reason": "testing"}
        )
        
        handoff_id = await session_service.create_handoff(handoff_data)
        print(f"✅ Created handoff: {handoff_id}")
        
        # Test 6: Get session analytics
        print("\n📝 Test 6: Getting session analytics...")
        analytics = await session_service.get_session_analytics(
            user_id="test_user",
            project_id="test_project",
            time_window_hours=24
        )
        
        print(f"✅ Session analytics:")
        print(f"   - Total sessions: {analytics.total_sessions}")
        print(f"   - Active sessions: {analytics.active_sessions}")
        print(f"   - Average success rate: {analytics.avg_success_rate:.2f}")
        print(f"   - Sessions by type: {analytics.sessions_by_type}")
        print(f"   - Sessions by state: {analytics.sessions_by_state}")
        
        # Test 7: Get recovery info
        print("\n📝 Test 7: Getting recovery information...")
        recovery_info = await session_service.get_recovery_info(session_id)
        
        print(f"✅ Recovery information:")
        print(f"   - Recoverable: {recovery_info.recoverable}")
        print(f"   - Recovery options: {len(recovery_info.recovery_options)}")
        print(f"   - Estimated data loss: {recovery_info.estimated_data_loss}")
        print(f"   - Recommended action: {recovery_info.recommended_action}")
        
        # Test 8: Create new session from handoff
        print("\n📝 Test 8: Restoring session from handoff...")
        new_session_data = SessionCreate(
            user_id="test_user",
            project_id="test_project",
            session_type=SessionType.INTERACTIVE,
            title="Restored Session",
            description="Session restored from handoff"
        )
        
        restored_session = await session_service.restore_from_handoff(
            handoff_id=handoff_id,
            new_session_data=new_session_data
        )
        
        print(f"✅ Restored session: {restored_session.id}")
        print(f"   - Title: {restored_session.title}")
        print(f"   - Context size: {restored_session.context_size}")
        
        print("\n🎉 All session management tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup
        await session_service.cleanup()
        print("🧹 Session service cleaned up")


async def main():
    """Main test function."""
    print("🚀 Starting Session Management Tests...")
    
    success = await test_session_management()
    
    if success:
        print("✅ All tests passed!")
        return 0
    else:
        print("❌ Some tests failed!")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)