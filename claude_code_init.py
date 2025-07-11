#!/usr/bin/env python3
"""
Claude Code Session Initializer
Automatically restores context when Claude Code starts a new conversation
"""

import os
import sys
import json
import requests
from pathlib import Path
from datetime import datetime


class ClaudeCodeInit:
    """Initialize Claude Code with automatic context restoration"""
    
    def __init__(self):
        self.api_base = "http://localhost:3000/api/claude-auto"
        self.cwd = os.getcwd()
        self.session_file = Path.home() / ".claude_session.json"
        
    def initialize(self):
        """Main initialization process"""
        print("🤖 Claude Code Session Initializer")
        print("=" * 60)
        
        # Start session
        print(f"\n📁 Working Directory: {self.cwd}")
        
        try:
            # Call session start endpoint
            response = requests.post(
                f"{self.api_base}/session/start",
                params={"cwd": self.cwd},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                session = data["session"]
                context = data["context"]
                
                print(f"\n✅ Session Started: {session['session_id']}")
                print(f"📦 Project: {session['project_name']} ({session['project_type']})")
                
                # Show restored context
                if session.get("previous_session"):
                    print(f"\n🔗 Continuing from: {session['previous_session']}")
                
                # Display handoff notes
                if context.get("handoff_notes"):
                    print("\n📋 Handoff Notes:")
                    for note in context["handoff_notes"]:
                        print(f"  - {note}")
                
                # Display unfinished tasks
                if context.get("unfinished_tasks"):
                    print("\n📝 Unfinished Tasks:")
                    for task in context["unfinished_tasks"]:
                        print(f"  - {task}")
                
                # Display recent errors
                if context.get("recent_errors"):
                    print("\n⚠️  Recent Errors:")
                    for error in context["recent_errors"]:
                        print(f"  - {error}")
                
                # Get task predictions
                pred_response = requests.get(f"{self.api_base}/tasks/predict", timeout=5)
                if pred_response.status_code == 200:
                    predictions = pred_response.json()
                    if predictions:
                        print("\n🎯 Suggested Next Tasks:")
                        for pred in predictions:
                            print(f"  - {pred['task']} ({pred['type']}, confidence: {pred['confidence']})")
                
                # Get proactive assistance
                print("\n" + "=" * 60)
                self._show_proactive_assistance(session['session_id'], session.get('project_id'))
                
                print("\n" + "=" * 60)
                print("🚀 Ready to continue your work!")
                
                # Save session info for other tools
                self._save_session_info(session)
                
                return True
                
            else:
                print(f"❌ Failed to start session: {response.status_code}")
                return False
                
        except requests.exceptions.ConnectionError:
            print("⚠️  KnowledgeHub not available - starting without context restoration")
            return self._fallback_init()
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    
    def _save_session_info(self, session):
        """Save session info for other tools to use"""
        try:
            with open(self.session_file, 'w') as f:
                json.dump(session, f, indent=2)
        except:
            pass
    
    def _fallback_init(self):
        """Fallback initialization when API is not available"""
        # Create basic session
        session_id = f"claude-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        project_name = Path(self.cwd).name
        
        print(f"\n📝 Created local session: {session_id}")
        print(f"📦 Project: {project_name}")
        
        # Check for local memory system
        memory_cli = Path("/opt/projects/memory-system/memory-cli")
        if memory_cli.exists():
            print("\n💾 Local memory system available")
            print("   Use memory-cli to save important context")
        
        return True
    
    def _show_proactive_assistance(self, session_id, project_id=None):
        """Show proactive assistance at session start"""
        try:
            # Get brief from proactive assistant
            response = requests.get(
                f"{self.api_base[:-12]}/proactive/brief",
                params={"session_id": session_id, "project_id": project_id},
                timeout=5
            )
            
            if response.status_code == 200:
                brief = response.json().get("brief", "")
                if brief:
                    print(brief)
            else:
                # Fallback to basic analysis
                self._show_basic_assistance()
        except:
            # Silent fail - don't interrupt session start
            pass
    
    def _show_basic_assistance(self):
        """Show basic assistance when API not available"""
        print("\n💡 Proactive Assistant")
        print("-" * 40)
        print("Tip: Use 'claude-check <action>' before risky operations")
        print("Tip: Run 'claude-report' to see mistake patterns")
    
    def create_handoff(self, summary, next_tasks=None, issues=None):
        """Create handoff note for next session"""
        try:
            params = {"content": summary}
            if next_tasks:
                params["next_tasks"] = next_tasks
            if issues:
                params["unresolved_issues"] = issues
            
            response = requests.post(
                f"{self.api_base}/session/handoff",
                params=params,
                timeout=10
            )
            
            if response.status_code == 200:
                print("✅ Handoff note created")
                return True
        except:
            pass
        
        print("⚠️  Could not create handoff note")
        return False


def main():
    """Main entry point"""
    init = ClaudeCodeInit()
    
    # Check for command line arguments
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "handoff" and len(sys.argv) > 2:
            # Create handoff note
            summary = sys.argv[2]
            next_tasks = sys.argv[3].split(',') if len(sys.argv) > 3 else None
            issues = sys.argv[4].split(',') if len(sys.argv) > 4 else None
            init.create_handoff(summary, next_tasks, issues)
        else:
            print("Usage:")
            print("  claude_code_init.py              - Initialize session")
            print("  claude_code_init.py handoff <summary> [tasks] [issues]")
    else:
        # Default: initialize session
        init.initialize()


if __name__ == "__main__":
    main()