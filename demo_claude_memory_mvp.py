#!/usr/bin/env python3
"""
MVP Demo: Claude-Code with Working Memory System

This demo showcases the complete memory system functionality:
- Session management with automatic linking
- Memory storage and retrieval
- Context injection for Claude-Code
- Cross-session memory persistence
- Intelligent session linking
"""

import requests
import json
import time
from datetime import datetime
from typing import Dict, List, Any

class ClaudeMemoryDemo:
    def __init__(self):
        self.base_url = "http://localhost:3000"
        self.user_id = "claude-demo@example.com"
        self.sessions = []
        self.memories = []
        
    def print_header(self, title: str):
        """Print a formatted header"""
        print("\n" + "=" * 60)
        print(f"🤖 {title}")
        print("=" * 60)
    
    def print_step(self, step: str, description: str):
        """Print a formatted step"""
        print(f"\n{step}. {description}")
        print("-" * 40)
    
    def print_result(self, data: Any, label: str = "Result"):
        """Print formatted result data"""
        if isinstance(data, (dict, list)):
            print(f"📋 {label}:")
            print(json.dumps(data, indent=2, default=str))
        else:
            print(f"📋 {label}: {data}")
    
    def verify_system_health(self) -> bool:
        """Verify all memory system components are healthy"""
        self.print_step("🔧", "Verifying System Health")
        
        endpoints = [
            ("/health", "Main API"),
            ("/api/memory/session/health", "Session Management"), 
            ("/api/memory/context/health", "Context Injection"),
            ("/api/memory/admin/cleanup/health", "Cleanup Service"),
            ("/api/memory/linking/health", "Session Linking")
        ]
        
        all_healthy = True
        
        for endpoint, service in endpoints:
            try:
                response = requests.get(f"{self.base_url}{endpoint}")
                if response.status_code == 200:
                    health_data = response.json()
                    status = health_data.get('status', 'unknown')
                    print(f"  ✅ {service}: {status}")
                else:
                    print(f"  ❌ {service}: HTTP {response.status_code}")
                    all_healthy = False
            except Exception as e:
                print(f"  💥 {service}: {str(e)}")
                all_healthy = False
        
        if all_healthy:
            print("\n🎉 All memory system components are healthy!")
        
        return all_healthy
    
    def create_demo_session(self, session_name: str, context: str, project_context: str = None) -> Dict:
        """Create a demo session with context"""
        self.print_step("📝", f"Creating Session: {session_name}")
        
        session_data = {
            "user_id": self.user_id,
            "metadata": {
                "demo": True,
                "session_name": session_name,
                "context": context,
                "created_at": datetime.now().isoformat()
            },
            "tags": ["demo", "mvp", session_name.lower().replace(" ", "-")]
        }
        
        if project_context:
            session_data["metadata"]["project"] = project_context
            session_data["tags"].append("project")
        
        response = requests.post(f"{self.base_url}/api/memory/session/start", json=session_data)
        
        if response.status_code == 200:
            session = response.json()
            self.sessions.append(session)
            
            print(f"  ✅ Session created: {session['id']}")
            if session.get('parent_session_id'):
                print(f"  🔗 Automatically linked to: {session['parent_session_id']}")
            
            self.print_result(session, "Session Details")
            return session
        else:
            print(f"  ❌ Failed to create session: {response.status_code}")
            return {}
    
    def add_memory_to_session(self, session_id: str, content: str, memory_type: str, 
                            importance: float = 0.7, entities: List[str] = None) -> Dict:
        """Add a memory to a session"""
        memory_data = {
            "session_id": session_id,
            "content": content,
            "memory_type": memory_type,
            "importance": importance,
            "confidence": 0.9,
            "entities": entities or [],
            "metadata": {"demo": True}
        }
        
        response = requests.post(f"{self.base_url}/api/memory/memories/", json=memory_data)
        
        if response.status_code == 200:
            memory = response.json()
            self.memories.append(memory)
            print(f"    💾 Memory stored: {memory['id'][:8]}... ({memory_type})")
            return memory
        else:
            print(f"    ❌ Failed to store memory: {response.status_code}")
            return {}
    
    def demonstrate_session_linking(self):
        """Demonstrate intelligent session linking"""
        self.print_step("🔗", "Demonstrating Session Linking Suggestions")
        
        # Get linking suggestions
        suggestion_data = {
            "user_id": self.user_id,
            "context_hint": "Python authentication JWT security development"
        }
        
        response = requests.post(f"{self.base_url}/api/memory/linking/suggestions", json=suggestion_data)
        
        if response.status_code == 200:
            suggestions = response.json()
            print(f"  🎯 Found {suggestions['total_candidates']} linking suggestions")
            
            if suggestions['best_suggestion']:
                best = suggestions['best_suggestion']
                print(f"  🏆 Best suggestion:")
                print(f"    - Session: {best['session_id']}")
                print(f"    - Type: {best['link_type']}")
                print(f"    - Confidence: {best['confidence']:.2f}")
                print(f"    - Reason: {best['reason']}")
            
            self.print_result(suggestions['suggestions'][:2], "Top Suggestions")
        
    def demonstrate_context_injection(self, query: str):
        """Demonstrate context injection for Claude-Code"""
        self.print_step("🧠", f"Demonstrating Context Injection for Query: '{query}'")
        
        # Quick context retrieval
        params = {
            "user_id": self.user_id,
            "query": query,
            "max_memories": 5,
            "max_tokens": 2000
        }
        
        response = requests.get(f"{self.base_url}/api/memory/context/quick/{self.user_id}", params=params)
        
        if response.status_code == 200:
            context = response.json()
            
            print(f"  📊 Context retrieved:")
            print(f"    - Total memories: {context['total_memories']}")
            print(f"    - Total tokens: {context['total_tokens']}")
            print(f"    - Max relevance: {context['max_relevance']:.3f}")
            print(f"    - Retrieval time: {context['retrieval_time_ms']:.1f}ms")
            
            print(f"\n  📝 Formatted Context for Claude-Code:")
            print("    " + "─" * 50)
            formatted_context = context['formatted_context']
            context_lines = formatted_context.split('\n')
            for line in context_lines[:10]:  # Show first 10 lines
                print(f"    {line}")
            if len(context_lines) > 10:
                remaining_lines = len(context_lines) - 10
                print(f"    ... ({remaining_lines} more lines)")
            print("    " + "─" * 50)
            
            return context
        else:
            print(f"  ❌ Context retrieval failed: {response.status_code}")
            return {}
    
    def demonstrate_memory_search(self, search_query: str):
        """Demonstrate memory search capabilities"""
        self.print_step("🔍", f"Demonstrating Memory Search: '{search_query}'")
        
        search_data = {
            "query": search_query,
            "user_id": self.user_id,
            "limit": 5
        }
        
        response = requests.post(f"{self.base_url}/api/memory/memories/search", json=search_data)
        
        if response.status_code == 200:
            search_response = response.json()
            results = search_response.get('results', [])
            print(f"  📊 Found {search_response.get('total', len(results))} matching memories")
            
            top_results = results[:3]
            for i, memory in enumerate(top_results, 1):
                print(f"    {i}. {memory['memory_type'].upper()}: {memory['content'][:80]}...")
                print(f"       Importance: {memory['importance']:.2f}, Created: {memory['created_at'][:10]}")
        
    def demonstrate_vector_search(self, query: str):
        """Demonstrate vector similarity search"""
        self.print_step("🎯", f"Demonstrating Vector Similarity Search: '{query}'")
        
        vector_data = {
            "query": query,
            "limit": 3,
            "min_similarity": 0.3
        }
        
        response = requests.post(f"{self.base_url}/api/memory/vector/search", json=vector_data)
        
        if response.status_code == 200:
            search_response = response.json()
            results = search_response.get('results', [])
            print(f"  📊 Found {search_response.get('total', len(results))} semantically similar memories")
            
            for result in results:
                memory = result['memory']
                similarity = result.get('similarity', result.get('similarity_score', 0))
                print(f"    🎯 Similarity: {similarity:.3f} - {memory['content'][:60]}...")
    
    def analyze_session_chain(self):
        """Demonstrate session chain analysis"""
        if not self.sessions:
            return
            
        self.print_step("📈", "Analyzing Session Chain")
        
        session_id = self.sessions[0]['id']
        response = requests.get(f"{self.base_url}/api/memory/linking/chain/{session_id}/analysis")
        
        if response.status_code == 200:
            analysis = response.json()
            
            print(f"  📊 Chain Analysis:")
            print(f"    - Chain length: {analysis['chain_length']} sessions")
            print(f"    - Total memories: {analysis['total_memories']}")
            print(f"    - Unique topics: {len(analysis.get('unique_topics', []))}")
            
            if analysis.get('unique_topics'):
                topics = analysis['unique_topics'][:5]
                print(f"    - Topics: {', '.join(topics)}")
            
            if analysis.get('continuation_patterns'):
                patterns = analysis['continuation_patterns']
                if 'average_gap_minutes' in patterns:
                    print(f"    - Avg gap between sessions: {patterns['average_gap_minutes']:.1f} minutes")
    
    def run_demo(self):
        """Run the complete MVP demo"""
        self.print_header("Claude-Code Working Memory System - MVP Demo")
        
        print("This demo showcases:")
        print("• Session management with automatic linking")
        print("• Memory storage and intelligent retrieval") 
        print("• Context injection optimized for Claude-Code")
        print("• Cross-session memory persistence")
        print("• Vector similarity search")
        print("• Session analytics and chain analysis")
        
        # 1. Verify system health
        if not self.verify_system_health():
            print("\n❌ System health check failed. Please ensure all services are running.")
            return False
        
        # 2. Create first session - Python development
        session1 = self.create_demo_session(
            "Python Development Session",
            "Working on authentication system using JWT tokens",
            "WebApp Security"
        )
        
        if session1:
            # Add memories to first session
            self.add_memory_to_session(
                session1['id'],
                "User authentication should use JWT tokens with 1-hour expiration for security",
                "decision",
                0.9,
                ["JWT", "authentication", "security", "tokens"]
            )
            
            self.add_memory_to_session(
                session1['id'], 
                "import jwt\nfrom datetime import datetime, timedelta\n\ntoken = jwt.encode({'user_id': user.id, 'exp': datetime.utcnow() + timedelta(hours=1)}, secret_key)",
                "code",
                0.8,
                ["JWT", "Python", "tokens", "datetime"]
            )
            
            self.add_memory_to_session(
                session1['id'],
                "User prefers Redis for session storage over database for performance",
                "preference", 
                0.7,
                ["Redis", "session", "performance", "database"]
            )
        
        # Wait a moment for indexing
        time.sleep(2)
        
        # 3. Create second session - Redis implementation
        session2 = self.create_demo_session(
            "Redis Implementation Session",
            "Implementing Redis caching for JWT session management",
            "WebApp Security"
        )
        
        if session2:
            self.add_memory_to_session(
                session2['id'],
                "Redis caching layer implemented with TTL matching JWT expiration time",
                "fact",
                0.8,
                ["Redis", "caching", "TTL", "JWT"]
            )
            
            self.add_memory_to_session(
                session2['id'],
                "Encountered issue with Redis connection timeout - fixed by increasing timeout to 5 seconds",
                "error",
                0.6,
                ["Redis", "timeout", "connection", "error"]
            )
        
        # Wait for processing
        time.sleep(2)
        
        # 4. Demonstrate session linking
        self.demonstrate_session_linking()
        
        # 5. Demonstrate context injection for Claude-Code
        context = self.demonstrate_context_injection(
            "How should I implement JWT authentication with Redis caching?"
        )
        
        # 6. Demonstrate memory search
        self.demonstrate_memory_search("authentication JWT tokens")
        
        # 7. Demonstrate vector similarity search
        self.demonstrate_vector_search("Redis session caching implementation")
        
        # 8. Analyze session chain
        self.analyze_session_chain()
        
        # 9. Create third session to show automatic linking
        self.print_step("🔄", "Creating Third Session to Show Auto-Linking")
        
        session3 = self.create_demo_session(
            "Security Review Session", 
            "Reviewing JWT and Redis security implementation",
            "WebApp Security"
        )
        
        # 10. Show comprehensive context for complex query
        self.print_step("🌟", "Comprehensive Context Retrieval")
        
        comprehensive_data = {
            "user_id": self.user_id,
            "query": "security best practices for JWT and Redis implementation",
            "max_tokens": 4000
        }
        
        response = requests.post(f"{self.base_url}/api/memory/context/comprehensive", json=comprehensive_data)
        
        if response.status_code == 200:
            comp_context = response.json()
            print(f"  🎯 Comprehensive context:")
            print(f"    - Memories: {comp_context['total_memories']}")
            print(f"    - Tokens: {comp_context['total_tokens']}")
            print(f"    - Sections: {len(comp_context['sections'])}")
            print(f"    - Max relevance: {comp_context['max_relevance']:.3f}")
            
            section_types = [s['context_type'] for s in comp_context['sections']]
            print(f"    - Context types: {', '.join(section_types)}")
        
        # 11. Cleanup demo sessions (end them)
        self.print_step("🧹", "Cleaning Up Demo Sessions")
        
        for session in self.sessions:
            response = requests.post(f"{self.base_url}/api/memory/session/{session['id']}/end")
            if response.status_code == 200:
                print(f"  ✅ Ended session: {session['metadata']['session_name']}")
        
        # 12. Show final summary
        self.print_header("MVP Demo Results Summary")
        
        print("🎉 Successfully demonstrated:")
        print(f"  • Created {len(self.sessions)} linked sessions")
        print(f"  • Stored {len(self.memories)} memories across sessions")
        print("  • Automatic session linking based on context")
        print("  • Context injection with relevance scoring")
        print("  • Memory search (text and vector similarity)")
        print("  • Session chain analysis and patterns")
        print("  • LLM-optimized context formatting")
        
        print("\n🚀 The Claude-Code working memory system is fully operational!")
        print("   Claude can now maintain context across multiple sessions,")
        print("   recall important information, and provide more intelligent")
        print("   responses based on historical context.")
        
        return True

def main():
    """Run the MVP demo"""
    demo = ClaudeMemoryDemo()
    
    try:
        success = demo.run_demo()
        if success:
            print("\n✅ MVP Demo completed successfully!")
            return 0
        else:
            print("\n❌ MVP Demo failed!")
            return 1
    except Exception as e:
        print(f"\n💥 Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())