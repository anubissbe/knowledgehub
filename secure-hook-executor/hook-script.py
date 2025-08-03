#!/usr/bin/env python3
"""
Secure hook executor for Claude Code integration
Runs in a sandboxed Docker container with minimal privileges
"""

import sys
import json
import os
import requests
from typing import Dict, Any

# Configuration from environment
API_URL = os.environ.get("KNOWLEDGEHUB_API_URL", "http://api:3000")
API_KEY = os.environ.get("KNOWLEDGEHUB_API_KEY", "")
USER_ID = os.environ.get("CLAUDE_USER_ID", "claude-code")
PROJECT_ID = os.environ.get("CLAUDE_PROJECT_ID", "")

def get_context_for_prompt(prompt: str) -> Dict[str, Any]:
    """
    Query the RAG system for relevant context
    """
    try:
        # Prepare RAG query
        query_data = {
            "query": prompt,
            "user_id": USER_ID,
            "project_id": PROJECT_ID,
            "top_k": 5,
            "use_hybrid": True,
            "stream": False
        }
        
        # Make API request
        response = requests.post(
            f"{API_URL}/api/rag/query",
            json=query_data,
            headers={"X-API-Key": API_KEY},
            timeout=10
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {
                "error": f"API returned status {response.status_code}",
                "response": "Unable to retrieve context."
            }
            
    except requests.exceptions.Timeout:
        return {
            "error": "Request timeout",
            "response": "Context retrieval timed out."
        }
    except Exception as e:
        return {
            "error": str(e),
            "response": "Failed to retrieve context."
        }

def format_context_for_claude(rag_result: Dict[str, Any], prompt: str) -> str:
    """
    Format RAG results as context for Claude
    """
    if "error" in rag_result:
        # Return minimal context on error
        return f"# Context\nUnable to retrieve additional context: {rag_result['error']}\n\n# User Query\n{prompt}"
    
    # Build context from RAG results
    context_parts = ["# Retrieved Context\n"]
    
    # Add the synthesized response
    if "response" in rag_result:
        context_parts.append(f"## Summary\n{rag_result['response']}\n")
    
    # Add source snippets
    if "source_nodes" in rag_result:
        context_parts.append("## Relevant Documentation\n")
        for i, node in enumerate(rag_result["source_nodes"][:3], 1):
            text = node.get("text", "").strip()
            if text:
                # Truncate long snippets
                if len(text) > 500:
                    text = text[:500] + "..."
                context_parts.append(f"### Source {i}\n{text}\n")
    
    # Add the original prompt
    context_parts.append(f"\n# User Query\n{prompt}")
    
    return "\n".join(context_parts)

def main():
    """
    Main execution - read prompt, get context, output result
    """
    try:
        # Read prompt from stdin
        prompt = sys.stdin.read().strip()
        
        if not prompt:
            print("# User Query\n(No prompt provided)")
            return
        
        # Get context from RAG system
        rag_result = get_context_for_prompt(prompt)
        
        # Format and output context
        formatted_context = format_context_for_claude(rag_result, prompt)
        print(formatted_context)
        
    except Exception as e:
        # On any error, output minimal context
        print(f"# Context\nError in context retrieval: {e}\n\n# User Query\n{prompt if 'prompt' in locals() else '(No prompt)'}")

if __name__ == "__main__":
    main()