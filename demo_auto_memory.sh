#!/bin/bash
# Auto-Memory Feature Demonstration

echo "=== KnowledgeHub Auto-Memory Demo ==="
echo ""

# Source the helper functions
source claude_code_helpers.sh

echo "1. Auto-Memory Status:"
claude-auto-memory status
echo ""

echo "2. Auto-Pattern Detection Examples:"
echo "   Detecting an error pattern..."
claude-auto-detect "ERROR: Connection refused to port 5432"
echo ""

echo "   Detecting a solution..."
claude-auto-detect "Fixed by updating firewall rules to allow PostgreSQL connections"
echo ""

echo "   Detecting a TODO..."
claude-auto-detect "TODO: Implement connection pooling for better performance"
echo ""

echo "   Detecting a decision..."
claude-auto-detect "Decided to use connection pooling with 10 connections max"
echo ""

echo "3. Auto-Memory can be toggled:"
echo "   Disabling auto-memory..."
claude-auto-memory off
echo ""

echo "   Status check..."
claude-auto-memory status
echo ""

echo "   Re-enabling auto-memory..."
claude-auto-memory on
echo ""

echo "=== Auto-Memory Features Summary ==="
echo ""
echo "Auto-memory automatically captures:"
echo "• Git commits and checkouts (when using git commands)"
echo "• npm/yarn installations and script runs"
echo "• Docker operations (run, compose up)"
echo "• Command failures (make, build commands)"
echo "• Important patterns (errors, solutions, TODOs, decisions)"
echo ""
echo "All captured memories use the hybrid memory system for:"
echo "• Ultra-fast local storage (<100ms)"
echo "• Automatic background sync to PostgreSQL"
echo "• Token optimization for AI context efficiency"
echo ""
echo "Commands available:"
echo "• claude-auto-memory [on|off|status] - Toggle auto-memory"
echo "• claude-auto-detect <text> - Manually detect patterns"
echo "• claude-auto-history - View auto-captured memories"