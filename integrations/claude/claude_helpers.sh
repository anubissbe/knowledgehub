#!/bin/bash

# KnowledgeHub Claude Code Helper Functions
# Source this file in your shell to enable KnowledgeHub integration with Claude Code

# Set default values if not already set
export KNOWLEDGEHUB_API="${KNOWLEDGEHUB_API:-http://localhost:3000}"
export KNOWLEDGEHUB_USER="${KNOWLEDGEHUB_USER:-claude-user}"

# Initialize session
claude-init() {
    local cwd="${1:-$(pwd)}"
    echo "🚀 Initializing KnowledgeHub session..."
    
    response=$(curl -s -X POST "${KNOWLEDGEHUB_API}/api/claude-auto/session/start?cwd=${cwd}")
    
    if [ $? -eq 0 ]; then
        export CLAUDE_SESSION_ID=$(echo $response | jq -r '.session.session_id')
        echo "✅ Session initialized: ${CLAUDE_SESSION_ID}"
        echo "📍 Working directory: ${cwd}"
    else
        echo "❌ Failed to initialize session"
        return 1
    fi
}

# Track a mistake
claude-error() {
    local error_type="$1"
    local error_message="$2"
    local solution="$3"
    local resolved="${4:-false}"
    
    if [ -z "$error_type" ] || [ -z "$error_message" ]; then
        echo "Usage: claude-error <type> <message> [solution] [resolved]"
        return 1
    fi
    
    curl -s -X POST "${KNOWLEDGEHUB_API}/api/mistake-learning/track" \
        -H "Content-Type: application/json" \
        -d "{
            \"user_id\": \"${KNOWLEDGEHUB_USER}\",
            \"error_type\": \"${error_type}\",
            \"error_message\": \"${error_message}\",
            \"solution\": \"${solution}\",
            \"resolved\": ${resolved}
        }" > /dev/null
    
    echo "✅ Error tracked: ${error_type}"
}

# Record a decision
claude-decide() {
    local choice="$1"
    local reasoning="$2"
    local alternatives="$3"
    local context="$4"
    local confidence="${5:-0.7}"
    
    if [ -z "$choice" ] || [ -z "$reasoning" ]; then
        echo "Usage: claude-decide <choice> <reasoning> [alternatives] [context] [confidence]"
        return 1
    fi
    
    curl -s -X POST "${KNOWLEDGEHUB_API}/api/decisions/record" \
        -H "Content-Type: application/json" \
        -d "{
            \"user_id\": \"${KNOWLEDGEHUB_USER}\",
            \"chosen_option\": \"${choice}\",
            \"reasoning\": \"${reasoning}\",
            \"alternatives\": ${alternatives:-[]},
            \"context\": ${context:-{}},
            \"confidence_score\": ${confidence}
        }" > /dev/null
    
    echo "✅ Decision recorded: ${choice}"
}

# Track performance
claude-track-performance() {
    local command="$1"
    local duration="$2"
    local success="${3:-true}"
    
    if [ -z "$command" ] || [ -z "$duration" ]; then
        echo "Usage: claude-track-performance <command> <duration_ms> [success]"
        return 1
    fi
    
    curl -s -X POST "${KNOWLEDGEHUB_API}/api/performance/track" \
        -H "Content-Type: application/json" \
        -d "{
            \"user_id\": \"${KNOWLEDGEHUB_USER}\",
            \"command\": \"${command}\",
            \"execution_time_ms\": ${duration},
            \"success\": ${success}
        }" > /dev/null
    
    echo "✅ Performance tracked: ${command} (${duration}ms)"
}

# Search for similar errors
claude-find-error() {
    local query="$1"
    
    if [ -z "$query" ]; then
        echo "Usage: claude-find-error <error-query>"
        return 1
    fi
    
    echo "🔍 Searching for similar errors..."
    curl -s -X GET "${KNOWLEDGEHUB_API}/api/mistake-learning/search?query=${query}" | jq
}

# Get current session stats
claude-stats() {
    echo "📊 KnowledgeHub Statistics"
    echo "========================="
    curl -s "${KNOWLEDGEHUB_API}/api/ai-features/summary" | jq
}

# Create session handoff
claude-handoff() {
    if [ -z "$CLAUDE_SESSION_ID" ]; then
        echo "❌ No active session. Run claude-init first."
        return 1
    fi
    
    echo "📋 Creating session handoff..."
    curl -s -X POST "${KNOWLEDGEHUB_API}/api/claude-auto/session/handoff/${CLAUDE_SESSION_ID}" | jq
}

# Search knowledge base
claude-search() {
    local query="$1"
    
    if [ -z "$query" ]; then
        echo "Usage: claude-search <query>"
        return 1
    fi
    
    echo "🔍 Searching knowledge base..."
    curl -s -X POST "${KNOWLEDGEHUB_API}/api/search" \
        -H "Content-Type: application/json" \
        -d "{\"query\": \"${query}\"}" | jq
}

# Get recent memories
claude-memories() {
    local limit="${1:-10}"
    
    echo "🧠 Recent memories:"
    curl -s "${KNOWLEDGEHUB_API}/api/memory?limit=${limit}" | jq
}

# Get predicted tasks
claude-tasks() {
    echo "🎯 Predicted next tasks:"
    curl -s "${KNOWLEDGEHUB_API}/api/proactive/predictions" | jq
}

# View learning patterns
claude-patterns() {
    echo "📈 Learning patterns:"
    curl -s "${KNOWLEDGEHUB_API}/api/pattern-recognition/patterns" | jq
}

# Check service health
claude-health() {
    echo "🏥 Service Health Check"
    echo "====================="
    
    echo -n "API Gateway: "
    curl -s "${KNOWLEDGEHUB_API}/health" | jq -r '.status' || echo "❌ Down"
    
    echo -n "Database: "
    curl -s "${KNOWLEDGEHUB_API}/health" | jq -r '.services.database' || echo "❌ Down"
    
    echo -n "Redis: "
    curl -s "${KNOWLEDGEHUB_API}/health" | jq -r '.services.redis' || echo "❌ Down"
    
    echo -n "AI Service: "
    curl -s "${KNOWLEDGEHUB_API}/health" | jq -r '.services.ai_service' || echo "❌ Down"
}

# Display available commands
claude-help() {
    echo "🤖 KnowledgeHub Claude Code Commands"
    echo "===================================="
    echo ""
    echo "Session Management:"
    echo "  claude-init [dir]          - Initialize session in directory"
    echo "  claude-handoff            - Create session handoff"
    echo "  claude-stats              - View current statistics"
    echo ""
    echo "Learning & Tracking:"
    echo "  claude-error              - Track an error/mistake"
    echo "  claude-decide             - Record a decision"
    echo "  claude-track-performance  - Track command performance"
    echo ""
    echo "Search & Discovery:"
    echo "  claude-search             - Search knowledge base"
    echo "  claude-find-error         - Find similar errors"
    echo "  claude-memories           - View recent memories"
    echo ""
    echo "Intelligence:"
    echo "  claude-tasks              - Get predicted next tasks"
    echo "  claude-patterns           - View learning patterns"
    echo ""
    echo "System:"
    echo "  claude-health             - Check service health"
    echo "  claude-help               - Show this help"
    echo ""
    echo "Environment:"
    echo "  KNOWLEDGEHUB_API=${KNOWLEDGEHUB_API}"
    echo "  KNOWLEDGEHUB_USER=${KNOWLEDGEHUB_USER}"
    echo "  CLAUDE_SESSION_ID=${CLAUDE_SESSION_ID:-not set}"
}

# Display banner on source
echo "🧠 KnowledgeHub Claude Code Integration Loaded"
echo "Type 'claude-help' for available commands"