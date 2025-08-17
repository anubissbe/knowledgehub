#!/bin/bash
# KnowledgeHub Code Intelligence Helper Functions for Claude Code
# Source this file to get access to code intelligence functions

KNOWLEDGEHUB_API="http://192.168.1.25:3000"
KNOWLEDGEHUB_PROJECT="/opt/projects/knowledgehub"

# Activate project for code intelligence
kh_activate_project() {
    local project_path=${1:-$KNOWLEDGEHUB_PROJECT}
    echo "🔍 Activating project: $project_path"
    curl -s -X POST "$KNOWLEDGEHUB_API/api/code-intelligence/activate-project" \
        -H "Content-Type: application/json" \
        -d "{\"project_path\": \"$project_path\"}" | jq .
}

# Get symbols overview
kh_get_symbols() {
    local project_path=${1:-$KNOWLEDGEHUB_PROJECT}
    local file_path=$2
    echo "📋 Getting symbols for: $project_path"
    
    local url="$KNOWLEDGEHUB_API/api/code-intelligence/symbols/overview?project_path=$project_path"
    if [ -n "$file_path" ]; then
        url="$url&file_path=$file_path"
    fi
    
    curl -s "$url" | jq .
}

# Find specific symbol
kh_find_symbol() {
    local project_path=${1:-$KNOWLEDGEHUB_PROJECT}
    local name_path=$2
    local include_body=${3:-false}
    
    if [ -z "$name_path" ]; then
        echo "Usage: kh_find_symbol [project_path] <name_path> [include_body]"
        return 1
    fi
    
    echo "🔎 Finding symbol: $name_path"
    curl -s -X POST "$KNOWLEDGEHUB_API/api/code-intelligence/symbols/find" \
        -H "Content-Type: application/json" \
        -d "{\"project_path\": \"$project_path\", \"name_path\": \"$name_path\", \"include_body\": $include_body}" | jq .
}

# Search pattern in code
kh_search_pattern() {
    local project_path=${1:-$KNOWLEDGEHUB_PROJECT}
    local pattern=$2
    local file_pattern=$3
    
    if [ -z "$pattern" ]; then
        echo "Usage: kh_search_pattern [project_path] <pattern> [file_pattern]"
        return 1
    fi
    
    echo "🔍 Searching pattern: $pattern"
    local payload="{\"project_path\": \"$project_path\", \"pattern\": \"$pattern\""
    if [ -n "$file_pattern" ]; then
        payload="$payload, \"file_pattern\": \"$file_pattern\""
    fi
    payload="$payload}"
    
    curl -s -X POST "$KNOWLEDGEHUB_API/api/code-intelligence/search/pattern" \
        -H "Content-Type: application/json" \
        -d "$payload" | jq .
}

# Save project memory
kh_save_memory() {
    local project_path=${1:-$KNOWLEDGEHUB_PROJECT}
    local name=$2
    local content=$3
    
    if [ -z "$name" ] || [ -z "$content" ]; then
        echo "Usage: kh_save_memory [project_path] <name> <content>"
        return 1
    fi
    
    echo "💾 Saving memory: $name"
    curl -s -X POST "$KNOWLEDGEHUB_API/api/code-intelligence/memory/save" \
        -H "Content-Type: application/json" \
        -d "{\"project_path\": \"$project_path\", \"name\": \"$name\", \"content\": \"$content\"}" | jq .
}

# Load project memory
kh_load_memory() {
    local project_path=${1:-$KNOWLEDGEHUB_PROJECT}
    local name=$2
    
    if [ -z "$name" ]; then
        echo "Usage: kh_load_memory [project_path] <name>"
        return 1
    fi
    
    echo "📖 Loading memory: $name"
    curl -s "$KNOWLEDGEHUB_API/api/code-intelligence/memory/load?project_path=$project_path&name=$name" | jq .
}

# List project memories
kh_list_memories() {
    local project_path=${1:-$KNOWLEDGEHUB_PROJECT}
    echo "📚 Listing memories for: $project_path"
    curl -s "$KNOWLEDGEHUB_API/api/code-intelligence/memory/list?project_path=$project_path" | jq .
}

# Health check
kh_health() {
    echo "🏥 KnowledgeHub Health Check"
    curl -s "$KNOWLEDGEHUB_API/api/code-intelligence/health" | jq .
}

# Quick project setup
kh_setup() {
    local project_path=${1:-$KNOWLEDGEHUB_PROJECT}
    echo "🚀 Setting up KnowledgeHub Code Intelligence for: $project_path"
    
    echo "1. Health check..."
    kh_health
    
    echo -e "\n2. Activating project..."
    kh_activate_project "$project_path"
    
    echo -e "\n3. Getting symbol count..."
    kh_get_symbols "$project_path" | jq '.count // 0'
    
    echo -e "\n✅ KnowledgeHub Code Intelligence ready!"
}

# Help function
kh_help() {
    echo "📚 KnowledgeHub Code Intelligence Helper Functions"
    echo ""
    echo "Available functions:"
    echo "  kh_activate_project [project_path]       - Activate project for analysis"
    echo "  kh_get_symbols [project_path] [file]     - Get symbols overview"
    echo "  kh_find_symbol [project_path] <name>     - Find specific symbol"
    echo "  kh_search_pattern [project_path] <pattern> [file_pattern] - Search code"
    echo "  kh_save_memory [project_path] <name> <content> - Save memory"
    echo "  kh_load_memory [project_path] <name>     - Load memory"
    echo "  kh_list_memories [project_path]          - List memories"
    echo "  kh_health                                - Health check"
    echo "  kh_setup [project_path]                  - Quick setup"
    echo "  kh_help                                  - Show this help"
    echo ""
    echo "Default project: $KNOWLEDGEHUB_PROJECT"
    echo "API URL: $KNOWLEDGEHUB_API"
}

echo "✅ KnowledgeHub Code Intelligence helpers loaded!"
echo "Run 'kh_help' for available functions or 'kh_setup' for quick start"