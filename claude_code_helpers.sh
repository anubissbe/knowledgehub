#!/bin/bash
# Claude Code Helper Functions for KnowledgeHub Integration
# This file provides all the CLI commands for seamless Claude Code integration

# Configuration
KNOWLEDGEHUB_API="http://192.168.1.25:3000"
CLAUDE_USER_ID="${CLAUDE_USER_ID:-claude}"
CLAUDE_SESSION_ID=""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Helper function for API calls
_api_call() {
    local method=$1
    local endpoint=$2
    local data=$3
    
    if [ -z "$data" ]; then
        curl -s -X "$method" "$KNOWLEDGEHUB_API$endpoint" \
            -H "Content-Type: application/json"
    else
        curl -s -X "$method" "$KNOWLEDGEHUB_API$endpoint" \
            -H "Content-Type: application/json" \
            -d "$data"
    fi
}

# Session Management Functions
claude-init() {
    echo -e "${GREEN}Initializing Claude session...${NC}"
    
    # Create new session
    response=$(_api_call POST "/api/claude-auto/session/initialize" "{\"user_id\": \"$CLAUDE_USER_ID\"}")
    CLAUDE_SESSION_ID=$(echo "$response" | jq -r '.session_id')
    
    if [ -n "$CLAUDE_SESSION_ID" ]; then
        export CLAUDE_SESSION_ID
        echo -e "${GREEN}✓ Session initialized: $CLAUDE_SESSION_ID${NC}"
        
        # Restore context
        context=$(_api_call GET "/api/memory/context/quick/$CLAUDE_USER_ID")
        echo -e "${GREEN}✓ Context restored: $(echo "$context" | jq -r '.memories_count') memories loaded${NC}"
        
        # Show predicted tasks
        claude-tasks
    else
        echo -e "${RED}✗ Failed to initialize session${NC}"
        return 1
    fi
}

claude-handoff() {
    local message=${1:-"Session handoff"}
    echo -e "${YELLOW}Creating session handoff...${NC}"
    
    response=$(_api_call POST "/api/claude-auto/session/handoff" "{
        \"session_id\": \"$CLAUDE_SESSION_ID\",
        \"message\": \"$message\",
        \"user_id\": \"$CLAUDE_USER_ID\"
    }")
    
    echo -e "${GREEN}✓ Handoff created: $(echo "$response" | jq -r '.handoff_id')${NC}"
}

claude-session() {
    echo -e "${GREEN}Current Session Status:${NC}"
    _api_call GET "/api/claude-auto/session/current" | jq '.'
}

claude-stats() {
    echo -e "${GREEN}Memory Statistics:${NC}"
    _api_call GET "/api/memory/stats" | jq '.'
}

claude-context() {
    echo -e "${GREEN}Current Context:${NC}"
    _api_call GET "/api/memory/context/quick/$CLAUDE_USER_ID" | jq '.'
}

# Error Learning Functions
claude-error() {
    local error_type=$1
    local error_message=$2
    local solution=$3
    local resolved=${4:-true}
    
    echo -e "${YELLOW}Recording error...${NC}"
    
    response=$(_api_call POST "/api/mistake-learning/record" "{
        \"error_type\": \"$error_type\",
        \"error_message\": \"$error_message\",
        \"solution\": \"$solution\",
        \"resolved\": $resolved,
        \"user_id\": \"$CLAUDE_USER_ID\",
        \"session_id\": \"$CLAUDE_SESSION_ID\"
    }")
    
    echo -e "${GREEN}✓ Error recorded${NC}"
}

claude-find-error() {
    local query=$1
    echo -e "${GREEN}Searching for similar errors...${NC}"
    
    _api_call POST "/api/mistake-learning/search" "{\"query\": \"$query\"}" | jq -r '.results[] | "[\(.score)] \(.error_type): \(.error_message)\n  Solution: \(.solution)\n"'
}

claude-lessons() {
    echo -e "${GREEN}Learned Lessons:${NC}"
    _api_call GET "/api/mistake-learning/lessons" | jq -r '.lessons[] | "• \(.pattern): \(.recommendation)"'
}

# Decision Tracking Functions
claude-decide() {
    local decision=$1
    local reasoning=$2
    local alternatives=$3
    local context=$4
    local confidence=${5:-0.8}
    
    echo -e "${YELLOW}Recording decision...${NC}"
    
    response=$(_api_call POST "/api/decisions/record" "{
        \"decision\": \"$decision\",
        \"reasoning\": \"$reasoning\",
        \"alternatives\": \"$alternatives\",
        \"context\": \"$context\",
        \"confidence\": $confidence,
        \"user_id\": \"$CLAUDE_USER_ID\",
        \"session_id\": \"$CLAUDE_SESSION_ID\"
    }")
    
    echo -e "${GREEN}✓ Decision recorded: $(echo "$response" | jq -r '.decision_id')${NC}"
}

claude-decisions() {
    echo -e "${GREEN}Recent Decisions:${NC}"
    _api_call GET "/api/decisions/recent" | jq -r '.decisions[] | "[\(.timestamp | split("T")[0])] \(.decision)\n  Reasoning: \(.reasoning)\n  Confidence: \(.confidence)\n"'
}

# Performance Tracking Functions
claude-track-performance() {
    local command=$1
    local duration=$2
    local success=${3:-true}
    
    _api_call POST "/api/performance/track" "{
        \"command\": \"$command\",
        \"duration\": $duration,
        \"success\": $success,
        \"user_id\": \"$CLAUDE_USER_ID\",
        \"session_id\": \"$CLAUDE_SESSION_ID\"
    }" > /dev/null
}

claude-performance-recommend() {
    echo -e "${GREEN}Performance Recommendations:${NC}"
    _api_call GET "/api/performance/recommendations" | jq -r '.recommendations[] | "• \(.category): \(.suggestion) (impact: \(.impact))"'
}

# Task Prediction Functions
claude-tasks() {
    echo -e "${GREEN}Predicted Next Tasks:${NC}"
    _api_call GET "/api/proactive/next-tasks" | jq -r '.tasks[] | "[\(.probability)] \(.task)\n  Context: \(.context)"'
}

claude-suggest() {
    local context=$1
    echo -e "${GREEN}AI Suggestions:${NC}"
    
    _api_call POST "/api/proactive/suggest" "{\"context\": \"$context\"}" | jq -r '.suggestions[] | "• \(.suggestion) (confidence: \(.confidence))"'
}

# Code Evolution Tracking
claude-track-code() {
    local file_path=$1
    local change_type=$2
    local description=$3
    
    echo -e "${YELLOW}Tracking code change...${NC}"
    
    _api_call POST "/api/code-evolution/track" "{
        \"file_path\": \"$file_path\",
        \"change_type\": \"$change_type\",
        \"description\": \"$description\",
        \"user_id\": \"$CLAUDE_USER_ID\",
        \"session_id\": \"$CLAUDE_SESSION_ID\"
    }" > /dev/null
    
    echo -e "${GREEN}✓ Code change tracked${NC}"
}

claude-code-history() {
    local file_path=$1
    echo -e "${GREEN}Code Evolution History:${NC}"
    
    _api_call GET "/api/code-evolution/history?file=$file_path" | jq -r '.changes[] | "[\(.timestamp | split("T")[0])] \(.change_type): \(.description)"'
}

# Pattern Recognition
claude-patterns() {
    echo -e "${GREEN}Recognized Patterns:${NC}"
    _api_call GET "/api/patterns/recent" | jq -r '.patterns[] | "• \(.pattern_type): \(.description)\n  Occurrences: \(.occurrences)"'
}

claude-apply-pattern() {
    local pattern_id=$1
    local target_file=$2
    
    echo -e "${YELLOW}Applying pattern...${NC}"
    
    response=$(_api_call POST "/api/patterns/apply" "{
        \"pattern_id\": \"$pattern_id\",
        \"target_file\": \"$target_file\",
        \"user_id\": \"$CLAUDE_USER_ID\"
    }")
    
    echo -e "${GREEN}✓ Pattern applied successfully${NC}"
}

# Workflow Integration
claude-workflow-start() {
    local workflow_name=$1
    echo -e "${YELLOW}Starting workflow: $workflow_name${NC}"
    
    response=$(_api_call POST "/api/claude-workflow/start" "{
        \"workflow_name\": \"$workflow_name\",
        \"user_id\": \"$CLAUDE_USER_ID\",
        \"session_id\": \"$CLAUDE_SESSION_ID\"
    }")
    
    workflow_id=$(echo "$response" | jq -r '.workflow_id')
    export CLAUDE_WORKFLOW_ID=$workflow_id
    echo -e "${GREEN}✓ Workflow started: $workflow_id${NC}"
}

claude-workflow-step() {
    local step_name=$1
    local status=${2:-"completed"}
    
    _api_call POST "/api/claude-workflow/step" "{
        \"workflow_id\": \"$CLAUDE_WORKFLOW_ID\",
        \"step_name\": \"$step_name\",
        \"status\": \"$status\"
    }" > /dev/null
}

claude-workflow-end() {
    local summary=$1
    echo -e "${YELLOW}Completing workflow...${NC}"
    
    _api_call POST "/api/claude-workflow/complete" "{
        \"workflow_id\": \"$CLAUDE_WORKFLOW_ID\",
        \"summary\": \"$summary\"
    }" > /dev/null
    
    unset CLAUDE_WORKFLOW_ID
    echo -e "${GREEN}✓ Workflow completed${NC}"
}

# Memory Management
claude-remember() {
    local content=$1
    local type=${2:-"general"}
    local tags=$3
    
    echo -e "${YELLOW}Storing memory...${NC}"
    
    response=$(_api_call POST "/api/memory" "{
        \"content\": \"$content\",
        \"type\": \"$type\",
        \"tags\": [$(echo "$tags" | sed 's/,/","/g' | sed 's/^/"/;s/$/"/')]
        \"user_id\": \"$CLAUDE_USER_ID\",
        \"session_id\": \"$CLAUDE_SESSION_ID\"
    }")
    
    echo -e "${GREEN}✓ Memory stored: $(echo "$response" | jq -r '.memory_id')${NC}"
}

claude-search() {
    local query=$1
    echo -e "${GREEN}Searching memories...${NC}"
    
    _api_call POST "/api/memory/search" "{\"query\": \"$query\"}" | jq -r '.results[] | "[\(.score)] \(.content | .[0:100])..."'
}

# LAN Service Testing
claude-test-lan-services() {
    echo -e "${GREEN}Testing LAN Service Connectivity...${NC}"
    
    # Test Synology NAS services
    echo -e "\n${YELLOW}Synology NAS (192.168.1.24):${NC}"
    for port in 3001 3002 3003 3004 3005 3006 3007 3008 3009 3010 3011 3012 3014; do
        if curl -s -f "http://192.168.1.24:$port/health" > /dev/null 2>&1; then
            echo -e "  Port $port: ${GREEN}✓${NC}"
        else
            echo -e "  Port $port: ${RED}✗${NC}"
        fi
    done
    
    # Test KnowledgeHub services
    echo -e "\n${YELLOW}KnowledgeHub Server (192.168.1.25):${NC}"
    for port in 3000 3008 3100 8002; do
        if curl -s -f "http://192.168.1.25:$port/health" > /dev/null 2>&1; then
            echo -e "  Port $port: ${GREEN}✓${NC}"
        else
            echo -e "  Port $port: ${RED}✗${NC}"
        fi
    done
}

claude-lan-health-check() {
    claude-test-lan-services
}

# Utility Functions
claude-check() {
    local action=$1
    echo -e "${YELLOW}Safety check for: $action${NC}"
    
    response=$(_api_call POST "/api/safety/check" "{\"action\": \"$action\"}")
    safe=$(echo "$response" | jq -r '.safe')
    
    if [ "$safe" = "true" ]; then
        echo -e "${GREEN}✓ Action is safe to proceed${NC}"
        return 0
    else
        echo -e "${RED}✗ Action may have risks: $(echo "$response" | jq -r '.warnings[]')${NC}"
        return 1
    fi
}

claude-sync() {
    echo -e "${YELLOW}Synchronizing with KnowledgeHub...${NC}"
    
    response=$(_api_call POST "/api/memory/sync" "{\"user_id\": \"$CLAUDE_USER_ID\"}")
    echo -e "${GREEN}✓ Synchronized $(echo "$response" | jq -r '.synced_count') items${NC}"
}

# Help function
claude-help() {
    echo -e "${GREEN}Claude Code Helper Commands:${NC}"
    echo -e "\n${YELLOW}Session Management:${NC}"
    echo "  claude-init              - Initialize session with context restoration"
    echo "  claude-handoff [msg]     - Create session handoff"
    echo "  claude-session           - Show current session status"
    echo "  claude-stats             - Show memory statistics"
    echo "  claude-context           - Show current context"
    
    echo -e "\n${YELLOW}Error Learning:${NC}"
    echo "  claude-error <type> <msg> <solution> [resolved] - Record error"
    echo "  claude-find-error <query>                       - Find similar errors"
    echo "  claude-lessons                                  - Show learned lessons"
    
    echo -e "\n${YELLOW}Decision Tracking:${NC}"
    echo "  claude-decide <decision> <reasoning> <alternatives> <context> [confidence] - Record decision"
    echo "  claude-decisions                                                          - Show recent decisions"
    
    echo -e "\n${YELLOW}Performance:${NC}"
    echo "  claude-track-performance <cmd> <duration> [success] - Track command performance"
    echo "  claude-performance-recommend                        - Get performance recommendations"
    
    echo -e "\n${YELLOW}Task Management:${NC}"
    echo "  claude-tasks             - Show predicted next tasks"
    echo "  claude-suggest <context> - Get AI suggestions"
    
    echo -e "\n${YELLOW}Code Evolution:${NC}"
    echo "  claude-track-code <file> <type> <description> - Track code change"
    echo "  claude-code-history <file>                    - Show file history"
    
    echo -e "\n${YELLOW}Patterns:${NC}"
    echo "  claude-patterns                         - Show recognized patterns"
    echo "  claude-apply-pattern <id> <target_file> - Apply pattern to file"
    
    echo -e "\n${YELLOW}Workflows:${NC}"
    echo "  claude-workflow-start <name> - Start workflow"
    echo "  claude-workflow-step <name> [status] - Record workflow step"
    echo "  claude-workflow-end <summary> - Complete workflow"
    
    echo -e "\n${YELLOW}Memory:${NC}"
    echo "  claude-remember <content> [type] [tags] - Store memory"
    echo "  claude-search <query>                   - Search memories"
    
    echo -e "\n${YELLOW}Source Management:${NC}"
    echo "  claude-add-source <url> [type] [name]  - Add knowledge source"
    echo "  claude-list-sources                    - List all knowledge sources"
    echo "  claude-refresh-source <id>             - Refresh/re-scrape a source"
    
    echo -e "\n${YELLOW}Utilities:${NC}"
    echo "  claude-check <action>      - Safety check for action"
    echo "  claude-sync                - Sync with KnowledgeHub"
    echo "  claude-test-lan-services   - Test LAN connectivity"
    echo "  claude-help                - Show this help"
}

# Source Management Functions
claude-add-source() {
    local url=$1
    local type=${2:-"website"}
    local name=${3:-"$url"}
    
    if [ -z "$url" ]; then
        echo -e "${RED}Usage: claude-add-source <url> [type] [name]${NC}"
        echo "Types: website, documentation, repository, api, wiki"
        return 1
    fi
    
    echo -e "${YELLOW}Adding knowledge source...${NC}"
    response=$(curl -s -X POST "${KNOWLEDGEHUB_API}/api/sources/" \
        -H "Content-Type: application/json" \
        -d "{
            \"url\": \"$url\",
            \"type\": \"$type\",
            \"name\": \"$name\",
            \"config\": {
                \"max_depth\": 3,
                \"max_pages\": 500,
                \"crawl_delay\": 1.0
            }
        }")
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Source added successfully${NC}"
        echo "$response" | jq '.' 2>/dev/null || echo "$response"
    else
        echo -e "${RED}✗ Failed to add source${NC}"
    fi
}

claude-list-sources() {
    echo -e "${YELLOW}Fetching knowledge sources...${NC}"
    response=$(curl -s "${KNOWLEDGEHUB_API}/api/sources/")
    
    if [ $? -eq 0 ]; then
        echo "$response" | jq -r '.[] | "[\(.id)] \(.name) - \(.url) (\(.status))"' 2>/dev/null || echo "$response"
    else
        echo -e "${RED}✗ Failed to fetch sources${NC}"
    fi
}

claude-refresh-source() {
    local source_id=$1
    
    if [ -z "$source_id" ]; then
        echo -e "${RED}Usage: claude-refresh-source <source_id>${NC}"
        return 1
    fi
    
    echo -e "${YELLOW}Refreshing source ${source_id}...${NC}"
    response=$(curl -s -X POST "${KNOWLEDGEHUB_API}/api/sources/${source_id}/refresh")
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Refresh triggered${NC}"
    else
        echo -e "${RED}✗ Failed to refresh source${NC}"
    fi
}

# Auto-initialize if not already done
if [ -z "$CLAUDE_SESSION_ID" ]; then
    echo -e "${YELLOW}No active session detected. Run 'claude-init' to start.${NC}"
fi

echo -e "${GREEN}Claude Code helpers loaded. Type 'claude-help' for available commands.${NC}"