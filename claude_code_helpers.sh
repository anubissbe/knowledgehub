#!/bin/bash
# Claude Code Helper Functions for KnowledgeHub Integration
# This file provides all the CLI commands for seamless Claude Code integration

# Configuration
KNOWLEDGEHUB_API="http://192.168.1.25:3000"
CLAUDE_USER_ID="${CLAUDE_USER_ID:-claude}"
CLAUDE_SESSION_ID=""
HYBRID_MEMORY_ENABLED="${HYBRID_MEMORY_ENABLED:-true}"
CLAUDE_AUTO_MEMORY="${CLAUDE_AUTO_MEMORY:-true}"
CLAUDE_AUTO_MEMORY_INTERVAL="${CLAUDE_AUTO_MEMORY_INTERVAL:-300}"  # 5 minutes

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

# Agent System Configuration
CLAUDE_AGENT_PATH="${CLAUDE_AGENT_PATH:-/opt/projects/memory-system/.claude/agents}"
CLAUDE_AGENT_AUTO="${CLAUDE_AGENT_AUTO:-true}"
CLAUDE_ACTIVE_AGENTS=""

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

# Hybrid memory store function
_hybrid_store() {
    local content=$1
    local type=${2:-"general"}
    local project=${3:-""}
    local tags=${4:-"[]"}
    
    if [ "$HYBRID_MEMORY_ENABLED" = "true" ]; then
        _api_call POST "/api/hybrid/quick-store" "{
            \"content\": \"$content\",
            \"type\": \"$type\",
            \"project\": \"$project\",
            \"tags\": $tags
        }"
    else
        # Fallback to regular memory API
        _api_call POST "/api/memory" "{
            \"user_id\": \"$CLAUDE_USER_ID\",
            \"content\": \"$content\",
            \"memory_type\": \"$type\",
            \"metadata\": {\"project\": \"$project\", \"tags\": $tags}
        }"
    fi
}

# Hybrid memory search function
_hybrid_search() {
    local query=$1
    local limit=${2:-10}
    
    if [ "$HYBRID_MEMORY_ENABLED" = "true" ]; then
        _api_call GET "/api/hybrid/quick-recall?query=$(echo "$query" | jq -sRr @uri)&limit=$limit"
    else
        # Fallback to regular search
        _api_call POST "/api/v1/search/unified" "{
            \"query\": \"$query\",
            \"search_type\": \"hybrid\",
            \"limit\": $limit
        }"
    fi
}

# Auto-memory functions
_auto_remember() {
    local content=$1
    local type=${2:-"auto"}
    local tags=${3:-"[\"auto-captured\"]"}
    
    if [ "$CLAUDE_AUTO_MEMORY" = "true" ] && [ "$HYBRID_MEMORY_ENABLED" = "true" ]; then
        # Get current project context
        local project_name=$(basename "$(pwd)")
        
        # Store in hybrid memory silently
        _hybrid_store "$content" "$type" "$project_name" "$tags" > /dev/null 2>&1
    fi
}

# Capture command execution for auto-memory
_capture_command() {
    local command=$1
    local exit_code=$2
    local output=$3
    
    if [ "$CLAUDE_AUTO_MEMORY" = "true" ]; then
        local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
        local cwd=$(pwd)
        
        # Capture successful commands
        if [ $exit_code -eq 0 ]; then
            _auto_remember "Command executed: $command in $cwd at $timestamp (success)" "command" "[\"command\", \"success\"]"
        else
            # Capture failed commands with error
            _auto_remember "Command failed: $command in $cwd at $timestamp (exit code: $exit_code)" "error" "[\"command\", \"error\"]"
        fi
    fi
}

# Auto-capture important patterns
_auto_capture_pattern() {
    local pattern_type=$1
    local content=$2
    
    case "$pattern_type" in
        "error")
            _auto_remember "Error pattern detected: $content" "error" "[\"pattern\", \"error\", \"auto-detected\"]"
            ;;
        "solution")
            _auto_remember "Solution pattern: $content" "learning" "[\"pattern\", \"solution\", \"auto-detected\"]"
            ;;
        "decision")
            _auto_remember "Decision made: $content" "decision" "[\"pattern\", \"decision\", \"auto-detected\"]"
            ;;
        "todo")
            _auto_remember "TODO/Task: $content" "task" "[\"pattern\", \"todo\", \"auto-detected\"]"
            ;;
    esac
}

# Session Management Functions
claude-init() {
    echo -e "${GREEN}Initializing Claude session...${NC}"
    
    # Check hybrid memory status
    if [ "$HYBRID_MEMORY_ENABLED" = "true" ]; then
        echo -e "${GREEN}✓ Hybrid memory system enabled (fast local + distributed sync)${NC}"
    fi
    
    # Check auto-memory status
    if [ "$CLAUDE_AUTO_MEMORY" = "true" ]; then
        echo -e "${GREEN}✓ Auto-memory enabled (automatically captures important information)${NC}"
    fi
    
    # Get current working directory
    current_dir=$(pwd)
    
    # Start session with KnowledgeHub
    response=$(_api_call POST "/api/claude-auto/session/start?cwd=$current_dir")
    if [ $? -eq 0 ]; then
        CLAUDE_SESSION_ID=$(echo "$response" | jq -r '.session.session_id // "unknown"')
        export CLAUDE_SESSION_ID
        echo -e "${GREEN}✓ Session initialized: $CLAUDE_SESSION_ID${NC}"
        
        # Show context info
        project_name=$(echo "$response" | jq -r '.session.project_name // "unknown"')
        project_type=$(echo "$response" | jq -r '.session.project_type // "unknown"')
        echo -e "${GREEN}✓ Project: $project_name (type: $project_type)${NC}"
        
        # Store session start in hybrid memory
        if [ "$HYBRID_MEMORY_ENABLED" = "true" ]; then
            _hybrid_store "Claude session started: $CLAUDE_SESSION_ID in project $project_name at $current_dir" \
                "session" \
                "$project_name" \
                "[\"session-start\", \"claude-init\"]" > /dev/null
        fi
        
        # Check for handoff notes
        handoff_count=$(echo "$response" | jq -r '.context.handoff_notes | length')
        if [ "$handoff_count" -gt 0 ]; then
            echo -e "${YELLOW}⚠ Found $handoff_count handoff notes from previous session${NC}"
        fi
        
        # Check for unfinished tasks
        task_count=$(echo "$response" | jq -r '.context.unfinished_tasks | length')
        if [ "$task_count" -gt 0 ]; then
            echo -e "${YELLOW}⚠ Found $task_count unfinished tasks${NC}"
        fi
    else
        # Fallback to simple session ID
        CLAUDE_SESSION_ID="session-$(date +%s)"
        export CLAUDE_SESSION_ID
        echo -e "${YELLOW}⚠ Using fallback session: $CLAUDE_SESSION_ID${NC}"
    fi
    
    # Show predicted tasks if available
    claude-tasks
}

claude-handoff() {
    local message=${1:-"Session handoff"}
    echo -e "${YELLOW}Creating session handoff...${NC}"
    
    # Store handoff as a memory item with special tag
    response=$(_api_call POST "/api/memory" "{
        \"content\": \"Handoff: $message\",
        \"tags\": [\"handoff\", \"session:$CLAUDE_SESSION_ID\"],
        \"user_id\": \"$CLAUDE_USER_ID\",
        \"session_id\": \"$CLAUDE_SESSION_ID\"
    }")
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Handoff created${NC}"
    else
        echo -e "${RED}✗ Failed to create handoff${NC}"
    fi
}

claude-session() {
    echo -e "${GREEN}Current Session Status:${NC}"
    _api_call GET "/api/claude-auto/session/current" | jq '.'
}

claude-stats() {
    echo -e "${GREEN}Memory Statistics:${NC}"
    _api_call GET "/api/claude-auto/memory/stats" | jq '.stats // {}'
}

claude-context() {
    echo -e "${GREEN}Current Context:${NC}"
    # Use session endpoint instead
    _api_call GET "/api/claude-auto/session/current" | jq '.'
}

# Error Learning Functions
claude-error() {
    local error_type=$1
    local error_message=$2
    local solution=$3
    local resolved=${4:-true}
    
    echo -e "${YELLOW}Recording error...${NC}"
    
    # URL encode the parameters
    error_type_encoded=$(echo -n "$error_type" | jq -sRr @uri)
    error_message_encoded=$(echo -n "$error_message" | jq -sRr @uri)
    solution_encoded=$(echo -n "$solution" | jq -sRr @uri)
    
    response=$(curl -s -X POST "$KNOWLEDGEHUB_API/api/mistake-learning/track?error_type=$error_type_encoded&error_message=$error_message_encoded&solution=$solution_encoded&resolved=$resolved")
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Error recorded${NC}"
    else
        echo -e "${RED}✗ Failed to record error${NC}"
    fi
}

claude-find-error() {
    local query=$1
    echo -e "${GREEN}Searching for similar errors...${NC}"
    
    local response=$(_api_call POST "/api/mistake-learning/search" "{\"query\": \"$query\"}")
    
    if [ -z "$response" ] || [ "$response" = "[]" ]; then
        echo -e "${YELLOW}No similar errors found${NC}"
    else
        echo "$response" | jq -r '.[] | "• \(.error_type): \(.error_message)\n  Solution: \(.solution)\n  Resolved: \(.resolved)"'
    fi
}

claude-lessons() {
    echo -e "${GREEN}Learned Lessons:${NC}"
    
    local response=$(_api_call GET "/api/mistake-learning/lessons")
    
    if [ -z "$response" ] || [ "$response" = "[]" ]; then
        echo -e "${YELLOW}No lessons learned yet${NC}"
    else
        echo "$response" | jq -r '.[] | "• [\(.error_type)] \(.lesson.summary)\n  Category: \(.lesson.category // "unknown")\n  Repetitions: \(.repetitions)"'
    fi
}

# Decision Tracking Functions
claude-decide() {
    local decision=$1
    local reasoning=$2
    local alternatives=$3
    local context=$4
    local confidence=${5:-0.8}
    
    echo -e "${YELLOW}Recording decision...${NC}"
    
    # URL encode the parameters
    decision_encoded=$(echo -n "$decision" | jq -sRr @uri)
    reasoning_encoded=$(echo -n "$reasoning" | jq -sRr @uri)
    
    # Build alternatives array
    alt_array="[{\"solution\": \"$alternatives\", \"pros\": [], \"cons\": [], \"risk_level\": \"medium\"}]"
    
    response=$(curl -s -X POST "$KNOWLEDGEHUB_API/api/decisions/record?decision_title=$decision_encoded&chosen_solution=$decision_encoded&reasoning=$reasoning_encoded&confidence=$confidence&session_id=$CLAUDE_SESSION_ID" \
        -H "Content-Type: application/json" \
        -d "{\"alternatives\": $alt_array, \"context\": {\"description\": \"$context\"}}")
    
    if echo "$response" | jq -e '.decision_id' > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Decision recorded: $(echo "$response" | jq -r '.decision_id')${NC}"
    else
        echo -e "${RED}✗ Failed to record decision${NC}"
    fi
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
    response=$(_api_call GET "/api/performance/recommendations")
    if [ -z "$response" ] || [ "$response" = "[]" ]; then
        echo "No recommendations at this time"
    else
        echo "$response" | jq -r '.[] | "• \(.category): \(.recommendation)\n  Priority: \(.priority) | Potential time saved: \(.potential_time_saved)s"'
    fi
}

# Task Prediction Functions
claude-tasks() {
    echo -e "${GREEN}Predicted Next Tasks:${NC}"
    response=$(_api_call GET "/api/proactive/next-tasks")
    if echo "$response" | jq -e '.tasks' > /dev/null 2>&1; then
        echo "$response" | jq -r '.tasks[] | "[\(.probability)] \(.task)\n  Context: \(.context)"'
    else
        echo -e "${YELLOW}No task predictions available yet${NC}"
    fi
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
    response=$(_api_call GET "/api/patterns/recent")
    if [ -z "$response" ] || [ "$response" = "[]" ]; then
        echo "No patterns recognized yet"
    else
        echo "$response" | jq -r '.[] | "• \(.pattern_type): \(.description)\n  Occurrences: \(.occurrences)"'
    fi
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
    
    # Build tags array
    if [ -n "$tags" ]; then
        tags_array="[$(echo "$tags" | sed 's/,/","/g' | sed 's/^/"/;s/$/"/')]"
    else
        tags_array="[\"$type\"]"
    fi
    
    # Get current project context
    local project_name=$(basename "$(pwd)")
    
    # Use hybrid memory if enabled
    if [ "$HYBRID_MEMORY_ENABLED" = "true" ]; then
        response=$(_hybrid_store "$content" "$type" "$project_name" "$tags_array")
        if [ $? -eq 0 ]; then
            memory_id=$(echo "$response" | jq -r '.memory_id // "unknown"')
            echo -e "${GREEN}✓ Memory stored in hybrid system: $memory_id (fast local + auto-sync)${NC}"
        else
            echo -e "${RED}✗ Failed to store memory${NC}"
        fi
    else
        # Original implementation
        response=$(_api_call POST "/api/memories/" "{
            \"content\": \"$content\",
            \"tags\": $tags_array,
            \"user_id\": \"$CLAUDE_USER_ID\",
            \"session_id\": \"$CLAUDE_SESSION_ID\"
        }")
        
        if echo "$response" | jq -e '.id' > /dev/null 2>&1; then
            echo -e "${GREEN}✓ Memory stored: $(echo "$response" | jq -r '.id')${NC}"
        else
            echo -e "${RED}✗ Failed to store memory (may need authentication)${NC}"
        fi
    fi
}

claude-search() {
    local query=$1
    echo -e "${GREEN}Searching memories...${NC}"
    
    # Use hybrid memory if enabled for faster search
    if [ "$HYBRID_MEMORY_ENABLED" = "true" ]; then
        response=$(_hybrid_search "$query" 10)
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✓ Using hybrid memory (ultra-fast local search)${NC}"
            # Display results
            if [ -n "$response" ] && [ "$response" != "[]" ]; then
                echo "$response" | jq -r '.[] | "  [\(.type)] \(.content | .[0:100])..."'
            else
                echo "No results found"
            fi
        else
            echo -e "${YELLOW}⚠ Hybrid search failed, falling back to regular search${NC}"
            # Fallback to regular search
            response=$(_api_call POST "/api/v1/search/unified" "{
                \"query\": \"$query\",
                \"search_type\": \"hybrid\",
                \"limit\": 10,
                \"include_memories\": true
            }")
        fi
    else
        # Use original unified search endpoint
        response=$(_api_call POST "/api/v1/search/unified" "{
            \"query\": \"$query\",
            \"search_type\": \"hybrid\",
            \"limit\": 10,
            \"include_memories\": true
        }")
        
        if echo "$response" | jq -e '.documents' > /dev/null 2>&1; then
            # Show documents
            echo -e "${YELLOW}Documents:${NC}"
            echo "$response" | jq -r '.documents[] | "  [\(.score)] \(.content | .[0:100])..."'
            
            # Show memories if any
            if echo "$response" | jq -e '.memories' > /dev/null 2>&1; then
                echo -e "\n${YELLOW}Memories:${NC}"
                echo "$response" | jq -r '.memories[] | "  [\(.importance)] \(.content | .[0:100])..."'
            fi
        else
            echo -e "${YELLOW}Search failed or no results found${NC}"
        fi
    fi
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
# Auto-memory wrapper functions for common commands
if [ "$CLAUDE_AUTO_MEMORY" = "true" ]; then
    # Wrap git commands
    git() {
        command git "$@"
        local exit_code=$?
        if [[ "$1" == "commit" ]] && [ $exit_code -eq 0 ]; then
            local commit_msg=$(command git log -1 --pretty=%B)
            _auto_remember "Git commit: $commit_msg" "development" "[\"git\", \"commit\", \"auto-captured\"]"
        elif [[ "$1" == "checkout" ]] && [ $exit_code -eq 0 ]; then
            _auto_remember "Git checkout: $2" "development" "[\"git\", \"checkout\", \"auto-captured\"]"
        fi
        return $exit_code
    }
    
    # Wrap npm/yarn commands
    npm() {
        command npm "$@"
        local exit_code=$?
        if [[ "$1" == "install" ]] && [ $exit_code -eq 0 ]; then
            _auto_remember "npm install in $(pwd)" "development" "[\"npm\", \"install\", \"auto-captured\"]"
        elif [[ "$1" == "run" ]] && [ $exit_code -eq 0 ]; then
            _auto_remember "npm run $2 in $(pwd)" "development" "[\"npm\", \"script\", \"auto-captured\"]"
        fi
        return $exit_code
    }
    
    # Wrap docker commands
    docker() {
        command docker "$@"
        local exit_code=$?
        if [[ "$1" == "run" ]] && [ $exit_code -eq 0 ]; then
            _auto_remember "Docker run: $*" "infrastructure" "[\"docker\", \"run\", \"auto-captured\"]"
        elif [[ "$1" == "compose" ]] && [[ "$2" == "up" ]] && [ $exit_code -eq 0 ]; then
            _auto_remember "Docker compose up in $(pwd)" "infrastructure" "[\"docker\", \"compose\", \"auto-captured\"]"
        fi
        return $exit_code
    }
    
    # Wrap error-prone commands
    make() {
        command make "$@"
        local exit_code=$?
        if [ $exit_code -ne 0 ]; then
            _auto_remember "Make failed: $* (exit code: $exit_code)" "error" "[\"make\", \"error\", \"auto-captured\"]"
        else
            _auto_remember "Make succeeded: $*" "development" "[\"make\", \"success\", \"auto-captured\"]"
        fi
        return $exit_code
    }
fi

# Function to toggle auto-memory
claude-auto-memory() {
    local action=${1:-status}
    
    case "$action" in
        on|enable)
            export CLAUDE_AUTO_MEMORY=true
            echo -e "${GREEN}✓ Auto-memory enabled${NC}"
            echo -e "${YELLOW}I will now automatically remember:${NC}"
            echo "  • Git commits and checkouts"
            echo "  • npm/yarn installations and scripts"
            echo "  • Docker operations"
            echo "  • Command failures"
            echo "  • Important patterns (errors, solutions, decisions)"
            ;;
        off|disable)
            export CLAUDE_AUTO_MEMORY=false
            echo -e "${YELLOW}✗ Auto-memory disabled${NC}"
            ;;
        status)
            if [ "$CLAUDE_AUTO_MEMORY" = "true" ]; then
                echo -e "${GREEN}Auto-memory is ENABLED${NC}"
                echo -e "Capturing: commands, errors, patterns, decisions"
            else
                echo -e "${YELLOW}Auto-memory is DISABLED${NC}"
            fi
            ;;
        *)
            echo "Usage: claude-auto-memory [on|off|status]"
            ;;
    esac
}

# Auto-remember function for manual pattern detection
claude-auto-detect() {
    local content=$1
    
    # Detect patterns
    if [[ "$content" =~ [Ee]rror|[Ff]ailed|[Ee]xception ]]; then
        _auto_capture_pattern "error" "$content"
        echo -e "${YELLOW}✓ Auto-captured error pattern${NC}"
    elif [[ "$content" =~ [Ff]ixed|[Ss]olved|[Rr]esolved ]]; then
        _auto_capture_pattern "solution" "$content"
        echo -e "${GREEN}✓ Auto-captured solution pattern${NC}"
    elif [[ "$content" =~ [Tt][Oo][Dd][Oo]|FIXME|XXX ]]; then
        _auto_capture_pattern "todo" "$content"
        echo -e "${YELLOW}✓ Auto-captured TODO${NC}"
    elif [[ "$content" =~ [Dd]ecided|[Cc]hoose|[Ss]elect ]]; then
        _auto_capture_pattern "decision" "$content"
        echo -e "${GREEN}✓ Auto-captured decision${NC}"
    else
        _auto_remember "$content" "general" "[\"manual\", \"auto-detect\"]"
        echo -e "${GREEN}✓ Auto-captured general information${NC}"
    fi
}

# Show auto-captured memories
claude-auto-history() {
    echo -e "${GREEN}Showing auto-captured memories...${NC}"
    
    if [ "$HYBRID_MEMORY_ENABLED" = "true" ]; then
        response=$(_hybrid_search "auto-captured" 20)
        if [ -n "$response" ] && [ "$response" \!= "[]" ]; then
            echo "$response" | jq -r '.[] | "[\(.type)] \(.created_at | .[0:19]) - \(.content | .[0:80])..."'
        else
            echo "No auto-captured memories found"
        fi
    else
        echo -e "${YELLOW}Hybrid memory not enabled${NC}"
    fi
}


# ============================================
# AGENT SYSTEM FUNCTIONS
# ============================================

# Initialize agent system
_init_agent_system() {
    if [ -f "$CLAUDE_AGENT_PATH/agent_profiles.json" ]; then
        echo -e "${GREEN}✓ Agent system initialized${NC}"
        export CLAUDE_AGENTS_AVAILABLE="true"
    else
        echo -e "${YELLOW}⚠ Agent profiles not found at $CLAUDE_AGENT_PATH${NC}"
        export CLAUDE_AGENTS_AVAILABLE="false"
    fi
}

# Analyze task and select agents
claude-agent-select() {
    local task="$*"
    if [ -z "$task" ]; then
        echo -e "${RED}Usage: claude-agent-select <task description>${NC}"
        return 1
    fi
    
    if [ "$CLAUDE_AGENTS_AVAILABLE" \!= "true" ]; then
        echo -e "${YELLOW}Agent system not available${NC}"
        return 1
    fi
    
    # Run agent selector
    local result=$(python3 /opt/projects/memory-system/claude_agent_selector.py "$task" 2>/dev/null)
    
    if [ -z "$result" ] || [ "$result" = '{"task": "'"$task"'", "agents": []}' ]; then
        echo -e "${YELLOW}No specific agents matched for this task${NC}"
        return 0
    fi
    
    # Parse and display results
    echo -e "${BLUE}🤖 Agent Analysis for:${NC} $task"
    echo -e "${BLUE}Selected Agents:${NC}"
    
    # Extract agent info and store in environment
    export CLAUDE_ACTIVE_AGENTS=$(echo "$result" | jq -r '.agents[].id' | tr '\n' ' ')
    
    echo "$result" | jq -r '.agents[] | "  • \(.name) (Score: \(.score))\n    Expertise: \(.path)"'
    
    # Store selection in memory if enabled
    if [ "$HYBRID_MEMORY_ENABLED" = "true" ]; then
        _hybrid_store "Agent selection: $task -> $CLAUDE_ACTIVE_AGENTS" "agent_selection" "" '["auto-agent"]' >/dev/null
    fi
}

# Manually activate specific agent(s)
claude-agent-activate() {
    local agents="$*"
    if [ -z "$agents" ]; then
        echo -e "${RED}Usage: claude-agent-activate <agent-id> [agent-id2...]${NC}"
        echo "Available agents:"
        claude-agent-list
        return 1
    fi
    
    export CLAUDE_ACTIVE_AGENTS="$agents"
    echo -e "${GREEN}✓ Activated agents: $CLAUDE_ACTIVE_AGENTS${NC}"
    
    # Show agent details
    for agent in $agents; do
        if [ -f "$CLAUDE_AGENT_PATH/agent_profiles.json" ]; then
            local info=$(jq -r ".agents[\"$agent\"] | \"  • \\(.name): \\(.expertise | join(\", \"))\"" "$CLAUDE_AGENT_PATH/agent_profiles.json" 2>/dev/null)
            if [ -n "$info" ] && [ "$info" \!= "  • null" ]; then
                echo "$info"
            fi
        fi
    done
}

# Deactivate all agents
claude-agent-clear() {
    export CLAUDE_ACTIVE_AGENTS=""
    echo -e "${GREEN}✓ All agents deactivated${NC}"
}

# Show currently active agents
claude-agent-status() {
    if [ -z "$CLAUDE_ACTIVE_AGENTS" ]; then
        echo -e "${YELLOW}No agents currently active${NC}"
        if [ "$CLAUDE_AGENT_AUTO" = "true" ]; then
            echo -e "${BLUE}ℹ Auto-selection is enabled${NC}"
        fi
    else
        echo -e "${BLUE}Active agents:${NC} $CLAUDE_ACTIVE_AGENTS"
        
        # Show details for each active agent
        for agent in $CLAUDE_ACTIVE_AGENTS; do
            if [ -f "$CLAUDE_AGENT_PATH/agent_profiles.json" ]; then
                jq -r ".agents[\"$agent\"] | \"  • \\(.name): \\(.expertise | join(\", \"))\"" "$CLAUDE_AGENT_PATH/agent_profiles.json" 2>/dev/null
            fi
        done
    fi
}

# List all available agents
claude-agent-list() {
    if [ "$CLAUDE_AGENTS_AVAILABLE" \!= "true" ]; then
        echo -e "${YELLOW}Agent system not available${NC}"
        return 1
    fi
    
    echo -e "${BLUE}Available Agents:${NC}"
    
    if [ -f "$CLAUDE_AGENT_PATH/agent_profiles.json" ]; then
        jq -r '.agents | to_entries[] | "  • \(.key): \(.value.name)\n    Expertise: \(.value.expertise | join(", "))"' "$CLAUDE_AGENT_PATH/agent_profiles.json"
    fi
}

# View agent details
claude-agent-info() {
    local agent_id="$1"
    if [ -z "$agent_id" ]; then
        echo -e "${RED}Usage: claude-agent-info <agent-id>${NC}"
        return 1
    fi
    
    if [ -f "$CLAUDE_AGENT_PATH/agent_profiles.json" ]; then
        local agent_info=$(jq -r ".agents[\"$agent_id\"]" "$CLAUDE_AGENT_PATH/agent_profiles.json" 2>/dev/null)
        
        if [ "$agent_info" = "null" ] || [ -z "$agent_info" ]; then
            echo -e "${RED}Agent '$agent_id' not found${NC}"
            return 1
        fi
        
        # Display agent details
        echo -e "${BLUE}Agent Profile: $agent_id${NC}"
        echo "$agent_info" | jq -r '"Name: \(.name)\nPath: \(.path)\nPriority: \(.priority)\n\nExpertise:\n\(.expertise | map("  • " + .) | join("\n"))\n\nKeywords:\n\(.keywords | map("  • " + .) | join("\n"))"'
        
        # Show the actual agent markdown if it exists
        local agent_path=$(echo "$agent_info" | jq -r '.path')
        if [ -f "$CLAUDE_AGENT_PATH/$agent_path" ]; then
            echo -e "\n${BLUE}Full Documentation:${NC}"
            echo "View at: $CLAUDE_AGENT_PATH/$agent_path"
        fi
    fi
}

# Enable/disable automatic agent selection
claude-agent-auto() {
    local setting="${1:-toggle}"
    
    case "$setting" in
        on|enable|true)
            export CLAUDE_AGENT_AUTO="true"
            echo -e "${GREEN}✓ Automatic agent selection enabled${NC}"
            ;;
        off|disable|false)
            export CLAUDE_AGENT_AUTO="false"
            echo -e "${YELLOW}✓ Automatic agent selection disabled${NC}"
            ;;
        toggle)
            if [ "$CLAUDE_AGENT_AUTO" = "true" ]; then
                claude-agent-auto off
            else
                claude-agent-auto on
            fi
            ;;
        status)
            if [ "$CLAUDE_AGENT_AUTO" = "true" ]; then
                echo -e "${GREEN}Automatic agent selection is enabled${NC}"
            else
                echo -e "${YELLOW}Automatic agent selection is disabled${NC}"
            fi
            ;;
        *)
            echo -e "${RED}Usage: claude-agent-auto [on|off|toggle|status]${NC}"
            return 1
            ;;
    esac
}

# Test agent selection with various tasks
claude-agent-test() {
    echo -e "${BLUE}Testing agent selection system...${NC}\n"
    
    local test_tasks=(
        "Build a REST API with authentication"
        "Optimize database query performance"
        "Create a mobile app for iOS and Android"
        "Design a microservices architecture"
        "Implement blockchain smart contracts"
        "Set up CI/CD pipeline with Docker"
        "Analyze security vulnerabilities"
        "Create data visualization dashboard"
        "Build a game with Unity"
        "Develop machine learning model"
    )
    
    for task in "${test_tasks[@]}"; do
        echo -e "${MAGENTA}Task:${NC} $task"
        claude-agent-select "$task"
        echo ""
    done
}

# Initialize agent system when helpers are loaded
_init_agent_system

# Update claude-init to include agent initialization
claude-init() {
    echo -e "${GREEN}Initializing Claude session...${NC}"
    
    # Generate session ID
    export CLAUDE_SESSION_ID="claude-$(date +%Y%m%d-%H%M%S)"
    
    # Initialize hybrid memory
    echo -e "${GREEN}✓ Hybrid memory system enabled (fast local + distributed sync)${NC}"
    
    # Initialize auto-memory if enabled
    if [ "$CLAUDE_AUTO_MEMORY" = "true" ]; then
        echo -e "${GREEN}✓ Auto-memory enabled (automatically captures important information)${NC}"
    fi
    
    # Initialize agent system
    if [ "$CLAUDE_AGENTS_AVAILABLE" = "true" ]; then
        echo -e "${GREEN}✓ Agent system available (${CLAUDE_AGENT_AUTO:-true} auto-selection)${NC}"
    fi
    
    # Create session memory
    if [ "$HYBRID_MEMORY_ENABLED" = "true" ]; then
        _hybrid_store "Session initialized: $CLAUDE_SESSION_ID" "session_event" "" '["session-start"]' >/dev/null
    fi
    
    # Restore context if available
    local context=$(_api_call GET "/api/memory/context/quick/$CLAUDE_USER_ID" | jq -r '.[0].content' 2>/dev/null)
    if [ -n "$context" ] && [ "$context" \!= "null" ]; then
        echo -e "${GREEN}✓ Previous context restored${NC}"
    fi
    
    echo -e "${GREEN}Session ID: $CLAUDE_SESSION_ID${NC}"
}

# Add agent help to main help function
claude-help() {
    echo -e "${BLUE}Claude Code Integration - KnowledgeHub Commands${NC}"
    echo ""
    echo -e "${GREEN}Session Management:${NC}"
    echo "  claude-init              - Initialize a new Claude session"
    echo "  claude-handoff [message] - Create a session handoff for continuity"
    echo ""
    echo -e "${GREEN}Memory Commands:${NC}"
    echo "  claude-memory <content>  - Store a memory/note"
    echo "  claude-search <query>    - Search memories"
    echo "  claude-context           - Get current context"
    echo ""
    echo -e "${GREEN}Project Context:${NC}"
    echo "  claude-project-init <name> - Initialize project context"
    echo "  claude-project-switch <name> - Switch project context"
    echo ""
    echo -e "${GREEN}Error Learning:${NC}"
    echo "  claude-error <type> <message> [solution] [resolved] - Track an error"
    echo "  claude-find-error <error> - Find similar errors and solutions"
    echo ""
    echo -e "${GREEN}Agent System:${NC}"
    echo "  claude-agent-select <task> - Auto-select agents for a task"
    echo "  claude-agent-activate <id> - Manually activate agent(s)"
    echo "  claude-agent-clear        - Deactivate all agents"
    echo "  claude-agent-status       - Show active agents"
    echo "  claude-agent-list         - List all available agents"
    echo "  claude-agent-info <id>    - View agent details"
    echo "  claude-agent-auto [on/off] - Toggle auto-selection"
    echo "  claude-agent-test         - Test agent selection"
    echo ""
    echo -e "${GREEN}Other Commands:${NC}"
    echo "  claude-decide <decision> <reasoning> - Track a decision"
    echo "  claude-stats             - Show memory statistics"
    echo "  claude-auto-history      - Show auto-captured memories"
    echo ""
    echo -e "${YELLOW}Environment Variables:${NC}"
    echo "  CLAUDE_USER_ID          - User identifier (default: claude)"
    echo "  HYBRID_MEMORY_ENABLED   - Use hybrid memory (default: true)"
    echo "  CLAUDE_AUTO_MEMORY      - Auto-capture memories (default: true)"
    echo "  CLAUDE_AGENT_AUTO       - Auto-select agents (default: true)"
}
