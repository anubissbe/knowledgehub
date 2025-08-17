#!/bin/bash

# IT-DEPT-200 Agent Integration for Claude Code
# Provides /agent command functionality

AGENT_LOADER="/opt/it-dept-200/runtime/agent_loader.py"
AGENT_CONFIG="/opt/it-dept-200/runtime/config.yaml"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to show agent system status
claude-agents() {
    if [ ! -f "$AGENT_CONFIG" ]; then
        echo -e "${RED}Agent system not found. Run setup first.${NC}"
        return 1
    fi
    
    echo -e "${GREEN}IT-DEPT-200 Agent System${NC}"
    python3 "$AGENT_LOADER" stats
}

# Function to list agents
claude-agent-list() {
    python3 "$AGENT_LOADER" list
}

# Function to search agents
claude-agent-search() {
    local query="$*"
    if [ -z "$query" ]; then
        echo -e "${RED}Usage: claude-agent-search <search terms>${NC}"
        return 1
    fi
    echo -e "${BLUE}Searching for: $query${NC}"
    python3 "$AGENT_LOADER" search "$query"
}

# Function to select best agent for a task
claude-agent-select() {
    local task="$*"
    if [ -z "$task" ]; then
        echo -e "${RED}Usage: claude-agent-select <task description>${NC}"
        return 1
    fi
    echo -e "${BLUE}Selecting agent for: $task${NC}"
    python3 "$AGENT_LOADER" select "$task"
}

# Main /agent command handler
agent() {
    local command="$1"
    shift
    
    case "$command" in
        "")
            claude-agents
            ;;
        list)
            claude-agent-list
            ;;
        search)
            claude-agent-search "$@"
            ;;
        select)
            claude-agent-select "$@"
            ;;
        help)
            echo -e "${GREEN}IT-DEPT-200 Agent System Commands:${NC}"
            echo "  /agent              - Show agent system status"
            echo "  /agent list         - List available agents"
            echo "  /agent search <q>   - Search agents by keyword"
            echo "  /agent select <task> - Select best agent for task"
            echo ""
            echo -e "${BLUE}200 specialized IT agents across 25 domains${NC}"
            ;;
        *)
            # Treat as task selection
            claude-agent-select "$command" "$@"
            ;;
    esac
}

# Export the functions
export -f claude-agents
export -f claude-agent-list
export -f claude-agent-search
export -f claude-agent-select
export -f agent

# Function that mimics /agent command
/agent() {
    agent "$@"
}

echo -e "${GREEN}✓ IT-DEPT-200 Agent System loaded (200 agents)${NC}"
echo -e "${BLUE}Type '/agent help' for commands${NC}"