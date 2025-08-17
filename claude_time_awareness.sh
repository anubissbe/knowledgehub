#!/bin/bash
# Claude automatic time awareness helper
# This script is sourced automatically to maintain time awareness

# Get current time context
get_time_context() {
    local brussels_time=$(python /opt/projects/mcp-servers-time/time_service.py "Europe/Brussels" 2>/dev/null | cut -d' ' -f1)
    local day_of_week=$(date +%A)
    local date=$(date +%Y-%m-%d)
    local hour=$(date +%H)
    
    # Determine part of day
    local part_of_day=""
    if [ $hour -ge 6 ] && [ $hour -lt 12 ]; then
        part_of_day="morning"
    elif [ $hour -ge 12 ] && [ $hour -lt 17 ]; then
        part_of_day="afternoon"
    elif [ $hour -ge 17 ] && [ $hour -lt 21 ]; then
        part_of_day="evening"
    else
        part_of_day="night"
    fi
    
    # Check if it's working hours (9-17 Brussels time)
    local working_hours="outside working hours"
    if [ $hour -ge 9 ] && [ $hour -lt 17 ] && [ "$(date +%u)" -le 5 ]; then
        working_hours="during working hours"
    fi
    
    echo "Time: $brussels_time Brussels | Day: $day_of_week | Part: $part_of_day | Status: $working_hours"
}

# Store time awareness in environment
export CLAUDE_TIME_AWARENESS=$(get_time_context)
export CLAUDE_CURRENT_TIME=$(python /opt/projects/mcp-servers-time/time_service.py "Europe/Brussels" 2>/dev/null)
export CLAUDE_TIMEZONE="Europe/Brussels"

# Function to refresh time awareness
claude-refresh-time() {
    export CLAUDE_TIME_AWARENESS=$(get_time_context)
    export CLAUDE_CURRENT_TIME=$(python /opt/projects/mcp-servers-time/time_service.py "Europe/Brussels" 2>/dev/null)
    echo "⏰ Time awareness updated: $CLAUDE_CURRENT_TIME"
}

# Auto-refresh time awareness every time a command is run (for long sessions)
PROMPT_COMMAND='export CLAUDE_CURRENT_TIME=$(python /opt/projects/mcp-servers-time/time_service.py "Europe/Brussels" 2>/dev/null 2>&1)'