#!/bin/bash
# Claude Code Helper Functions
# Source this file in your .bashrc or .zshrc: source /opt/projects/knowledgehub/claude_code_helpers.sh

# Initialize Claude Code session (call at start of conversation)
claude-init() {
    /opt/projects/knowledgehub/claude_code_init.py
}

# Create handoff note (call at end of conversation)
claude-handoff() {
    local summary="$1"
    local tasks="$2"
    local issues="$3"
    
    if [ -z "$summary" ]; then
        echo "Usage: claude-handoff \"summary\" [\"task1,task2\"] [\"issue1,issue2\"]"
        return 1
    fi
    
    /opt/projects/knowledgehub/claude_code_init.py handoff "$summary" "$tasks" "$issues"
}

# Record an error with solution
claude-error() {
    local error_type="$1"
    local error_msg="$2"
    local solution="$3"
    local worked="${4:-false}"
    
    if [ -z "$error_type" ] || [ -z "$error_msg" ]; then
        echo "Usage: claude-error \"ErrorType\" \"error message\" [\"solution\"] [true/false]"
        return 1
    fi
    
    curl -X POST "http://localhost:3000/api/claude-auto/error/record" \
        -G \
        --data-urlencode "error_type=$error_type" \
        --data-urlencode "error_message=$error_msg" \
        --data-urlencode "solution=$solution" \
        --data-urlencode "worked=$worked" \
        2>/dev/null | jq -r '.recorded // "Failed to record error"'
}

# Get current session info
claude-session() {
    curl -s "http://localhost:3000/api/claude-auto/session/current" | jq '.'
}

# Get memory stats
claude-stats() {
    curl -s "http://localhost:3000/api/claude-auto/memory/stats" | jq '.'
}

# Quick memory add (using local memory system)
claude-remember() {
    local content="$1"
    local type="${2:-context}"
    local priority="${3:-high}"
    
    if [ -z "$content" ]; then
        echo "Usage: claude-remember \"content\" [type] [priority]"
        return 1
    fi
    
    cd /opt/projects/memory-system && ./memory-cli add "$content" -t "$type" -p "$priority"
}

# Search memories
claude-search() {
    local query="$1"
    
    if [ -z "$query" ]; then
        echo "Usage: claude-search \"query\""
        return 1
    fi
    
    cd /opt/projects/memory-system && ./memory-cli search "$query"
}

# Create checkpoint
claude-checkpoint() {
    local description="$1"
    
    if [ -z "$description" ]; then
        echo "Usage: claude-checkpoint \"description\""
        return 1
    fi
    
    cd /opt/projects/memory-system && ./memory-cli checkpoint -d "$description"
}

# Show recent context
claude-context() {
    local count="${1:-10}"
    cd /opt/projects/memory-system && ./memory-cli context -n "$count"
}

# Predict next tasks
claude-tasks() {
    curl -s "http://localhost:3000/api/claude-auto/tasks/predict" | jq -r '.[] | "[\(.confidence)] \(.task) (\(.type))"'
}

# Find similar errors
claude-find-error() {
    local error_type="$1"
    local error_msg="$2"
    
    if [ -z "$error_type" ] || [ -z "$error_msg" ]; then
        echo "Usage: claude-find-error \"ErrorType\" \"error message\""
        return 1
    fi
    
    curl -X GET "http://localhost:3000/api/claude-auto/error/similar" \
        -G \
        --data-urlencode "error_type=$error_type" \
        --data-urlencode "error_message=$error_msg" \
        2>/dev/null | jq -r '.[] | "[\(.worked // false)] \(.solution // "No solution")"'
}

# Auto-init on new shell (optional - uncomment to enable)
# claude-init 2>/dev/null || true

# Check before risky action
claude-check() {
    local action="$1"
    
    if [ -z "$action" ]; then
        echo "Usage: claude-check \"action to perform\""
        return 1
    fi
    
    response=$(curl -s -X POST "http://localhost:3000/api/mistake-learning/check-action" \
        -G \
        --data-urlencode "action=$action" \
        -H "Content-Type: application/json" \
        -d '{}' \
        2>/dev/null)
    
    should_proceed=$(echo "$response" | jq -r '.should_proceed')
    
    if [ "$should_proceed" = "false" ]; then
        echo "⚠️  WARNING: $(echo "$response" | jq -r '.message')"
        return 1
    else
        echo "✅ $(echo "$response" | jq -r '.message')"
        return 0
    fi
}

# Get lessons learned
claude-lessons() {
    local category="$1"
    local days="${2:-30}"
    
    url="http://localhost:3000/api/mistake-learning/lessons?days=$days"
    if [ -n "$category" ]; then
        url="${url}&category=$category"
    fi
    
    curl -s "$url" | jq -r '.[] | "[\(.repetitions)x] \(.lesson.summary // .lesson)"' | head -10
}

# Get mistake report
claude-report() {
    local days="${1:-7}"
    
    report=$(curl -s "http://localhost:3000/api/mistake-learning/report?days=$days")
    
    echo "📊 Mistake Report (last $days days)"
    echo "================================"
    echo "Total mistakes: $(echo "$report" | jq -r '.total_mistakes')"
    echo "Repeated: $(echo "$report" | jq -r '.repeated_mistakes') ($(echo "$report" | jq -r '.repetition_rate * 100 | floor')%)"
    echo "Solved: $(echo "$report" | jq -r '.solved_mistakes') ($(echo "$report" | jq -r '.solution_rate * 100 | floor')%)"
    echo ""
    echo "Categories:"
    echo "$report" | jq -r '.categories | to_entries[] | "  \(.key): \(.value)"'
    echo ""
    echo "Top Lessons:"
    echo "$report" | jq -r '.top_lessons[:3][] | "  - \(.lesson.summary // "N/A")"'
}

# Get proactive assistance
claude-assist() {
    local session_id=$(jq -r '.session_id' ~/.claude_session.json 2>/dev/null)
    
    if [ -z "$session_id" ]; then
        echo "No active session. Run claude-init first."
        return 1
    fi
    
    curl -s "http://localhost:3000/api/proactive/brief?session_id=$session_id" | jq -r '.brief'
}

# Get incomplete tasks
claude-todos() {
    local session_id=$(jq -r '.session_id' ~/.claude_session.json 2>/dev/null)
    
    if [ -z "$session_id" ]; then
        echo "No active session. Run claude-init first."
        return 1
    fi
    
    todos=$(curl -s "http://localhost:3000/api/proactive/incomplete-tasks?session_id=$session_id")
    
    if [ -n "$todos" ] && [ "$todos" != "[]" ]; then
        echo "📝 Incomplete Tasks:"
        echo "$todos" | jq -r '.[] | "[\(.priority)] \(.task) (\(.age_hours | floor)h ago)"'
    else
        echo "✅ No incomplete tasks!"
    fi
}

# Get reminders
claude-reminders() {
    local session_id=$(jq -r '.session_id' ~/.claude_session.json 2>/dev/null)
    
    if [ -z "$session_id" ]; then
        echo "No active session."
        return 1
    fi
    
    reminders=$(curl -s "http://localhost:3000/api/proactive/reminders?session_id=$session_id")
    
    if [ -n "$reminders" ] && [ "$reminders" != "[]" ]; then
        echo "🔔 Reminders:"
        echo "$reminders" | jq -r '.[] | "[\(.priority)] \(.message)\n   Action: \(.action)"'
    else
        echo "No reminders at this time."
    fi
}

# Record a decision with reasoning and alternatives
claude-decide() {
    local title="$1"
    local chosen="$2"
    local reasoning="$3"
    local confidence="${4:-0.7}"
    
    if [ -z "$title" ] || [ -z "$chosen" ] || [ -z "$reasoning" ]; then
        echo "Usage: claude-decide \"decision title\" \"chosen solution\" \"reasoning\" [confidence 0-1]"
        echo "Example: claude-decide \"Use React for UI\" \"React with TypeScript\" \"Better type safety and community support\" 0.8"
        return 1
    fi
    
    # Get session info
    local session_id=$(jq -r '.session_id' ~/.claude_session.json 2>/dev/null)
    local project_id=$(pwd | xargs basename)
    
    curl -X POST "http://localhost:3000/api/decisions/record" \
        -G \
        --data-urlencode "decision_title=$title" \
        --data-urlencode "chosen_solution=$chosen" \
        --data-urlencode "reasoning=$reasoning" \
        --data-urlencode "confidence=$confidence" \
        --data-urlencode "project_id=$project_id" \
        --data-urlencode "session_id=$session_id" \
        -H "Content-Type: application/json" \
        -d '{"alternatives": [{"solution": "Alternative not specified", "reason_rejected": "Only one option considered"}], "context": {}}' \
        2>/dev/null | jq -r 'if .decision_id then "✅ Decision recorded: \(.decision_id)" else "❌ Failed to record decision" end'
}

# Explain a past decision
claude-explain() {
    local decision_id="$1"
    
    if [ -z "$decision_id" ]; then
        echo "Usage: claude-explain <decision_id>"
        echo "Get decision_id from claude-search-decisions"
        return 1
    fi
    
    explanation=$(curl -s "http://localhost:3000/api/decisions/explain/$decision_id")
    
    if echo "$explanation" | jq -e '.decision' > /dev/null; then
        echo "📋 Decision: $(echo "$explanation" | jq -r '.decision')"
        echo "📅 Made on: $(echo "$explanation" | jq -r '.made_on' | cut -c1-19)"
        echo "🎯 Category: $(echo "$explanation" | jq -r '.category')"
        echo "✅ Chosen: $(echo "$explanation" | jq -r '.what_was_chosen')"
        echo "🤔 Why: $(echo "$explanation" | jq -r '.why')"
        echo "📊 Confidence: $(echo "$explanation" | jq -r '.confidence_level')"
        echo "📈 Outcome: $(echo "$explanation" | jq -r '.outcome')"
        
        echo ""
        echo "🔄 Alternatives considered:"
        echo "$explanation" | jq -r '.alternatives_considered[] | "  • \(.option): \(.rejected_because)"'
    else
        echo "❌ Decision not found or error occurred"
    fi
}

# Search decisions
claude-search-decisions() {
    local query="$1"
    local category="$2"
    local limit="${3:-5}"
    
    if [ -z "$query" ]; then
        echo "Usage: claude-search-decisions \"search terms\" [category] [limit]"
        return 1
    fi
    
    url="http://localhost:3000/api/decisions/search?query=$query&limit=$limit"
    if [ -n "$category" ]; then
        url="${url}&category=$category"
    fi
    
    results=$(curl -s "$url")
    
    if [ "$results" != "[]" ]; then
        echo "🔍 Decision Search Results:"
        echo "$results" | jq -r '.[] | "  [\(.decision_id[0:8])] \(.title) - \(.chosen) (\(.confidence * 100 | floor)% confident)"'
    else
        echo "No decisions found matching: $query"
    fi
}

# Get decision suggestions for a problem
claude-suggest-decision() {
    local problem="$1"
    
    if [ -z "$problem" ]; then
        echo "Usage: claude-suggest-decision \"problem description\""
        return 1
    fi
    
    suggestion=$(curl -s -X GET "http://localhost:3000/api/decisions/suggest" \
        -G \
        --data-urlencode "problem=$problem" \
        -H "Content-Type: application/json" \
        -d '{}')
    
    if echo "$suggestion" | jq -e '.suggested_approach' > /dev/null; then
        echo "💡 Suggestion for: $problem"
        echo "📋 Category: $(echo "$suggestion" | jq -r '.category')"
        echo "✅ Suggested approach: $(echo "$suggestion" | jq -r '.suggested_approach // "No specific suggestion"')"
        echo "📊 Confidence: $(echo "$suggestion" | jq -r '.confidence * 100 | floor')%"
        
        if echo "$suggestion" | jq -e '.based_on[0]' > /dev/null; then
            echo ""
            echo "📚 Based on:"
            echo "$suggestion" | jq -r '.based_on[] | "  • \(.decision) (\(.confidence * 100 | floor)% confident, outcome: \(.outcome))"'
        fi
        
        if echo "$suggestion" | jq -e '.reasoning_patterns[0]' > /dev/null; then
            echo ""
            echo "🧠 Reasoning patterns:"
            echo "$suggestion" | jq -r '.reasoning_patterns[] | "  • \(.)"'
        fi
    else
        echo "❌ Could not generate suggestion"
    fi
}

# Update decision outcome
claude-update-decision() {
    local decision_id="$1"
    local outcome="$2"
    
    if [ -z "$decision_id" ] || [ -z "$outcome" ]; then
        echo "Usage: claude-update-decision <decision_id> <outcome>"
        echo "Outcome: successful | failed | mixed"
        return 1
    fi
    
    curl -X POST "http://localhost:3000/api/decisions/update-outcome" \
        -G \
        --data-urlencode "decision_id=$decision_id" \
        --data-urlencode "outcome=$outcome" \
        -H "Content-Type: application/json" \
        -d '{"impact": {"description": "Updated via CLI"}}' \
        2>/dev/null | jq -r 'if .updated then "✅ Decision outcome updated" else "❌ Failed to update: \(.error // "Unknown error")" end'
}

# Get confidence report
claude-confidence-report() {
    local category="$1"
    
    url="http://localhost:3000/api/decisions/confidence-report"
    if [ -n "$category" ]; then
        url="${url}?category=$category"
    fi
    
    report=$(curl -s "$url")
    
    echo "📊 Decision Confidence Report"
    echo "=============================="
    echo "Overall accuracy: $(echo "$report" | jq -r '.overall_accuracy * 100 | floor')%"
    echo ""
    echo "By category:"
    echo "$report" | jq -r '.categories | to_entries[] | "  \(.key): \(.value.accuracy * 100 | floor)% (\(.value.total_decisions) decisions)"'
    
    if echo "$report" | jq -e '.recommendations[0]' > /dev/null; then
        echo ""
        echo "Recommendations:"
        echo "$report" | jq -r '.recommendations[] | "  • \(.)"'
    fi
}

# Track code change evolution
claude-track-change() {
    local file_path="$1"
    local description="$2"
    local reason="$3"
    local before_file="$4"
    local after_file="$5"
    
    if [ -z "$file_path" ] || [ -z "$description" ] || [ -z "$reason" ]; then
        echo "Usage: claude-track-change \"file/path\" \"description\" \"reason\" [before_file] [after_file]"
        echo "Example: claude-track-change \"src/main.py\" \"Add error handling\" \"Improve robustness\" before.py after.py"
        return 1
    fi
    
    # Get session info
    local session_id=$(jq -r '.session_id' ~/.claude_session.json 2>/dev/null)
    local project_id=$(pwd | xargs basename)
    
    # If before/after files provided, read them
    local before_code=""
    local after_code=""
    
    if [ -n "$before_file" ] && [ -f "$before_file" ]; then
        before_code=$(cat "$before_file")
    fi
    
    if [ -n "$after_file" ] && [ -f "$after_file" ]; then
        after_code=$(cat "$after_file")
    fi
    
    # If no files provided, try to get from git
    if [ -z "$before_code" ] && [ -z "$after_code" ] && command -v git >/dev/null 2>&1; then
        # Get current content
        if [ -f "$file_path" ]; then
            after_code=$(cat "$file_path")
        fi
        
        # Get previous version from git
        before_code=$(git show HEAD~1:"$file_path" 2>/dev/null || echo "")
    fi
    
    if [ -z "$before_code" ] || [ -z "$after_code" ]; then
        echo "❌ Need before and after code. Provide files or ensure git history exists."
        return 1
    fi
    
    curl -X POST "http://localhost:3000/api/code-evolution/track-change" \
        -G \
        --data-urlencode "file_path=$file_path" \
        --data-urlencode "change_description=$description" \
        --data-urlencode "change_reason=$reason" \
        --data-urlencode "project_id=$project_id" \
        --data-urlencode "session_id=$session_id" \
        -H "Content-Type: application/json" \
        -d "{\"before_code\": $(echo "$before_code" | jq -R -s .), \"after_code\": $(echo "$after_code" | jq -R -s .)}" \
        2>/dev/null | jq -r 'if .change_id then "✅ Change tracked: \(.change_id) - \(.patterns_detected) patterns detected, \(.quality_improvement * 100 | floor)% quality improvement" else "❌ Failed to track change" end'
}

# Compare code versions
claude-compare-change() {
    local change_id="$1"
    
    if [ -z "$change_id" ]; then
        echo "Usage: claude-compare-change <change_id>"
        echo "Get change_id from claude-evolution-history"
        return 1
    fi
    
    comparison=$(curl -s "http://localhost:3000/api/code-evolution/compare/$change_id")
    
    if echo "$comparison" | jq -e '.change_id' > /dev/null; then
        echo "📁 File: $(echo "$comparison" | jq -r '.file_path')"
        echo "📋 Change: $(echo "$comparison" | jq -r '.description')"
        echo "🎯 Type: $(echo "$comparison" | jq -r '.analysis.change_type')"
        echo "📏 Scope: $(echo "$comparison" | jq -r '.analysis.change_scope')"
        echo "⚠️ Risk: $(echo "$comparison" | jq -r '.analysis.risk_level')"
        
        echo ""
        echo "🔧 Patterns detected:"
        echo "$comparison" | jq -r '.patterns[] | "  • \(.pattern): \(.description) (\(.confidence * 100 | floor)% confidence)"'
        
        echo ""
        echo "📊 Quality metrics:"
        echo "  Before: $(echo "$comparison" | jq -r '.quality_before.complexity_estimate') complexity, $(echo "$comparison" | jq -r '.quality_before.total_lines') lines"
        echo "  After:  $(echo "$comparison" | jq -r '.quality_after.complexity_estimate') complexity, $(echo "$comparison" | jq -r '.quality_after.total_lines') lines"
        echo "  Improvement: $(echo "$comparison" | jq -r '.quality_improvement.overall_improvement * 100 | floor')%"
    else
        echo "❌ Change not found or error occurred"
    fi
}

# Get evolution history
claude-evolution-history() {
    local file_path="$1"
    local change_type="$2"
    local limit="${3:-10}"
    
    url="http://localhost:3000/api/code-evolution/history?limit=$limit"
    if [ -n "$file_path" ]; then
        url="${url}&file_path=$file_path"
    fi
    if [ -n "$change_type" ]; then
        url="${url}&change_type=$change_type"
    fi
    
    history=$(curl -s "$url")
    
    if [ "$history" != "[]" ]; then
        echo "📚 Code Evolution History:"
        echo "$history" | jq -r '.[] | "  [\(.change_id[0:8])] \(.description) - \(.type) (\(.quality_improvement * 100 | floor)% improvement)"'
    else
        echo "No evolution history found"
    fi
}

# Get refactoring suggestions
claude-suggest-refactoring() {
    local file_path="$1"
    local code_file="$2"
    
    if [ -z "$file_path" ]; then
        echo "Usage: claude-suggest-refactoring \"file/path\" [code_file]"
        return 1
    fi
    
    local code=""
    if [ -n "$code_file" ] && [ -f "$code_file" ]; then
        code=$(cat "$code_file")
    elif [ -f "$file_path" ]; then
        code=$(cat "$file_path")
    else
        echo "❌ Could not read code from $file_path"
        return 1
    fi
    
    local project_id=$(pwd | xargs basename)
    
    suggestion=$(curl -s -X POST "http://localhost:3000/api/code-evolution/suggest-refactoring" \
        -G \
        --data-urlencode "file_path=$file_path" \
        --data-urlencode "project_id=$project_id" \
        -H "Content-Type: application/json" \
        -d "{\"code\": $(echo "$code" | jq -R -s .)}")
    
    if echo "$suggestion" | jq -e '.improvement_opportunities[0]' > /dev/null; then
        echo "💡 Refactoring Suggestions for: $file_path"
        echo ""
        echo "📊 Current metrics:"
        echo "  Lines: $(echo "$suggestion" | jq -r '.current_metrics.total_lines')"
        echo "  Complexity: $(echo "$suggestion" | jq -r '.current_metrics.complexity_estimate')"
        echo "  Functions: $(echo "$suggestion" | jq -r '.current_metrics.function_count')"
        
        echo ""
        echo "🔧 Improvement opportunities:"
        echo "$suggestion" | jq -r '.improvement_opportunities[] | "  • \(.description) [\(.type)]"'
        
        if echo "$suggestion" | jq -e '.based_on_history[0]' > /dev/null; then
            echo ""
            echo "📚 Based on successful changes:"
            echo "$suggestion" | jq -r '.based_on_history[] | "  • \(.description) (\(.quality_improvement * 100 | floor)% improvement)"'
        fi
    else
        echo "❌ Could not generate suggestions"
    fi
}

# Update change impact
claude-update-impact() {
    local change_id="$1"
    local success_rating="$2"
    local impact_notes="$3"
    
    if [ -z "$change_id" ] || [ -z "$success_rating" ] || [ -z "$impact_notes" ]; then
        echo "Usage: claude-update-impact <change_id> <success_rating> \"impact_notes\""
        echo "Success rating: 0.0-1.0 (0=failed, 1=perfect)"
        return 1
    fi
    
    curl -X POST "http://localhost:3000/api/code-evolution/update-impact" \
        -G \
        --data-urlencode "change_id=$change_id" \
        --data-urlencode "success_rating=$success_rating" \
        -H "Content-Type: application/json" \
        -d "{\"impact_notes\": \"$impact_notes\"}" \
        2>/dev/null | jq -r 'if .updated then "✅ Impact updated for change \(.change_id)" else "❌ Failed to update: \(.error // "Unknown error")" end'
}

# Get evolution trends
claude-evolution-trends() {
    local days="${1:-30}"
    local project_id=$(pwd | xargs basename)
    
    trends=$(curl -s "http://localhost:3000/api/code-evolution/trends?days=$days&project_id=$project_id")
    
    echo "📈 Code Evolution Trends (last $days days)"
    echo "=========================================="
    echo "Total changes: $(echo "$trends" | jq -r '.total_changes')"
    echo "Avg quality improvement: $(echo "$trends" | jq -r '.avg_quality_improvement * 100 | floor')%"
    echo ""
    echo "Change types:"
    echo "$trends" | jq -r '.change_types | to_entries[] | "  \(.key): \(.value)"'
    echo ""
    echo "Popular patterns:"
    echo "$trends" | jq -r '.pattern_usage | to_entries[] | "  \(.key): \(.value) times"' | head -5
}

# Get pattern analytics
claude-pattern-analytics() {
    analytics=$(curl -s "http://localhost:3000/api/code-evolution/patterns/analytics")
    
    echo "🧠 Refactoring Pattern Analytics"
    echo "================================"
    
    if echo "$analytics" | jq -e '.pattern_success_rates | keys[0]' > /dev/null; then
        echo "Success rates by change type:"
        echo "$analytics" | jq -r '.pattern_success_rates | to_entries[] | "  \(.key): \(.value.avg_improvement * 100 | floor)% avg improvement (\(.value.total_attempts) attempts)"'
    else
        echo "No pattern data available yet"
    fi
    
    echo ""
    echo "Common improvement types:"
    echo "$analytics" | jq -r '.common_improvement_types | to_entries[] | "  \(.key): \(.value) changes"'
}

# Search evolution records
claude-search-evolution() {
    local query="$1"
    local change_type="$2"
    local min_improvement="$3"
    local limit="${4:-5}"
    
    if [ -z "$query" ]; then
        echo "Usage: claude-search-evolution \"search terms\" [change_type] [min_improvement] [limit]"
        return 1
    fi
    
    url="http://localhost:3000/api/code-evolution/search?query=$query&limit=$limit"
    if [ -n "$change_type" ]; then
        url="${url}&change_type=$change_type"
    fi
    if [ -n "$min_improvement" ]; then
        url="${url}&min_quality_improvement=$min_improvement"
    fi
    
    results=$(curl -s "$url")
    
    if [ "$results" != "[]" ]; then
        echo "🔍 Evolution Search Results:"
        echo "$results" | jq -r '.[] | "  [\(.change_id[0:8])] \(.description) - \(.type) (\(.quality_improvement * 100 | floor)% improvement)"'
    else
        echo "No evolution records found matching: $query"
    fi
}

# Track command performance
claude-track-performance() {
    local cmd_type="$1"
    local exec_time="$2"
    local success="$3"
    local output_size="$4"
    local error_msg="$5"
    
    if [ -z "$cmd_type" ] || [ -z "$exec_time" ] || [ -z "$success" ]; then
        echo "Usage: claude-track-performance \"command_type\" exec_time success [output_size] [error_msg]"
        echo "Example: claude-track-performance \"file_read\" 0.5 true 1024"
        return 1
    fi
    
    # Get session info
    local session_id=$(jq -r '.session_id' ~/.claude_session.json 2>/dev/null)
    local project_id=$(pwd | xargs basename)
    
    # Build query params
    local url="http://localhost:3000/api/performance/track"
    url="${url}?command_type=$cmd_type"
    url="${url}&execution_time=$exec_time"
    url="${url}&success=$success"
    
    if [ -n "$output_size" ]; then
        url="${url}&output_size=$output_size"
    fi
    
    if [ -n "$error_msg" ]; then
        url="${url}&error_message=$(echo "$error_msg" | jq -R -r @uri)"
    fi
    
    if [ -n "$project_id" ]; then
        url="${url}&project_id=$project_id"
    fi
    
    if [ -n "$session_id" ]; then
        url="${url}&session_id=$session_id"
    fi
    
    # Track performance
    result=$(curl -s -X POST "$url" \
        -H "Content-Type: application/json" \
        -d "{\"command_details\": {\"timestamp\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}, \"context\": {\"pwd\": \"$(pwd)\"}}")
    
    if echo "$result" | jq -e '.execution_id' > /dev/null; then
        echo "✅ Performance tracked: $(echo "$result" | jq -r '.execution_id') - $(echo "$result" | jq -r '.performance_rating'), quality: $(echo "$result" | jq -r '.quality_score * 100 | floor')%"
        
        if [ "$(echo "$result" | jq -r '.optimization_available')" = "true" ]; then
            echo "💡 $(echo "$result" | jq -r '.suggestions_count') optimization suggestions available"
        fi
    else
        echo "❌ Failed to track performance"
    fi
}

# Get performance report
claude-performance-report() {
    local category="$1"
    local days="${2:-7}"
    local project_id=$(pwd | xargs basename)
    
    url="http://localhost:3000/api/performance/report?time_range=$days"
    if [ -n "$category" ]; then
        url="${url}&category=$category"
    fi
    if [ -n "$project_id" ]; then
        url="${url}&project_id=$project_id"
    fi
    
    report=$(curl -s "$url")
    
    if echo "$report" | jq -e '.summary' > /dev/null; then
        echo "📊 Performance Report (last $days days)"
        echo "========================================"
        echo "Total commands: $(echo "$report" | jq -r '.summary.total_commands')"
        echo "Success rate: $(echo "$report" | jq -r '.summary.success_rate * 100 | floor')%"
        echo "Avg execution time: $(echo "$report" | jq -r '.summary.average_execution_time | . * 100 | floor / 100')s"
        echo "Avg quality score: $(echo "$report" | jq -r '.summary.average_quality_score * 100 | floor')%"
        echo ""
        echo "Categories:"
        echo "$report" | jq -r '.summary.categories | to_entries[] | "  \(.key): \(.value) commands"'
        
        if echo "$report" | jq -e '.optimization_opportunities[0]' > /dev/null; then
            echo ""
            echo "Top optimization opportunities:"
            echo "$report" | jq -r '.optimization_opportunities[:3][] | "  • \(.description) [\(.strategy)]"'
        fi
    else
        echo "❌ No performance data available"
    fi
}

# Predict command performance
claude-predict-performance() {
    local cmd_type="$1"
    
    if [ -z "$cmd_type" ]; then
        echo "Usage: claude-predict-performance \"command_type\""
        return 1
    fi
    
    prediction=$(curl -s -X POST "http://localhost:3000/api/performance/predict" \
        -G \
        --data-urlencode "command_type=$cmd_type" \
        -H "Content-Type: application/json" \
        -d "{\"command_details\": {}, \"context\": {\"pwd\": \"$(pwd)\"}}")
    
    if echo "$prediction" | jq -e '.predicted_execution_time' > /dev/null; then
        echo "🔮 Performance Prediction for: $cmd_type"
        echo "  Predicted time: $(echo "$prediction" | jq -r '.predicted_execution_time | . * 100 | floor / 100')s"
        echo "  Success rate: $(echo "$prediction" | jq -r '.predicted_success_rate * 100 | floor')%"
        echo "  Confidence: $(echo "$prediction" | jq -r '.confidence * 100 | floor')%"
        echo "  Based on: $(echo "$prediction" | jq -r '.based_on_samples') samples"
        
        if echo "$prediction" | jq -e '.risk_factors[0]' > /dev/null; then
            echo ""
            echo "⚠️ Risk factors:"
            echo "$prediction" | jq -r '.risk_factors[] | "  • \(.description) (frequency: \(.frequency * 100 | floor)%)"'
        fi
        
        if echo "$prediction" | jq -e '.optimization_suggestions[0]' > /dev/null; then
            echo ""
            echo "💡 Optimization suggestions:"
            echo "$prediction" | jq -r '.optimization_suggestions[] | "  • \(.description) [\(.strategy)]"'
        fi
    else
        echo "❌ Could not predict performance"
    fi
}

# Analyze command patterns
claude-analyze-patterns() {
    local days="${1:-7}"
    local min_freq="${2:-3}"
    
    patterns=$(curl -s "http://localhost:3000/api/performance/patterns?time_range=$days&min_frequency=$min_freq")
    
    if echo "$patterns" | jq -e '.command_frequency' > /dev/null; then
        echo "🔍 Command Pattern Analysis (last $days days)"
        echo "==========================================="
        
        echo "Most frequent commands:"
        echo "$patterns" | jq -r '.command_frequency | to_entries | sort_by(.value) | reverse | .[:5][] | "  \(.key): \(.value) times"'
        
        if echo "$patterns" | jq -e '.command_sequences | to_entries[0]' > /dev/null; then
            echo ""
            echo "Common command sequences:"
            echo "$patterns" | jq -r '.command_sequences | to_entries | sort_by(.value) | reverse | .[:3][] | "  \(.key): \(.value) times"'
        fi
        
        if echo "$patterns" | jq -e '.optimization_candidates[0]' > /dev/null; then
            echo ""
            echo "Commands needing optimization:"
            echo "$patterns" | jq -r '.optimization_candidates[] | "  • \(.command_type): \(.reason) (avg \(.average_time | . * 100 | floor / 100)s)"'
        fi
    else
        echo "❌ No pattern data available"
    fi
}

# Get performance recommendations
claude-performance-recommend() {
    local limit="${1:-5}"
    
    recommendations=$(curl -s "http://localhost:3000/api/performance/recommendations?limit=$limit")
    
    if [ "$recommendations" != "[]" ]; then
        echo "💡 Performance Recommendations"
        echo "=============================="
        echo "$recommendations" | jq -r '.[] | "[\(.priority)] \(.recommendation)\n  Action: \(.action)"'
    else
        echo "✅ No performance issues detected!"
    fi
}

# View performance trends
claude-performance-trends() {
    local metric="${1:-execution_time}"
    local days="${2:-30}"
    local project_id=$(pwd | xargs basename)
    
    trends=$(curl -s "http://localhost:3000/api/performance/trends?metric=$metric&time_range=$days&project_id=$project_id")
    
    if echo "$trends" | jq -e '.summary' > /dev/null; then
        echo "📈 Performance Trends: $metric (last $days days)"
        echo "============================================="
        echo "Direction: $(echo "$trends" | jq -r '.summary.direction')"
        echo "Change: $(echo "$trends" | jq -r '.summary.change_percentage | floor')%"
        echo "Current value: $(echo "$trends" | jq -r '.summary.current_value | . * 100 | floor / 100')"
        
        if [ "$(echo "$trends" | jq -r '.summary.direction')" = "degrading" ]; then
            echo ""
            echo "⚠️ Performance is getting worse! Consider optimization."
        elif [ "$(echo "$trends" | jq -r '.summary.direction')" = "improving" ]; then
            echo ""
            echo "✅ Performance is improving!"
        fi
    else
        echo "❌ No trend data available"
    fi
}

# Benchmark a command
claude-benchmark() {
    local cmd_type="$1"
    local iterations="${2:-5}"
    
    if [ -z "$cmd_type" ]; then
        echo "Usage: claude-benchmark \"command_type\" [iterations]"
        return 1
    fi
    
    echo "🏃 Benchmarking $cmd_type ($iterations iterations)..."
    
    result=$(curl -s -X POST "http://localhost:3000/api/performance/benchmark" \
        -G \
        --data-urlencode "command_type=$cmd_type" \
        --data-urlencode "iterations=$iterations" \
        -H "Content-Type: application/json" \
        -d '{"command_details": {}, "context": {}}')
    
    if echo "$result" | jq -e '.execution_times' > /dev/null; then
        echo ""
        echo "Results:"
        echo "  Success rate: $(echo "$result" | jq -r '.success_rate * 100 | floor')%"
        echo "  Min time: $(echo "$result" | jq -r '.execution_times.min | . * 1000 | floor')ms"
        echo "  Max time: $(echo "$result" | jq -r '.execution_times.max | . * 1000 | floor')ms"
        echo "  Mean time: $(echo "$result" | jq -r '.execution_times.mean | . * 1000 | floor')ms"
        echo "  Median time: $(echo "$result" | jq -r '.execution_times.median | . * 1000 | floor')ms"
        echo "  Std dev: $(echo "$result" | jq -r '.execution_times.stdev | . * 1000 | floor')ms"
        echo "  Rating: $(echo "$result" | jq -r '.performance_rating')"
    else
        echo "❌ Benchmark failed"
    fi
}

# Get optimization history
claude-optimization-history() {
    local strategy="$1"
    local project_id=$(pwd | xargs basename)
    
    url="http://localhost:3000/api/performance/optimization-history"
    if [ -n "$strategy" ]; then
        url="${url}?strategy=$strategy"
    fi
    if [ -n "$project_id" ]; then
        url="${url}&project_id=$project_id"
    fi
    
    history=$(curl -s "$url")
    
    if [ "$history" != "[]" ]; then
        echo "🔧 Optimization History"
        echo "======================="
        echo "$history" | jq -r '.[:10][] | "[\(.strategy)] \(.command_type): \(.expected_improvement * 100 | floor)% improvement expected (\(.current_time | . * 100 | floor / 100)s → \(.expected_time | . * 100 | floor / 100)s)"'
    else
        echo "No optimization history available"
    fi
}

# ============================================
# Claude Workflow Integration Commands
# ============================================

# Capture memories from conversation text
claude-capture-conversation() {
    local text="$1"
    local session_id="${2:-}"
    local project_id="${3:-}"
    
    if [ -z "$text" ]; then
        echo "Usage: claude-capture-conversation \"conversation text\" [session_id] [project_id]"
        return 1
    fi
    
    result=$(curl -s -X POST "http://localhost:3000/api/claude-workflow/capture/conversation" \
        -G \
        --data-urlencode "session_id=$session_id" \
        --data-urlencode "project_id=$project_id" \
        -H "Content-Type: application/json" \
        -d "{\"conversation_text\": \"$text\"}")
    
    if echo "$result" | jq -e '.memories_created' > /dev/null; then
        echo "✅ Captured $(echo "$result" | jq -r '.memories_created') memories from conversation"
        
        if [ $(echo "$result" | jq '.memories_created') -gt 0 ]; then
            echo "📚 Patterns found:"
            echo "$result" | jq -r '.patterns_found | to_entries[] | "  • \(.key): \(.value | length) items"'
        fi
    else
        echo "❌ Failed to capture conversation memories"
    fi
}

# Extract context from terminal output
claude-capture-terminal() {
    local output="$1"
    local command="$2"
    local exit_code="${3:-0}"
    local exec_time="${4:-}"
    
    if [ -z "$output" ] || [ -z "$command" ]; then
        echo "Usage: claude-capture-terminal \"output\" \"command\" [exit_code] [execution_time]"
        return 1
    fi
    
    params="-G --data-urlencode \"command=$command\" --data-urlencode \"exit_code=$exit_code\""
    if [ -n "$exec_time" ]; then
        params="$params --data-urlencode \"execution_time=$exec_time\""
    fi
    
    result=$(curl -s -X POST "http://localhost:3000/api/claude-workflow/capture/terminal" \
        $params \
        -H "Content-Type: application/json" \
        -d "{\"terminal_output\": \"$output\"}")
    
    if echo "$result" | jq -e '.insights_extracted' > /dev/null; then
        echo "✅ Extracted $(echo "$result" | jq -r '.insights_extracted') insights from terminal"
        echo "📝 Command: $command (exit code: $exit_code)"
    else
        echo "❌ Failed to extract terminal context"
    fi
}

# Capture tool usage
claude-capture-tool() {
    local tool_name="$1"
    local exec_time="$2"
    local params="$3"
    local result="$4"
    local session_id="${5:-}"
    
    if [ -z "$tool_name" ] || [ -z "$exec_time" ]; then
        echo "Usage: claude-capture-tool \"tool_name\" execution_time \"{params}\" \"{result}\" [session_id]"
        return 1
    fi
    
    capture_result=$(curl -s -X POST "http://localhost:3000/api/claude-workflow/capture/tool-usage" \
        -G \
        --data-urlencode "tool_name=$tool_name" \
        --data-urlencode "execution_time=$exec_time" \
        --data-urlencode "session_id=$session_id" \
        -H "Content-Type: application/json" \
        -d "{\"tool_params\": $params, \"tool_result\": $result}")
    
    if echo "$capture_result" | jq -e '.memories_created' > /dev/null; then
        echo "✅ Captured tool usage: $tool_name ($(echo "$capture_result" | jq -r '.memories_created') memories)"
    else
        echo "❌ Failed to capture tool usage"
    fi
}

# Save a discovery
claude-save-discovery() {
    local type="$1"
    local content="$2"
    local context="${3:-{}}"
    local importance="${4:-high}"
    local tags="${5:-}"
    
    if [ -z "$type" ] || [ -z "$content" ]; then
        echo "Usage: claude-save-discovery \"type\" \"content\" [context_json] [importance] [tags]"
        echo "Types: pattern, solution, bug_fix, optimization, architecture, algorithm, configuration"
        return 1
    fi
    
    params="-G --data-urlencode \"discovery_type=$type\" --data-urlencode \"importance=$importance\""
    if [ -n "$tags" ]; then
        params="$params --data-urlencode \"tags=$tags\""
    fi
    
    result=$(curl -s -X POST "http://localhost:3000/api/claude-workflow/save/discovery" \
        $params \
        -H "Content-Type: application/json" \
        -d "{\"content\": \"$content\", \"context\": $context}")
    
    if echo "$result" | jq -e '.discovery_id' > /dev/null; then
        echo "✅ Discovery saved: $(echo "$result" | jq -r '.discovery_id[0:8]')"
        echo "📌 Type: $type ($(echo "$result" | jq -r '.memory_type'))"
        echo "⭐ Importance: $importance"
    else
        echo "❌ Failed to save discovery"
    fi
}

# Extract insights from messages
claude-extract-insights() {
    local message="$1"
    local type="${2:-assistant}"
    local session_id="${3:-}"
    
    if [ -z "$message" ]; then
        echo "Usage: claude-extract-insights \"message\" [message_type] [session_id]"
        return 1
    fi
    
    result=$(curl -s -X POST "http://localhost:3000/api/claude-workflow/extract/insights" \
        -G \
        --data-urlencode "message_type=$type" \
        --data-urlencode "session_id=$session_id" \
        -H "Content-Type: application/json" \
        -d "{\"message\": \"$message\"}")
    
    if echo "$result" | jq -e '.insights_found' > /dev/null; then
        echo "✅ Extracted $(echo "$result" | jq -r '.insights_found') insights"
        
        if [ $(echo "$result" | jq '.insights_found') -gt 0 ]; then
            echo "$result" | jq -r '.insights[] | "  • [\(.type)] \(.content // .language // ""))"'
        fi
    else
        echo "❌ Failed to extract insights"
    fi
}

# Get workflow integration stats
claude-workflow-stats() {
    local session_id="${1:-}"
    local days="${2:-7}"
    
    params="-G --data-urlencode \"time_range=$days\""
    if [ -n "$session_id" ]; then
        params="$params --data-urlencode \"session_id=$session_id\""
    fi
    
    result=$(curl -s "http://localhost:3000/api/claude-workflow/stats" $params)
    
    if echo "$result" | jq -e '.stats' > /dev/null; then
        echo "📊 Workflow Integration Stats (last $days days)"
        echo "============================================"
        echo "  Auto-captured insights: $(echo "$result" | jq -r '.stats.auto_captured_insights')"
        echo "  Tool usage memories: $(echo "$result" | jq -r '.stats.tool_usage_memories')"
        echo "  Discoveries saved: $(echo "$result" | jq -r '.stats.discoveries_saved')"
        echo "  Total workflow memories: $(echo "$result" | jq -r '.stats.total_workflow_memories')"
        
        if echo "$result" | jq -e '.breakdown.by_type' > /dev/null; then
            echo ""
            echo "By type:"
            echo "$result" | jq -r '.breakdown.by_type | to_entries[] | "  • \(.key): \(.value)"'
        fi
    else
        echo "❌ Failed to get workflow stats"
    fi
}

echo "Claude Code helpers loaded! Commands available:"
echo "  claude-init              - Start new session with context restoration"
echo "  claude-handoff           - Create handoff note for next session"
echo "  claude-error             - Record error and solution"
echo "  claude-session           - Show current session info"
echo "  claude-stats             - Show memory statistics"
echo "  claude-remember          - Add to memory"
echo "  claude-search            - Search memories"
echo "  claude-checkpoint        - Create checkpoint"
echo "  claude-context           - Show recent context"
echo "  claude-tasks             - Predict next tasks"
echo "  claude-find-error        - Find similar errors with solutions"
echo "  claude-check             - Check if action might cause known error"
echo "  claude-lessons           - View lessons learned from mistakes"
echo "  claude-report            - Get mistake analysis report"
echo "  claude-assist            - Get proactive assistance"
echo "  claude-todos             - Show incomplete tasks"
echo "  claude-reminders         - Get helpful reminders"
echo "  claude-decide            - Record decision with reasoning and alternatives"
echo "  claude-explain           - Explain reasoning behind past decision"
echo "  claude-search-decisions  - Search through past decisions"
echo "  claude-suggest-decision  - Get decision suggestion based on past experience"
echo "  claude-update-decision   - Update decision with actual outcome"
echo "  claude-confidence-report - Get report on decision confidence accuracy"
echo "  claude-track-change      - Track code changes with before/after analysis"
echo "  claude-compare-change    - Compare code versions from specific change"
echo "  claude-evolution-history - View code evolution history"
echo "  claude-suggest-refactoring - Get refactoring suggestions based on patterns"
echo "  claude-update-impact     - Update change record with measured impact"
echo "  claude-evolution-trends  - View code evolution trends over time"
echo "  claude-pattern-analytics - Get refactoring pattern analytics"
echo "  claude-search-evolution  - Search through code evolution records"
echo "  claude-track-performance - Track command execution performance"
echo "  claude-performance-report - Get comprehensive performance report"
echo "  claude-predict-performance - Predict command performance before execution"
echo "  claude-analyze-patterns  - Analyze command execution patterns"
echo "  claude-performance-recommend - Get performance optimization recommendations"
echo "  claude-performance-trends - View performance trends over time"
echo "  claude-benchmark         - Benchmark command performance"
echo "  claude-optimization-history - View optimization suggestions history"
echo ""
echo "Workflow Integration Commands:"
echo "  claude-capture-conversation - Capture memories from conversation text"
echo "  claude-capture-terminal  - Extract context from terminal output"
echo "  claude-capture-tool      - Capture tool usage memories"
echo "  claude-save-discovery    - Save important discoveries"
echo "  claude-extract-insights  - Extract insights from messages"
echo "  claude-workflow-stats    - Get workflow integration statistics"