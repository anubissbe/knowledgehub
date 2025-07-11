# KnowledgeHub Shell Commands Reference

## Overview

KnowledgeHub provides 45 shell commands to interact with all AI intelligence features directly from your terminal. These commands are available after sourcing the helper script.

## Setup

```bash
# Load all commands into your shell
source /opt/projects/knowledgehub/claude_code_helpers.sh

# Or add to your shell profile for permanent access
echo "source /opt/projects/knowledgehub/claude_code_helpers.sh" >> ~/.bashrc
```

## Commands by Category

### 🔗 Session Management (5 commands)

#### claude-init
Start a new Claude Code session with automatic context restoration.
```bash
claude-init
# Output: Session ID, restored memories, handoff notes, unfinished tasks
```

#### claude-handoff
Create a handoff note for the next session.
```bash
claude-handoff "Implemented auth system, next: add unit tests" "TODO: Add JWT refresh" "Issue: Rate limiting needs tuning"
# Parameters: summary, tasks (optional), issues (optional)
```

#### claude-session
Show current session information.
```bash
claude-session
# Output: Session ID, project, duration, memory count
```

#### claude-stats
Display memory system statistics.
```bash
claude-stats
# Output: Total memories, types breakdown, recent activity
```

#### claude-checkpoint
Create a milestone checkpoint.
```bash
claude-checkpoint "Completed authentication implementation"
# Parameter: description
```

---

### 📁 Project Context (3 commands)

#### claude-project-context
Show current project context information.
```bash
claude-project-context
# Output: Project name, type, total memories, conventions
```

#### claude-project-preference
Save a project-specific preference.
```bash
claude-project-preference "indent_style" "spaces" "Code style"
# Parameters: key, value, description
```

#### claude-project-patterns
View learned project patterns.
```bash
claude-project-patterns
# Output: Code patterns, naming conventions, common practices
```

---

### 🧠 Error Learning (5 commands)

#### claude-error
Record an error and its solution.
```bash
claude-error "ImportError" "No module named requests" "pip install requests" true
# Parameters: error_type, error_message, solution, worked (true/false)
```

#### claude-find-error
Find similar errors with solutions.
```bash
claude-find-error "No module named"
# Parameter: error_message_part
# Output: Similar errors with their solutions
```

#### claude-check
Check if an action might cause a known error.
```bash
claude-check "delete database migration"
# Parameter: action_description
# Output: Warnings and prevention tips
```

#### claude-lessons
View all learned lessons from mistakes.
```bash
claude-lessons
# Output: List of error types with their solutions
```

#### claude-report
Get a comprehensive mistake analysis report.
```bash
claude-report 30  # Last 30 days
# Parameter: days (optional, default: 7)
# Output: Error patterns, most common mistakes, success rates
```

---

### ⚡ Proactive Assistance (7 commands)

#### claude-tasks
Predict likely next tasks based on context.
```bash
claude-tasks
# Output: Predicted tasks with confidence scores
```

#### claude-brief
Get a quick project status brief.
```bash
claude-brief
# Output: Current focus, recent changes, suggestions
```

#### claude-todos
Show incomplete tasks from previous sessions.
```bash
claude-todos
# Output: Unfinished work items
```

#### claude-reminders
Get contextual reminders and tips.
```bash
claude-reminders
# Output: Best practices, warnings, suggestions
```

#### claude-assist
Get comprehensive proactive assistance.
```bash
claude-assist
# Output: Full analysis with predictions, reminders, and suggestions
```

#### claude-suggestions
Get specific suggestions for current context.
```bash
claude-suggestions
# Output: Actionable suggestions based on patterns
```

#### claude-interrupt
Check if it's a good time to switch tasks.
```bash
claude-interrupt
# Output: Whether to continue or switch, with reasoning
```

---

### 📊 Decision Tracking (7 commands)

#### claude-decide
Record a decision with reasoning and alternatives.
```bash
claude-decide "Use PostgreSQL" "Better JSON support" "MySQL,MongoDB" "Need ACID compliance" 0.85
# Parameters: decision, reasoning, alternatives (comma-separated), context, confidence (0-1)
```

#### claude-explain-decision
Explain the reasoning behind a past decision.
```bash
claude-explain-decision abc123def  # Decision ID
# Parameter: decision_id (first 8 chars)
# Output: Full decision context with reasoning
```

#### claude-search-decisions
Search through past decisions.
```bash
claude-search-decisions "database" "backend" 10
# Parameters: query, category (optional), limit (optional)
# Output: Matching decisions with confidence scores
```

#### claude-suggest-decision
Get decision suggestions based on past experience.
```bash
claude-suggest-decision "Which testing framework to use"
# Parameter: problem description
# Output: Suggestions based on similar past decisions
```

#### claude-update-decision
Update a decision with its actual outcome.
```bash
claude-update-decision abc123def "success" "Improved performance by 50%"
# Parameters: decision_id, outcome (success/failure/mixed), impact
```

#### claude-confidence-report
Analyze decision confidence accuracy.
```bash
claude-confidence-report 30
# Parameter: days (optional)
# Output: Confidence calibration analysis
```

#### claude-decision-patterns
View decision patterns by category.
```bash
claude-decision-patterns "architecture"
# Parameter: category (optional)
# Output: Common patterns and reasoning
```

---

### 🔄 Code Evolution (8 commands)

#### claude-track-change
Track a code change with analysis.
```bash
claude-track-change "Extract validation logic into separate module" "Improve modularity"
# Parameters: description, reason
```

#### claude-code-history
View code evolution history.
```bash
claude-code-history "/src/api/auth.py" 10
# Parameters: file_path (optional), limit (optional)
# Output: Change history with quality metrics
```

#### claude-compare-change
Compare before/after versions of a change.
```bash
claude-compare-change abc123def
# Parameter: change_id
# Output: Diff with improvements highlighted
```

#### claude-suggest-refactoring
Get refactoring suggestions for a file.
```bash
claude-suggest-refactoring "/src/api/views.py"
# Parameter: file_path
# Output: Suggested improvements based on patterns
```

#### claude-update-impact
Update a code change with its measured impact.
```bash
claude-update-impact abc123def 0.75 "Reduced complexity, improved readability"
# Parameters: change_id, quality_score (0-1), notes
```

#### claude-evolution-trends
View code quality trends over time.
```bash
claude-evolution-trends 30
# Parameter: days (optional)
# Output: Quality trends, common patterns
```

#### claude-pattern-analytics
Analyze refactoring patterns.
```bash
claude-pattern-analytics
# Output: Most common refactoring types with success rates
```

#### claude-search-evolution
Search through code evolution records.
```bash
claude-search-evolution "validation" 20
# Parameters: query, limit (optional)
# Output: Matching code changes
```

---

### 📈 Performance Metrics (8 commands)

#### claude-track-performance
Track command execution performance.
```bash
claude-track-performance "npm build" 45.5 true 8192000 ""
# Parameters: command_type, execution_time, success, output_size (optional), error_msg (optional)
```

#### claude-performance-report
Get comprehensive performance report.
```bash
claude-performance-report "build_operations" 7
# Parameters: category (optional), days (optional)
# Output: Stats, optimization opportunities
```

#### claude-predict-performance
Predict command performance before execution.
```bash
claude-predict-performance "large_file_search"
# Parameter: command_type
# Output: Predicted time, success rate, risks
```

#### claude-analyze-patterns
Analyze command execution patterns.
```bash
claude-analyze-patterns 7 3
# Parameters: days (optional), min_frequency (optional)
# Output: Common sequences, optimization candidates
```

#### claude-performance-recommend
Get performance optimization recommendations.
```bash
claude-performance-recommend 5
# Parameter: limit (optional)
# Output: Prioritized recommendations
```

#### claude-performance-trends
View performance trends over time.
```bash
claude-performance-trends "execution_time" 30
# Parameters: metric (optional), days (optional)
# Output: Trend analysis with charts
```

#### claude-benchmark
Benchmark command performance.
```bash
claude-benchmark "file_read" 10
# Parameters: command_type, iterations (optional)
# Output: Statistical analysis
```

#### claude-optimization-history
View optimization suggestion history.
```bash
claude-optimization-history "caching"
# Parameter: strategy (optional)
# Output: Past optimizations with results
```

---

### 🤖 Workflow Integration (6 commands)

#### claude-capture-conversation
Capture memories from conversation text.
```bash
claude-capture-conversation "Found bug in auth. Fixed by adding middleware validation."
# Parameter: conversation_text
# Output: Extracted memories count
```

#### claude-capture-terminal
Extract insights from terminal output.
```bash
claude-capture-terminal "Build failed: missing dependency" "npm install" 1 0.5
# Parameters: output, command, exit_code, execution_time (optional)
```

#### claude-capture-tool
Track Claude Code tool usage.
```bash
claude-capture-tool "Write" 0.25 '{"file_path": "test.py"}' '{"success": true}' "session-123"
# Parameters: tool_name, execution_time, params_json, result_json, session_id (optional)
```

#### claude-save-discovery
Save an important discovery.
```bash
claude-save-discovery "optimization" "Use Redis for session caching" '{"impact": "50% faster"}' "high" "performance,caching"
# Parameters: type, content, context_json (optional), importance (optional), tags (optional)
# Types: pattern, solution, bug_fix, optimization, architecture, algorithm, configuration
```

#### claude-extract-insights
Extract insights from messages.
```bash
claude-extract-insights "I found that using async/await improves performance" "assistant"
# Parameters: message, message_type (optional)
# Output: Extracted insights
```

#### claude-workflow-stats
View workflow integration statistics.
```bash
claude-workflow-stats "session-123" 7
# Parameters: session_id (optional), days (optional)
# Output: Capture statistics
```

---

### 💾 Memory Operations (4 commands)

#### claude-remember
Quick memory addition to the system.
```bash
claude-remember "Redis cache invalidation pattern" "pattern" "high"
# Parameters: content, type (optional), priority (optional)
# Types: fact, preference, code, decision, error, pattern, entity
```

#### claude-search
Search through memories.
```bash
claude-search "cache invalidation"
# Parameter: query
# Output: Matching memories with relevance scores
```

#### claude-context
Show recent context memories.
```bash
claude-context 20
# Parameter: count (optional, default: 10)
# Output: Recent relevant memories
```

---

## Tips and Best Practices

### 1. Start Every Session
Always run `claude-init` at the beginning of a Claude Code session to restore context.

### 2. Track Decisions
Use `claude-decide` for important architectural or implementation decisions.

### 3. Learn from Errors
Always use `claude-error` when you solve a problem to help future sessions.

### 4. Regular Checkpoints
Use `claude-checkpoint` at major milestones for better context preservation.

### 5. Review Performance
Run `claude-performance-report` weekly to identify optimization opportunities.

### 6. Capture Discoveries
Use `claude-save-discovery` for important insights or patterns you discover.

## Command Output Formats

Most commands return formatted text output suitable for terminal display. Some commands also save results to the memory system for future reference.

### JSON Output
Add `| jq` to any command for JSON formatting:
```bash
claude-session | jq
```

### Filtering Output
Use standard Unix tools to filter results:
```bash
claude-lessons | grep "Import"
claude-performance-report | head -20
```

## Troubleshooting

### Commands Not Found
```bash
# Ensure the helper script is sourced
source /opt/projects/knowledgehub/claude_code_helpers.sh
```

### API Connection Issues
```bash
# Check if API is running
curl http://localhost:3000/health
```

### No Results
Some commands require existing data. Use the system for a while to build up memory.

---

**Version**: 2.0.0  
**Total Commands**: 45  
**Last Updated**: 2025-07-11