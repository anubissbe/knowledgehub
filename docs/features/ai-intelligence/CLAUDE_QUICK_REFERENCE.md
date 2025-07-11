# Claude Code Quick Reference

## 🚀 Start Any New Session
```bash
/opt/projects/knowledgehub/claude_code_init.py
# OR
source /opt/projects/knowledgehub/claude_code_helpers.sh && claude-init
```

## 📍 What's Implemented

### 1. Session Continuity ✅
- Auto-restores context from previous sessions
- Files: `~/.claude_session.json`, `~/.claude_context.json`
- API: `/api/claude-auto/session/*`

### 2. Project Profiles ✅  
- Auto-detects Python/Node/Rust/Go projects
- Maintains project-specific context
- API: `/api/claude-auto/session/start?cwd=/path`

### 3. Error Learning ✅
- Records errors with solutions
- Finds similar errors from history
- Commands: `claude-error "Type" "Message" "Solution" true/false`
- API: `/api/claude-auto/error/*`

### 4. Task Prediction ✅
- Predicts next tasks from handoff notes
- Shows unfinished work
- Command: `claude-tasks`
- API: `/api/claude-auto/tasks/predict`

## 💾 Key Files
- `/opt/projects/knowledgehub/src/api/services/claude_session_manager.py` - Core logic
- `/opt/projects/knowledgehub/src/api/routers/claude_auto.py` - API endpoints
- `/opt/projects/knowledgehub/claude_code_init.py` - CLI initializer
- `/opt/projects/knowledgehub/claude_code_helpers.sh` - Shell commands

## 🧪 Test Everything
```bash
python3 /opt/projects/knowledgehub/test_claude_session_continuity.py
```

## 📝 Create Handoff
```bash
claude-handoff "Work summary" "task1,task2" "issue1,issue2"
```

---
**REMEMBER**: Always run `claude-init` at start of conversation!