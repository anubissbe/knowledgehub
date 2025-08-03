#!/bin/bash

# KnowledgeHub Service Recovery System Shutdown Script
# Stops the automated recovery and self-healing system

set -e

echo "🛑 Stopping KnowledgeHub Service Recovery System..."

# Configuration
RECOVERY_PID_FILE="/var/run/knowledgehub-recovery.pid"

# Check if running as systemd service
if systemctl is-active --quiet knowledgehub-recovery.service 2>/dev/null; then
    echo "🔄 Stopping systemd service..."
    sudo systemctl stop knowledgehub-recovery.service
    
    # Wait for clean shutdown
    echo "⏳ Waiting for graceful shutdown..."
    sleep 3
    
    # Verify stopped
    if systemctl is-active --quiet knowledgehub-recovery.service; then
        echo "⚠️  Service still running, forcing stop..."
        sudo systemctl kill knowledgehub-recovery.service
        sleep 2
    fi
    
    echo "✅ Systemd service stopped"
    
else
    echo "ℹ️  Systemd service not running"
fi

# Check for standalone process
if [ -f "$RECOVERY_PID_FILE" ]; then
    OLD_PID=$(cat "$RECOVERY_PID_FILE")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "🔄 Stopping standalone process (PID: $OLD_PID)..."
        kill -TERM "$OLD_PID"
        
        # Wait for graceful shutdown
        sleep 5
        
        # Force kill if still running
        if kill -0 "$OLD_PID" 2>/dev/null; then
            echo "⚠️  Process still running, forcing shutdown..."
            kill -KILL "$OLD_PID"
        fi
        
        echo "✅ Standalone process stopped"
    fi
    
    rm -f "$RECOVERY_PID_FILE"
fi

# Check for any remaining recovery processes
RECOVERY_PROCESSES=$(pgrep -f "service_recovery" || true)
if [ -n "$RECOVERY_PROCESSES" ]; then
    echo "🧹 Cleaning up remaining recovery processes..."
    echo "$RECOVERY_PROCESSES" | xargs kill -TERM 2>/dev/null || true
    sleep 2
    
    # Force kill any stubborn processes
    REMAINING=$(pgrep -f "service_recovery" || true)
    if [ -n "$REMAINING" ]; then
        echo "🚫 Force killing stubborn processes..."
        echo "$REMAINING" | xargs kill -KILL 2>/dev/null || true
    fi
fi

echo ""
echo "✅ Recovery system shutdown complete!"

# Show final status
echo ""
echo "📊 Final Status Check:"
echo "  🔍 Systemd service: $(systemctl is-active knowledgehub-recovery.service 2>/dev/null || echo 'inactive')"
echo "  🔍 Running processes: $(pgrep -f "service_recovery" | wc -l) recovery processes found"

echo ""
echo "💡 To restart the recovery system:"
echo "  ./scripts/start-recovery-system.sh"

echo ""
echo "🗑️  To disable the systemd service permanently:"
echo "  sudo systemctl disable knowledgehub-recovery.service"
echo "  sudo rm /etc/systemd/system/knowledgehub-recovery.service"
echo "  sudo systemctl daemon-reload"