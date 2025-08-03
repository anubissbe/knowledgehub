#!/bin/bash

# KnowledgeHub Service Recovery System Startup Script
# Starts the automated recovery and self-healing system

set -e

echo "🔄 Starting KnowledgeHub Service Recovery System..."

# Configuration
RECOVERY_LOG_DIR="/var/log/knowledgehub/recovery"
RECOVERY_PID_FILE="/var/run/knowledgehub-recovery.pid"
RECOVERY_CONFIG_FILE="/opt/projects/knowledgehub/config/recovery.json"

# Create log directory if it doesn't exist
if [ ! -d "$RECOVERY_LOG_DIR" ]; then
    echo "📁 Creating recovery log directory..."
    sudo mkdir -p "$RECOVERY_LOG_DIR"
    sudo chown -R $USER:$USER "$RECOVERY_LOG_DIR"
fi

# Create default recovery configuration if it doesn't exist
if [ ! -f "$RECOVERY_CONFIG_FILE" ]; then
    echo "⚙️  Creating default recovery configuration..."
    mkdir -p "$(dirname "$RECOVERY_CONFIG_FILE")"
    cat > "$RECOVERY_CONFIG_FILE" << 'EOF'
{
    "enabled": true,
    "monitoring_interval": 30,
    "recovery_timeout": 300,
    "max_concurrent_recoveries": 3,
    "notification": {
        "email_enabled": false,
        "slack_enabled": false,
        "webhook_url": null
    },
    "services": {
        "database": {
            "enabled": true,
            "priority": "high",
            "max_retries": 3,
            "recovery_actions": ["reconnect", "restart"]
        },
        "redis": {
            "enabled": true,
            "priority": "medium",
            "max_retries": 3,
            "recovery_actions": ["reconnect", "restart", "reset_cache"]
        },
        "weaviate": {
            "enabled": true,
            "priority": "high",
            "max_retries": 2,
            "recovery_actions": ["restart"]
        },
        "neo4j": {
            "enabled": true,
            "priority": "medium",
            "max_retries": 2,
            "recovery_actions": ["restart"]
        },
        "minio": {
            "enabled": true,
            "priority": "low",
            "max_retries": 2,
            "recovery_actions": ["restart"]
        }
    }
}
EOF
    echo "✅ Default recovery configuration created at $RECOVERY_CONFIG_FILE"
fi

# Check if recovery system is already running
if [ -f "$RECOVERY_PID_FILE" ]; then
    OLD_PID=$(cat "$RECOVERY_PID_FILE")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "⚠️  Recovery system already running with PID $OLD_PID"
        echo "🛑 Stopping existing recovery system..."
        kill "$OLD_PID"
        sleep 3
    fi
    rm -f "$RECOVERY_PID_FILE"
fi

# Start recovery system as a background service
echo "🚀 Starting recovery system daemon..."

# Create systemd service if it doesn't exist
if [ ! -f "/etc/systemd/system/knowledgehub-recovery.service" ]; then
    echo "📋 Creating systemd service..."
    sudo tee /etc/systemd/system/knowledgehub-recovery.service > /dev/null << EOF
[Unit]
Description=KnowledgeHub Service Recovery System
After=network.target
Wants=network.target

[Service]
Type=simple
User=$USER
Group=$USER
WorkingDirectory=/opt/projects/knowledgehub
Environment=PYTHONPATH=/opt/projects/knowledgehub
ExecStart=/usr/bin/python3 -m api.services.service_recovery
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=knowledgehub-recovery

# Security settings
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/opt/projects/knowledgehub /var/log/knowledgehub /var/run
CapabilityBoundingSet=CAP_NET_BIND_SERVICE
AmbientCapabilities=CAP_NET_BIND_SERVICE

[Install]
WantedBy=multi-user.target
EOF

    # Reload systemd and enable service
    sudo systemctl daemon-reload
    sudo systemctl enable knowledgehub-recovery.service
    echo "✅ Systemd service created and enabled"
fi

# Start the service
echo "🔄 Starting recovery system service..."
sudo systemctl start knowledgehub-recovery.service

# Wait a moment for startup
sleep 2

# Check service status
if sudo systemctl is-active --quiet knowledgehub-recovery.service; then
    echo "✅ Recovery system started successfully!"
    
    # Get service status
    echo ""
    echo "📊 Service Status:"
    sudo systemctl status knowledgehub-recovery.service --no-pager -l
    
    echo ""
    echo "📝 Recent logs:"
    sudo journalctl -u knowledgehub-recovery.service --no-pager -l -n 10
    
else
    echo "❌ Failed to start recovery system"
    echo ""
    echo "🔍 Checking logs for errors:"
    sudo journalctl -u knowledgehub-recovery.service --no-pager -l -n 20
    exit 1
fi

echo ""
echo "🎯 Recovery System Information:"
echo "  📋 Service: knowledgehub-recovery.service"
echo "  📁 Config: $RECOVERY_CONFIG_FILE"
echo "  📄 Logs: journalctl -u knowledgehub-recovery.service -f"
echo "  🛑 Stop: sudo systemctl stop knowledgehub-recovery.service"
echo "  📊 Status: sudo systemctl status knowledgehub-recovery.service"

echo ""
echo "🌐 API Endpoints:"
echo "  📈 Status: http://localhost:3000/api/recovery/status"
echo "  📊 Health: http://localhost:3000/api/recovery/health"
echo "  📋 Services: http://localhost:3000/api/recovery/services"
echo "  📉 Metrics: http://localhost:3000/api/recovery/metrics"
echo "  🚨 Alerts: http://localhost:3000/api/recovery/alerts"

echo ""
echo "✅ Recovery system startup complete!"

# Perform initial health check
echo ""
echo "🔍 Performing initial health check..."
sleep 5

# Test API endpoint
if curl -sf http://localhost:3000/api/recovery/health > /dev/null 2>&1; then
    echo "✅ Recovery API responding correctly"
    
    # Show initial status
    echo ""
    echo "📊 Initial System Status:"
    curl -s http://localhost:3000/api/recovery/status | python3 -m json.tool || echo "Status endpoint not yet ready"
    
else
    echo "⚠️  Recovery API not yet responding (may still be starting up)"
    echo "💡 Check logs with: sudo journalctl -u knowledgehub-recovery.service -f"
fi

echo ""
echo "🎉 Recovery system is now monitoring and protecting your services!"