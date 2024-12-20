#!/bin/bash

# install.sh
#
# Purpose: Install and configure Atlantium RAG update service
# Author: Your Name
# Date: December 2024
#
# This script:
# 1. Creates necessary directories and user
# 2. Sets up permissions
# 3. Installs the update service
# 4. Configures logging
# 5. Starts the service
#
# Usage: sudo ./install.sh

#!/bin/bash

# install.sh
# Installs the Atlantium RAG update service for the current user

set -e

# Detect user environment
CURRENT_USER=${SUDO_USER:-$USER}
USER_HOME=$(eval echo ~$CURRENT_USER)
DEFAULT_APP_DIR="$USER_HOME/Projects/Atlantium_LLM"

# Allow override of installation directory
APP_DIR=${APP_DIR:-$DEFAULT_APP_DIR}

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "Please run with sudo"
    exit 1
fi

echo "Installing Atlantium RAG update service..."
echo "User: $CURRENT_USER"
echo "Home: $USER_HOME"
echo "App Directory: $APP_DIR"

# Check if Docker is installed
if ! command -v docker >/dev/null 2>&1; then
    echo "Error: Docker is not installed"
    exit 1
fi

# Add user to docker group if needed
if ! groups "$CURRENT_USER" | grep -q docker; then
    echo "Adding $CURRENT_USER to docker group..."
    usermod -aG docker "$CURRENT_USER"
fi

# Create project directory structure
echo "Creating directory structure..."
mkdir -p "$APP_DIR/scripts/update_service"
mkdir -p "$APP_DIR/logs/updates"
mkdir -p "$APP_DIR/backups"

# Install the update script
echo "Installing update script..."
cp release_update.sh "$APP_DIR/scripts/update_service"
chmod +x "$APP_DIR/scripts/update_service/release_update.sh"

# Create systemd service file
echo "Creating systemd service..."
cat > /etc/systemd/system/atlantium-update.service << EOF
[Unit]
Description=Atlantium RAG Release Update Service
Documentation=https://github.com/kertser/Atlantium_LLM
After=network.target docker.service
Wants=docker.service

[Service]
Type=simple
User=$CURRENT_USER
Group=docker
WorkingDirectory=$APP_DIR

# Environment variables
Environment="APP_DIR=$APP_DIR"
Environment="LOG_DIR=$APP_DIR/logs"
Environment="LOG_LEVEL=INFO"

# Execution
ExecStartPre=/bin/mkdir -p \${LOG_DIR}/updates
ExecStart=$APP_DIR/scripts/update_service/release_update.sh

# Restart configuration
Restart=on-failure
RestartSec=60
StartLimitInterval=300
StartLimitBurst=3

# Security hardening
NoNewPrivileges=yes
ProtectSystem=full
ProtectHome=read-only
PrivateTmp=yes
ProtectKernelEnables=yes
ProtectKernelModules=yes
ProtectControlGroups=yes
RestrictAddressFamilies=AF_UNIX AF_INET AF_INET6
RestrictNamespaces=yes
RestrictRealtime=yes
SystemCallArchitectures=native

[Install]
WantedBy=multi-user.target
EOF

# Set proper permissions
echo "Setting permissions..."
chown -R "$CURRENT_USER":"$CURRENT_USER" "$APP_DIR"
chmod 755 "$APP_DIR/scripts/update_service/release_update.sh"

# Setup log rotation
echo "Configuring log rotation..."
cat > /etc/logrotate.d/atlantium-update << EOF
$APP_DIR/logs/updates/update.log {
    daily
    rotate 7
    compress
    missingok
    notifempty
    create 0640 $CURRENT_USER $CURRENT_USER
}
EOF

# Reload systemd and start service
echo "Starting service..."
systemctl daemon-reload
systemctl enable atlantium-update
systemctl start atlantium-update

# Show status
echo -e "\nInstallation complete!"
echo "Service status:"
systemctl status atlantium-update

# Show helpful information
echo -e "\nUseful commands:"
echo "  Check service status: systemctl status atlantium-update"
echo "  View service logs: journalctl -u atlantium-update -f"
echo "  View update logs: tail -f $APP_DIR/logs/updates/update.log"
echo -e "\nUninstall commands:"
echo "  sudo systemctl stop atlantium-update"
echo "  sudo systemctl disable atlantium-update"
echo "  sudo rm /etc/systemd/system/atlantium-update.service"

# Warning about docker group
if groups "$CURRENT_USER" | grep -q docker; then
    echo -e "\nIMPORTANT: You've been added to the docker group."
    echo "Please log out and back in for this change to take effect."
fi