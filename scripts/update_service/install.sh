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
USER_HOME=$(eval echo ~"$CURRENT_USER")
DEFAULT_APP_DIR="$USER_HOME/Projects/Atlantium_LLM"

# Allow override of installation directory
APP_DIR=${APP_DIR:-$DEFAULT_APP_DIR}

validate_environment() {
    # Check if we're in the correct repository
    if [ ! -d "$APP_DIR/.git" ]; then
        echo "Warning: Git repository not found in $APP_DIR"
        read -p "Continue anyway? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi

    # Validate Python installation
    if ! command -v python3 >/dev/null 2>&1; then
        echo "Error: Python 3 is required but not installed"
        exit 1
    fi

    # Check disk space
    MIN_SPACE_MB=1000
    AVAILABLE_SPACE=$(df -m "$APP_DIR" | awk 'NR==2 {print $4}')
    if [ "$AVAILABLE_SPACE" -lt "$MIN_SPACE_MB" ]; then
        echo "Error: Insufficient disk space. Need at least ${MIN_SPACE_MB}MB"
        exit 1
    fi
}

# Call the validation function
validate_environment

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

# Create project directory structure and set permissions
echo "Creating directory structure and setting permissions..."
mkdir -p "$APP_DIR/scripts/update_service"
mkdir -p "$APP_DIR/logs/updates"
mkdir -p "$APP_DIR/backups"

# Set proper directory ownership and permissions
echo "Setting directory permissions..."
chown -R "$CURRENT_USER:$CURRENT_USER" "$APP_DIR"
chmod -R u+rwX,g+rX,o+rX "$APP_DIR"  # Base permissions
chmod -R 755 "$APP_DIR/scripts"       # Executable permissions for scripts
chmod -R 755 "$APP_DIR/logs"          # Ensure logs directory is writable
chmod -R 755 "$APP_DIR/backups"       # Ensure backups directory is writable

# Ensure specific permissions for logs directory
find "$APP_DIR/logs" -type d -exec chmod 755 {} \;
find "$APP_DIR/logs" -type f -exec chmod 644 {} \;

# Create log file with proper permissions if it doesn't exist
touch "$APP_DIR/logs/updates/update.log"
chown "$CURRENT_USER:$CURRENT_USER" "$APP_DIR/logs/updates/update.log"
chmod 644 "$APP_DIR/logs/updates/update.log"

# Verify permissions
echo "Verifying permissions..."
if ! sudo -u "$CURRENT_USER" test -w "$APP_DIR/logs/updates/update.log"; then
    echo "Error: Log file is not writable by $CURRENT_USER"
    exit 1
fi

# Install the update script
echo "Installing update script..."
chmod +x "$APP_DIR/scripts/update_service/release_update.sh"

# Create systemd service file
echo "Creating systemd service..."
cat > /etc/systemd/system/atlantium-update.service << EOF
[Unit]
Description=Atlantium RAG Release Update Service
Documentation=https://github.com/kertser/Atlantium_LLM
After=network.target docker.service
Wants=docker.service
StartLimitIntervalSec=300
StartLimitBurst=3

[Service]
Type=simple
User=$CURRENT_USER
Group=docker
WorkingDirectory=$APP_DIR

# Environment variables
Environment="APP_DIR=$APP_DIR"
Environment="LOG_DIR=$APP_DIR/logs"
Environment="LOG_LEVEL=INFO"
Environment="DOCKER_BUILDKIT=1"
Environment="HOME=$USER_HOME"
Environment="SCRIPTS_DIR=$APP_DIR/scripts/update_service"
Environment="PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

# Docker socket access
SupplementaryGroups=docker

# Execution
ExecStartPre=/bin/mkdir -p \${LOG_DIR}/updates
ExecStart=/bin/bash $APP_DIR/scripts/update_service/release_update.sh

# Restart configuration
Restart=on-failure
RestartSec=60

# Security hardening (but allow docker and home access)
NoNewPrivileges=yes
ProtectSystem=full
ProtectHome=read-only
PrivateTmp=yes
ProtectKernelTunables=yes
ProtectKernelModules=yes
ProtectControlGroups=no  # Required for Docker
RestrictAddressFamilies=AF_UNIX AF_INET AF_INET6
RestrictNamespaces=no    # Required for Docker
RestrictRealtime=yes

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
if ! systemctl daemon-reload; then
    echo "Error: Failed to reload systemd configuration"
    exit 1
fi

if ! systemctl enable atlantium-update; then
    echo "Error: Failed to enable service"
    exit 1
fi

if ! systemctl start atlantium-update; then
    echo "Error: Failed to start service"
    echo "Checking service logs..."
    journalctl -u atlantium-update -n 50 --no-pager
    exit 1
fi

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

# Verify service can write to logs
echo "Verifying service can write to logs..."
if ! sudo -u "$CURRENT_USER" touch "$APP_DIR/logs/updates/test.log" 2>/dev/null; then
    echo "Warning: Service might have issues writing to log directory"
    echo "Please check permissions and systemd service configuration"
    # Don't exit here, just warn the user
fi
rm -f "$APP_DIR/logs/updates/test.log"

# Verify service is running correctly
if ! systemctl is-active --quiet atlantium-update; then
    echo "Warning: Service is not running. Checking logs..."
    journalctl -u atlantium-update -n 50 --no-pager
fi