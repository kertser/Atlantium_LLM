#!/bin/bash

# install.sh
#
# Purpose: Install and configure Atlantium RAG update service
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
    MIN_SPACE_MB=6000 # 6GB
    AVAILABLE_SPACE=$(df -m "$APP_DIR" | awk 'NR==2 {print $4}')
    if [ "$AVAILABLE_SPACE" -lt "$MIN_SPACE_MB" ]; then
        echo "Error: Insufficient disk space. Need at least ${MIN_SPACE_MB}MB"
        exit 1
    fi
}

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "Please run with sudo"
    exit 1
fi

# Call the validation function
validate_environment

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
    sudo usermod -aG docker "$CURRENT_USER"
fi

# Create project directory structure and set permissions
echo "Creating directory structure and setting permissions..."
mkdir -p "$APP_DIR/scripts/update_service"
mkdir -p "$APP_DIR/logs/updates"
mkdir -p "$APP_DIR/backups"

# Set proper directory ownership and permissions
echo "Setting directory permissions..."
sudo chown -R "$CURRENT_USER:$CURRENT_USER" "$APP_DIR"
find "$APP_DIR" -type d -exec chmod 775 {} \;
find "$APP_DIR" -type f -exec chmod 664 {} \;
find "$APP_DIR/scripts" -type f -name "*.sh" -exec chmod +x {} \;

# Ensure specific permissions for logs directory
find "$APP_DIR/logs" -type d -exec chmod 775 {} \;
find "$APP_DIR/logs" -type f -exec chmod 664 {} \;

# Create log file with proper permissions if it doesn't exist
touch "$APP_DIR/logs/updates/update.log"
sudo chown "$CURRENT_USER:$CURRENT_USER" "$APP_DIR/logs/updates/update.log"
sudo chmod 664 "$APP_DIR/logs/updates/update.log"

# Set ACL permissions if available
if command -v setfacl >/dev/null 2>&1; then
    echo "Setting ACL permissions..."
    setfacl -R -m u:"$CURRENT_USER":rwx "$APP_DIR/logs"
    setfacl -R -m u:"$CURRENT_USER":rwx "$APP_DIR/backups"
    setfacl -R -d -m u:"$CURRENT_USER":rwx "$APP_DIR/logs"
    setfacl -R -d -m u:"$CURRENT_USER":rwx "$APP_DIR/backups"
fi

# Verify permissions
echo "Verifying permissions..."
if ! sudo -u "$CURRENT_USER" test -w "$APP_DIR/logs/updates/update.log"; then
    echo "Error: Log file is not writable by $CURRENT_USER"
    exit 1
fi

# Install the update script
echo "Installing update script..."
sudo chmod +x "$APP_DIR/scripts/update_service/release_update.sh"

# Create systemd service file
echo "Creating systemd service..."
sudo cp "$APP_DIR"/scripts/update_service/atlantium-update.service /etc/systemd/system/

# Setup log rotation
echo "Configuring log rotation..."
cat > /etc/logrotate.d/atlantium-update << EOF
$APP_DIR/logs/updates/update.log {
    daily
    rotate 7
    compress
    missingok
    notifempty
    create 0664 $CURRENT_USER $CURRENT_USER
}
EOF

# Set SELinux context if SELinux is enabled
if command -v selinuxenabled >/dev/null 2>&1 && selinuxenabled; then
    echo "Setting SELinux context..."
    chcon -R -t container_file_t "$APP_DIR/logs"
    chcon -R -t container_file_t "$APP_DIR/backups"
    semanage fcontext -a -t container_file_t "$APP_DIR/logs(/.*)?"
    semanage fcontext -a -t container_file_t "$APP_DIR/backups(/.*)?"
    restorecon -R "$APP_DIR"
fi

# Reload and restart service
echo "Starting service..."
systemctl daemon-reload

# Stop the service if it's running
systemctl stop atlantium-update || true

# Clear any failed status
systemctl reset-failed atlantium-update || true

if ! systemctl enable atlantium-update; then
    echo "Error: Failed to enable service"
    exit 1
fi

if ! systemctl start atlantium-update; then
    echo "Error: Failed to start service"
    echo "Checking logs..."
    journalctl -u atlantium-update -n 50 --no-pager
    exit 1
fi

# Wait for service to stabilize
sleep 2

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

# Final verification
echo "Performing final verification..."
if ! systemctl is-active --quiet atlantium-update; then
    echo "Warning: Service is not running. Checking logs..."
    journalctl -u atlantium-update -n 50 --no-pager
else
    echo "Service is running correctly."
fi