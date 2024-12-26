# Update Service Guide

## Overview

The update service provides automated system updates by monitoring the GitHub release branch and maintaining data persistence. It integrates with the Docker-based deployment and ensures system stability during updates.

## Core Components

### Update Service Script (`release_update.sh`)
Main update script with the following functions:
```bash
# Core Functions
log() # Logging with timestamp
error_exit() # Error handling and cleanup
cleanup() # Cleanup temporary files
setup_temp() # Set up temporary directory
setup_directories() # Create required directories
check_docker() # Verify Docker status
verify_installation() # Check installation integrity
create_backup() # Create system backup
detect_gpu_configuration() # Check GPU availability
update_code() # Update from repository
update_docker() # Update Docker containers
verify_update() # Verify update success
```

### Service Configuration (`atlantium-update.service`)
```ini
[Unit]
Description=Atlantium RAG Update Service
After=network.target

[Service]
Type=simple
User=your_user
WorkingDirectory=/path/to/Atlantium_LLM
ExecStart=/path/to/Atlantium_LLM/scripts/update_service/release_update.sh
Restart=on-failure

[Install]
WantedBy=multi-user.net
```

### Installation Script (`install.sh`)
```bash
# Core Functions
validate_environment() # Check system requirements
fix_directory_permissions() # Set correct permissions
setup_logging() # Configure log rotation
install_service() # Install systemd service
configure_selinux() # Set SELinux context if enabled
```

## Update Process

### Initialization
1. Environment validation
2. Directory structure verification
3. Docker status check
4. GPU configuration detection

### Backup Process
1. Create timestamped backup
2. Store in configured backup location
3. Maintain backup rotation (keep last 5)
4. Verify backup integrity

### Update Sequence
1. Check for new releases
2. Create system backup
3. Update code from repository
4. Rebuild Docker containers
5. Verify system integrity
6. Update configuration if needed

### Verification Steps
1. Container status check
2. Service response verification
3. Log inspection
4. GPU status verification (if applicable)

## Service Management

### Installation
```bash
# Install service
cd ~/Projects/Atlantium_LLM
sudo chmod +x scripts/update_service/install.sh
sudo ./scripts/update_service/install.sh
```

### Control Commands
```bash
# Start service
sudo systemctl start atlantium-update

# Stop service
sudo systemctl stop atlantium-update

# Check status
sudo systemctl status atlantium-update

# Enable at boot
sudo systemctl enable atlantium-update

# View logs
journalctl -u atlantium-update -f
```

## Data Persistence

### Protected Data
- RAG database (embeddings, indexes)
- Uploaded documents
- System configurations
- User preferences
- Log files

### Backup Strategy
```bash
# Backup locations
RAG_Data/          # Vector database
Raw Documents/     # User documents
logs/              # System logs
.env               # Configuration
```

## Error Handling

### Common Issues
1. Network Connection Failures
2. Docker Service Issues
3. Permission Problems
4. GPU Configuration Errors
5. Backup Failures

### Recovery Procedures
```bash
# Restore from backup
cd ~/Projects/Atlantium_LLM/backups
tar -xzf backup_YYYYMMDD.tar.gz -C /path/to/restore

# Fix permissions
sudo chown -R your_user:your_group /path/to/Atlantium_LLM

# Restart service
sudo systemctl restart atlantium-update
```

## Security Considerations

### Authentication
- Service runs with limited privileges
- Uses separate service account
- Protected access to Docker daemon

### File Permissions
- Strict directory permissions
- Controlled access to sensitive files
- SELinux context management

## Monitoring

### Log Files
```bash
# System logs
/var/log/atlantium-update.log

# Application logs
~/Projects/Atlantium_LLM/logs/system.log

# Docker logs
docker logs -f atlantium_llm-web-app-1
```

### Health Checks
1. Container status monitoring
2. Service response verification
3. Resource usage tracking
4. Update status monitoring

## Related Documentation

- [Installation Guide](../docs/installation.md)
- [Technical Reference](../docs/technical-reference.md)
- [Frontend Documentation](../docs/frontend.md)
- [Models Documentation](../docs/models.md)