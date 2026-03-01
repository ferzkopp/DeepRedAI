#!/bin/bash

# LMStudio Update Script
# Usage: sudo ./update_lmstudio.sh <version>
# Example: sudo ./update_lmstudio.sh 0.3.35-1

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if version argument is provided
if [ -z "$1" ]; then
    echo -e "${RED}Error: Version number required${NC}"
    echo "Usage: sudo $0 <version>"
    echo "Example: sudo $0 0.3.35-1"
    exit 1
fi

VERSION="$1"
INSTALL_DIR="/opt/lm-studio"
APPIMAGE_NAME="LM-Studio-${VERSION}-x64.AppImage"
DOWNLOAD_URL="https://installers.lmstudio.ai/linux/x64/${VERSION}/${APPIMAGE_NAME}"

echo -e "${GREEN}=== LMStudio Update Script ===${NC}"
echo "Version: ${VERSION}"
echo "Installation directory: ${INSTALL_DIR}"
echo ""

# Configuration
CLI_SOURCE="/root/.lmstudio/bin/lms"
CLI_DEST="${INSTALL_DIR}/bin/lms"
CLI_WAIT_TIME=30  # Seconds to wait for CLI binary to be updated
PASSKEY_SOURCE="/root/.lmstudio/.internal/lms-key-2"
CLI_USERS=("wiki" "aschiffler")  # Users who need CLI access

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo -e "${RED}Error: This script must be run as root${NC}"
    echo "Please run: sudo $0 $VERSION"
    exit 1
fi

# Step 1: Navigate to installation directory
echo -e "${YELLOW}[1/8] Navigating to installation directory...${NC}"
cd "$INSTALL_DIR" || { echo -e "${RED}Error: Could not access ${INSTALL_DIR}${NC}"; exit 1; }

# Step 2: Download the new version
echo -e "${YELLOW}[2/8] Downloading LMStudio ${VERSION}...${NC}"
if wget -q --show-progress "$DOWNLOAD_URL"; then
    echo -e "${GREEN}Download successful${NC}"
else
    echo -e "${RED}Error: Download failed${NC}"
    echo "URL: $DOWNLOAD_URL"
    exit 1
fi

# Step 3: Make the new AppImage executable
echo -e "${YELLOW}[3/8] Making AppImage executable...${NC}"
chmod +x "$APPIMAGE_NAME"

# Step 4: Stop the LMStudio service
echo -e "${YELLOW}[4/8] Stopping LMStudio service...${NC}"
if systemctl is-active --quiet lmstudio.service; then
    systemctl stop lmstudio.service
    echo -e "${GREEN}Service stopped${NC}"
else
    echo -e "${YELLOW}Service was not running${NC}"
fi

# Step 5: Update the symlink
echo -e "${YELLOW}[5/8] Updating symlink...${NC}"
ln -sf "$APPIMAGE_NAME" LM-Studio.AppImage
echo -e "${GREEN}Symlink updated${NC}"

# Verify symlink
echo ""
echo "Current installation:"
ls -lh "$INSTALL_DIR" | grep -E 'LM-Studio|AppImage'
echo ""

# Step 6: Restart the LMStudio service
echo -e "${YELLOW}[6/8] Starting LMStudio service...${NC}"
systemctl start lmstudio.service

# Wait a moment for service to start
sleep 2

# Verify service status
if systemctl is-active --quiet lmstudio.service; then
    echo -e "${GREEN}Service started successfully${NC}"
else
    echo -e "${RED}Warning: Service may have failed to start${NC}"
    echo "Check status with: systemctl status lmstudio.service"
fi

# Step 7: Wait for CLI binary to be updated
echo ""
echo -e "${YELLOW}[7/8] Waiting ${CLI_WAIT_TIME} seconds for LMStudio to update CLI binary...${NC}"
echo "    (LMStudio updates the CLI on first launch after upgrade)"
for i in $(seq $CLI_WAIT_TIME -1 1); do
    printf "\r    Waiting: %2d seconds remaining..." "$i"
    sleep 1
done
echo -e "\r    ${GREEN}Wait complete${NC}                        "

# Step 8: Update CLI for multi-user access
echo -e "${YELLOW}[8/9] Updating CLI for multi-user access...${NC}"
if [ -f "$CLI_SOURCE" ]; then
    # Create bin directory if it doesn't exist
    mkdir -p "$(dirname "$CLI_DEST")"
    
    # Copy CLI binary
    cp "$CLI_SOURCE" "$CLI_DEST"
    chmod 755 "$CLI_DEST"
    
    # Verify the copy
    if [ -x "$CLI_DEST" ]; then
        echo -e "${GREEN}CLI updated successfully: ${CLI_DEST}${NC}"
    else
        echo -e "${RED}Warning: CLI copy may have failed${NC}"
    fi
else
    echo -e "${YELLOW}Warning: CLI source not found at ${CLI_SOURCE}${NC}"
    echo "    The CLI may not have been updated yet. You can manually copy it later:"
    echo "    sudo cp ${CLI_SOURCE} ${CLI_DEST}"
    echo "    sudo chmod 755 ${CLI_DEST}"
fi

# Step 9: Sync passkey for non-root users
echo -e "${YELLOW}[9/9] Syncing authentication passkey for non-root users...${NC}"
if [ -f "$PASSKEY_SOURCE" ]; then
    for username in "${CLI_USERS[@]}"; do
        # Check if user exists and get their home directory
        if id "$username" &>/dev/null; then
            # Get the actual home directory from passwd database
            user_home=$(getent passwd "$username" | cut -d: -f6)
            user_lmstudio_dir="${user_home}/.lmstudio/.internal"
            
            # Create the .lmstudio/.internal directory
            mkdir -p "$user_lmstudio_dir"
            
            # Copy the passkey file
            cp "$PASSKEY_SOURCE" "${user_lmstudio_dir}/lms-key-2"
            
            # Set proper ownership
            chown -R "${username}:${username}" "${user_home}/.lmstudio"
            chmod 700 "${user_home}/.lmstudio"
            chmod 700 "$user_lmstudio_dir"
            chmod 600 "${user_lmstudio_dir}/lms-key-2"
            
            echo -e "  ${GREEN}✓ Passkey synced for user: ${username} (${user_home})${NC}"
        else
            echo -e "  ${YELLOW}⚠ User not found, skipping: ${username}${NC}"
        fi
    done
else
    echo -e "${YELLOW}Warning: Passkey file not found at ${PASSKEY_SOURCE}${NC}"
    echo "    Non-root users will need to use 'sudo' with the lms CLI"
fi

echo ""
echo -e "${GREEN}=== Update Complete ===${NC}"
echo "LMStudio has been updated to version ${VERSION}"
echo ""
echo "To verify the service is running:"
echo "  sudo systemctl status lmstudio.service"
echo ""
echo "To view logs:"
echo "  journalctl -u lmstudio.service -f"
echo ""
echo "To remove old versions (after testing):"
echo "  ls -lh ${INSTALL_DIR}"
echo "  sudo rm ${INSTALL_DIR}/LM-Studio-<old-version>-x64.AppImage"
