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

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo -e "${RED}Error: This script must be run as root${NC}"
    echo "Please run: sudo $0 $VERSION"
    exit 1
fi

# Step 1: Navigate to installation directory
echo -e "${YELLOW}[1/6] Navigating to installation directory...${NC}"
cd "$INSTALL_DIR" || { echo -e "${RED}Error: Could not access ${INSTALL_DIR}${NC}"; exit 1; }

# Step 2: Download the new version
echo -e "${YELLOW}[2/6] Downloading LMStudio ${VERSION}...${NC}"
if wget -q --show-progress "$DOWNLOAD_URL"; then
    echo -e "${GREEN}Download successful${NC}"
else
    echo -e "${RED}Error: Download failed${NC}"
    echo "URL: $DOWNLOAD_URL"
    exit 1
fi

# Step 3: Make the new AppImage executable
echo -e "${YELLOW}[3/6] Making AppImage executable...${NC}"
chmod +x "$APPIMAGE_NAME"

# Step 4: Stop the LMStudio service
echo -e "${YELLOW}[4/6] Stopping LMStudio service...${NC}"
if systemctl is-active --quiet lmstudio.service; then
    systemctl stop lmstudio.service
    echo -e "${GREEN}Service stopped${NC}"
else
    echo -e "${YELLOW}Service was not running${NC}"
fi

# Step 5: Update the symlink
echo -e "${YELLOW}[5/6] Updating symlink...${NC}"
ln -sf "$APPIMAGE_NAME" LM-Studio.AppImage
echo -e "${GREEN}Symlink updated${NC}"

# Verify symlink
echo ""
echo "Current installation:"
ls -lh "$INSTALL_DIR" | grep -E 'LM-Studio|AppImage'
echo ""

# Step 6: Restart the LMStudio service
echo -e "${YELLOW}[6/6] Starting LMStudio service...${NC}"
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
