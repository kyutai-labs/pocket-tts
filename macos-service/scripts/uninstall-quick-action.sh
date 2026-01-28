#!/bin/bash

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "======================================"
echo "Pocket TTS Quick Action Uninstaller"
echo "======================================"
echo

# Step 1: Remove CLI binary
echo "Step 1: Removing CLI binary..."
echo "--------------------------------------"

INSTALL_PATH="/usr/local/bin/pocket-tts-quick-action"

if [ -f "$INSTALL_PATH" ]; then
    echo "Removing $INSTALL_PATH (requires sudo)..."
    if sudo rm "$INSTALL_PATH"; then
        echo -e "${GREEN}✓${NC} Binary removed"
    else
        echo -e "${RED}✗${NC} Failed to remove binary"
        exit 1
    fi
else
    echo -e "${YELLOW}⚠${NC} Binary not found (already removed?)"
fi

# Step 2: Remove Quick Action workflow
echo
echo "Step 2: Removing Quick Action..."
echo "--------------------------------------"

WORKFLOW_DEST="$HOME/Library/Services/Read Selection with Pocket TTS.workflow"

if [ -d "$WORKFLOW_DEST" ]; then
    echo "Removing workflow from ~/Library/Services/..."
    if rm -rf "$WORKFLOW_DEST"; then
        echo -e "${GREEN}✓${NC} Workflow removed"
    else
        echo -e "${RED}✗${NC} Failed to remove workflow"
        exit 1
    fi
else
    echo -e "${YELLOW}⚠${NC} Workflow not found (already removed?)"
fi

# Note about config
echo
echo "Note: Configuration files were NOT removed."
echo "They are located at:"
echo "  ~/Library/Application Support/Pocket TTS/"
echo
echo "To remove them manually, run:"
echo "  rm -rf ~/Library/Application\ Support/Pocket\ TTS/"
echo

# Success message
echo "======================================"
echo -e "${GREEN}Uninstallation Complete!${NC}"
echo "======================================"
echo
