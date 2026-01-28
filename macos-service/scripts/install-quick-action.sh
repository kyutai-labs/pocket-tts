#!/bin/bash

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "======================================"
echo "Pocket TTS Quick Action Installer"
echo "======================================"
echo

# Get the script's directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
QUICK_ACTION_DIR="$PROJECT_ROOT/PocketTTSQuickAction"
WORKFLOW_DIR="$PROJECT_ROOT/quick-actions"

# Check if Swift is installed
if ! command -v swift &> /dev/null; then
    echo -e "${RED}Error: Swift is not installed${NC}"
    echo "Please install Xcode or Swift from https://developer.apple.com/xcode/"
    exit 1
fi

echo -e "${GREEN}✓${NC} Swift found: $(swift --version | head -n1)"

# Step 1: Build the Swift CLI
echo
echo "Step 1: Building Swift CLI..."
echo "--------------------------------------"

cd "$QUICK_ACTION_DIR"

# Clean previous builds
echo "Cleaning previous builds..."
swift package clean 2>/dev/null || true

# Build in release mode
echo "Building in release mode..."
if swift build -c release; then
    echo -e "${GREEN}✓${NC} Build successful"
else
    echo -e "${RED}✗${NC} Build failed"
    exit 1
fi

# Step 2: Install CLI binary
echo
echo "Step 2: Installing CLI binary..."
echo "--------------------------------------"

BINARY_PATH="$QUICK_ACTION_DIR/.build/release/pocket-tts-quick-action"
INSTALL_PATH="/usr/local/bin/pocket-tts-quick-action"

if [ ! -f "$BINARY_PATH" ]; then
    echo -e "${RED}Error: Binary not found at $BINARY_PATH${NC}"
    exit 1
fi

# Check if /usr/local/bin exists, create if not
if [ ! -d "/usr/local/bin" ]; then
    echo "Creating /usr/local/bin directory..."
    sudo mkdir -p /usr/local/bin
fi

# Copy binary (requires sudo)
echo "Installing binary to $INSTALL_PATH (requires sudo)..."
if sudo cp "$BINARY_PATH" "$INSTALL_PATH"; then
    sudo chmod +x "$INSTALL_PATH"
    echo -e "${GREEN}✓${NC} Binary installed"
else
    echo -e "${RED}✗${NC} Failed to install binary"
    exit 1
fi

# Verify installation
if [ -x "$INSTALL_PATH" ]; then
    echo -e "${GREEN}✓${NC} Binary is executable"
else
    echo -e "${RED}✗${NC} Binary is not executable"
    exit 1
fi

# Step 3: Install Quick Action workflow
echo
echo "Step 3: Installing Quick Action..."
echo "--------------------------------------"

WORKFLOW_SOURCE="$WORKFLOW_DIR/Read Selection with Pocket TTS.workflow"
WORKFLOW_DEST="$HOME/Library/Services/Read Selection with Pocket TTS.workflow"

if [ ! -d "$WORKFLOW_SOURCE" ]; then
    echo -e "${RED}Error: Workflow not found at $WORKFLOW_SOURCE${NC}"
    exit 1
fi

# Remove existing workflow if present
if [ -d "$WORKFLOW_DEST" ]; then
    echo "Removing existing workflow..."
    rm -rf "$WORKFLOW_DEST"
fi

# Copy workflow
echo "Copying workflow to ~/Library/Services/..."
if cp -r "$WORKFLOW_SOURCE" "$WORKFLOW_DEST"; then
    echo -e "${GREEN}✓${NC} Workflow installed"
else
    echo -e "${RED}✗${NC} Failed to install workflow"
    exit 1
fi

# Step 4: Create config directory if needed
echo
echo "Step 4: Setting up configuration..."
echo "--------------------------------------"

# Use pocket-tts-electron directory to match Electron app and Menu Bar app
CONFIG_DIR="$HOME/Library/Application Support/pocket-tts-electron"
CONFIG_FILE="$CONFIG_DIR/config.json"

if [ ! -d "$CONFIG_DIR" ]; then
    echo "Creating config directory..."
    mkdir -p "$CONFIG_DIR"
    echo -e "${GREEN}✓${NC} Config directory created"
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Creating default config.json..."
    cat > "$CONFIG_FILE" << 'EOF'
{
  "selectedVoiceId": "alba",
  "selectedVoiceType": "predefined",
  "serverPort": 8765,
  "autoStartServer": true,
  "version": "1.0.0"
}
EOF
    echo -e "${GREEN}✓${NC} Default config created"
else
    echo -e "${GREEN}✓${NC} Config file already exists (shared with Electron/Menu Bar apps)"
fi

# Step 5: Verify server is running
echo
echo "Step 5: Checking TTS server..."
echo "--------------------------------------"

if curl -s http://localhost:8765/health > /dev/null 2>&1; then
    echo -e "${GREEN}✓${NC} TTS server is running"
else
    echo -e "${YELLOW}⚠${NC} TTS server is not running"
    echo "Please start the server with:"
    echo "  cd $PROJECT_ROOT/.."
    echo "  uv run pocket-tts serve --port 8765"
    echo
    echo "Or install and start the LaunchAgent:"
    echo "  $SCRIPT_DIR/install-service.sh"
fi

# Success message
echo
echo "======================================"
echo -e "${GREEN}Installation Complete!${NC}"
echo "======================================"
echo
echo "Next steps:"
echo "1. Enable the Quick Action:"
echo "   System Settings > Keyboard > Shortcuts > Services"
echo "   → Find 'Read Selection with Pocket TTS'"
echo "   → Check the box to enable it"
echo "   → Optional: Assign a keyboard shortcut (e.g., ⌥⌘R)"
echo
echo "2. Make sure the TTS server is running:"
echo "   - Use the LaunchAgent (auto-start on login)"
echo "   - Or run manually: uv run pocket-tts serve"
echo
echo "3. Test it:"
echo "   - Select text anywhere on your Mac"
echo "   - Right-click → Services → 'Read Selection with Pocket TTS'"
echo "   - Or use your keyboard shortcut"
echo
echo "Enjoy your new Quick Action!"
echo
