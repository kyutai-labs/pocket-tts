#!/bin/bash

# Pocket TTS macOS Service - Development Testing Script
# This script rebuilds and installs all macOS service components

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'  # No Color

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

print_header() {
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
}

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}\")\" && pwd)"
MACOS_SERVICE_DIR="$(dirname "$SCRIPT_DIR")"

print_header "Pocket TTS macOS Service - Dev Test"

# Step 1: Kill existing instances
print_header "Step 1: Cleaning Up Existing Processes"

# Kill menu bar app
if pgrep -x "PocketTTSMenuBar" > /dev/null; then
    print_info "Killing existing PocketTTSMenuBar instances..."
    killall PocketTTSMenuBar 2>/dev/null || true
    sleep 1
    print_success "Menu bar app terminated"
else
    print_info "No running menu bar app found"
fi

# Kill Quick Action CLI
if pgrep -x "pocket-tts-quick-action" > /dev/null; then
    print_info "Killing existing Quick Action instances..."
    killall pocket-tts-quick-action 2>/dev/null || true
    sleep 1
    print_success "Quick Action CLI terminated"
else
    print_info "No running Quick Action CLI found"
fi

# Kill Automator workflow services
WORKFLOW_COUNT=$(pgrep -lf "Automator.*Pocket TTS" | wc -l)
if [ "$WORKFLOW_COUNT" -gt 0 ]; then
    print_info "Killing $WORKFLOW_COUNT Automator workflow instances..."
    pkill -f "Automator.*Pocket TTS" 2>/dev/null || true
    sleep 1
    print_success "Automator workflows terminated"
else
    print_info "No running Automator workflows found"
fi

# Step 2: Build Menu Bar App
print_header "Step 2: Building Menu Bar App"

cd "$MACOS_SERVICE_DIR/PocketTTSMenuBar"

if [ -f "/Users/system-backup/bin/xcodebuild-clean" ]; then
    print_info "Building PocketTTSMenuBar with xcodebuild-clean..."
    /Users/system-backup/bin/xcodebuild-clean -scheme PocketTTSMenuBar -destination 'platform=macOS' -configuration Debug build
    print_success "Menu bar app built successfully"
else
    print_warning "xcodebuild-clean not found, using swift build..."
    swift build -c debug
    print_success "Menu bar app built successfully (using swift build)"
fi

# Step 3: Build Quick Action CLI
print_header "Step 3: Building Quick Action CLI"

cd "$MACOS_SERVICE_DIR/PocketTTSQuickAction"

if [ -f "/Users/system-backup/bin/xcodebuild-clean" ]; then
    print_info "Building pocket-tts-quick-action with xcodebuild-clean..."
    /Users/system-backup/bin/xcodebuild-clean -scheme pocket-tts-quick-action -destination 'platform=macOS' -configuration Release build

    # Find the built binary in DerivedData
    DERIVED_DATA_PATH="$HOME/Library/Developer/Xcode/DerivedData"
    QUICK_ACTION_BINARY=$(find "$DERIVED_DATA_PATH" -name "pocket-tts-quick-action" -path "*/Release/*" -type f -perm +111 | head -n 1)

    if [ -n "$QUICK_ACTION_BINARY" ]; then
        BUILD_PATH="$QUICK_ACTION_BINARY"
    else
        print_error "Could not find built binary in DerivedData"
        exit 1
    fi

    print_success "Quick Action CLI built successfully"
else
    print_warning "xcodebuild-clean not found, using swift build..."
    swift build -c release
    BUILD_PATH=".build/release/pocket-tts-quick-action"
    print_success "Quick Action CLI built successfully (using swift build)"
fi

# Step 4: Install Quick Action CLI
print_header "Step 4: Installing Quick Action CLI"

if [ ! -f "$BUILD_PATH" ]; then
    print_error "Build executable not found at: $BUILD_PATH"
    exit 1
fi

print_info "Installing to /usr/local/bin/pocket-tts-quick-action..."
sudo mkdir -p /usr/local/bin
sudo cp "$BUILD_PATH" /usr/local/bin/pocket-tts-quick-action
sudo chmod +x /usr/local/bin/pocket-tts-quick-action
print_success "CLI installed to /usr/local/bin/pocket-tts-quick-action"

# Step 5: Install Quick Action Workflow
print_header "Step 5: Installing Quick Action Workflow"

WORKFLOW_SOURCE="$MACOS_SERVICE_DIR/quick-actions/Read Selection with Pocket TTS.workflow"
WORKFLOW_DEST="$HOME/Library/Services/Read Selection with Pocket TTS.workflow"

if [ -d "$WORKFLOW_SOURCE" ]; then
    print_info "Installing Automator workflow..."

    # Remove existing workflow if present
    if [ -d "$WORKFLOW_DEST" ]; then
        rm -rf "$WORKFLOW_DEST"
        print_info "Removed existing workflow"
    fi

    cp -r "$WORKFLOW_SOURCE" "$WORKFLOW_DEST"
    print_success "Workflow installed to ~/Library/Services/"

    print_info "Restarting pbs to refresh Services menu..."
    /System/Library/CoreServices/pbs -flush 2>/dev/null || true
    killall pbs 2>/dev/null || true
    sleep 1
    print_success "Services menu refreshed"
else
    print_warning "Workflow not found at: $WORKFLOW_SOURCE"
fi

# Step 6: Open Menu Bar App in Xcode
print_header "Step 6: Opening Menu Bar App in Xcode"

cd "$MACOS_SERVICE_DIR/PocketTTSMenuBar"
print_info "Opening Package.swift in Xcode..."
open Package.swift
print_success "Xcode opened"

# Summary
print_header "Setup Complete!"

echo ""
echo "Next steps:"
echo ""
echo "1. Menu Bar App:"
echo "   - Press ⌘R in Xcode to run"
echo "   - Click the microphone icon in menu bar"
echo "   - Menu should appear with voice options"
echo ""
echo "2. Quick Action:"
echo "   - Enable in: System Settings → Keyboard → Shortcuts → Services"
echo "   - Look for 'Read Selection with Pocket TTS'"
echo "   - Assign keyboard shortcut (optional, e.g., ⌥⌘R)"
echo ""
echo "3. Test Quick Action:"
echo "   - Select text in any app"
echo "   - Right-click → Services → 'Read Selection with Pocket TTS'"
echo "   - Audio should play completely without cutting off"
echo ""
echo "4. Check logs:"
echo "   - Xcode console for menu bar app debug output"
echo "   - Terminal for Quick Action output"
echo ""

print_success "All components built and installed successfully!"
echo ""
