#!/bin/bash

# Pocket TTS LaunchAgent Uninstaller
# Removes the LaunchAgent that auto-starts the Pocket TTS server

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'  # No Color

# Print colored message
print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

echo "Pocket TTS LaunchAgent Uninstaller"
echo "===================================="
echo ""

PLIST_PATH="$HOME/Library/LaunchAgents/com.kyutai.pocket-tts.server.plist"

# Check if plist exists
if [ ! -f "$PLIST_PATH" ]; then
    print_warning "LaunchAgent not found at: $PLIST_PATH"
    echo ""
    echo "The service may not be installed or has already been removed."
    exit 0
fi

# Check if service is loaded
if launchctl list | grep -q "com.kyutai.pocket-tts.server"; then
    echo "Unloading service..."
    if launchctl unload "$PLIST_PATH" 2>/dev/null; then
        print_success "Service unloaded"
    else
        print_warning "Failed to unload service (it may not be running)"
    fi
else
    print_warning "Service is not currently loaded"
fi

# Remove plist file
echo ""
echo "Removing LaunchAgent plist..."
rm "$PLIST_PATH"
print_success "Removed plist: $PLIST_PATH"

# Verify service is no longer running
echo ""
echo "Verifying removal..."
sleep 1

if launchctl list | grep -q "com.kyutai.pocket-tts.server"; then
    print_error "Service is still running"
    echo ""
    echo "You may need to restart your computer for the changes to take effect."
    exit 1
else
    print_success "Service is no longer running"
fi

# Optional: Ask about removing logs and config
echo ""
echo "===================================="
print_success "Uninstallation complete!"
echo ""
echo "Note: The following directories were NOT removed:"
echo "  • ~/Library/Logs/PocketTTS/ (logs)"
echo "  • ~/Library/Application Support/PocketTTS/ (config and voices)"
echo ""
read -p "Would you like to remove these directories? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    LOGS_DIR="$HOME/Library/Logs/PocketTTS"
    APP_SUPPORT_DIR="$HOME/Library/Application Support/PocketTTS"

    if [ -d "$LOGS_DIR" ]; then
        rm -rf "$LOGS_DIR"
        print_success "Removed logs: $LOGS_DIR"
    fi

    if [ -d "$APP_SUPPORT_DIR" ]; then
        echo ""
        print_warning "About to remove all configs and custom voices!"
        echo "Location: $APP_SUPPORT_DIR"
        read -p "Are you sure? [y/N] " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf "$APP_SUPPORT_DIR"
            print_success "Removed config and voices: $APP_SUPPORT_DIR"
        else
            print_warning "Kept config and voices: $APP_SUPPORT_DIR"
        fi
    fi

    echo ""
    print_success "All data removed"
else
    echo ""
    print_warning "Logs and config preserved"
    echo ""
    echo "To remove them manually later, delete:"
    echo "  ~/Library/Logs/PocketTTS/"
    echo "  ~/Library/Application Support/PocketTTS/"
fi

echo ""
