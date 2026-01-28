#!/bin/bash

# Pocket TTS Menu Bar App Builder
# Builds the Swift menu bar application

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
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

echo "Pocket TTS Menu Bar App Builder"
echo "================================"
echo ""

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR/../PocketTTSMenuBar"

# Check if swift is available
if ! command -v swift &> /dev/null; then
    print_error "Swift is not installed"
    echo ""
    echo "Please install Xcode from the Mac App Store or"
    echo "Command Line Tools: xcode-select --install"
    echo ""
    exit 1
fi

print_success "Found Swift $(swift --version | head -n1)"

# Change to project directory
cd "$PROJECT_DIR"

# Build the app
echo ""
echo "Building release version..."
swift build -c release

if [ $? -eq 0 ]; then
    print_success "Build completed successfully"
else
    print_error "Build failed"
    exit 1
fi

# Get build output path
BUILD_PATH=".build/release/PocketTTSMenuBar"

if [ ! -f "$BUILD_PATH" ]; then
    print_error "Build executable not found at: $BUILD_PATH"
    exit 1
fi

echo ""
echo "Build output: $BUILD_PATH"
echo ""

# Ask if user wants to install
read -p "Install to /usr/local/bin? [y/N] " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "Installing to /usr/local/bin..."

    # Create directory if it doesn't exist
    sudo mkdir -p /usr/local/bin

    # Copy executable
    sudo cp "$BUILD_PATH" /usr/local/bin/PocketTTSMenuBar
    sudo chmod +x /usr/local/bin/PocketTTSMenuBar

    print_success "Installed to /usr/local/bin/PocketTTSMenuBar"
    echo ""
    echo "You can now run the app with:"
    echo "  PocketTTSMenuBar"
    echo ""
    echo "Or add it to Login Items:"
    echo "  System Settings → General → Login Items"
fi

# Ask if user wants to create app bundle
echo ""
read -p "Create application bundle? [y/N] " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "Creating application bundle..."

    APP_NAME="Pocket TTS Menu Bar.app"
    APP_DIR="$SCRIPT_DIR/../$APP_NAME"

    # Remove existing bundle
    rm -rf "$APP_DIR"

    # Create bundle structure
    mkdir -p "$APP_DIR/Contents/MacOS"
    mkdir -p "$APP_DIR/Contents/Resources"

    # Copy executable
    cp "$BUILD_PATH" "$APP_DIR/Contents/MacOS/PocketTTSMenuBar"
    chmod +x "$APP_DIR/Contents/MacOS/PocketTTSMenuBar"

    # Copy Info.plist
    cp "Sources/PocketTTSMenuBar/Resources/Info.plist" "$APP_DIR/Contents/"

    print_success "Created app bundle: $APP_DIR"
    echo ""
    echo "To install:"
    echo "  mv '$APP_DIR' ~/Applications/"
    echo ""
    echo "Then add to Login Items:"
    echo "  System Settings → General → Login Items"
fi

echo ""
print_success "Done!"
echo ""
