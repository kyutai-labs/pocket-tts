#!/bin/bash

# Pocket TTS LaunchAgent Installer
# Installs a LaunchAgent that auto-starts the Pocket TTS server on login

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

echo "Pocket TTS LaunchAgent Installer"
echo "================================="
echo ""

# Step 1: Check if uv is installed
echo "Checking for uv package manager..."
if ! command -v uv &> /dev/null; then
    print_error "uv is not installed"
    echo ""
    echo "Please install uv first:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo ""
    exit 1
fi

UV_PATH=$(which uv)
print_success "Found uv at: $UV_PATH"

# Step 2: Get project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo ""
echo "Project root: $PROJECT_ROOT"

# Step 3: Verify pocket-tts is available
echo ""
echo "Verifying pocket-tts installation..."
cd "$PROJECT_ROOT"
if ! uv run pocket-tts --help &> /dev/null; then
    print_error "pocket-tts command not found"
    echo ""
    echo "Please ensure you're in the pocket-tts project directory and run:"
    echo "  uv sync"
    echo ""
    exit 1
fi
print_success "pocket-tts is available"

# Step 4: Get username
USERNAME=$(whoami)

# Step 5: Create necessary directories
echo ""
echo "Creating directories..."

LAUNCH_AGENTS_DIR="$HOME/Library/LaunchAgents"
LOGS_DIR="$HOME/Library/Logs/PocketTTS"
APP_SUPPORT_DIR="$HOME/Library/Application Support/PocketTTS"

mkdir -p "$LAUNCH_AGENTS_DIR"
print_success "LaunchAgents directory: $LAUNCH_AGENTS_DIR"

mkdir -p "$LOGS_DIR"
print_success "Logs directory: $LOGS_DIR"

mkdir -p "$APP_SUPPORT_DIR"
print_success "Application Support directory: $APP_SUPPORT_DIR"

# Step 6: Create default config if it doesn't exist
CONFIG_FILE="$APP_SUPPORT_DIR/config.json"
if [ ! -f "$CONFIG_FILE" ]; then
    echo ""
    echo "Creating default config.json..."
    cat > "$CONFIG_FILE" << EOF
{
  "selectedVoiceId": "alba",
  "selectedVoiceType": "predefined",
  "serverPort": 8765,
  "autoStartServer": true,
  "version": "1.0.0"
}
EOF
    print_success "Created default config: $CONFIG_FILE"
else
    print_warning "Config already exists: $CONFIG_FILE"
fi

# Step 7: Process plist template
echo ""
echo "Installing LaunchAgent..."

PLIST_TEMPLATE="$SCRIPT_DIR/../launchd/com.kyutai.pocket-tts.server.plist"
PLIST_TARGET="$LAUNCH_AGENTS_DIR/com.kyutai.pocket-tts.server.plist"

if [ ! -f "$PLIST_TEMPLATE" ]; then
    print_error "Template plist not found: $PLIST_TEMPLATE"
    exit 1
fi

# Replace template variables
sed -e "s|{{USERNAME}}|$USERNAME|g" \
    -e "s|{{UV_PATH}}|$UV_PATH|g" \
    -e "s|{{PROJECT_ROOT}}|$PROJECT_ROOT|g" \
    "$PLIST_TEMPLATE" > "$PLIST_TARGET"

print_success "Installed plist: $PLIST_TARGET"

# Step 8: Unload existing service (if any)
if launchctl list | grep -q "com.kyutai.pocket-tts.server"; then
    echo ""
    echo "Unloading existing service..."
    launchctl unload "$PLIST_TARGET" 2>/dev/null || true
    print_success "Unloaded existing service"
fi

# Step 9: Load the LaunchAgent
echo ""
echo "Loading LaunchAgent..."
launchctl load "$PLIST_TARGET"
print_success "LaunchAgent loaded"

# Step 10: Verify the service is running
echo ""
echo "Verifying service..."
sleep 2  # Give the service time to start

if launchctl list | grep -q "com.kyutai.pocket-tts.server"; then
    print_success "Service is running"
else
    print_error "Service failed to start"
    echo ""
    echo "Check logs at: $LOGS_DIR/server-error.log"
    exit 1
fi

# Step 11: Test server health
echo ""
echo "Testing server connection..."
MAX_ATTEMPTS=10
ATTEMPT=1

while [ $ATTEMPT -le $MAX_ATTEMPTS ]; do
    if curl -s http://localhost:8765/health > /dev/null 2>&1; then
        print_success "Server is responding at http://localhost:8765"
        break
    fi

    if [ $ATTEMPT -eq $MAX_ATTEMPTS ]; then
        print_error "Server is not responding after $MAX_ATTEMPTS attempts"
        echo ""
        echo "Check logs at:"
        echo "  $LOGS_DIR/server.log"
        echo "  $LOGS_DIR/server-error.log"
        exit 1
    fi

    echo "Waiting for server to start (attempt $ATTEMPT/$MAX_ATTEMPTS)..."
    sleep 1
    ATTEMPT=$((ATTEMPT + 1))
done

# Success!
echo ""
echo "================================="
print_success "Installation complete!"
echo ""
echo "The Pocket TTS server will now:"
echo "  • Start automatically on login"
echo "  • Restart automatically if it crashes"
echo "  • Listen on http://localhost:8765"
echo ""
echo "Logs are available at:"
echo "  $LOGS_DIR/server.log"
echo "  $LOGS_DIR/server-error.log"
echo ""
echo "To uninstall, run:"
echo "  ./uninstall-service.sh"
echo ""
