#!/bin/bash
# modl-mcp skill helper script
# Checks if modl is properly configured and running

set -e

echo "Checking modl MCP setup..."

# Check if modl is installed
if ! command -v modl &> /dev/null; then
    echo "❌ modl is not installed"
    echo ""
    echo "Install with:"
    echo "  curl -fsSL https://modl.run/install.sh | sh"
    exit 1
fi

echo "✓ modl is installed"

# Check modl version
MODL_VERSION=$(modl --version 2>/dev/null || echo "unknown")
echo "  Version: $MODL_VERSION"

# Check if modl serve is running
if curl -s http://127.0.0.1:3333/api/models > /dev/null 2>&1; then
    echo "✓ modl serve is running on port 3333"
else
    echo "⚠ modl serve is not running"
    echo ""
    echo "Start it with:"
    echo "  modl serve"
    echo ""
    echo "Or as a background service:"
    echo "  modl serve --install-service"
fi

echo ""
echo "MCP Configuration:"
echo "==================="
echo ""
echo "Add this to your MCP client config:"
echo ""
cat << 'EOF'
{
  "mcpServers": {
    "modl": {
      "command": "modl",
      "args": ["mcp"]
    }
  }
}
EOF

echo ""
echo "For Claude Desktop: ~/Library/Application Support/Claude/claude_desktop_config.json"
echo "For Cursor: Settings > MCP > Add Server"
