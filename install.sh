#!/bin/sh
# mods installer — downloads the latest release binary for your platform
# Usage: curl -fsSL https://raw.githubusercontent.com/modshq/mods/main/install.sh | sh

set -e

REPO="pedropaf/mods"  # TODO: change to modshq/mods after org transfer
INSTALL_DIR="${MODS_INSTALL_DIR:-/usr/local/bin}"

# Detect OS and architecture
OS="$(uname -s)"
ARCH="$(uname -m)"

case "$OS" in
    Linux)
        case "$ARCH" in
            x86_64)  TARGET="x86_64-unknown-linux-gnu" ;;
            aarch64) TARGET="aarch64-unknown-linux-gnu" ;;
            *)       echo "Unsupported architecture: $ARCH"; exit 1 ;;
        esac
        ;;
    Darwin)
        case "$ARCH" in
            x86_64)  TARGET="x86_64-apple-darwin" ;;
            arm64)   TARGET="aarch64-apple-darwin" ;;
            *)       echo "Unsupported architecture: $ARCH"; exit 1 ;;
        esac
        ;;
    *)
        echo "Unsupported OS: $OS (use Windows installer or cargo install mods)"
        exit 1
        ;;
esac

# Get latest release tag
echo "Detecting latest version..."
LATEST=$(curl -fsSL "https://api.github.com/repos/${REPO}/releases/latest" | grep '"tag_name"' | head -1 | cut -d'"' -f4)

if [ -z "$LATEST" ]; then
    echo "Could not determine latest version. Install with: cargo install mods"
    exit 1
fi

echo "Installing mods ${LATEST} for ${TARGET}..."

# Download
URL="https://github.com/${REPO}/releases/download/${LATEST}/mods-${LATEST}-${TARGET}.tar.gz"
TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT

curl -fsSL "$URL" -o "$TMPDIR/mods.tar.gz"
tar xzf "$TMPDIR/mods.tar.gz" -C "$TMPDIR"

# Install
if [ -w "$INSTALL_DIR" ]; then
    mv "$TMPDIR/mods" "$INSTALL_DIR/mods"
else
    echo "Installing to $INSTALL_DIR (requires sudo)..."
    sudo mv "$TMPDIR/mods" "$INSTALL_DIR/mods"
fi

chmod +x "$INSTALL_DIR/mods"

echo ""
echo "✓ mods ${LATEST} installed to ${INSTALL_DIR}/mods"
echo ""
echo "Get started:"
echo "  mods init       # Configure your setup"
echo "  mods install flux-dev   # Install a model"
echo ""
