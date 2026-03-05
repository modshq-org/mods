#!/usr/bin/env bash
# Launch the full dev environment for UI work:
#   1. Build the Rust CLI (if needed)
#   2. Start the persistent GPU worker (keeps models in VRAM)
#   3. Start the backend API server (modl serve on :3333)
#   4. Start the Vite dev server (hot-reload on :5173)
#
# Open http://localhost:5173 to work on the UI.
# Ctrl+C stops everything.
#
# Usage: ./scripts/dev.sh [--no-worker]

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

NO_WORKER=false
for arg in "$@"; do
    case "$arg" in
        --no-worker) NO_WORKER=true ;;
    esac
done

# Track background PIDs for cleanup
PIDS=()
cleanup() {
    echo ""
    echo "→ Shutting down..."

    # Stop worker gracefully
    if [[ "$NO_WORKER" == false ]]; then
        ./target/debug/modl worker stop 2>/dev/null || true
    fi

    # Kill background processes
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done

    # Kill any stale Vite
    pkill -f "node.*vite" 2>/dev/null || true

    echo "✓ Dev environment stopped"
}
trap cleanup EXIT

# ── 1. Build CLI ──────────────────────────────────────────────
echo "→ Building CLI..."
cargo build --quiet 2>&1

# ── 2. Start worker (persistent GPU daemon) ──────────────────
if [[ "$NO_WORKER" == false ]]; then
    echo "→ Starting persistent worker..."
    ./target/debug/modl worker start 2>&1 || echo "  (worker start skipped — no GPU or runtime not ready)"
fi

# ── 3. Start backend (modl serve) ────────────────────────────
echo "→ Starting backend on :3333..."
./target/debug/modl serve --foreground --no-open &
PIDS+=($!)
sleep 1

# ── 4. Start Vite dev server ─────────────────────────────────
echo "→ Starting Vite dev server on :5173..."
pkill -f "node.*vite" 2>/dev/null || true
sleep 0.3

cd src/ui/web
npm run dev &
PIDS+=($!)
cd "$PROJECT_DIR"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  UI:      http://localhost:5173"
echo "  Backend: http://localhost:3333"
echo "  Worker:  modl worker status"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Ctrl+C to stop everything"
echo ""

# Wait for any background process to exit
wait
