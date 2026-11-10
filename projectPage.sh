#!/usr/bin/env bash
set -euo pipefail

# Pianoid project page launcher.
#
#   ./projectPage.sh            start the dev server (default)
#   ./projectPage.sh dev        same as above
#   ./projectPage.sh build      production build into projectPage/dist
#   ./projectPage.sh preview    build, then serve the built bundle
#   ./projectPage.sh lint       run eslint
#   ./projectPage.sh stop       stop whatever is serving PORT
#
# Environment:
#   PORT=5173     port to bind (default 5173)
#   HOST=0.0.0.0  interface to bind (default 0.0.0.0, so a remote box is reachable)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAGE_DIR="$ROOT_DIR/projectPage"
CMD="${1:-dev}"
PORT="${PORT:-5173}"
HOST="${HOST:-0.0.0.0}"

if [ ! -d "$PAGE_DIR" ]; then
  echo "Error: projectPage directory not found at $PAGE_DIR"
  exit 1
fi

# PIDs listening on $PORT, newline separated, empty if the port is free.
port_pids() {
  if command -v lsof >/dev/null 2>&1; then
    lsof -t -i "TCP:$PORT" -s TCP:LISTEN 2>/dev/null || true
  elif command -v ss >/dev/null 2>&1; then
    ss -ltnp "sport = :$PORT" 2>/dev/null \
      | grep -o 'pid=[0-9]*' | cut -d= -f2 | sort -u || true
  elif command -v fuser >/dev/null 2>&1; then
    fuser -n tcp "$PORT" 2>/dev/null | tr -s ' ' '\n' | grep -E '^[0-9]+$' || true
  fi
}

# Vite binds with --strictPort so the URL stays predictable. Explain the clash
# instead of letting vite fail with a bare "Port is already in use".
require_free_port() {
  local pids
  pids="$(port_pids)"
  [ -z "$pids" ] && return 0

  echo "Error: port $PORT is already in use by:"
  for pid in $pids; do
    echo "  PID $pid  $(ps -p "$pid" -o args= 2>/dev/null | cut -c1-90)"
  done
  echo
  echo "Fix it with either:"
  echo "  ./projectPage.sh stop        # stop the process holding $PORT"
  echo "  PORT=5174 ./projectPage.sh   # use a different port"
  exit 1
}

cd "$PAGE_DIR"

if ! command -v npm >/dev/null 2>&1; then
  echo "Error: npm not found. Install Node.js 20+ first."
  exit 1
fi

NODE_MAJOR="$(node -p 'process.versions.node.split(".")[0]' 2>/dev/null || echo 0)"
if [ "$NODE_MAJOR" -lt 20 ]; then
  echo "Error: Node.js 20+ required, found $(node -v 2>/dev/null || echo 'none')."
  exit 1
fi

if [ ! -d node_modules ]; then
  echo "Installing dependencies (first run)..."
  npm install
fi

# The showcase clips (~68 MB of .bin/.mp3) are gitignored. Without them the page
# loads but every viewer panel stays empty, which is a confusing failure mode.
CLIP_COUNT="$(find public/data -mindepth 2 -name '*.bin' 2>/dev/null | wc -l | tr -d ' ')"
if [ "$CLIP_COUNT" -eq 0 ]; then
  echo "Warning: no clip data found under projectPage/public/data/."
  echo "         The page will render but the 3D panels will be empty."
  echo "         Restore it with: git archive origin/gh-pages data | tar -x -C public/"
  echo
fi

case "$CMD" in
  dev)
    require_free_port
    echo "Pianoid project page -> http://localhost:$PORT/"
    echo "($CLIP_COUNT clip files available; Ctrl-C to stop)"
    echo
    exec npx vite --host "$HOST" --port "$PORT" --strictPort
    ;;
  build)
    exec npm run build
    ;;
  preview)
    require_free_port
    npm run build
    echo "Preview -> http://localhost:$PORT/"
    exec npx vite preview --host "$HOST" --port "$PORT" --strictPort
    ;;
  lint)
    exec npm run lint
    ;;
  stop)
    PIDS="$(port_pids)"
    if [ -z "$PIDS" ]; then
      echo "Nothing is listening on port $PORT."
      exit 0
    fi
    for pid in $PIDS; do
      echo "Stopping PID $pid ($(ps -p "$pid" -o comm= 2>/dev/null))"
      kill "$pid" 2>/dev/null || true
    done
    sleep 1
    if [ -n "$(port_pids)" ]; then
      echo "Still alive, sending SIGKILL."
      for pid in $(port_pids); do kill -9 "$pid" 2>/dev/null || true; done
    fi
    echo "Port $PORT is free."
    ;;
  *)
    echo "Usage: ./projectPage.sh [dev|build|preview|lint|stop]"
    exit 1
    ;;
esac
