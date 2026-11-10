#!/bin/bash
# Download released Pianoid checkpoints and verify integrity.
#
# Pulls a single ~188 MB zip from the repository's GitHub Release, unpacks it
# under checkpoints/release/, and verifies SHA256 checksums.
#
# Usage:
#   bash scripts/download_checkpoints.sh
set -e
cd "$(dirname "$0")/.."

URL="https://github.com/joonhyungbae/pianoid/releases/download/v1.0.0/pianoid_checkpoints.zip"
DEST="checkpoints/release"
ZIP="checkpoints/pianoid.zip"

mkdir -p "$DEST"

echo ">>> Downloading checkpoints (~188 MB) from GitHub Releases..."
if command -v curl >/dev/null 2>&1; then
    curl -L --fail -o "$ZIP" "$URL"
elif command -v wget >/dev/null 2>&1; then
    wget -O "$ZIP" "$URL"
else
    echo "Error: need curl or wget" >&2; exit 1
fi

echo ">>> Unpacking into $DEST/"
unzip -o -q "$ZIP" -d "$DEST"
rm -f "$ZIP"

echo ">>> Verifying SHA256 checksums..."
( cd "$DEST" && sha256sum -c SHA256SUMS.txt )

echo ""
echo "Done. Released checkpoints under $DEST/:"
ls -1 "$DEST"
