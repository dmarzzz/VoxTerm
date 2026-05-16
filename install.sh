#!/bin/bash
# voxterm.xyz/install.sh — evergreen shim.
#
# Always re-fetches the canonical installer from the latest GitHub release
# asset (Cache-Control: no-cache end-to-end), so this URL never serves a
# stale installer. The real install.sh lives in the repo and is uploaded
# as an asset on every release.
#
# Source: https://github.com/dmarzzz/VoxTerm/blob/main/install.sh

set -euo pipefail
INSTALL_URL="https://github.com/dmarzzz/VoxTerm/releases/latest/download/install.sh"
curl -fsSL --retry 3 --retry-delay 2 --retry-connrefused "$INSTALL_URL" | bash -s -- "$@"
