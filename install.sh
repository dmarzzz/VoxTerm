#!/bin/bash
set -euo pipefail

# ── VoxTerm Installer ──────────────────────────────────────
#
# Install:    curl -fsSL https://raw.githubusercontent.com/dmarzzz/VoxTerm/main/install.sh | bash
# Specific:   curl ... | bash -s -- --version v0.1.0
# Uninstall:  curl ... | bash -s -- --uninstall

REPO="dmarzzz/VoxTerm"
REPO_URL="https://github.com/$REPO"
INSTALL_DIR="$HOME/.local/share/voxterm"
BIN_DIR="$HOME/.local/bin"
VENV_DIR="$INSTALL_DIR/.venv"
VERSION_FILE="$INSTALL_DIR/.installed-version"

# Colors
GREEN='\033[0;32m'
CYAN='\033[0;36m'
RED='\033[0;31m'
DIM='\033[2m'
BOLD='\033[1m'
RESET='\033[0m'

info()  { echo -e "${CYAN}▸${RESET} $1"; }
done_() { echo -e "${GREEN}✓${RESET} $1"; }
dim()   { echo -e "${DIM}  $1${RESET}"; }
err()   { echo -e "${RED}✗${RESET} $1"; }

# ── Parse args ────────────────────────────────────────────
REQUESTED_VERSION=""
UNINSTALL=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --version)  REQUESTED_VERSION="$2"; shift 2 ;;
        --version=*) REQUESTED_VERSION="${1#*=}"; shift ;;
        --uninstall) UNINSTALL=true; shift ;;
        --help|-h)
            echo "VoxTerm installer"
            echo ""
            echo "Usage: curl -fsSL .../install.sh | bash [-s -- OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --version VERSION   Install a specific version (e.g. v0.1.0)"
            echo "  --uninstall         Remove VoxTerm completely"
            echo "  --help              Show this help"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── Uninstall ─────────────────────────────────────────────
if $UNINSTALL; then
    echo ""
    echo -e "${BOLD}Uninstalling VoxTerm...${RESET}"
    rm -rf "$INSTALL_DIR"
    rm -f "$BIN_DIR/voxterm"
    done_ "removed $INSTALL_DIR"
    done_ "removed $BIN_DIR/voxterm"
    echo ""
    echo -e "${DIM}voice data at ~/Library/Application Support/voxterm/ was NOT removed${RESET}"
    echo -e "${DIM}to remove voice data too: rm -rf ~/Library/Application\\ Support/voxterm${RESET}"
    echo ""
    exit 0
fi

# ── Header ────────────────────────────────────────────────
echo ""
echo -e "${BOLD}VOXTERM${RESET} — local voice transcription"
echo -e "${DIM}everything runs on your machine, nothing leaves${RESET}"
echo ""

# ── Resolve version ───────────────────────────────────────
if [ -z "$REQUESTED_VERSION" ]; then
    info "checking latest release..."
    # Only look for v* tags (skip utility releases like onnx-models)
    REQUESTED_VERSION=$(curl -fsSL "https://api.github.com/repos/$REPO/releases" 2>/dev/null \
        | grep '"tag_name"' | grep '"v' | head -1 | sed 's/.*"tag_name": *"\([^"]*\)".*/\1/' || echo "")

    if [ -z "$REQUESTED_VERSION" ]; then
        REQUESTED_VERSION="main"
        dim "no releases found, using main branch"
    else
        done_ "latest release: $REQUESTED_VERSION"
    fi
fi

# ── Check if already up to date ───────────────────────────
if [ -f "$VERSION_FILE" ]; then
    INSTALLED=$(cat "$VERSION_FILE")
    if [ "$INSTALLED" = "$REQUESTED_VERSION" ]; then
        done_ "already up to date ($INSTALLED)"
        echo ""
        exit 0
    fi
    info "updating $INSTALLED → $REQUESTED_VERSION"
fi

# ── Check Python ──────────────────────────────────────────
info "checking python..."

PYTHON=""
for cmd in python3.12 python3.11 python3.10 python3.9 python3; do
    if command -v "$cmd" &>/dev/null; then
        version=$("$cmd" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "0.0")
        major=$(echo "$version" | cut -d. -f1)
        minor=$(echo "$version" | cut -d. -f2)
        if [ "$major" -ge 3 ] && [ "$minor" -ge 9 ]; then
            PYTHON="$cmd"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    err "Python 3.9+ required but not found."
    echo ""
    echo "   Install it with:"
    echo "     brew install python@3.12    (macOS)"
    echo "     sudo apt install python3    (Linux)"
    exit 1
fi

done_ "found $PYTHON ($($PYTHON --version 2>&1))"

# ── Download release ──────────────────────────────────────
info "downloading voxterm $REQUESTED_VERSION..."

if [ "$REQUESTED_VERSION" = "main" ]; then
    ARCHIVE_URL="$REPO_URL/archive/refs/heads/main.tar.gz"
else
    ARCHIVE_URL="$REPO_URL/archive/refs/tags/$REQUESTED_VERSION.tar.gz"
fi

# Download and extract to a temp dir, then swap
TMPDIR_DL=$(mktemp -d)
trap "rm -rf $TMPDIR_DL" EXIT

curl -fsSL "$ARCHIVE_URL" | tar -xz -C "$TMPDIR_DL" --strip-components=1

# Preserve venv if it exists (avoid re-downloading all deps)
if [ -d "$VENV_DIR" ]; then
    mv "$VENV_DIR" "$TMPDIR_DL/.venv"
fi

# Preserve voice data symlinks or local state
rm -rf "$INSTALL_DIR"
mv "$TMPDIR_DL" "$INSTALL_DIR"

done_ "downloaded"

# ── Create venv & install deps ────────────────────────────
info "installing dependencies..."
dim "this may take a minute on first install"

if [ ! -d "$VENV_DIR" ]; then
    "$PYTHON" -m venv "$VENV_DIR"
fi

"$VENV_DIR/bin/pip" install --quiet --upgrade pip 2>/dev/null
if [ -f "$INSTALL_DIR/pyproject.toml" ]; then
    "$VENV_DIR/bin/pip" install --quiet -e "$INSTALL_DIR"
elif [ -f "$INSTALL_DIR/requirements.txt" ]; then
    "$VENV_DIR/bin/pip" install --quiet -r "$INSTALL_DIR/requirements.txt"
else
    err "neither pyproject.toml nor requirements.txt found in $INSTALL_DIR"
    exit 1
fi

done_ "dependencies installed"

# ── Record installed version ──────────────────────────────
echo "$REQUESTED_VERSION" > "$VERSION_FILE"

# ── Create launcher ───────────────────────────────────────
info "creating voxterm command..."

mkdir -p "$BIN_DIR"
cat > "$BIN_DIR/voxterm" << 'LAUNCHER'
#!/bin/bash
INSTALL_DIR="$HOME/.local/share/voxterm"
INSTALL_URL="https://raw.githubusercontent.com/dmarzzz/VoxTerm/main/install.sh"

case "${1:-}" in
    update)
        shift
        if [ $# -gt 0 ]; then
            # voxterm update <version>  → pin to a specific tag
            exec bash -c "curl -fsSL '$INSTALL_URL' | bash -s -- --version '$1'"
        else
            exec bash -c "curl -fsSL '$INSTALL_URL' | bash"
        fi
        ;;
    uninstall)
        exec bash -c "curl -fsSL '$INSTALL_URL' | bash -s -- --uninstall"
        ;;
    version|-V)
        if [ -f "$INSTALL_DIR/.installed-version" ]; then
            cat "$INSTALL_DIR/.installed-version"
        else
            echo "unknown"
        fi
        exit 0
        ;;
esac

cd "$INSTALL_DIR"
export PYTHONWARNINGS="ignore::UserWarning"
"$INSTALL_DIR/.venv/bin/python" -m tui.app "$@"
exit 0
LAUNCHER
chmod +x "$BIN_DIR/voxterm"

done_ "installed to $BIN_DIR/voxterm"

# ── Check PATH ────────────────────────────────────────────
if ! echo "$PATH" | tr ':' '\n' | grep -q "$BIN_DIR"; then
    echo ""
    echo -e "${CYAN}▸${RESET} add this to your shell profile (~/.zshrc or ~/.bashrc):"
    echo ""
    echo "    export PATH=\"\$HOME/.local/bin:\$PATH\""
    echo ""
    echo "  then restart your terminal, or run:"
    echo ""
    echo "    source ~/.zshrc"
    echo ""
fi

# ── Done ──────────────────────────────────────────────────
echo ""
echo -e "${GREEN}${BOLD}voxterm $REQUESTED_VERSION installed!${RESET}"
echo ""
echo "  run it:      voxterm"
echo "  update:      voxterm update"
echo "  uninstall:   voxterm uninstall"
echo "  pin version: voxterm update v0.1.0"
echo "  show version: voxterm version"
echo ""
