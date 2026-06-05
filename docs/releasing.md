# Releasing VoxTerm

## How to cut a release

1. **Bump the version** in `config.py`:
   ```python
   VERSION = "0.1.0"
   ```

2. **Commit and tag**:
   ```bash
   git commit -am "Release v0.1.0"
   git tag v0.1.0
   ```

3. **Push**:
   ```bash
   git push && git push --tags
   ```

That's it. GitHub Actions handles the rest — creates the GitHub Release with auto-generated release notes and install instructions.

## What happens automatically

- The `release.yml` workflow triggers on any `v*` tag push
- It verifies that `config.py VERSION` matches the tag (prevents mismatches)
- Creates a GitHub Release at `github.com/dmarzzz/VoxTerm/releases`
- The install script (`install.sh`) fetches the latest release tag automatically

## How users install

```bash
# Latest stable release
curl -fsSL https://raw.githubusercontent.com/dmarzzz/VoxTerm/main/install.sh | bash

# Specific version
curl -fsSL https://raw.githubusercontent.com/dmarzzz/VoxTerm/main/install.sh | bash -s -- --version v0.1.0

# Uninstall
curl -fsSL https://raw.githubusercontent.com/dmarzzz/VoxTerm/main/install.sh | bash -s -- --uninstall
```

## Version format

We use semantic versioning: `MAJOR.MINOR.PATCH`

- **PATCH** (0.0.1 → 0.0.2): bug fixes, no new features
- **MINOR** (0.1.0 → 0.2.0): new features, backward compatible
- **MAJOR** (0.x → 1.0): breaking changes, major milestones

The version lives in one place: `config.py VERSION`. It's used by:
- The TUI header (`VOXTERM v0.1.0`)
- mDNS peer discovery (peers see each other's version)
- The install script (tracks what's installed at `~/.local/share/voxterm/.installed-version`)
- GitHub Release tags

## Checklist before releasing

- [ ] All tests pass (`python3 -m pytest tests/`)
- [ ] Tested locally (`python3 -m tui.app`)
- [ ] `VERSION` in `config.py` is updated
- [ ] No uncommitted changes
- [ ] Tag matches VERSION (e.g., `VERSION = "0.1.0"` → tag `v0.1.0`)

## Install script details

The install script at `install.sh`:
- Lives on `main` branch (users always curl from main)
- Queries GitHub API for latest release tag
- Downloads the release tarball (not git clone — faster, no .git directory)
- Creates a venv at `~/.local/share/voxterm/.venv/`
- Installs dependencies from `pyproject.toml` (via `pip install -e .`)
- Symlinks `voxterm` to `~/.local/bin/voxterm`
- Preserves existing venv on updates (avoids re-downloading all deps)
- Skips entirely if already on the requested version
