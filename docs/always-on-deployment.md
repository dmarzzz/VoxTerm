# Always-On Room Deployment

This runbook covers the code-side deployment shape for issue #95: a small set
of dedicated room devices that boot into VoxTerm, expose the room UI on the LAN
behind a token, and update from the release channel. Hardware purchasing,
receipts, and physical placement remain external work.

## Room Device Checklist

- Create one OS user per device, for example `voxterm-room`.
- Install VoxTerm from the release installer:

  ```bash
  curl -fsSL https://github.com/dmarzzz/VoxTerm/releases/latest/download/install.sh | bash
  ```

- Create a stable room token.

  macOS:

  ```bash
  mkdir -p "$HOME/.config/voxterm"
  python3 -c 'import secrets; print(secrets.token_urlsafe(24))' > "$HOME/.config/voxterm/room-token"
  chmod 600 "$HOME/.config/voxterm/room-token"
  ```

  Linux:

  ```bash
  mkdir -p "$HOME/.config/voxterm"
  printf 'VOXTERM_GUI_TOKEN=%s\n' "$(python3 -c 'import secrets; print(secrets.token_urlsafe(24))')" > "$HOME/.config/voxterm/room.env"
  chmod 600 "$HOME/.config/voxterm/room.env"
  ```

- Set the device label in the room inventory table.
- Enable auto-login or a locked-down kiosk session for the room user.
- Install the matching service templates below.
- Plug in and select the room mic.
- Verify the room UI from another device on the same LAN:

  ```bash
  curl -fsS -H "X-VoxTerm-Token: $(cat ~/.config/voxterm/room-token 2>/dev/null || awk -F= '/^VOXTERM_GUI_TOKEN=/ {print $2}' ~/.config/voxterm/room.env)" \
    http://127.0.0.1:8740/api/options
  ```

## macOS LaunchAgent

The GUI server needs user-session access for browser and mic permissions, so use
a LaunchAgent under the dedicated room user rather than a system LaunchDaemon.

Install:

```bash
mkdir -p "$HOME/Library/LaunchAgents" "$HOME/Library/Logs"
python3 - <<'PY'
from pathlib import Path
home = str(Path.home())
src = Path("deploy/macos/com.voxterm.room-gui.plist")
dst = Path.home() / "Library/LaunchAgents/com.voxterm.room-gui.plist"
dst.write_text(src.read_text().replace("__HOME__", home))
src = Path("deploy/macos/com.voxterm.update.plist")
dst = Path.home() / "Library/LaunchAgents/com.voxterm.update.plist"
dst.write_text(src.read_text().replace("__HOME__", home))
PY
launchctl bootstrap "gui/$(id -u)" "$HOME/Library/LaunchAgents/com.voxterm.room-gui.plist"
launchctl bootstrap "gui/$(id -u)" "$HOME/Library/LaunchAgents/com.voxterm.update.plist"
launchctl kickstart -k "gui/$(id -u)/com.voxterm.room-gui"
```

The room server binds `0.0.0.0:8740` and requires the token in
`~/.config/voxterm/room-token`. The update agent runs the existing release
installer every Tuesday at 04:00 local time.

## Linux systemd User Units

Install:

```bash
mkdir -p "$HOME/.config/systemd/user" "$HOME/.config/autostart"
cp deploy/linux/voxterm-room-gui.service "$HOME/.config/systemd/user/"
cp deploy/linux/voxterm-update.service "$HOME/.config/systemd/user/"
cp deploy/linux/voxterm-update.timer "$HOME/.config/systemd/user/"
cp deploy/linux/voxterm-kiosk.desktop "$HOME/.config/autostart/"
systemctl --user daemon-reload
systemctl --user enable --now voxterm-room-gui.service
systemctl --user enable --now voxterm-update.timer
loginctl enable-linger "$USER"
```

The user service binds `0.0.0.0:8740` and reads
`~/.config/voxterm/room.env`. The desktop autostart file opens Chromium in
kiosk mode against `http://127.0.0.1:8740/?token=...`; adjust the browser
binary if the image uses Chrome, Firefox, or another kiosk shell.

## Room Inventory

Track these fields for each deployed device:

| Room | Device label | Hostname | OS | Mic | Token location | Update channel | Notes |
|---|---|---|---|---|---|---|---|
| TBD | voxterm-room-01 | TBD | macOS/Linux | TBD | user config | latest release | |
| TBD | voxterm-room-02 | TBD | macOS/Linux | TBD | user config | latest release | |
| TBD | voxterm-room-03 | TBD | macOS/Linux | TBD | user config | latest release | |
| TBD | voxterm-room-04 | TBD | macOS/Linux | TBD | user config | latest release | |
| TBD | voxterm-room-05 | TBD | macOS/Linux | TBD | user config | latest release | |

## Operator Checks

- Status: `launchctl print gui/$(id -u)/com.voxterm.room-gui` on macOS, or
  `systemctl --user status voxterm-room-gui.service` on Linux.
- Logs: `~/Library/Logs/voxterm-room-gui*.log` on macOS, or
  `journalctl --user -u voxterm-room-gui.service` on Linux.
- Restart: `launchctl kickstart -k gui/$(id -u)/com.voxterm.room-gui` on macOS,
  or `systemctl --user restart voxterm-room-gui.service` on Linux.
- Update now: rerun the installer command, then restart the room service.
- Token rotation: overwrite the token file/env, restart the room service, and
  update any paired tablets or signage.

## Acceptance Evidence To Collect In The Room

- Device photo with label and mic.
- `voxterm-room-gui` service status after reboot.
- Browser/kiosk opened to the local room UI.
- Token-gated `/api/options` check from another LAN device.
- A short saved transcript proving the selected mic works.
- Update timer/LaunchAgent installed and enabled.
