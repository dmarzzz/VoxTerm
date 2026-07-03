import configparser
import plistlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _ini(path: str) -> configparser.RawConfigParser:
    parser = configparser.RawConfigParser(interpolation=None)
    parser.optionxform = str
    with (ROOT / path).open(encoding="utf-8") as handle:
        parser.read_file(handle)
    return parser


def test_macos_room_launchagent_template_is_valid():
    with (ROOT / "deploy/macos/com.voxterm.room-gui.plist").open("rb") as handle:
        plist = plistlib.load(handle)

    assert plist["Label"] == "com.voxterm.room-gui"
    assert plist["RunAtLoad"] is True
    assert plist["KeepAlive"] is True
    command = " ".join(plist["ProgramArguments"])
    assert "VOXTERM_GUI_LAN=1" in command
    assert "-m gui.server" in command
    assert "room-token" in command


def test_macos_update_launchagent_template_is_valid():
    with (ROOT / "deploy/macos/com.voxterm.update.plist").open("rb") as handle:
        plist = plistlib.load(handle)

    assert plist["Label"] == "com.voxterm.update"
    command = " ".join(plist["ProgramArguments"])
    assert "releases/latest/download/install.sh" in command
    assert plist["StartCalendarInterval"]["Weekday"] == 2


def test_linux_room_service_template_is_valid():
    service = _ini("deploy/linux/voxterm-room-gui.service")

    assert service["Unit"]["Description"] == "VoxTerm room GUI server"
    assert service["Service"]["Environment"] == "VOXTERM_GUI_LAN=1 VOXTERM_GUI_PORT=8740"
    assert service["Service"]["EnvironmentFile"] == "%h/.config/voxterm/room.env"
    assert service["Service"]["ExecStart"].endswith("-m gui.server")
    assert service["Install"]["WantedBy"] == "default.target"


def test_linux_update_timer_templates_are_valid():
    service = _ini("deploy/linux/voxterm-update.service")
    timer = _ini("deploy/linux/voxterm-update.timer")

    assert "install.sh" in service["Service"]["ExecStart"]
    assert timer["Timer"]["OnCalendar"] == "weekly"
    assert timer["Timer"]["Persistent"] == "true"
    assert timer["Install"]["WantedBy"] == "timers.target"


def test_linux_kiosk_desktop_template_is_valid():
    desktop = _ini("deploy/linux/voxterm-kiosk.desktop")

    assert desktop["Desktop Entry"]["Type"] == "Application"
    assert desktop["Desktop Entry"]["Name"] == "VoxTerm Kiosk"
    assert "chromium --kiosk" in desktop["Desktop Entry"]["Exec"]
    assert "127.0.0.1:8740" in desktop["Desktop Entry"]["Exec"]


def test_runbook_links_all_templates():
    doc = (ROOT / "docs/always-on-deployment.md").read_text(encoding="utf-8")

    for path in [
        "deploy/macos/com.voxterm.room-gui.plist",
        "deploy/macos/com.voxterm.update.plist",
        "deploy/linux/voxterm-room-gui.service",
        "deploy/linux/voxterm-update.service",
        "deploy/linux/voxterm-update.timer",
        "deploy/linux/voxterm-kiosk.desktop",
    ]:
        assert path in doc
