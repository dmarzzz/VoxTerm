from tui.app import _hivemind_header_state
from tui.widgets.header import CyberHeader
from tui.widgets.transcript import Log, _LOG_CATEGORIES


class _Sink:
    host = "convent.local"
    node = "convent"


class _Client:
    def __init__(self, *, push_enabled=True, sink=None, pinned=""):
        self.push_enabled = push_enabled
        self.pinned_sink_pubkey = pinned
        self._sink = sink

    def active_sink(self):
        return self._sink


def test_hivemind_log_category_is_distinct():
    assert Log.HIVE == "hive"
    assert _LOG_CATEGORIES[Log.HIVE]["tag"] == "HIVE"


def test_hivemind_header_state_requires_push_enabled():
    assert _hivemind_header_state(None) == (False, "")
    assert _hivemind_header_state(_Client(push_enabled=False, sink=_Sink())) == (False, "")


def test_hivemind_header_state_uses_active_sink_label():
    assert _hivemind_header_state(_Client(sink=_Sink())) == (True, "convent")


def test_hivemind_header_state_falls_back_to_pinned_or_waiting():
    assert _hivemind_header_state(_Client(sink=None, pinned="ed25519:abcdef")) == (
        True,
        "abcdef..",
    )
    assert _hivemind_header_state(_Client(sink=None, pinned="")) == (True, "waiting")


def test_cyber_header_tracks_hivemind_chip_state():
    header = CyberHeader()

    header.set_hivemind(True, "convent")
    assert header._hivemind_active is True
    assert header._hivemind_label == "convent"

    header.set_hivemind(False)
    assert header._hivemind_active is False
    assert header._hivemind_label == ""
