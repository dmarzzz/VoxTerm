from tui.widgets.transcript import TranscriptPanel


def test_reaction_entries_export_to_text_and_markdown(monkeypatch):
    panel = TranscriptPanel()
    monkeypatch.setattr(panel, "write", lambda *_args, **_kwargs: None)

    panel.add_transcript("hello", "Speaker 1", 1)
    panel.add_reaction("👏 agreed", "button-pad")

    entries = panel.get_entries()
    assert entries[0][1] == "transcript"
    assert entries[1][1] == "reaction"
    assert entries[1][2] == "👏 agreed"
    assert entries[1][3] == "button-pad"

    plain = panel.get_plain_text()
    assert "[reaction:button-pad] 👏 agreed" in plain

    markdown = panel.get_markdown()
    assert "**button-pad reacted:** 👏 agreed" in markdown
