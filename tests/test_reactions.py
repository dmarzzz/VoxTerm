import json

from tui.reactions import (
    append_reaction_event,
    normalize_reaction,
    parse_reaction_line,
    reaction_label,
)


def test_normalize_reaction_accepts_aliases():
    assert normalize_reaction("clap", author="pad") == {
        "emoji": "👏",
        "text": "",
        "author": "pad",
    }


def test_reaction_label_combines_emoji_and_text():
    assert reaction_label("💡", "connect this to demos") == "💡 connect this to demos"
    assert reaction_label("", "brb") == "brb"


def test_parse_reaction_line_accepts_reaction_json():
    payload = parse_reaction_line(
        json.dumps({"kind": "reaction", "emoji": "question", "text": "source?", "author": "button"})
    )

    assert payload == {"emoji": "❓", "text": "source?", "author": "button"}


def test_parse_reaction_line_ignores_bad_or_non_reaction_lines():
    assert parse_reaction_line("{bad") is None
    assert parse_reaction_line(json.dumps({"kind": "text", "text": "hello"})) is None
    assert parse_reaction_line(json.dumps({"kind": "reaction"})) is None


def test_append_reaction_event_writes_jsonl(tmp_path):
    path = tmp_path / "inbox" / "reactions.jsonl"

    payload = append_reaction_event(path, "idea", "save this", "macro-pad")

    assert payload == {"emoji": "💡", "text": "save this", "author": "macro-pad"}
    line = path.read_text(encoding="utf-8").strip()
    record = json.loads(line)
    assert record["kind"] == "reaction"
    assert record["emoji"] == "💡"
    assert record["text"] == "save this"
    assert record["author"] == "macro-pad"
    assert isinstance(record["t"], float)
