import json
from io import StringIO

from tui.reactions import (
    append_reaction_event,
    append_reaction_stream,
    main,
    normalize_reaction,
    parse_reaction_command,
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


def test_parse_reaction_command_accepts_simple_lines_and_quotes():
    assert parse_reaction_command("clap", author="pad") == {
        "emoji": "👏",
        "text": "",
        "author": "pad",
    }
    assert parse_reaction_command('question "what source?"', author="pad") == {
        "emoji": "❓",
        "text": "what source?",
        "author": "pad",
    }


def test_parse_reaction_command_accepts_jsonl_shape():
    payload = parse_reaction_command(
        json.dumps(
            {
                "kind": "reaction",
                "emoji": "idea",
                "text": "clip this",
                "author": "serial-pad",
            }
        ),
        author="ignored",
    )

    assert payload == {"emoji": "💡", "text": "clip this", "author": "serial-pad"}


def test_parse_reaction_command_ignores_bad_lines():
    assert parse_reaction_command("") is None
    assert parse_reaction_command('"unterminated') is None
    assert parse_reaction_command("{}") is None


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


def test_append_reaction_stream_writes_valid_lines_and_skips_bad(tmp_path):
    path = tmp_path / "inbox" / "reactions.jsonl"

    count = append_reaction_stream(
        path,
        [
            "clap",
            'question "source?"',
            "{bad",
            json.dumps({"kind": "reaction", "emoji": "idea", "text": "save"}),
        ],
        author="serial-pad",
        stderr=None,
    )

    assert count == 3
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [row["emoji"] for row in rows] == ["👏", "❓", "💡"]
    assert rows[1]["text"] == "source?"
    assert rows[0]["author"] == "serial-pad"
    assert rows[2]["author"] == "external"


def test_cli_stdin_bridge(monkeypatch, tmp_path, capsys):
    inbox = tmp_path / "reactions.jsonl"
    monkeypatch.setattr("sys.stdin", StringIO("clap\nquestion source?\n"))

    rc = main(["--stdin", "--author", "macro-pad", "--inbox", str(inbox)])

    assert rc == 0
    assert "sent 2 reactions" in capsys.readouterr().out
    rows = [json.loads(line) for line in inbox.read_text(encoding="utf-8").splitlines()]
    assert [row["emoji"] for row in rows] == ["👏", "❓"]
    assert rows[1]["text"] == "source?"
    assert rows[0]["author"] == "macro-pad"
