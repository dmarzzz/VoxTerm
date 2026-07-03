import json

import pytest

import batch.__main__ as batch_cli
from batch import core, rode


def test_transcribe_file_uses_headless_pipeline(monkeypatch, tmp_path):
    src = tmp_path / "meeting.wav"
    src.write_bytes(b"fake wav")
    out_dir = tmp_path / "out"
    captured = {}

    def fake_load(path):
        captured["loaded"] = path
        return "audio-buffer"

    def fake_transcribe_audio(audio, target, **kwargs):
        captured["audio"] = audio
        captured["target"] = target
        captured["kwargs"] = kwargs
        return {
            "events_path": str(target / "meeting-events.jsonl"),
            "transcript_path": str(target / "meeting-transcript.md"),
            "n_turns": 2,
            "n_speakers": 1,
        }

    monkeypatch.setattr(core.transcribe, "load_wav_16k_mono", fake_load)
    monkeypatch.setattr(core.transcribe, "transcribe_audio", fake_transcribe_audio)

    result = core.transcribe_file(
        src,
        out_dir=out_dir,
        model="fw-base",
        language="en",
        diarize=False,
    )

    assert captured["loaded"] == src
    assert captured["audio"] == "audio-buffer"
    assert captured["target"] == out_dir
    assert captured["kwargs"]["model"] == "fw-base"
    assert captured["kwargs"]["language"] == "en"
    assert captured["kwargs"]["diarize"] is False
    assert result["source_path"] == str(src)
    assert result["transcript_path"].endswith("meeting-transcript.md")


def test_transcribe_file_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        core.transcribe_file(tmp_path / "missing.wav")


def test_find_rode_recordings_scans_root_and_child_volumes(tmp_path):
    root = tmp_path / "Volumes"
    volume = root / "RODE"
    volume.mkdir(parents=True)
    direct = root / "00001_Wireless_GO.WAV"
    nested = volume / "00002_Wireless_GO.wav"
    ignored = volume / "notes.txt"
    direct.write_bytes(b"one")
    nested.write_bytes(b"two")
    ignored.write_text("nope")

    assert rode.find_rode_recordings([root]) == [direct, nested]


def test_import_recordings_copy_only_then_skip(tmp_path):
    root = tmp_path / "Volumes"
    volume = root / "RODE"
    volume.mkdir(parents=True)
    source = volume / "00001_Wireless_GO.WAV"
    source.write_bytes(b"audio")
    import_dir = tmp_path / "imports"
    manifest = tmp_path / "manifest.json"

    first = rode.import_recordings(
        roots=[root],
        import_dir=import_dir,
        manifest_path=manifest,
        transcribe=False,
    )
    second = rode.import_recordings(
        roots=[root],
        import_dir=import_dir,
        manifest_path=manifest,
        transcribe=False,
    )

    assert [result.action for result in first] == ["copy"]
    assert first[0].local_path.exists()
    assert [result.action for result in second] == ["skip"]
    raw = json.loads(manifest.read_text())
    assert len(raw) == 1
    entry = next(iter(raw.values()))
    assert entry["source_name"] == "00001_Wireless_GO.WAV"
    assert entry["transcript_path"] == ""


def test_import_recordings_transcribes_previously_copied_file(monkeypatch, tmp_path):
    root = tmp_path / "Volumes"
    volume = root / "RODE"
    volume.mkdir(parents=True)
    source = volume / "00001_Wireless_GO.WAV"
    source.write_bytes(b"audio")
    import_dir = tmp_path / "imports"
    manifest = tmp_path / "manifest.json"
    out_dir = tmp_path / "out"

    rode.import_recordings(
        roots=[root],
        import_dir=import_dir,
        manifest_path=manifest,
        transcribe=False,
    )

    def fake_transcribe(path, **kwargs):
        assert path == import_dir / "00001_Wireless_GO.WAV"
        assert kwargs["out_dir"] == out_dir
        transcript = out_dir / "done-transcript.md"
        transcript.parent.mkdir(parents=True)
        transcript.write_text("# done")
        return {"transcript_path": str(transcript)}

    monkeypatch.setattr(rode, "core_transcribe_file", fake_transcribe)
    results = rode.import_recordings(
        roots=[root],
        import_dir=import_dir,
        manifest_path=manifest,
        out_dir=out_dir,
        transcribe=True,
    )

    assert [result.action for result in results] == ["transcribe"]
    assert results[0].transcript_path == out_dir / "done-transcript.md"
    raw = json.loads(manifest.read_text())
    entry = next(iter(raw.values()))
    assert entry["transcript_path"].endswith("done-transcript.md")


def test_import_recordings_imports_and_transcribes_new_file(monkeypatch, tmp_path):
    root = tmp_path / "Volumes"
    volume = root / "RODE"
    volume.mkdir(parents=True)
    source = volume / "00001_Wireless_GO.WAV"
    source.write_bytes(b"audio")
    out_dir = tmp_path / "out"

    def fake_transcribe(path, **kwargs):
        transcript = out_dir / "new-transcript.md"
        transcript.parent.mkdir(parents=True)
        transcript.write_text("# done")
        return {"transcript_path": str(transcript)}

    monkeypatch.setattr(rode, "core_transcribe_file", fake_transcribe)
    results = rode.import_recordings(
        roots=[root],
        import_dir=tmp_path / "imports",
        manifest_path=tmp_path / "manifest.json",
        out_dir=out_dir,
        transcribe=True,
    )

    assert [result.action for result in results] == ["import"]
    assert results[0].transcript_path == out_dir / "new-transcript.md"


def test_watch_recordings_imports_new_files_without_skip_spam(tmp_path):
    root = tmp_path / "Volumes"
    volume = root / "RODE"
    volume.mkdir(parents=True)
    source = volume / "00001_Wireless_GO.WAV"
    source.write_bytes(b"audio")
    seen = []
    sleeps = []

    handled = rode.watch_recordings(
        roots=[root],
        import_dir=tmp_path / "imports",
        manifest_path=tmp_path / "manifest.json",
        transcribe=False,
        interval_seconds=0.25,
        scan_limit=2,
        sleep=sleeps.append,
        on_results=seen.extend,
    )

    assert handled == 1
    assert [result.action for result in seen] == ["copy"]
    assert sleeps == [0.25]


def test_watch_recordings_rejects_nonpositive_interval(tmp_path):
    with pytest.raises(ValueError, match="interval_seconds"):
        rode.watch_recordings(
            roots=[tmp_path],
            interval_seconds=0,
            scan_limit=1,
        )


def test_cli_rode_watch_wires_options_and_prints_results(monkeypatch, tmp_path, capsys):
    root = tmp_path / "Volumes"
    source = root / "00001_Wireless_GO.WAV"
    local = tmp_path / "imports" / source.name
    calls = {}

    def fake_watch(**kwargs):
        calls.update(kwargs)
        kwargs["on_results"](
            [
                rode.ImportResult(
                    source_path=source,
                    local_path=local,
                    action="copy",
                    sha256="abc123",
                )
            ]
        )
        return 1

    monkeypatch.setattr(batch_cli, "watch_recordings", fake_watch)

    rc = batch_cli.main(
        [
            "--rode",
            "--watch",
            "--watch-interval",
            "0.5",
            "--mount-root",
            str(root),
            "--import-dir",
            str(tmp_path / "imports"),
            "--manifest",
            str(tmp_path / "manifest.json"),
            "--no-transcribe",
            "--model",
            "fw-base",
        ]
    )

    assert rc == 0
    assert calls["roots"] == [str(root)]
    assert calls["interval_seconds"] == 0.5
    assert calls["transcribe"] is False
    assert calls["model"] == "fw-base"
    out = capsys.readouterr().out
    assert "Watching for Rode recordings every 0.5s" in out
    assert "[copy] 00001_Wireless_GO.WAV" in out


def test_cli_watch_requires_rode():
    with pytest.raises(SystemExit):
        batch_cli.main(["--watch"])
