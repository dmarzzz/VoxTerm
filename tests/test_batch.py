import json

import pytest

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
