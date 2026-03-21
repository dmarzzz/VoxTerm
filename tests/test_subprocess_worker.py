"""Tests for diarization/subprocess_worker.py — _dispatch routing logic."""

import numpy as np
import pytest

from diarization.subprocess_worker import _dispatch
from diarization.ipc import (
    MSG_ACK,
    MSG_ERROR,
    MSG_GET_EMBEDDINGS,
    MSG_GET_NAME,
    MSG_GET_STATE,
    MSG_IDENTIFY,
    MSG_NUM_SPEAKERS,
    MSG_PING,
    MSG_PONG,
    MSG_RESET,
    MSG_RESULT,
    MSG_SET_NAME,
    encode_array,
    decode_array,
)

EMBEDDING_DIM = 192
SAMPLE_RATE = 16000


def _make_audio(duration_sec=2.5, freq=440.0):
    """Generate a synthetic sine-wave audio chunk (16kHz float32)."""
    t = np.linspace(0, duration_sec, int(SAMPLE_RATE * duration_sec), dtype=np.float32)
    return 0.5 * np.sin(2 * np.pi * freq * t)


class TestDispatch:

    def test_identify_returns_valid_response(self, loaded_mock_engine, sample_audio):
        audio = sample_audio(duration_sec=2.5)
        msg = {
            "type": MSG_IDENTIFY,
            "audio": encode_array(audio),
            "sample_rate": SAMPLE_RATE,
        }
        resp = _dispatch(loaded_mock_engine, MSG_IDENTIFY, msg)

        assert resp["type"] == MSG_RESULT
        assert "label" in resp
        assert "speaker_id" in resp
        assert "color" in resp
        assert "centroid" in resp

    def test_set_and_get_name(self, loaded_mock_engine, sample_audio):
        # First identify a speaker so one exists
        audio = sample_audio(duration_sec=2.5)
        identify_resp = _dispatch(loaded_mock_engine, MSG_IDENTIFY, {
            "type": MSG_IDENTIFY,
            "audio": encode_array(audio),
            "sample_rate": SAMPLE_RATE,
        })
        speaker_id = identify_resp["speaker_id"]

        # Set a custom name
        set_resp = _dispatch(loaded_mock_engine, MSG_SET_NAME, {
            "type": MSG_SET_NAME,
            "speaker_id": speaker_id,
            "name": "Alice",
        })
        assert set_resp["type"] == MSG_ACK

        # Retrieve the name
        get_resp = _dispatch(loaded_mock_engine, MSG_GET_NAME, {
            "type": MSG_GET_NAME,
            "speaker_id": speaker_id,
        })
        assert get_resp["type"] == MSG_RESULT
        assert get_resp["name"] == "Alice"

    def test_get_state_after_identify(self, loaded_mock_engine, sample_audio):
        audio = sample_audio(duration_sec=2.5)
        _dispatch(loaded_mock_engine, MSG_IDENTIFY, {
            "type": MSG_IDENTIFY,
            "audio": encode_array(audio),
            "sample_rate": SAMPLE_RATE,
        })

        resp = _dispatch(loaded_mock_engine, MSG_GET_STATE, {
            "type": MSG_GET_STATE,
        })
        assert resp["type"] == MSG_RESULT
        assert len(resp["session_speakers"]) > 0

    def test_num_speakers(self, loaded_mock_engine, sample_audio):
        audio = sample_audio(duration_sec=2.5)
        _dispatch(loaded_mock_engine, MSG_IDENTIFY, {
            "type": MSG_IDENTIFY,
            "audio": encode_array(audio),
            "sample_rate": SAMPLE_RATE,
        })

        resp = _dispatch(loaded_mock_engine, MSG_NUM_SPEAKERS, {
            "type": MSG_NUM_SPEAKERS,
        })
        assert resp["type"] == MSG_RESULT
        assert resp["count"] >= 1

    def test_reset_clears_state(self, loaded_mock_engine, sample_audio):
        audio = sample_audio(duration_sec=2.5)
        _dispatch(loaded_mock_engine, MSG_IDENTIFY, {
            "type": MSG_IDENTIFY,
            "audio": encode_array(audio),
            "sample_rate": SAMPLE_RATE,
        })

        # Reset session
        reset_resp = _dispatch(loaded_mock_engine, MSG_RESET, {
            "type": MSG_RESET,
        })
        assert reset_resp["type"] == MSG_ACK

        # State should now be empty
        state_resp = _dispatch(loaded_mock_engine, MSG_GET_STATE, {
            "type": MSG_GET_STATE,
        })
        assert state_resp["type"] == MSG_RESULT
        assert len(state_resp["session_speakers"]) == 0

    def test_ping_pong(self, loaded_mock_engine):
        resp = _dispatch(loaded_mock_engine, MSG_PING, {"type": MSG_PING})
        assert resp["type"] == MSG_PONG

    def test_unknown_message_type(self, loaded_mock_engine):
        resp = _dispatch(loaded_mock_engine, "bogus", {"type": "bogus"})
        assert resp["type"] == MSG_ERROR

    def test_get_embeddings(self, loaded_mock_engine, sample_audio):
        audio = sample_audio(duration_sec=2.5)
        identify_resp = _dispatch(loaded_mock_engine, MSG_IDENTIFY, {
            "type": MSG_IDENTIFY,
            "audio": encode_array(audio),
            "sample_rate": SAMPLE_RATE,
        })
        speaker_id = identify_resp["speaker_id"]

        resp = _dispatch(loaded_mock_engine, MSG_GET_EMBEDDINGS, {
            "type": MSG_GET_EMBEDDINGS,
            "speaker_id": speaker_id,
        })
        assert resp["type"] == MSG_RESULT
        assert isinstance(resp["embeddings"], list)
        assert len(resp["embeddings"]) >= 1
        # Each entry should have embedding and duration
        entry = resp["embeddings"][0]
        assert "embedding" in entry
        assert "duration" in entry
