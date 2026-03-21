#!/usr/bin/env python3
"""Diarizer subprocess worker — loads ECAPA-TDNN and processes IPC requests.

This process imports PyTorch/SpeechBrain only (never MLX), preventing C++
runtime conflicts with the main process which uses MLX for transcription.

Protocol: reads length-prefixed JSON messages from stdin, writes responses
to stdout. See ipc.py for details.

Usage:
    python -m diarization.subprocess_worker
"""

from __future__ import annotations

import os
import signal
import sys

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Prevent PyTorch from spawning threads that could conflict
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np

from diarization.engine import DiarizationEngine
from diarization.ipc import (
    MSG_ACK, MSG_ERROR, MSG_GET_CENTROID, MSG_GET_COLOR,
    MSG_GET_EMBEDDINGS, MSG_GET_NAME, MSG_GET_NAMES, MSG_GET_STATE,
    MSG_IDENTIFY, MSG_IDENTIFY_MULTI, MSG_IS_MATCHED, MSG_IS_STABLE,
    MSG_MARK_MATCHED, MSG_MERGE, MSG_NUM_SPEAKERS, MSG_PING, MSG_PONG,
    MSG_READY, MSG_RESET, MSG_RESULT, MSG_SET_NAME, MSG_SHUTDOWN,
    decode_array, encode_array, recv_msg, send_msg,
)

_running = True


def _handle_sigterm(signum, frame):
    global _running
    _running = False


def main():
    signal.signal(signal.SIGTERM, _handle_sigterm)

    # Use binary stdin/stdout for the IPC protocol
    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer
    # Redirect stderr so PyTorch warnings don't pollute stdout (used for IPC).
    # Parent process captures stderr via subprocess.PIPE for diagnostics.
    try:
        sys.stderr = open(os.devnull, "w")
    except Exception:
        pass

    engine = DiarizationEngine()

    try:
        engine.load()
    except Exception as e:
        send_msg(stdout, {"type": MSG_ERROR, "error": f"model load failed: {e}"})
        sys.exit(1)

    send_msg(stdout, {"type": MSG_READY})

    while _running:
        msg = recv_msg(stdin)
        if msg is None:
            break  # EOF — parent closed pipe

        msg_type = msg.get("type", "")

        try:
            response = _dispatch(engine, msg_type, msg)
            send_msg(stdout, response)
        except Exception as e:
            send_msg(stdout, {"type": MSG_ERROR, "error": str(e)})


def _dispatch(engine: DiarizationEngine, msg_type: str, msg: dict) -> dict:
    """Route a message to the appropriate engine method."""

    if msg_type == MSG_IDENTIFY:
        audio = decode_array(msg["audio"])
        sample_rate = msg.get("sample_rate", 16000)
        import numpy as np
        audio_rms = float(np.sqrt(np.mean(audio ** 2)))
        audio_len = len(audio)
        label, speaker_id = engine.identify(audio, sample_rate)
        color = engine.get_speaker_color(speaker_id)
        # Return the embedding too — main process needs it for cross-session matching
        centroid = engine.get_session_centroid(speaker_id)
        return {
            "type": MSG_RESULT,
            "label": label,
            "speaker_id": speaker_id,
            "color": color,
            "centroid": encode_array(centroid) if centroid is not None else None,
            "debug_rms": round(audio_rms, 4),
            "debug_samples": audio_len,
            "debug_speakers": engine.num_speakers,
        }

    elif msg_type == MSG_IDENTIFY_MULTI:
        audio = decode_array(msg["audio"])
        sample_rate = msg.get("sample_rate", 16000)
        # Use overlap-aware multi-speaker identification
        segments = engine.identify_multi(audio, sample_rate)
        seg_results = []
        for label, speaker_id, start, end in segments:
            color = engine.get_speaker_color(speaker_id)
            seg_results.append({
                "label": label,
                "speaker_id": speaker_id,
                "color": color,
                "start_sample": start,
                "end_sample": end,
            })
        return {
            "type": MSG_RESULT,
            "segments": seg_results,
            "debug_speakers": engine.num_speakers,
        }

    elif msg_type == MSG_SET_NAME:
        engine.set_speaker_name(msg["speaker_id"], msg["name"])
        return {"type": MSG_ACK}

    elif msg_type == MSG_GET_STATE:
        session_speakers = engine.get_all_session_speakers()
        names = engine.get_speaker_names()
        colors = {sid: engine.get_speaker_color(sid) for sid in session_speakers}
        return {
            "type": MSG_RESULT,
            "session_speakers": {str(k): v for k, v in session_speakers.items()},
            "names": {str(k): v for k, v in names.items()},
            "colors": {str(k): v for k, v in colors.items()},
        }

    elif msg_type == MSG_GET_EMBEDDINGS:
        sid = msg["speaker_id"]
        seg_data = engine.get_segment_embeddings(sid)
        return {
            "type": MSG_RESULT,
            "embeddings": [
                {"embedding": encode_array(emb), "duration": dur}
                for emb, dur in seg_data
            ],
        }

    elif msg_type == MSG_GET_CENTROID:
        sid = msg["speaker_id"]
        centroid = engine.get_session_centroid(sid)
        return {
            "type": MSG_RESULT,
            "centroid": encode_array(centroid) if centroid is not None else None,
        }

    elif msg_type == MSG_IS_STABLE:
        sid = msg["speaker_id"]
        return {"type": MSG_RESULT, "stable": engine.is_speaker_stable(sid)}

    elif msg_type == MSG_MARK_MATCHED:
        engine.mark_matched(msg["speaker_id"])
        return {"type": MSG_ACK}

    elif msg_type == MSG_IS_MATCHED:
        return {
            "type": MSG_RESULT,
            "matched": engine.is_matched(msg["speaker_id"]),
        }

    elif msg_type == MSG_MERGE:
        engine.merge_speakers(msg["source_id"], msg["target_id"])
        return {"type": MSG_ACK}

    elif msg_type == MSG_RESET:
        engine.reset_session()
        return {"type": MSG_ACK}

    elif msg_type == MSG_PING:
        return {"type": MSG_PONG}

    elif msg_type == MSG_SHUTDOWN:
        global _running
        _running = False
        return {"type": MSG_ACK}

    elif msg_type == MSG_GET_COLOR:
        color = engine.get_speaker_color(msg["speaker_id"])
        return {"type": MSG_RESULT, "color": color}

    elif msg_type == MSG_GET_NAME:
        name = engine.get_speaker_name(msg["speaker_id"])
        return {"type": MSG_RESULT, "name": name}

    elif msg_type == MSG_GET_NAMES:
        names = engine.get_speaker_names()
        return {
            "type": MSG_RESULT,
            "names": {str(k): v for k, v in names.items()},
        }

    elif msg_type == MSG_NUM_SPEAKERS:
        return {"type": MSG_RESULT, "count": engine.num_speakers}

    else:
        return {"type": MSG_ERROR, "error": f"unknown message type: {msg_type}"}


if __name__ == "__main__":
    main()
