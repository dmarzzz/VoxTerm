"""Default-model selection policy.

The PyTorch Qwen3-ASR path is fast on a GPU but effectively unusable on CPU
(minutes to load, sub-real-time decode), so a CPU-only machine must not pick it
as the out-of-box default. These tests guard that policy and the cheap,
torch-free GPU probe that drives it (config._has_nvidia_gpu).
"""

import sys

import pytest

import config


def test_probe_returns_bool():
    assert isinstance(config._has_nvidia_gpu(), bool)


def test_config_import_is_torch_free():
    # Deciding the default model must not pull torch in at config-import time —
    # importing torch slows every startup (--list-models, the test suite, …).
    # Check in a clean subprocess so other tests that imported torch can't mask
    # a regression.
    import os
    import subprocess

    repo_root = os.path.dirname(os.path.abspath(config.__file__))
    out = subprocess.run(
        [sys.executable, "-c", "import sys, config; print('torch' in sys.modules)"],
        capture_output=True,
        text=True,
        cwd=repo_root,
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False", f"config import pulled torch: {out.stdout!r}"


def test_default_model_is_in_registry():
    assert config.DEFAULT_MODEL in config.AVAILABLE_MODELS


@pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="Linux CPU-default policy"
)
def test_cpu_only_linux_does_not_default_to_qwen():
    if config._has_nvidia_gpu():
        pytest.skip("GPU present — qwen3 default is appropriate here")
    # No GPU: must default to the CPU-friendly faster-whisper, never the slow
    # PyTorch qwen3 path.
    assert config.DEFAULT_MODEL in config.FASTER_WHISPER_MODELS
    assert config.DEFAULT_MODEL == "fw-base"


def test_torch_cpu_only_wheel_detected_via_metadata(monkeypatch):
    import importlib.metadata as md

    monkeypatch.setattr(md, "version", lambda name: "2.11.0+cpu")
    assert config._torch_is_cpu_only() is True
    monkeypatch.setattr(md, "version", lambda name: "2.11.0+cu121")
    assert config._torch_is_cpu_only() is False
    monkeypatch.setattr(md, "version", lambda name: "2.11.0")  # default PyPI (CUDA-capable)
    assert config._torch_is_cpu_only() is False

    def _raise(_name):
        raise md.PackageNotFoundError
    monkeypatch.setattr(md, "version", _raise)
    assert config._torch_is_cpu_only() is False  # unknown -> don't suppress


def test_cpu_only_torch_suppresses_gpu_default_even_with_driver(monkeypatch):
    # The review's exact false-positive case: an NVIDIA driver is present but
    # torch is a +cpu wheel -> the GPU model can't actually run, so the heuristic
    # must NOT pick it.
    monkeypatch.setattr(config, "_nvidia_driver_present", lambda: True)
    monkeypatch.setattr(config, "_torch_is_cpu_only", lambda: True)
    assert config._has_nvidia_gpu() is False
    monkeypatch.setattr(config, "_torch_is_cpu_only", lambda: False)
    assert config._has_nvidia_gpu() is True
    monkeypatch.setattr(config, "_nvidia_driver_present", lambda: False)
    assert config._has_nvidia_gpu() is False  # no driver -> never GPU default
