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
