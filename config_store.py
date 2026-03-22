"""Typed, thread-safe configuration store with atomic writes and merge semantics."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any


# Schema: key → default value
_DEFAULTS: dict[str, Any] = {
    "last_model": "",
    "last_language": "",
    "audio_retention": False,
    "export_format": "markdown",
    "summarization_model": "",
    "summarization_strength": "medium",
}

# Expected types per key (for validation)
_TYPES: dict[str, type] = {
    "last_model": str,
    "last_language": str,
    "audio_retention": bool,
    "export_format": str,
    "summarization_model": str,
    "summarization_strength": str,
}


class ConfigStore:
    """Persistent configuration with typed schema, merge semantics, and atomic writes.

    Reads existing .state.json on init (backward compatible with bare 2-key files).
    Writes use tmp+rename for atomicity.
    """

    def __init__(self, path: Path | None = None) -> None:
        if path is None:
            path = Path.home() / "Documents" / "voxterm" / ".state.json"
        self._path = path
        self._lock = threading.Lock()
        self._data: dict[str, Any] = dict(_DEFAULTS)
        self._load()

    def _load(self) -> None:
        """Load from disk, merging with defaults. Unknown keys are preserved."""
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                for key, value in raw.items():
                    # Validate type for known keys; keep unknown keys as-is
                    expected = _TYPES.get(key)
                    if expected is None or isinstance(value, expected):
                        self._data[key] = value
        except Exception:
            pass

    def _save(self) -> None:
        """Atomic write: write to .tmp then os.replace()."""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._data, indent=2), encoding="utf-8")
            os.replace(tmp, self._path)
        except Exception:
            pass

    def get(self, key: str) -> Any:
        """Get a config value. Returns the default if key is in schema, else None."""
        with self._lock:
            return self._data.get(key, _DEFAULTS.get(key))

    def set(self, key: str, value: Any) -> None:
        """Set a config value and persist to disk (merge, not overwrite)."""
        with self._lock:
            expected = _TYPES.get(key)
            if expected is not None and not isinstance(value, expected):
                raise TypeError(
                    f"Invalid type for config key '{key}': "
                    f"expected {expected.__name__}, got {type(value).__name__}"
                )
            self._data[key] = value
            self._save()

    def update(self, values: dict[str, Any]) -> None:
        """Set multiple config values and persist once (single disk write)."""
        with self._lock:
            for key, value in values.items():
                expected = _TYPES.get(key)
                if expected is not None and not isinstance(value, expected):
                    raise TypeError(
                        f"Invalid type for config key '{key}': "
                        f"expected {expected.__name__}, got {type(value).__name__}"
                    )
                self._data[key] = value
            self._save()

    @property
    def data(self) -> dict[str, Any]:
        """Return a snapshot of all config data."""
        with self._lock:
            return dict(self._data)
