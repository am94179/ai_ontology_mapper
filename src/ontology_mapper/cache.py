"""Small, local JSON cache for repeatable schema-mapping stages."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

CACHE_FORMAT_VERSION = 1
DEFAULT_CACHE_DIRECTORY = Path(__file__).resolve().parents[2] / ".ontology_mapper_cache"


def fingerprint(value: Any) -> str:
    """Return a stable SHA-256 fingerprint for JSON-compatible input."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=_json_default)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def file_fingerprint(path: str | Path) -> str:
    """Fingerprint file content rather than its name or modification time."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class FileCache:
    """A deliberately small JSON cache with malformed-entry-as-miss behavior."""

    def __init__(self, directory: str | Path = DEFAULT_CACHE_DIRECTORY) -> None:
        self.directory = Path(directory)

    def get(self, namespace: str, key: str) -> Any | None:
        path = self._path(namespace, key)
        try:
            envelope = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(envelope, dict) or envelope.get("version") != CACHE_FORMAT_VERSION:
            return None
        if envelope.get("key") != key or "value" not in envelope:
            return None
        return envelope["value"]

    def set(self, namespace: str, key: str, value: Any) -> None:
        path = self._path(namespace, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        envelope = {"version": CACHE_FORMAT_VERSION, "key": key, "value": value}
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=".pending-", delete=False
        ) as temporary:
            json.dump(envelope, temporary, ensure_ascii=False, sort_keys=True, default=_json_default)
            temporary.write("\n")
            temporary_path = Path(temporary.name)
        os.replace(temporary_path, path)

    def _path(self, namespace: str, key: str) -> Path:
        if not namespace.replace("_", "").isalnum():
            raise ValueError("cache namespace must contain only letters, numbers, and underscores")
        if len(key) != 64 or any(character not in "0123456789abcdef" for character in key):
            raise ValueError("cache key must be a SHA-256 hexadecimal digest")
        return self.directory / namespace / f"{key}.json"


def _json_default(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot fingerprint or cache value of type {type(value).__name__}")
