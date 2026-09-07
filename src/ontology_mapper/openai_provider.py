"""OpenAI Responses API adapter for structured mapping reasoning."""

from __future__ import annotations

import json
import os
from pathlib import Path
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from dotenv import load_dotenv

from .reasoning import ReasoningError

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class OpenAIConfig:
    """Runtime-only OpenAI configuration; no values are persisted by the project."""

    api_key: str
    base_url: str
    model: str

    @classmethod
    def from_environment(cls, dotenv_path: str | Path | None = None) -> "OpenAIConfig":
        load_dotenv(dotenv_path=dotenv_path or PROJECT_ROOT / ".env", override=False)
        values = {
            "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", "").strip(),
            "OPENAI_URL": os.getenv("OPENAI_URL", "").strip(),
            "OPENAI_MODEL": os.getenv("OPENAI_MODEL", "").strip(),
        }
        missing = [name for name, value in values.items() if not value]
        if missing:
            raise ReasoningError(f"Missing required environment variables: {', '.join(missing)}")
        return cls(values["OPENAI_API_KEY"], values["OPENAI_URL"], values["OPENAI_MODEL"])


class OpenAIReasoningClient:
    """One OpenAI provider implementation using strict structured output."""

    def __init__(self, config: OpenAIConfig) -> None:
        from openai import OpenAI

        self._client = OpenAI(api_key=config.api_key, base_url=config.base_url)
        self._model = config.model

    @classmethod
    def from_environment(cls) -> "OpenAIReasoningClient":
        return cls(OpenAIConfig.from_environment())

    def reason(self, prompt: str) -> Mapping[str, Any]:
        response = self._client.responses.create(
            model=self._model,
            store=False,
            input=[
                {
                    "role": "system",
                    "content": "Return only the requested structured mapping assessment.",
                },
                {"role": "user", "content": prompt},
            ],
            text={"format": _REASONING_RESPONSE_SCHEMA},
        )
        try:
            return json.loads(response.output_text)
        except json.JSONDecodeError as error:
            raise ReasoningError("OpenAI response was not valid JSON") from error


    def reason_batch(self, prompt: str) -> Mapping[str, Any]:
        response = self._client.responses.create(
            model=self._model,
            store=False,
            input=[
                {
                    "role": "system",
                    "content": "Return only the requested structured batch mapping assessments.",
                },
                {"role": "user", "content": prompt},
            ],
            text={"format": _BATCH_REASONING_RESPONSE_SCHEMA},
        )
        try:
            return json.loads(response.output_text)
        except json.JSONDecodeError as error:
            raise ReasoningError("OpenAI batch response was not valid JSON") from error


_REASONING_RESPONSE_SCHEMA = {
    "type": "json_schema",
    "name": "mapping_reasoning",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "relationship": {
                "type": "string",
                "enum": ["MATCH", "RELATED", "CONFLICT", "NO_MATCH"],
            },
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "explanation": {"type": "string"},
            "conflicts": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["relationship", "confidence", "explanation", "conflicts"],
        "additionalProperties": False,
    },
}

_BATCH_REASONING_RESPONSE_SCHEMA = {
    "type": "json_schema",
    "name": "batch_mapping_reasoning",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "assessments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "target_column": {"type": "string"},
                        "relationship": {"type": "string", "enum": ["MATCH", "RELATED", "CONFLICT", "NO_MATCH"]},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        "explanation": {"type": "string"},
                        "conflicts": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["target_column", "relationship", "confidence", "explanation", "conflicts"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["assessments"],
        "additionalProperties": False,
    },
}
