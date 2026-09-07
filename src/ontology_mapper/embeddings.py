"""PyTorch-free Hugging Face ONNX embedding candidate generation."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Protocol

from .matching import normalize_column_name
from .models import CandidateMapping, ColumnProfile, DatasetProfile, MatchMethod

DEFAULT_MODEL_ID = "Xenova/all-MiniLM-L6-v2"


class TextEncoder(Protocol):
    """Minimal interface used by embedding candidate generation."""

    def encode(self, texts: Sequence[str]) -> list[list[float]]: ...


class OnnxEmbeddingEncoder:
    """Load a Hugging Face ONNX feature-extraction model without PyTorch."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,

    ) -> None:
        import numpy as np
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer

        self._np = np
        tokenizer_path = hf_hub_download(repo_id=model_id, filename="tokenizer.json")
        model_path = hf_hub_download(repo_id=model_id, filename="onnx/model.onnx")
        self._tokenizer = Tokenizer.from_file(tokenizer_path)
        self._tokenizer.enable_truncation(max_length=256)
        self._tokenizer.enable_padding()
        self._session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        """Tokenize, mean-pool, and return one embedding per input text."""
        if not texts:
            return []
        encodings = self._tokenizer.encode_batch(list(texts))
        input_ids = self._np.asarray([item.ids for item in encodings], dtype=self._np.int64)
        attention_mask = self._np.asarray([item.attention_mask for item in encodings], dtype=self._np.int64)
        token_types = self._np.asarray([item.type_ids for item in encodings], dtype=self._np.int64)
        available = {item.name for item in self._session.get_inputs()}
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if "token_type_ids" in available:
            inputs["token_type_ids"] = token_types
        hidden_states = self._session.run(None, inputs)[0]
        attention_mask = attention_mask[..., self._np.newaxis]
        pooled = (hidden_states * attention_mask).sum(axis=1) / self._np.clip(
            attention_mask.sum(axis=1), 1e-9, None
        )
        return pooled.tolist()


def column_representation(column: ColumnProfile) -> str:
    """Create compact semantic text from a column's observed metadata."""
    details = [
        f"Column name: {_humanize_name(column.name)}.",
        f"Observed data type: {column.observed_dtype}.",
    ]
    if column.minimum is not None and column.maximum is not None:
        details.append(f"Observed numeric range: {column.minimum} to {column.maximum}.")
    elif column.min_length is not None and column.max_length is not None:
        details.append(f"Observed string length: {column.min_length} to {column.max_length} characters.")
    if column.sample_values:
        details.append(f"Sample values: {', '.join(column.sample_values[:3])}.")
    return " ".join(details)


def generate_embedding_candidates(
    source: DatasetProfile,
    target: DatasetProfile,
    encoder: TextEncoder,
    *,
    top_k: int = 3,
    minimum_score: float = 0.35,
) -> list[CandidateMapping]:
    """Rank target columns by cosine similarity of Hugging Face embeddings."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    if not -1.0 <= minimum_score <= 1.0:
        raise ValueError("minimum_score must be between -1 and 1")

    source_columns = source.columns
    target_columns = target.columns
    source_vectors = encoder.encode([column_representation(column) for column in source_columns])
    target_vectors = encoder.encode([column_representation(column) for column in target_columns])
    if len(source_vectors) != len(source_columns) or len(target_vectors) != len(target_columns):
        raise ValueError("encoder must return one embedding per input text")

    candidates: list[CandidateMapping] = []
    for source_column, source_vector in zip(source_columns, source_vectors, strict=True):
        source_normalized = normalize_column_name(source_column.name)
        ranked = sorted(
            (
                CandidateMapping(
                    source_dataset=source.dataset.identifier,
                    source_column=source_column.name,
                    target_dataset=target.dataset.identifier,
                    target_column=target_column.name,
                    score=cosine_similarity(source_vector, target_vector),
                    method=MatchMethod.EMBEDDING,
                    source_normalized=source_normalized,
                    target_normalized=normalize_column_name(target_column.name),
                )
                for target_column, target_vector in zip(target_columns, target_vectors, strict=True)
            ),
            key=lambda item: (-item.score, item.target_column),
        )
        candidates.extend(candidate for candidate in ranked[:top_k] if candidate.score >= minimum_score)
    return candidates


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Compute cosine similarity without requiring a vector database or NumPy API."""
    if len(left) != len(right):
        raise ValueError("embedding vectors must have the same dimension")
    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = sum(value * value for value in left) ** 0.5
    right_norm = sum(value * value for value in right) ** 0.5
    return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0


def _humanize_name(name: str) -> str:
    split_case = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", name)
    return " ".join(re.findall(r"[A-Za-z0-9]+", split_case)).lower()
