import tempfile
from pathlib import Path
import os
import unittest
from unittest.mock import patch

from ontology_mapper.models import CandidateMapping, ColumnProfile, MatchMethod, Relationship
from ontology_mapper.openai_provider import OpenAIConfig
from ontology_mapper.reasoning import (
    ReasoningError,
    build_reasoning_prompt,
    reason_candidate,
    validate_reasoning_payload,
)


class FakeReasoningClient:
    def reason(self, prompt: str) -> dict[str, object]:
        self.prompt = prompt
        return {
            "relationship": "MATCH",
            "confidence": 0.91,
            "explanation": "Both fields identify the same concept.",
            "conflicts": [],
        }


def profile(name: str) -> ColumnProfile:
    return ColumnProfile(name, "string", 2, 0, 0.0, 2, 1.0, ["one", "two"])


def candidate() -> CandidateMapping:
    return CandidateMapping(
        "source", "customer_id", "target", "client_identifier", 0.8,
        MatchMethod.EMBEDDING, "customerid", "clientidentifier"
    )


class ReasoningTests(unittest.TestCase):
    def test_prompt_contains_only_the_candidate_profiles_and_evidence(self) -> None:
        prompt = build_reasoning_prompt(profile("customer_id"), profile("client_identifier"), candidate())

        self.assertIn('"source_column_profile"', prompt)
        self.assertIn('"target_column_profile"', prompt)
        self.assertIn('"candidate"', prompt)
        self.assertNotIn("annual_income", prompt)

    def test_fake_client_produces_structured_result(self) -> None:
        client = FakeReasoningClient()
        result = reason_candidate(profile("customer_id"), profile("client_identifier"), candidate(), client)

        self.assertEqual(result.relationship, Relationship.MATCH)
        self.assertEqual(result.candidate_method, MatchMethod.EMBEDDING)
        self.assertIn("customer_id", client.prompt)

    def test_all_relationship_labels_are_accepted(self) -> None:
        for relationship in Relationship:
            result = validate_reasoning_payload(
                {"relationship": relationship.value, "confidence": 0.5, "explanation": "valid", "conflicts": []}
            )
            self.assertEqual(result[0], relationship)

    def test_invalid_payloads_are_rejected(self) -> None:
        with self.assertRaises(ReasoningError):
            validate_reasoning_payload({"relationship": "MAYBE"})
        with self.assertRaises(ReasoningError):
            validate_reasoning_payload(
                {"relationship": "MATCH", "confidence": 1.2, "explanation": "x", "conflicts": []}
            )

    def test_openai_environment_configuration_requires_all_values(self) -> None:
        with patch.dict(os.environ, {"OPENAI_API_KEY": "", "OPENAI_URL": "", "OPENAI_MODEL": ""}):
            with self.assertRaises(ReasoningError):
                OpenAIConfig.from_environment()

    def test_openai_configuration_loads_project_local_dotenv(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dotenv_path = Path(directory) / ".env"
            dotenv_path.write_text("OPENAI_API_KEY=dotenv-key\nOPENAI_URL=https://example.test/v1\nOPENAI_MODEL=test-model\n")
            with patch.dict(os.environ, {}, clear=True):
                config = OpenAIConfig.from_environment(dotenv_path)

        self.assertEqual(config.api_key, "dotenv-key")
        self.assertEqual(config.base_url, "https://example.test/v1")
        self.assertEqual(config.model, "test-model")

    def test_shell_environment_overrides_dotenv_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dotenv_path = Path(directory) / ".env"
            dotenv_path.write_text("OPENAI_API_KEY=dotenv-key\nOPENAI_URL=https://dotenv.test/v1\nOPENAI_MODEL=dotenv-model\n")
            with patch.dict(os.environ, {
                "OPENAI_API_KEY": "shell-key",
                "OPENAI_URL": "https://shell.test/v1",
                "OPENAI_MODEL": "shell-model",
            }, clear=True):
                config = OpenAIConfig.from_environment(dotenv_path)

        self.assertEqual(config.api_key, "shell-key")
        self.assertEqual(config.base_url, "https://shell.test/v1")
        self.assertEqual(config.model, "shell-model")
