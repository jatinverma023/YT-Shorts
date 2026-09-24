"""
Phase 2 Tests: Eliminate Unsafe Dynamic Model Selection.
Verifies that:
1. TEST 5: Explicit configured model exists -> exactly that model is used.
2. TEST 6: Explicit configured model does not exist -> clear configuration failure (ModelConfigurationError), no arbitrary model substitution.
3. TEST 7: Model configuration is absent -> deterministic documented default (DEFAULT_GROQ_CHAT_MODEL) is used.
"""
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))

import clip_detection
from clip_detection import _resolve_groq_model, ModelConfigurationError
import config


class TestPhase2ModelSelection(unittest.TestCase):

    def _make_mock_client(self, model_ids):
        mock_client = MagicMock()
        mock_data = []
        for mid in model_ids:
            m = MagicMock()
            m.id = mid
            mock_data.append(m)
        mock_client.models.list.return_value.data = mock_data
        return mock_client

    def test_test5_explicit_configured_model_exists_is_used(self):
        """TEST 5: Explicit configured model exists -> exactly that model is used."""
        client = self._make_mock_client(["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "openai/gpt-oss-120b"])
        resolved = _resolve_groq_model(client, configured_model="llama-3.3-70b-versatile")
        self.assertEqual(resolved, "llama-3.3-70b-versatile")

    def test_test6_explicit_configured_model_not_exist_fails_fast(self):
        """TEST 6: Explicit configured model does not exist -> clear configuration failure, NO arbitrary substitution."""
        # Provider only has gpt-oss-120b and llama-3.1-8b-instant, but user configured non_existent_model
        client = self._make_mock_client(["openai/gpt-oss-120b", "llama-3.1-8b-instant"])

        with self.assertRaises(ModelConfigurationError) as ctx:
            _resolve_groq_model(client, configured_model="non_existent_model_v1")

        self.assertIn("non_existent_model_v1", str(ctx.exception))
        # Ensure it NEVER returned gpt-oss-120b or any arbitrary model!

    def test_test7_model_configuration_absent_uses_documented_default(self):
        """TEST 7: Model configuration is absent -> deterministic documented behavior (DEFAULT_GROQ_CHAT_MODEL)."""
        client = self._make_mock_client(["llama-3.3-70b-versatile", "llama-3.1-8b-instant"])
        # Pass empty/None configured_model and empty config.GROQ_CHAT_MODEL
        with patch.object(clip_detection, "GROQ_CHAT_MODEL", ""):
            resolved = _resolve_groq_model(client, configured_model=None)
            self.assertEqual(resolved, config.DEFAULT_GROQ_CHAT_MODEL)
            self.assertEqual(resolved, "llama-3.3-70b-versatile")

    def test_test7b_absent_default_model_fails_if_default_not_on_provider(self):
        """TEST 7b: If default model is also not on provider, fail fast with ModelConfigurationError."""
        client = self._make_mock_client(["random-other-model"])
        with patch.object(clip_detection, "GROQ_CHAT_MODEL", ""):
            with self.assertRaises(ModelConfigurationError):
                _resolve_groq_model(client, configured_model=None)


if __name__ == "__main__":
    unittest.main()
