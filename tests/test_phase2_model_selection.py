"""
Phase 2 Tests: Eliminate Unsafe Dynamic Model Selection & Verify Production Model.
Verifies that:
1. TEST 5: Explicit configured model exists -> exactly that model is used.
2. TEST 6: Explicit configured model does not exist -> clear configuration failure (ModelConfigurationError), no arbitrary model substitution.
3. TEST 7: Model configuration is absent -> deterministic documented default (DEFAULT_GROQ_CHAT_MODEL) is used.
4. TEST 7b: If default model is absent on provider -> fail fast with ModelConfigurationError.
5. All three AI subsystems (clip discovery, metadata generation, hook generation) resolve the identical model.
6. Workflow default in .github/workflows/pipeline.yml strictly matches config.py DEFAULT_GROQ_CHAT_MODEL.
7. No reference to obsolete llama-3.3-70b-versatile remains in active production configuration.
8. No automatic fallback to openai/gpt-oss-120b exists.
"""
import os
import re
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))

import clip_detection
from clip_detection import _resolve_groq_model, ModelConfigurationError
import config
import hook_generator
import metadata_ai


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

    def test_new_production_model_is_configured_default(self):
        """Phase 4.1: The new production model (openai/gpt-oss-20b) is the configured default."""
        self.assertEqual(config.DEFAULT_GROQ_CHAT_MODEL, "openai/gpt-oss-20b")

    def test_test5_explicit_configured_model_exists_is_used(self):
        """TEST 5: Explicit configured model exists -> exactly that model is used."""
        client = self._make_mock_client(["openai/gpt-oss-20b", "qwen/qwen3.8-27b", "openai/gpt-oss-120b"])
        resolved = _resolve_groq_model(client, configured_model="openai/gpt-oss-20b")
        self.assertEqual(resolved, "openai/gpt-oss-20b")

    def test_test6_explicit_configured_model_not_exist_fails_fast(self):
        """TEST 6: Explicit configured model does not exist -> clear configuration failure, NO arbitrary substitution."""
        # Provider only has gpt-oss-120b and qwen/qwen3.8-27b, but user configured non_existent_model
        client = self._make_mock_client(["openai/gpt-oss-120b", "qwen/qwen3.8-27b"])

        with self.assertRaises(ModelConfigurationError) as ctx:
            _resolve_groq_model(client, configured_model="non_existent_model_v1")

        self.assertIn("non_existent_model_v1", str(ctx.exception))
        # Ensure it NEVER returned gpt-oss-120b or any arbitrary model!

    def test_test7_model_configuration_absent_uses_documented_default(self):
        """TEST 7: Model configuration is absent -> deterministic documented behavior (DEFAULT_GROQ_CHAT_MODEL)."""
        client = self._make_mock_client(["openai/gpt-oss-20b", "qwen/qwen3.8-27b"])
        # Pass empty/None configured_model and empty config.GROQ_CHAT_MODEL
        with patch.object(clip_detection, "GROQ_CHAT_MODEL", ""):
            resolved = _resolve_groq_model(client, configured_model=None)
            self.assertEqual(resolved, config.DEFAULT_GROQ_CHAT_MODEL)
            self.assertEqual(resolved, "openai/gpt-oss-20b")

    def test_test7b_absent_default_model_fails_if_default_not_on_provider(self):
        """TEST 7b: If default model is also not on provider, fail fast with ModelConfigurationError."""
        client = self._make_mock_client(["random-other-model"])
        with patch.object(clip_detection, "GROQ_CHAT_MODEL", ""):
            with self.assertRaises(ModelConfigurationError):
                _resolve_groq_model(client, configured_model=None)

    def test_all_three_subsystems_resolve_same_model(self):
        """Phase 4.2: All three AI subsystems (discovery, metadata, hooks) resolve the same model."""
        client = self._make_mock_client(["openai/gpt-oss-20b"])
        with patch.object(clip_detection, "GROQ_CHAT_MODEL", ""):
            model_discovery = clip_detection._resolve_groq_model(client, configured_model=None)
            model_metadata = metadata_ai._resolve_groq_model(client, configured_model=None)
            model_hooks = hook_generator._resolve_groq_model(client, configured_model=None)

            self.assertEqual(model_discovery, "openai/gpt-oss-20b")
            self.assertEqual(model_metadata, "openai/gpt-oss-20b")
            self.assertEqual(model_hooks, "openai/gpt-oss-20b")
            self.assertEqual(model_discovery, model_metadata)
            self.assertEqual(model_metadata, model_hooks)

    def test_workflow_default_matches_config(self):
        """Phase 4.5: The workflow default in pipeline.yml matches config.py."""
        workflow_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", ".github", "workflows", "pipeline.yml")
        )
        self.assertTrue(os.path.exists(workflow_path), f"Workflow file not found at {workflow_path}")
        with open(workflow_path, "r", encoding="utf-8") as f:
            workflow_content = f.read()

        match = re.search(r"GROQ_CHAT_MODEL:\s*\${{\s*vars\.GROQ_CHAT_MODEL\s*\|\|\s*'([^']+)'\s*}}", workflow_content)
        self.assertIsNotNone(match, "Could not find GROQ_CHAT_MODEL default in pipeline.yml")
        workflow_default = match.group(1)
        self.assertEqual(workflow_default, config.DEFAULT_GROQ_CHAT_MODEL)
        self.assertEqual(workflow_default, "openai/gpt-oss-20b")

    def test_no_obsolete_llama_in_active_production_config(self):
        """Phase 4.6: No reference to llama-3.3-70b-versatile remains in active production configuration."""
        self.assertNotEqual(config.DEFAULT_GROQ_CHAT_MODEL, "llama-3.3-70b-versatile")

        workflow_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", ".github", "workflows", "pipeline.yml")
        )
        with open(workflow_path, "r", encoding="utf-8") as f:
            workflow_content = f.read()
        self.assertNotIn("llama-3.3-70b-versatile", workflow_content)

        config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts", "config.py"))
        with open(config_path, "r", encoding="utf-8") as f:
            config_content = f.read()
        self.assertNotIn("llama-3.3-70b-versatile", config_content)

    def test_no_gpt_oss_120b_as_automatic_fallback(self):
        """Phase 4.7: No reference to openai/gpt-oss-120b exists as an automatic fallback."""
        # If openai/gpt-oss-120b is the only model available from provider, but configured default is gpt-oss-20b,
        # the system must NOT automatically fall back to gpt-oss-120b.
        client = self._make_mock_client(["openai/gpt-oss-120b"])
        with patch.object(clip_detection, "GROQ_CHAT_MODEL", ""):
            with self.assertRaises(ModelConfigurationError) as ctx:
                _resolve_groq_model(client, configured_model=None)
            self.assertIn("openai/gpt-oss-20b", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
