"""
Unit tests for the YouTube Shorts Content Packaging Upgrade:
- Package Combination Optimization (5x5x5 -> top 3x3x3 -> up to 27 combinations)
- Authoritative Python package scoring
- Complementarity and non-duplication (Hook + Title + Description)
- Emoji validation (0-2 emojis, chains rejected, duplicate emojis rejected)
- Batch 2 retry and deterministic grounded fallback
"""

import unittest
from unittest.mock import MagicMock, patch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))

import config
import hook_generator
import metadata_ai


class TestPackageOptimization(unittest.TestCase):
    """Test suite for package combination evaluation, scoring, and emoji handling."""

    def setUp(self):
        self.transcript = (
            "Smoking cigarettes introduces toxic chemicals that significantly reduce sperm count "
            "and motility, directly damaging male reproductive health and fertility over time. "
            "Quitting allows cellular recovery within ninety days."
        )
        self.filename = "fertility_health_expert.mp4"

    def test_01_hook_emoji_validation_rules(self):
        """0-2 emojis allowed, chains and duplicates rejected."""
        # 0 emojis: valid
        h0 = {"hook": "WHY DOES SMOKING DAMAGE FERTILITY?", "supported_by_clip": True}
        val, _ = hook_generator.validate_hook(h0, transcript=self.transcript)
        self.assertTrue(val)

        # 1 emoji: valid
        h1 = {"hook": "WHY DOES SMOKING DAMAGE FERTILITY? 🚬", "supported_by_clip": True}
        val, _ = hook_generator.validate_hook(h1, transcript=self.transcript)
        self.assertTrue(val)

        # 2 distinct emojis separated by words: valid
        h2 = {"hook": "WHY DOES SMOKING 🚬 DAMAGE SPERM? 🧬", "supported_by_clip": True}
        val, _ = hook_generator.validate_hook(h2, transcript=self.transcript)
        self.assertTrue(val)

        # Emoji chain (consecutive emojis): invalid
        h_chain = {"hook": "WHY SMOKING KILLS FERTILITY 🚬🔥", "supported_by_clip": True}
        val, reason = hook_generator.validate_hook(h_chain, transcript=self.transcript)
        self.assertFalse(val)
        self.assertIn("chain", reason.lower())

        # Duplicate emojis: invalid
        h_dup = {"hook": "SMOKING 🚬 DAMAGES YOUR SPERM 🚬", "supported_by_clip": True}
        val, reason = hook_generator.validate_hook(h_dup, transcript=self.transcript)
        self.assertFalse(val)
        self.assertIn("duplicate", reason.lower())

        # 3 emojis (exceeds default MAX_HOOK_EMOJIS=2): invalid
        h3 = {"hook": "SMOKING 🚬 HURTS 🧬 SPERM ⚡", "supported_by_clip": True}
        val, reason = hook_generator.validate_hook(h3, transcript=self.transcript)
        self.assertFalse(val)
        self.assertIn("multiple emojis", reason.lower())

    def test_02_hook_scoring_semantic_emoji_boost(self):
        """Relevant emoji receives a moderate boost, meaningless/excessive does not."""
        h_no_emoji = {"hook": "WHY DOES SMOKING DAMAGE FERTILITY?", "supported_by_clip": True}
        score_no = hook_generator.score_hook(h_no_emoji, transcript=self.transcript)

        # Relevant emoji
        h_rel_emoji = {"hook": "WHY DOES SMOKING DAMAGE FERTILITY? 🚬", "supported_by_clip": True}
        score_rel = hook_generator.score_hook(h_rel_emoji, transcript=self.transcript)
        self.assertGreater(score_rel, score_no)

    def test_03_top_3_candidates_selected_for_package_evaluation(self):
        """Top 3 valid candidates from each category are evaluated in up to 27 combinations."""
        # 5 hooks
        hooks = [
            {"hook": f"HOOK TEST CANDIDATE NUMBER {i}", "supported_by_clip": True}
            for i in range(1, 6)
        ]
        # 5 titles
        titles = [
            {"title": f"How Smoking Damages Male Reproductive Health Part {i}", "grounding_score": 9.0}
            for i in range(1, 6)
        ]
        # 5 descriptions
        descs = [
            {"description": f"Exploring how cigarette toxins impact reproductive cellular health over time step {i}.", "strategy": "micro_teaser"}
            for i in range(1, 6)
        ]
        tags = ["#Shorts", "#Health", "#Fertility"]

        best_pkg = metadata_ai.select_best_package(
            hook_candidates=hooks,
            title_candidates=titles,
            desc_candidates=descs,
            hashtags=tags,
            transcript=self.transcript,
            filename=self.filename,
        )
        self.assertIsNotNone(best_pkg)
        self.assertIn("package_score", best_pkg)
        self.assertGreaterEqual(best_pkg["package_score"], config.MIN_PACKAGE_QUALITY_SCORE)
        self.assertIn("hook", best_pkg)
        self.assertIn("title", best_pkg)
        self.assertIn("description", best_pkg)

    def test_04_complementarity_rejects_duplicate_hook_title(self):
        """Combinations where title merely echoes or duplicates hook are penalized or skipped."""
        identical_text = "Smoking Damages Male Fertility Severely"
        is_dup, reason = metadata_ai.is_title_hook_duplicate(identical_text, identical_text.upper())
        self.assertTrue(is_dup)

        diff_hook = "WHY DOES SMOKING DAMAGE FERTILITY?"
        diff_title = "How Toxic Cigarette Smoke Reduces Male Sperm Count"
        is_dup2, _ = metadata_ai.is_title_hook_duplicate(diff_title, diff_hook)
        self.assertFalse(is_dup2)

    def test_05_package_scoring_formula_weights(self):
        """Package scoring accurately reflects the 7 weighted dimensions."""
        h = {"hook": "WHY DOES SMOKING DAMAGE FERTILITY?", "score": 85.0}
        t = {"title": "How Cigarette Toxins Reduce Sperm Count And Quality", "title_score": 85.0}
        d = {"description": "Examines how chemical toxins impair motility and why quitting triggers recovery.", "desc_score": 80.0}
        tags = ["#Shorts", "#Fertility", "#Health"]

        pkg_score, dims = metadata_ai.score_package(h, t, d, tags, transcript=self.transcript)
        self.assertGreaterEqual(pkg_score, 75.0)
        self.assertIn("scroll_stop", dims)
        self.assertIn("curiosity", dims)
        self.assertIn("grounding", dims)
        self.assertIn("specificity", dims)
        self.assertIn("complementarity", dims)
        self.assertIn("payoff_alignment", dims)
        self.assertIn("clarity", dims)

    def test_06_batch_2_fallback_when_batch_1_below_threshold(self):
        """When Batch 1 yields low quality or invalid candidates, Batch 2 is invoked."""
        with patch("metadata_ai.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client

            # Batch 1 mock with low quality titles
            mock_resp = MagicMock()
            mock_resp.choices = [MagicMock(message=MagicMock(content='''{
                "punchline": "SMOKING IMPACT",
                "title_candidates": [
                    {"title": "Click Here To Find Out"},
                    {"title": "Too Short"}
                ],
                "description": "Explains smoking reproductive effects.",
                "hashtags": ["#Shorts", "#Health", "#Fertility"]
            }'''))]
            mock_client.chat.completions.create.return_value = mock_resp

            # Mock generate_title_candidates for Batch 2
            with patch("metadata_ai.generate_title_candidates") as mock_b2:
                mock_b2.return_value = [
                    {"title": "How Smoking Reduces Male Sperm Count Significantly", "grounding_score": 9.5}
                ]
                with patch.object(config, "GROQ_API_KEY", "gsk_test"):
                    meta = metadata_ai.generate_shorts_metadata(self.filename, transcript=self.transcript)

                mock_b2.assert_called_once()
                self.assertEqual(meta["title"], "How Smoking Reduces Male Sperm Count Significantly")
                self.assertGreaterEqual(meta["package_quality_score"], config.MIN_PACKAGE_QUALITY_SCORE)

    def test_07_deterministic_grounded_fallback_on_complete_failure(self):
        """When AI generation completely fails or API key is absent, deterministic fallback is returned."""
        with patch.object(config, "GROQ_API_KEY", ""), patch.object(config, "OPENAI_API_KEY", ""):
            with patch.dict(os.environ, {"GROQ_API_KEY": "", "OPENAI_API_KEY": ""}):
                meta = metadata_ai.generate_shorts_metadata(self.filename, transcript=self.transcript)

        self.assertIn("title", meta)
        self.assertIn("title_variants", meta)
        self.assertIn("description", meta)
        self.assertIn("punchline", meta)
        self.assertIn("generated_hook", meta)
        self.assertIn("hashtags", meta)
        self.assertIn("tags", meta)
        self.assertIn("package_quality_score", meta)
        self.assertTrue(len(meta["title_variants"]) >= 3)
        self.assertTrue(meta["hashtags"][0] == "#Shorts")

    def test_08_claim_strength_preservation_hook(self):
        """Hook escalating a moderate effect to extreme/permanent/guaranteed claim is rejected."""
        mod_transcript = "Smoking can reduce sperm motility and affect male reproductive health."

        # Escalates to 'permanently destroys'
        cand_extreme = {
            "hook": "SMOKING PERMANENTLY DESTROYS SPERM",
            "supported_by_clip": True,
        }
        val, reason = hook_generator.validate_hook(cand_extreme, transcript=mod_transcript)
        self.assertFalse(val)
        self.assertIn("unsupported claim", reason.lower())

        # Escalates to 'guaranteed'
        cand_guar = {
            "hook": "SMOKING GUARANTEES INFERTILITY",
            "supported_by_clip": True,
        }
        val, reason = hook_generator.validate_hook(cand_guar, transcript=mod_transcript)
        self.assertFalse(val)
        self.assertIn("unsupported claim", reason.lower())

        # Preserves moderate claim strength
        cand_grounded = {
            "hook": "HOW SMOKING AFFECTS SPERM MOTILITY",
            "supported_by_clip": True,
        }
        val, reason = hook_generator.validate_hook(cand_grounded, transcript=mod_transcript)
        self.assertTrue(val, f"Expected grounded hook to pass, got: {reason}")

    def test_09_claim_strength_preservation_title(self):
        """Title escalating moderate or hedged findings to extreme claims is rejected."""
        mod_transcript = "Cigarette smoke contains toxins that may reduce sperm count over time."

        # Escalates to 'destroys' / 'incurable'
        cand_extreme = {"title": "How Cigarette Smoke Destroys Male Fertility Permanently"}
        val, reason = metadata_ai.validate_title(cand_extreme, transcript=mod_transcript)
        self.assertFalse(val)
        self.assertIn("unsupported claim", reason.lower())

        # Preserves qualified claim strength
        cand_valid = {"title": "How Cigarette Smoke Toxins Can Reduce Sperm Count"}
        val, reason = metadata_ai.validate_title(cand_valid, transcript=mod_transcript)
        self.assertTrue(val, f"Expected valid title to pass, got: {reason}")

    def test_10_claim_strength_preservation_description(self):
        """Description escalating moderate findings to absolute or fatal claims is rejected."""
        mod_transcript = "Cigarette smoke contains toxins that can reduce sperm motility over time."

        # Escalates to 'deadly' and 'guaranteed'
        cand_extreme = {
            "description": "Smoking is guaranteed to cause deadly damage to reproductive cells in every smoker."
        }
        val, reason = metadata_ai.validate_description(cand_extreme, transcript=mod_transcript)
        self.assertFalse(val)
        self.assertIn("unsupported claim", reason.lower())

        # Preserves grounded claim strength
        cand_valid = {
            "description": "Toxins in cigarette smoke can significantly reduce sperm motility over extended time."
        }
        val, reason = metadata_ai.validate_description(cand_valid, transcript=mod_transcript)
        self.assertTrue(val, f"Expected valid description to pass, got: {reason}")

    def test_11_hedged_claim_escalation_rejected(self):
        """When transcript hedges (e.g. 'might', 'preliminary'), claims of certainty are rejected."""
        hedged_tr = "Preliminary research suggests drinking green tea might potentially support metabolic health."

        # Hook claiming absolute proof / cure
        h_cand = {"hook": "GREEN TEA CURES METABOLIC DISEASE", "supported_by_clip": True}
        val, reason = hook_generator.validate_hook(h_cand, transcript=hedged_tr)
        self.assertFalse(val)
        self.assertIn("unsupported claim", reason.lower())

        # Title claiming 'always' / 'guaranteed'
        t_cand = {"title": "Why Green Tea Always Guarantees Faster Metabolism"}
        val, reason = metadata_ai.validate_title(t_cand, transcript=hedged_tr)
        self.assertFalse(val)
        self.assertIn("unsupported claim", reason.lower())

        # Grounded hedged title passes
        t_grounded = {"title": "How Green Tea Might Support Metabolic Health"}
        val, reason = metadata_ai.validate_title(t_grounded, transcript=hedged_tr)
        self.assertTrue(val, f"Expected hedged title to pass, got: {reason}")

    def test_12_escalation_may_to_will_rejected(self):
        """1. may -> will escalation rejected."""
        tr = "Eating walnuts may increase cognitive function and protect memory in older adults."
        is_val, reason = hook_generator.validate_claim_strength("Eating walnuts will increase cognitive function", transcript=tr)
        self.assertFalse(is_val)
        self.assertIn("will", reason.lower())

    def test_13_escalation_can_to_guarantee_rejected(self):
        """2. can -> guarantee escalation rejected."""
        tr = "Starting early can improve compound interest returns over several decades."
        is_val, reason = hook_generator.validate_claim_strength("Starting early guarantees high returns", transcript=tr)
        self.assertFalse(is_val)
        self.assertIn("guarantee", reason.lower())

    def test_14_escalation_associated_with_to_causes_rejected(self):
        """3. associated with -> causes rejected."""
        tr = "Skipping breakfast is associated with elevated cortisol levels during morning hours."
        is_val, reason = hook_generator.validate_claim_strength("Skipping breakfast causes elevated cortisol", transcript=tr)
        self.assertFalse(is_val)
        self.assertIn("causes", reason.lower())

    def test_15_escalation_suggests_to_proves_rejected(self):
        """4. suggests -> proves rejected."""
        tr = "Recent clinical studies suggest intermittent fasting supports cellular autophagy."
        is_val, reason = hook_generator.validate_claim_strength("Science proves fasting triggers autophagy", transcript=tr)
        self.assertFalse(is_val)
        self.assertIn("proves", reason.lower())

    def test_16_escalation_historically_to_always_rejected(self):
        """5. historically -> always rejected."""
        tr = "Historically, value stocks outperformed growth stocks during prolonged high inflation."
        is_val, reason = hook_generator.validate_claim_strength("Value stocks always outperform in inflation", transcript=tr)
        self.assertFalse(is_val)
        self.assertIn("always", reason.lower())

    def test_17_escalation_some_to_everyone_rejected(self):
        """6. some -> everyone rejected."""
        tr = "Some people experience mild headaches after consuming artificial sweeteners."
        is_val, reason = hook_generator.validate_claim_strength("Why everyone experiences headaches from sweeteners", transcript=tr)
        self.assertFalse(is_val)
        self.assertIn("everyone", reason.lower())

    def test_18_escalation_sometimes_to_always_rejected(self):
        """7. sometimes -> always rejected."""
        tr = "High intensity training sometimes leads to joint discomfort if recovery is skipped."
        is_val, reason = hook_generator.validate_claim_strength("High intensity training always hurts your joints", transcript=tr)
        self.assertFalse(is_val)
        self.assertIn("always", reason.lower())

    def test_19_escalation_possible_to_certain_rejected(self):
        """8. possible -> certain / the reason is rejected."""
        tr = "One possible explanation for sleep disruption is late evening blue light exposure."
        is_val, reason = hook_generator.validate_claim_strength("The reason is late blue light exposure", transcript=tr)
        self.assertFalse(is_val)
        self.assertIn("the reason is", reason.lower())

        is_val2, reason2 = hook_generator.validate_claim_strength("Late blue light is the reason you cannot sleep", transcript=tr)
        self.assertFalse(is_val2)
        self.assertIn("the reason", reason2.lower())

    def test_20_qualified_source_to_equally_qualified_accepted(self):
        """9. qualified source -> equally qualified output accepted."""
        tr = "Smoking cigarettes may reduce sperm motility and can affect male fertility."
        is_val, _ = hook_generator.validate_claim_strength("HOW SMOKING MAY AFFECT SPERM", transcript=tr)
        self.assertTrue(is_val)
        is_val_title, _ = hook_generator.validate_claim_strength("Why Smoking Can Reduce Sperm Quality", transcript=tr)
        self.assertTrue(is_val_title)

    def test_21_concise_strong_hook_accepted(self):
        """10. concise but non-escalated hook accepted."""
        tr = "Starting early can dramatically increase long-term wealth through compound growth."
        is_val, reason = hook_generator.validate_claim_strength("STARTING EARLY CHANGES THE MATH 💰", transcript=tr)
        self.assertTrue(is_val, f"Expected non-escalated concise hook to pass, got: {reason}")

    def test_22_grounded_hook_with_valid_strong_wording_accepted(self):
        """11. grounded hook with valid strong wording accepted when source supports it."""
        tr = "Clinical trials prove that the vaccine reduces severe hospitalization by ninety percent."
        is_val, reason = hook_generator.validate_claim_strength("PROVEN: 90% DROP IN HOSPITALIZATION", transcript=tr)
        self.assertTrue(is_val, f"Expected grounded strong hook to pass, got: {reason}")

    def test_23_title_escalation_rejected(self):
        """12. title escalation rejected."""
        tr = "Coffee consumption is linked to lower cardiovascular mortality in observational data."
        cand = {"title": "Why Coffee Consumption Causes Lower Mortality"}
        is_val, reason = metadata_ai.validate_title(cand, transcript=tr)
        self.assertFalse(is_val)
        self.assertIn("unsupported claim", reason.lower())

    def test_24_description_escalation_rejected(self):
        """13. description escalation rejected."""
        tr = "Daily walking could potentially improve insulin sensitivity in sedentary adults."
        cand = {"description": "Daily walking will definitely guarantee perfect blood sugar control."}
        is_val, reason = metadata_ai.validate_description(cand, transcript=tr)
        self.assertFalse(is_val)
        self.assertIn("unsupported claim", reason.lower())

    def test_25_fallback_remains_claim_safe(self):
        """14. fallback remains claim-safe."""
        tr = "Investing early in index funds might yield substantial compounded wealth over decades."
        fb_hook = hook_generator.get_fallback_hook(transcript=tr)
        fb_title = metadata_ai.extract_grounded_fallback_title(tr)
        fb_desc = metadata_ai.extract_grounded_fallback_description(tr)

        val_h, r_h = hook_generator.validate_claim_strength(fb_hook["hook"], transcript=tr)
        self.assertTrue(val_h, f"Fallback hook failed claim strength: {r_h}")

        val_t, r_t = hook_generator.validate_claim_strength(fb_title, transcript=tr)
        self.assertTrue(val_t, f"Fallback title failed claim strength: {r_t}")

        val_d, r_d = hook_generator.validate_claim_strength(fb_desc, transcript=tr)
        self.assertTrue(val_d, f"Fallback description failed claim strength: {r_d}")

    def test_26_valid_package_selectable_when_stronger_candidates_rejected(self):
        """15. valid package remains selectable when stronger candidates are rejected."""
        tr = "Cigarette smoking is associated with reduced sperm motility and lower fertility."
        # Mix of escalated and grounded candidates
        hooks = [
            {"hook": "SMOKING PERMANENTLY DESTROYS FERTILITY", "supported_by_clip": True}, # escalated -> rejected
            {"hook": "SMOKING GUARANTEES IMPOTENCE", "supported_by_clip": True},          # escalated -> rejected
            {"hook": "HOW SMOKING IMPACTS SPERM MOTILITY", "supported_by_clip": True},     # qualified -> valid
        ]
        titles = [
            {"title": "Why Smoking Always Causes Complete Infertility"},                   # escalated -> rejected
            {"title": "How Cigarette Smoking Is Linked To Lower Sperm Motility", "grounding_score": 9.5}, # valid
        ]
        descs = [
            {"description": "Smoking will definitely destroy your reproductive cells.", "strategy": "micro_teaser"}, # escalated -> rejected
            {"description": "Explores how chemical toxins are associated with reduced cellular motility.", "strategy": "micro_teaser"}, # valid
        ]
        tags = ["#Shorts", "#Health", "#Fertility"]

        best_pkg = metadata_ai.select_best_package(
            hook_candidates=hooks,
            title_candidates=titles,
            desc_candidates=descs,
            hashtags=tags,
            transcript=tr,
        )
        self.assertIsNotNone(best_pkg)
        self.assertEqual(best_pkg["hook"], "HOW SMOKING IMPACTS SPERM MOTILITY")
        self.assertEqual(best_pkg["title"], "How Cigarette Smoking Is Linked To Lower Sperm Motility")
        self.assertIn("associated with", best_pkg["description"])


if __name__ == "__main__":
    unittest.main()

