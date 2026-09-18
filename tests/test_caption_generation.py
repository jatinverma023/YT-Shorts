"""
Unit tests for Caption Upgrade #1: Professional Roman Hindi / Hinglish Captions.
Verifies:
1. English words preserved (never translated or corrupted).
2. Devanagari Hindi correctly romanized to readable Roman Hindi.
3. Mixed Hinglish / technical terms ('coding', 'AI', 'students', 'degree') stay intact.
4. Strict 1:1 word-level timestamp preservation (start and end timestamps unchanged).
5. Phrase grouping into 1-2 balanced lines with optimal '\\N' line break.
6. Active word karaoke highlighting (yellow highlight & 110% scale) vs inactive base white.
7. Shorts safe positioning: Alignment 2 (Bottom Center) & MarginV 280.
8. Top punchline / hook header: Alignment 8 (Top Center) remains fully intact.
9. Special character safety ({, }, \\ stripped/escaped to prevent ASS breakage).
10. Empty words list handling.
"""
import os
import sys
import tempfile
import unittest

# Ensure scripts directory is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))

import config
from transliterate import (
    has_devanagari,
    romanize_single_word,
    romanize_words,
)
import transcribe


class TestCaptionGeneration(unittest.TestCase):

    def test_english_words_preserved(self):
        """Pure English input words must remain English words, not corrupted or translated."""
        words = [
            {"word": "Consistency", "start": 0.0, "end": 0.6},
            {"word": "is", "start": 0.6, "end": 0.8},
            {"word": "the", "start": 0.8, "end": 0.9},
            {"word": "key", "start": 0.9, "end": 1.2},
            {"word": "to", "start": 1.2, "end": 1.4},
            {"word": "success", "start": 1.4, "end": 2.0},
        ]
        romanized = romanize_words(words, uppercase=True)
        self.assertEqual(len(romanized), len(words))
        expected = ["CONSISTENCY", "IS", "THE", "KEY", "TO", "SUCCESS"]
        actual = [w["word"] for w in romanized]
        self.assertEqual(actual, expected)

    def test_hindi_devanagari_to_roman_hindi(self):
        """Hindi speech in Devanagari is converted to standard, readable Roman Hindi."""
        sentence = "अगर आप सीख रहे हो तो ये गलती मत करना"
        words = [{"word": w, "start": i * 0.4, "end": (i + 1) * 0.4} for i, w in enumerate(sentence.split())]
        romanized = romanize_words(words, uppercase=True)
        result_words = [w["word"] for w in romanized]
        self.assertEqual(
            result_words,
            ["AGAR", "AAP", "SEEKH", "RAHE", "HO", "TOH", "YE", "GALTI", "MAT", "KARNA"]
        )

    def test_hinglish_technical_terms(self):
        """Spoken English technical terms inside Hindi speech remain English."""
        # 'अगर आप coding सीख रहे हो toh ye mistake mat karna'
        words = [
            {"word": "अगर", "start": 0.0, "end": 0.3},
            {"word": "आप", "start": 0.3, "end": 0.6},
            {"word": "coding", "start": 0.6, "end": 1.0},
            {"word": "सीख", "start": 1.0, "end": 1.3},
            {"word": "रहे", "start": 1.3, "end": 1.5},
            {"word": "हो", "start": 1.5, "end": 1.8},
            {"word": "toh", "start": 1.8, "end": 2.0},
            {"word": "ye", "start": 2.0, "end": 2.2},
            {"word": "mistake", "start": 2.2, "end": 2.6},
            {"word": "मत", "start": 2.6, "end": 2.9},
            {"word": "करना", "start": 2.9, "end": 3.3},
        ]
        romanized = romanize_words(words, uppercase=True)
        actual = [w["word"] for w in romanized]
        expected = ["AGAR", "AAP", "CODING", "SEEKH", "RAHE", "HO", "TOH", "YE", "MISTAKE", "MAT", "KARNA"]
        self.assertEqual(actual, expected)

    def test_word_timestamps_preserved_1_to_1(self):
        """Transliteration must preserve strict 1:1 mapping of word count, start and end timestamps."""
        words = [
            {"word": "India", "start": 1.15, "end": 1.45},
            {"word": "में", "start": 1.48, "end": 1.62},
            {"word": "students", "start": 1.65, "end": 2.10},
            {"word": "डिग्री", "start": 2.12, "end": 2.50},
            {"word": "ke", "start": 2.52, "end": 2.70},
            {"word": "पीछे", "start": 2.72, "end": 3.05},
            {"word": "भागते", "start": 3.08, "end": 3.42},
            {"word": "हैं", "start": 3.45, "end": 3.70},
        ]
        romanized = romanize_words(words)
        self.assertEqual(len(romanized), len(words))

        for orig, rom in zip(words, romanized):
            self.assertEqual(orig["start"], rom["start"])
            self.assertEqual(orig["end"], rom["end"])
            # Word string should not be empty
            self.assertTrue(len(rom["word"]) > 0)

    def test_phrase_grouping_two_lines_balanced(self):
        """Long phrases are grouped and split into 2 visually balanced lines with \\N."""
        # 6 words that exceed max chars per line
        words = [
            {"word": "AGAR", "start": 0.0, "end": 0.3},
            {"word": "AAP", "start": 0.3, "end": 0.6},
            {"word": "CODING", "start": 0.6, "end": 1.0},
            {"word": "SEEKH", "start": 1.0, "end": 1.3},
            {"word": "RAHE", "start": 1.3, "end": 1.6},
            {"word": "HO", "start": 1.6, "end": 1.9},
        ]
        phrases = transcribe.group_words_into_phrases(words, max_words=7, max_chars=60)
        self.assertEqual(len(phrases), 1)
        phrase = phrases[0]

        # Verify split index produces balanced lines
        split_k = transcribe.find_phrase_split(phrase, max_chars_per_line=18)
        self.assertTrue(1 <= split_k < len(phrase))
        line1 = " ".join(w["word"] for w in phrase[:split_k])
        line2 = " ".join(w["word"] for w in phrase[split_k:])
        self.assertTrue(len(line1) > 0)
        self.assertTrue(len(line2) > 0)
        # Line lengths should be reasonably close
        self.assertLessEqual(abs(len(line1) - len(line2)), 12)

    def test_karaoke_active_word_styling(self):
        """Active word has yellow highlight color and 110% scale, inactive words are base white."""
        words = [
            {"word": "CODING", "start": 0.0, "end": 0.5},
            {"word": "SEEKH", "start": 0.5, "end": 1.0},
            {"word": "LO", "start": 1.0, "end": 1.5},
        ]
        with tempfile.NamedTemporaryFile(suffix=".ass", delete=False) as f:
            ass_path = f.name

        try:
            transcribe.generate_ass_captions(words, ass_path)
            with open(ass_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Verify active word highlight code and scale
            self.assertIn(r"\fscx110\fscy110", content)
            self.assertIn(config.CAPTION_HIGHLIGHT_COLOR, content)
            # Verify base white color and normal scale reset
            self.assertIn(r"\fscx100\fscy100", content)
            self.assertIn(config.CAPTION_BASE_COLOR, content)
        finally:
            if os.path.exists(ass_path):
                os.remove(ass_path)

    def test_margin_bottom_and_alignment(self):
        """ASS file must specify Alignment: 2 (Bottom Center) and safe bottom margin."""
        words = [{"word": "TEST", "start": 0.0, "end": 1.0}]
        with tempfile.NamedTemporaryFile(suffix=".ass", delete=False) as f:
            ass_path = f.name

        try:
            transcribe.generate_ass_captions(words, ass_path)
            with open(ass_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Style line format check: Alignment 2 and MarginV 280
            # Style: Default,Arial,52,&H00FFFFFF&,&H000000FF,&H00000000&,&H80000000,-1,0,0,0,100,100,0,0,1,3.5,1.5,2,40,40,280,1
            self.assertIn(",2,40,40,280,1", content)
        finally:
            if os.path.exists(ass_path):
                os.remove(ass_path)

    def test_hook_text_intact(self):
        """Top punchline hook header (Alignment: 8) must remain intact and functional."""
        words = [{"word": "START", "start": 0.0, "end": 1.0}]
        hook_text = "Khan Sir with Raj Shamani 🥰"
        with tempfile.NamedTemporaryFile(suffix=".ass", delete=False) as f:
            ass_path = f.name

        try:
            transcribe.generate_ass_captions(words, ass_path, punchline=hook_text, total_duration=25.0)
            with open(ass_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Check HeaderPunchline style definition (Alignment 8)
            self.assertIn("Style: HeaderPunchline", content)
            self.assertIn(f",8,50,50,{config.PUNCHLINE_MARGIN_TOP},1", content)
            # Check HeaderPunchline dialogue line
            self.assertIn("Dialogue: 1,0:00:00.00,0:00:25.00,HeaderPunchline,,0,0,0,,Khan Sir with Raj Shamani 🥰", content)
        finally:
            if os.path.exists(ass_path):
                os.remove(ass_path)

    def test_special_character_escaping(self):
        """Special characters like braces and backslashes in words must not break ASS formatting."""
        words = [
            {"word": "{dangerous}", "start": 0.0, "end": 0.5},
            {"word": "normal\\backslash", "start": 0.5, "end": 1.0},
        ]
        with tempfile.NamedTemporaryFile(suffix=".ass", delete=False) as f:
            ass_path = f.name

        try:
            transcribe.generate_ass_captions(words, ass_path)
            with open(ass_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Braces inside word text must have been stripped
            self.assertNotIn("{{dangerous}}", content)
            self.assertIn("DANGEROUS", content)
            self.assertIn("NORMALBACKSLASH", content)
        finally:
            if os.path.exists(ass_path):
                os.remove(ass_path)

    def test_empty_words_handling(self):
        """Empty words list returns valid ASS file with script headers and no crashes."""
        with tempfile.NamedTemporaryFile(suffix=".ass", delete=False) as f:
            ass_path = f.name

        try:
            res_path = transcribe.generate_ass_captions([], ass_path, punchline="Test Header")
            self.assertEqual(res_path, ass_path)
            with open(ass_path, "r", encoding="utf-8") as f:
                content = f.read()

            self.assertIn("[Script Info]", content)
            self.assertIn("Style: Default", content)
            self.assertIn("Style: HeaderPunchline", content)
            self.assertIn("Test Header", content)
        finally:
            if os.path.exists(ass_path):
                os.remove(ass_path)

    def test_word_by_word_karaoke_timing_and_progression(self):
        """
        Verify true word-by-word karaoke timing on:
        'अगर आप coding सीख रहे हो तो focus'
        1. Generates exactly 8 Dialogue events for the 8 words in the phrase.
        2. Exactly one word is highlighted per event with yellow highlight & 110% scale.
        3. Highlight moves sequentially: AGAR -> AAP -> CODING -> SEEKH -> RAHE -> HO -> TOH -> FOCUS.
        4. Independent word timing across 2 balanced lines joined by \\N.
        5. Exact start and end timestamps match original word timestamps without overlap.
        """
        words = [
            {"word": "अगर", "start": 0.20, "end": 0.50},
            {"word": "आप", "start": 0.50, "end": 0.80},
            {"word": "coding", "start": 0.80, "end": 1.30},
            {"word": "सीख", "start": 1.30, "end": 1.70},
            {"word": "रहे", "start": 1.70, "end": 2.00},
            {"word": "हो", "start": 2.00, "end": 2.30},
            {"word": "तो", "start": 2.30, "end": 2.60},
            {"word": "focus", "start": 2.60, "end": 3.10},
        ]
        with tempfile.NamedTemporaryFile(suffix=".ass", delete=False) as f:
            ass_path = f.name

        try:
            transcribe.generate_ass_captions(words, ass_path, punchline="Podcast Hook 🔥", total_duration=3.5)
            with open(ass_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            # Separate header dialogue and caption dialogue lines
            header_events = [l.strip() for l in lines if "HeaderPunchline" in l and l.startswith("Dialogue:")]
            caption_events = [l.strip() for l in lines if "Default" in l and l.startswith("Dialogue:")]

            # 1. Header punchline is separate and unaffected
            self.assertEqual(len(header_events), 1)
            self.assertIn("Podcast Hook 🔥", header_events[0])

            # 2. Exactly 8 dialogue events for the 8-word phrase
            self.assertEqual(len(caption_events), 8)

            expected_words = ["AGAR", "AAP", "CODING", "SEEKH", "RAHE", "HO", "TOH", "FOCUS"]
            expected_times = [
                ("0:00:00.20", "0:00:00.50"),
                ("0:00:00.50", "0:00:00.80"),
                ("0:00:00.80", "0:00:01.30"),
                ("0:00:01.30", "0:00:01.70"),
                ("0:00:01.70", "0:00:02.00"),
                ("0:00:02.00", "0:00:02.30"),
                ("0:00:02.30", "0:00:02.60"),
                ("0:00:02.60", "0:00:03.10"),
            ]

            for i, (event_line, target_w, (t_start, t_end)) in enumerate(zip(caption_events, expected_words, expected_times)):
                # Verify exact timing
                self.assertIn(f"Dialogue: 0,{t_start},{t_end},Default,", event_line)

                # Verify active word is highlighted with yellow and 110% scale
                active_pattern = f"{config.CAPTION_HIGHLIGHT_COLOR}\\fscx110\\fscy110}}{target_w}"
                self.assertIn(active_pattern, event_line)

                # Verify only 1 word is highlighted per event
                self.assertEqual(event_line.count(config.CAPTION_HIGHLIGHT_COLOR), 1)

                # Verify line break \\N exists in every event
                self.assertIn(r"\N", event_line)

        finally:
            if os.path.exists(ass_path):
                os.remove(ass_path)


if __name__ == "__main__":
    unittest.main()
