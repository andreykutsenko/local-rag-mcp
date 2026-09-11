"""Tests 4-6 from the spec: keyword parsing and fallback. No model calls."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rag.keywords import KEYWORD_PROMPT, expand_query, extract_keywords, extract_keywords_detailed


def fake_model(reply):
    def ask(prompt):
        assert "Question:" in prompt
        return reply
    return ask


class KeywordParsingTest(unittest.TestCase):
    def test_4_well_formed_reply_is_parsed_into_a_list(self):
        keywords = extract_keywords("настройки asyncpg", ask=fake_model("asyncpg, настройки, пул соединений"))
        self.assertEqual(keywords, ["asyncpg", "настройки", "пул соединений"])

    def test_4b_bulleted_reply_with_duplicates_is_parsed_tolerantly(self):
        reply = "- FSD\n- FSD\n* слои\n1. Feature-Sliced Design\n\n"
        result = extract_keywords_detailed("что такое FSD", ask=fake_model(reply))
        self.assertEqual(result.keywords, ["FSD", "слои", "Feature-Sliced Design"])
        self.assertTrue(result.needed_tolerant_parse)
        self.assertFalse(result.used_fallback)

    def test_4c_items_longer_than_60_chars_are_dropped(self):
        reply = "asyncpg, " + "x" * 61 + ", pool"
        self.assertEqual(extract_keywords("q", ask=fake_model(reply)), ["asyncpg", "pool"])

    def test_4d_echoed_prompt_label_is_stripped(self):
        result = extract_keywords_detailed("q", ask=fake_model("Keywords: asyncpg, pool"))
        self.assertEqual(result.keywords, ["asyncpg", "pool"])
        self.assertTrue(result.needed_tolerant_parse)

    def test_5_empty_reply_falls_back_to_the_question_without_raising(self):
        result = extract_keywords_detailed("как накатить миграции", ask=fake_model("  \n- \n"))
        self.assertEqual(result.keywords, ["как накатить миграции"])
        self.assertTrue(result.used_fallback)

    def test_6_reply_longer_than_200_chars_falls_back(self):
        reply = "asyncpg, " * 40
        result = extract_keywords_detailed("q", ask=fake_model(reply))
        self.assertTrue(result.used_fallback)
        self.assertEqual(result.keywords, ["q"])

    def test_prompt_contains_the_question(self):
        self.assertIn("{question}", KEYWORD_PROMPT)

    def test_expand_query_keeps_the_original_question(self):
        self.assertEqual(expand_query("вопрос", ["a", "b"]), "вопрос\na, b")
        self.assertEqual(expand_query("вопрос", ["вопрос"]), "вопрос")


if __name__ == "__main__":
    unittest.main()
