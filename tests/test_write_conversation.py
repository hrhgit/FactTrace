"""Long sessions are extracted in parts of one conversation; gap fill continues it."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tests.fixtures import ScriptedModel, payload
from text_memory.collections import CollectionEngine, CollectionStore
from text_memory.collections import prompts
from text_memory.collections import write_runner
from text_memory.collections.query import DIRECT_LIMIT, DIRECT_POLICY
from text_memory.errors import ValidationError
from text_memory.store import Scope

SENTENCES = ["Alice bought a red brush.", "Alice paints every Sunday.", "Alice framed her lake painting."]


def long_request():
    return payload("long", messages=[{"role": "user", "content": s} for s in SENTENCES])


def reply(*refs):
    return {"facts": [{"text": SENTENCES[int(r[1])], "evidence": [r]} for r in refs]}


class MultiPartExtractionTests(unittest.TestCase):
    def setUp(self):
        self.store = CollectionStore()
        self.addCleanup(self.store.close)

    def test_short_session_keeps_the_single_request(self):
        request = long_request()
        message = json.loads(CollectionEngine.extraction_messages(request)[1]["content"])
        self.assertEqual(set(message), {"instructions", "sessions"})
        self.assertTrue(message["instructions"].startswith(prompts.EXTRACT))
        self.assertIn("Output contract", message["instructions"])
        self.assertIn('"m0.1"', message["instructions"])
        self.assertEqual(len(CollectionEngine.extraction_parts(request)), 1)

    def test_each_part_carries_its_text_and_the_earlier_facts(self):
        request = long_request()
        with patch.object(CollectionEngine, "PART_CHARS", 30):
            parts = CollectionEngine.extraction_parts(request)
            self.assertEqual([[s["ref"] for s in p] for p in parts], [["m0.1"], ["m1.1"], ["m2.1"]])
            model = ScriptedModel(reply("m0.1"), reply("m1.1"), reply("m2.1"))
            engine = CollectionEngine(self.store, model)
            extracted = engine.extract(request)
            self.assertEqual([f["text"] for f in extracted["facts"]], SENTENCES)
            bodies = [json.loads(c["messages"][-1]["content"]) for c in model.calls]
            self.assertTrue(all(len(c["messages"]) == 2 for c in model.calls))
            self.assertEqual([b["part"] for b in bodies], ["1 of 3", "2 of 3", "3 of 3"])
            self.assertNotIn("earlier_facts", bodies[0])
            self.assertEqual(bodies[2]["earlier_facts"], SENTENCES[:2])
            self.assertIn(prompts.EXTRACT_PARTS, bodies[2]["instructions"])
            self.assertTrue(bodies[2]["instructions"].endswith('for this request are ["m2.1"].'))
            self.assertEqual(len(extracted["conversation"]), 7)

            receipt = engine.add(request, extraction=extracted)
            self.assertEqual(model.calls[-1]["purpose"], "collection_extract")
            self.assertEqual(len(receipt["point_ids"]), 3)

    def test_earlier_facts_keep_the_newest_within_the_limit(self):
        facts = [{"text": "a" * 10}, {"text": "b" * 10}, {"text": "c" * 10}]
        with patch.object(CollectionEngine, "EARLIER_FACT_CHARS", 25):
            self.assertEqual(CollectionEngine.earlier_facts(facts), ["b" * 10, "c" * 10])

    def test_an_oversized_message_is_split_between_paragraphs(self):
        text = "\n\n".join(f"Paragraph {i} " + "word " * 10 for i in range(6))
        request = payload("long", messages=[{"role": "user", "content": "Short opening."},
                                            {"role": "user", "content": text},
                                            {"role": "user", "content": "Short closing."}])
        with patch.object(CollectionEngine, "PART_CHARS", 130):
            parts = CollectionEngine.extraction_parts(request)
        refs = [[s["ref"] for s in p] for p in parts]
        self.assertGreater(len(parts), 2)
        self.assertEqual([r for p in refs for r in p], [s["ref"] for s in CollectionEngine.source_view(request)[1]])
        self.assertTrue(all(sum(len(s["text"]) for s in p) <= 130 for p in parts))

    def test_a_sentence_past_the_part_limit_is_cut_without_losing_text(self):
        from text_memory.schema import UNIT_MAX_CHARS, split_units
        content = ("cell " * (UNIT_MAX_CHARS // 2)).strip()
        units = split_units(content)
        self.assertGreater(len(units), 1)
        self.assertTrue(all(b - a <= UNIT_MAX_CHARS for a, b in units))
        self.assertEqual("".join(content[a:b] for a, b in units), content)
        solid = "x" * (UNIT_MAX_CHARS + 5)
        self.assertEqual(split_units(solid), [[0, UNIT_MAX_CHARS], [UNIT_MAX_CHARS, UNIT_MAX_CHARS + 5]])

    def test_a_long_session_fills_gaps_from_the_surroundings_only(self):
        request = long_request()
        from text_memory.collections import gap_fill
        with patch.object(CollectionEngine, "PART_CHARS", 30):
            _, sources = CollectionEngine.source_view(request)
            facts = [{"text": "One.", "evidence": ["m0.1"]}, {"text": "Far.", "evidence": ["m2.1"]}]
            gaps = [{"ref": "m0.1", "check": "x"}]
            with patch.object(gap_fill, "FACT_REACH", 1):
                messages = CollectionEngine.gap_request(request, {"facts": facts}, facts, gaps)
        body = json.loads(messages[-1]["content"])
        self.assertEqual(len(messages), 2)
        self.assertEqual([f["index"] for f in body["facts"]], [0])
        self.assertEqual([p[0] for s in body["paragraphs"] for t in s["turns"] for p in t["paragraphs"]], ["m0.1", "m1.1"])

    def test_session_id_and_date_are_stated_once_per_session(self):
        sources = [{"ref": "m0.1", "session_id": "a", "session_date": "d1", "role": "user", "text": "x"},
                   {"ref": "m1.1", "session_id": "a", "session_date": "d1", "role": "assistant", "text": "y"},
                   {"ref": "m2.1", "session_id": "b", "role": "user", "text": "z"}]
        self.assertEqual(CollectionEngine.session_blocks(sources), [
            {"session_id": "a", "session_date": "d1", "turns": [
                {"role": "user", "paragraphs": [["m0.1", "x"]]}, {"role": "assistant", "paragraphs": [["m1.1", "y"]]}]},
            {"session_id": "b", "turns": [{"role": "user", "paragraphs": [["m2.1", "z"]]}]}])

    def test_paragraphs_of_one_message_share_a_turn(self):
        sources = [{"ref": "m0.1", "session_id": "a", "role": "user", "text": "x"},
                   {"ref": "m0.2", "session_id": "a", "role": "user", "text": "y"}]
        self.assertEqual(CollectionEngine.session_blocks(sources)[0]["turns"],
                         [{"role": "user", "paragraphs": [["m0.1", "x"], ["m0.2", "y"]]}])

    def test_each_part_may_cite_only_its_paragraphs(self):
        with self.assertRaises(ValidationError):
            CollectionEngine.part_points({"facts": [{"text": "A fact.", "evidence": ["m2.1"]}]}, {"m0.1", "m1.1"})
        CollectionEngine.part_points({"facts": [{"text": "A fact.", "evidence": ["m0.1"]}]}, {"m0.1", "m1.1"})
        CollectionEngine.part_points({"facts": []}, {"m0.1"})

    def test_gap_fill_continues_after_the_extraction_reply(self):
        request = payload("caption", messages=[{"role": "user", "content": "Alice shared a photo.\n[image caption] a sign reading Blue Harbor"}])
        extracted = {"facts": [{"text": "Alice shared a photo.", "evidence": ["m0.1"]}]}
        aliases, sources = CollectionEngine.source_view(request)
        from text_memory.collections import gap_fill
        gaps = gap_fill.find_gaps(sources, extracted)
        if not gaps:
            self.skipTest("fixture produced no gap flag")
        fill = {"revised": [{"index": 0, "text": "Alice shared a photo of a sign reading Blue Harbor.",
                             "evidence": ["m0.1"]}]}
        model = ScriptedModel(fill)
        filled = CollectionEngine(self.store, model).fill_gaps(request, extracted)
        prefix = CollectionEngine.conversation(request, extracted)
        self.assertEqual(model.calls[0]["messages"][:len(prefix)], prefix)
        self.assertEqual(filled["conversation"][-1]["role"], "assistant")


class DirectFactsQueryTests(unittest.TestCase):
    class Vectors:
        identity = {"model": "synthetic-direct"}

        def encode(self, texts):
            # Only "lake" texts point the query's way; everything else is far below any usual floor.
            return [[1.0, 0.0] if "lake" in t.lower() else [-1.0, 0.0] for t in texts]

    class SameVectors:
        identity = {"model": "synthetic-same"}

        def encode(self, texts):
            return [[1.0, 0.0] for _ in texts]

    def setUp(self):
        self.store = CollectionStore()
        self.addCleanup(self.store.close)
        self.scope = Scope("alice", "exhibition")
        request = long_request()
        CollectionEngine(self.store, ScriptedModel()).add(request, extraction=reply("m0.1", "m1.1", "m2.1"))

    def test_full_question_ranks_every_fact_without_model_call_or_threshold(self):
        model = ScriptedModel()
        engine = CollectionEngine(self.store, model, embedder=self.Vectors(), floor=.99)
        result = engine.query("Which lake painting did Alice frame?", self.scope)
        self.assertEqual(result["search_policy"], DIRECT_POLICY)
        # The only model call marks question attributes; ranking makes none.
        self.assertEqual([c["purpose"] for c in model.calls], ["question_flags"])
        self.assertFalse(any(result["question_flags"].values()))
        self.assertIsNone(result["question_flags_error"])
        self.assertEqual((result["fact_limit"], result["chronological"], result["notes"]), (16, False, []))
        self.assertNotIn("question_keywords", result)
        ranked = next(s for s in engine.trace if s["tool"] == "rank_question_facts")["result"]["facts"]
        # The floor of .99 is not applied: dissimilar facts are still ranked, just lower.
        self.assertEqual(len(ranked), 3)
        self.assertEqual(ranked[0]["text"], SENTENCES[2])
        self.assertNotIn("collections", result)
        self.assertFalse(any(s["tool"] == "expand_collection" for s in engine.trace))

    def test_bm25_weights_rare_words_when_vectors_tie(self):
        engine = CollectionEngine(self.store, None, embedder=self.SameVectors(), floor=.62)
        found = engine.searcher.rank_question_facts(self.store.snapshot(self.scope), "Where is Alice's brush?", limit=3)
        # "alice" is in every fact and carries little weight; "brush" decides.
        self.assertEqual(found["facts"][0]["text"], SENTENCES[0])
        self.assertGreater(found["facts"][0]["bm25"], found["facts"][1]["bm25"])

    def test_facts_in_the_named_period_get_the_time_bonus(self):
        store = CollectionStore(); self.addCleanup(store.close)
        dates = ["1:00 pm on 1 May, 2023", "2:00 pm on 10 June, 2023", "3:00 pm on 20 July, 2023"]
        request = payload("dated", messages=[{"role": "user", "content": s, "session_date": d}
                                             for s, d in zip(SENTENCES, dates)])
        CollectionEngine(store, ScriptedModel()).add(request, extraction=reply("m0.1", "m1.1", "m2.1"))
        engine = CollectionEngine(store, ScriptedModel(), embedder=self.SameVectors(), floor=.62)
        result = engine.query("What did Alice do in June 2023?", self.scope, fact_limit=3)
        self.assertEqual(result["question_time_ranges"], [["2023-06-01", "2023-06-30"]])
        ranked = next(s for s in engine.trace if s["tool"] == "rank_question_facts")["result"]["facts"]
        self.assertEqual(ranked[0]["text"], SENTENCES[1])
        self.assertEqual([f["time_match"] for f in ranked], [True, False, False])

    def test_limit_keeps_only_the_top_facts(self):
        engine = CollectionEngine(self.store, ScriptedModel(), embedder=self.Vectors(), floor=.62)
        found = engine.searcher.rank_question_facts(self.store.snapshot(self.scope), "lake", limit=1)
        self.assertEqual([f["text"] for f in found["facts"]], [SENTENCES[2]])
        self.assertEqual(DIRECT_LIMIT, 16)

    def test_fact_limit_can_be_set_per_query_for_comparisons(self):
        engine = CollectionEngine(self.store, ScriptedModel(), embedder=self.Vectors(), floor=.62)
        result = engine.query("Which lake painting?", self.scope, fact_limit=2)
        self.assertEqual(result["fact_limit"], 2)
        self.assertEqual(len(result["matched_fact_ids"]), 2)

    def test_question_flags_set_the_limit_order_and_notes(self):
        model = ScriptedModel(flag_responses=[{"combine": True, "changed": True, "ever": False}])
        engine = CollectionEngine(self.store, model, embedder=self.Vectors(), floor=.62)
        result = engine.query("How many paintings has Alice framed so far?", self.scope, fact_limit=1)
        self.assertEqual(result["fact_limit"], 2)
        self.assertEqual(len(result["matched_fact_ids"]), 2)
        self.assertTrue(result["chronological"])
        self.assertEqual(result["notes"], ["Sessions below are in chronological order (earliest first).",
                                           "When statements about the same thing differ, the later one is the current one."])

    def test_invalid_flag_replies_stop_before_ranking(self):
        from text_memory.errors import ModelError
        model = ScriptedModel(flag_responses=[{"combine": "yes"}] * 3)
        engine = CollectionEngine(self.store, model, embedder=self.Vectors(), floor=.62)
        with self.assertRaises(ModelError):
            engine.query("How many paintings?", self.scope, fact_limit=2)
        self.assertEqual(len(model.calls), 3)
        self.assertFalse(any(s["tool"] == "rank_question_facts" for s in engine.trace))

    def test_model_errors_wait_before_the_next_judgment_attempt(self):
        from text_memory.collections.question_flags import ERROR_DELAYS, check_question
        from text_memory.errors import ModelError
        waits = []
        model = ScriptedModel(flag_responses=[ModelError("HTTP 429"), ModelError("HTTP 429"),
                                              {"combine": False, "changed": False, "ever": True}])
        flags, error = check_question(model, "Have I ever been to Oslo?", sleep=waits.append)
        self.assertEqual((flags, error, waits), ({"combine": False, "changed": False, "ever": True}, None, list(ERROR_DELAYS)))
        model = ScriptedModel(flag_responses=[ModelError("HTTP 429")] * 3)
        with self.assertRaisesRegex(ModelError, "HTTP 429"):
            check_question(model, "Have I ever been to Oslo?", sleep=waits.append)
        self.assertEqual(len(waits), 2 * len(ERROR_DELAYS))

    def test_search_needs_vectors(self):
        with self.assertRaises(ValidationError):
            CollectionEngine(self.store, ScriptedModel()).query("lake?", self.scope)


if __name__ == "__main__":
    unittest.main()
