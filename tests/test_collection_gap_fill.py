"""Check gap flags and merge rules; scripted replies do not measure model quality."""
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tests.fixtures import ScriptedModel, payload
from text_memory.collections import CollectionEngine, CollectionStore
from text_memory.collections import gap_fill
from text_memory.collections import write_runner as runner
from text_memory.errors import ModelError, SearchLoopError, ValidationError
from text_memory.store import Scope

READING = ("It was a transgender poetry reading where people shared their stories through poetry."
           "\n[image caption] a photography of a sign that says trans lives matter"
           "\n[image search query] transgender poetry reading trans pride flags")
STRENGTH = "They're really amazing. Wish I was that resilient too. But they give me the strength to keep going."


def history():
    return payload("gaps", messages=[{"role": "user", "content": READING},
                                     {"role": "user", "content": STRENGTH},
                                     {"role": "user", "content": "Bye, Deb! See you later!"}])


EXTRACTION = {"facts": [{"text": "Caroline attended a transgender poetry reading.", "evidence": ["m0.1"]}]}
FILL = {"revised": [{"index": 0, "evidence": ["m0.1"],
                     "text": "Caroline attended a transgender poetry reading where a sign said 'trans lives matter'."}],
        "facts": [{"text": "Melanie says her family gives her the strength to keep going.", "evidence": ["m1.1"]}]}


class GapFillTests(unittest.TestCase):
    def test_flags_missing_caption_and_uncited_statement_but_not_short_farewell(self):
        _, sources = CollectionEngine.source_view(history())
        gaps = gap_fill.find_gaps(sources, EXTRACTION)
        self.assertEqual([g["ref"] for g in gaps], ["m0.1", "m1.1"])
        covered = {"facts": [{"text": "Caroline attended a transgender poetry reading where people shared their "
                              "stories through poetry and saw a sign that says trans lives matter.", "evidence": ["m0.1"]},
                             {"text": "Melanie says her family gives her the strength to keep going; she wishes "
                              "she were that resilient.", "evidence": ["m1.1"]}]}
        self.assertEqual(gap_fill.find_gaps(sources, covered), [])

    def test_listener_reactions_are_not_flagged(self):
        listener = payload("listener", messages=[{"role": "user", "content":
            "Oh man, sorry to hear that, Melanie. I hope you're okay. Pottery's a great way to relax, so it must "
            "have been tough taking a break. Need any help?"}])
        _, sources = CollectionEngine.source_view(listener)
        self.assertEqual(gap_fill.find_gaps(sources, {"facts": []}), [])

    def test_assistant_self_statement_is_not_flagged_but_user_statement_is(self):
        statement = ("I can help you compare local stores and recommend options after you share your city or zip code "
                     "so the search can be tailored to your location and preferences.")
        assistant = payload("assistant-reply", messages=[{"role": "assistant", "content": statement}])
        user = payload("user-statement", messages=[{"role": "user", "content": statement}])
        _, assistant_sources = CollectionEngine.source_view(assistant)
        _, user_sources = CollectionEngine.source_view(user)
        self.assertEqual(gap_fill.find_gaps(assistant_sources, {"facts": []}), [])
        self.assertEqual([g["ref"] for g in gap_fill.find_gaps(user_sources, {"facts": []})], ["m0.1"])

    def test_assistant_image_caption_is_still_checked(self):
        assistant = payload("assistant-caption", messages=[{"role": "assistant", "content":
            "Here is the image you shared.\n[image caption] a red bicycle beside a lake"}])
        _, sources = CollectionEngine.source_view(assistant)
        extracted = {"facts": [{"text": "The assistant described the shared image.", "evidence": ["m0.1"]}]}
        gaps = gap_fill.find_gaps(sources, extracted)
        self.assertEqual([g["ref"] for g in gaps], ["m0.1"])
        self.assertEqual(gaps[0]["check"], "image caption content is not reflected in the facts citing this paragraph")

    def test_covered_quote_must_exist_and_use_the_paragraph_wording(self):
        aliases, sources = CollectionEngine.source_view(history())
        texts = {s["ref"]: s["text"] for s in sources}
        loose = [{"text": "Melanie considers her family her biggest motivation.", "evidence": ["m2.1"]}]
        gaps = [g for g in gap_fill.find_gaps(sources, {"facts": loose}) if g["ref"] == "m1.1"]
        self.assertEqual(len(gaps), 1)
        synonym = {"covered": [{"ref": "m1.1", "index": 0, "quote": "her biggest motivation"}]}
        self.assertIn("用词不一致", gap_fill.apply_fill(loose, synonym, gaps, aliases, texts)[1]["m1.1"])
        invented = {"covered": [{"ref": "m1.1", "index": 0, "quote": "strength to keep going"}]}
        self.assertIn("引文不在事实", gap_fill.apply_fill(loose, invented, gaps, aliases, texts)[1]["m1.1"])
        revised = {"revised": [{"index": 0, "evidence": ["m2.1", "m1.1"], "text":
                   "Melanie considers her family her biggest motivation and says they give her the strength to keep going."}]}
        filled, errors = gap_fill.apply_fill(loose, revised, gaps, aliases, texts)
        self.assertEqual(errors, {})
        self.assertIn("strength", filled[0]["text"])
        quoted = {"covered": [{"ref": "m1.1", "index": 0, "quote": "Give her the STRENGTH, to keep going"}]}
        self.assertEqual(gap_fill.apply_fill(filled, quoted, gaps, aliases, texts)[1], {})
        disguised = {"no_fact": [{"ref": "m1.1", "reason": "already recorded in fact 0"}]}
        self.assertIn("理由称内容已保存", gap_fill.apply_fill(loose, disguised, gaps, aliases, texts)[1]["m1.1"])

    def test_add_revises_and_appends_facts_in_the_same_conversation(self):
        model = ScriptedModel(EXTRACTION, FILL)
        with CollectionStore() as store:
            receipt = CollectionEngine(store, model).add(history())
            texts = [store.get(Scope("alice", "exhibition"), "point", i)["text"] for i in receipt["point_ids"]]
        self.assertEqual(texts, [FILL["revised"][0]["text"], FILL["facts"][0]["text"]])
        self.assertEqual([c["purpose"] for c in model.calls],
                         ["collection_extract", "collection_gap_fill"])
        fill_messages = model.calls[1]["messages"]
        self.assertEqual(fill_messages[:2], CollectionEngine.extraction_messages(history()))
        self.assertEqual(json.loads(fill_messages[2]["content"]), EXTRACTION)
        flagged = json.loads(fill_messages[3]["content"])["flagged"]
        self.assertEqual([g["ref"] for g in flagged], ["m0.1", "m1.1"])

    def test_no_flag_means_no_gap_call(self):
        quiet = payload("quiet", messages=[{"role": "user", "content": "Alice plans a trip."}])
        model = ScriptedModel({"facts": [{"text": "Alice plans a trip.", "evidence": ["m0.1"]}]})
        with CollectionStore() as store:
            CollectionEngine(store, model).add(quiet)
        self.assertEqual([c["purpose"] for c in model.calls], ["collection_extract"])

    def test_merge_keeps_valid_items_and_reports_dropped_evidence_and_unhandled_flags(self):
        aliases, sources = CollectionEngine.source_view(history())
        texts = {s["ref"]: s["text"] for s in sources}
        gaps = gap_fill.find_gaps(sources, EXTRACTION)
        facts = EXTRACTION["facts"]
        dropped = {"revised": [{"index": 0, "text": "Caroline went.", "evidence": ["m1.1"]}],
                   "no_fact": [{"ref": "m0.1", "reason": "x"}]}
        merged, errors = gap_fill.apply_fill(facts, dropped, gaps, aliases, texts)
        self.assertEqual(merged, facts)
        self.assertEqual(list(errors), ["m1.1"])
        self.assertIn("保留原有依据", errors["m1.1"])
        _, errors = gap_fill.apply_fill(facts, {"no_fact": [{"ref": "m0.1", "reason": "repeat"}]}, gaps, aliases, texts)
        self.assertEqual(list(errors), ["m1.1"])
        merged, errors = gap_fill.apply_fill(facts, FILL, gaps, aliases, texts)
        self.assertEqual(errors, {})
        self.assertEqual(len(merged), 2)
        with self.assertRaises(ValidationError):
            gap_fill.check_shape({"extra": []})

    def test_only_rejected_paragraphs_are_asked_again(self):
        no_fact = {"no_fact": [{"ref": "m0.1", "reason": "greeting"}]}
        settled = {"no_fact": [{"ref": "m1.1", "reason": "encouragement"}]}
        model = ScriptedModel(no_fact, settled)
        filled = CollectionEngine(CollectionStore(), model).fill_gaps(history(), EXTRACTION)
        self.assertEqual(filled["facts"], EXTRACTION["facts"])
        retry = json.loads(model.calls[1]["messages"][-1]["content"])
        self.assertEqual([g["ref"] for g in retry["flagged"]], ["m1.1"])
        self.assertEqual(list(retry["rejected"]), ["m1.1"])
        self.assertEqual(len(model.calls[1]["messages"]), len(model.calls[0]["messages"]))

    def test_repeating_the_same_rejected_answer_stops_visibly(self):
        same = {"no_fact": [{"ref": "m0.1", "reason": "greeting"}]}
        model = ScriptedModel(same, {"no_fact": [{"ref": "m1.1", "reason": "already saved"}]},
                              {"no_fact": [{"ref": "m1.1", "reason": "already saved"}]})
        with self.assertRaises(SearchLoopError):
            CollectionEngine(CollectionStore(), model).fill_gaps(history(), EXTRACTION)

    def test_flagged_paragraphs_are_asked_in_batches(self):
        with patch.object(gap_fill, "BATCH", 1):
            model = ScriptedModel({"no_fact": [{"ref": "m0.1", "reason": "greeting"}]},
                                  {"no_fact": [{"ref": "m1.1", "reason": "encouragement"}]})
            CollectionEngine(CollectionStore(), model).fill_gaps(history(), EXTRACTION)
        self.assertEqual([[g["ref"] for g in json.loads(c["messages"][-1]["content"])["flagged"]] for c in model.calls],
                         [["m0.1"], ["m1.1"]])

    def test_runner_reuses_gap_checkpoint_after_interruption(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(StringIO()):
            output = Path(directory)
            with patch.object(CollectionStore, 'commit', side_effect=OSError('disk unavailable')), self.assertRaises(OSError):
                runner.run([history()], output, lambda h: ScriptedModel(EXTRACTION, FILL))
            saved = json.loads((output / "gap_fills" / (runner.token(history()) + ".json")).read_text())
            self.assertEqual(len(saved["extraction"]["facts"]), 2)
            resumed = ScriptedModel()
            result = runner.run([history()], output, lambda h: resumed)
        self.assertEqual([c["purpose"] for c in resumed.calls], [])
        self.assertEqual(result["completed_writes"], 1)


if __name__ == "__main__":
    unittest.main()
