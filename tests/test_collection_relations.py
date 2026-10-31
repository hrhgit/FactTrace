"""Write-time relation stage and its answer notes, with explicit synthetic relation replies."""
import json
import unittest
from unittest.mock import patch

from tests.fixtures import ScriptedModel, payload
from text_memory.collections import CollectionStore
from text_memory.collections.answer_context import render_fact_context
from text_memory.collections.engine import CollectionEngine
from text_memory.collections.relations import PROMPT_HASH, dates_differ, event_spans
from text_memory.store import Scope


class Embedder:
    identity = {"model": "synthetic-constant"}

    def encode(self, texts):
        return [[1.0, 0.0] for _ in texts]


def session(request_id, date, *messages):
    return payload(request_id, session_id=request_id, messages=[
        {"role": role, "content": content, "session_date": date} for role, content in messages])


def extraction(*facts):
    return {"facts": [{"text": text, "evidence": [f"m{index}.1"]} for text, index in facts]}


class EventDateTests(unittest.TestCase):
    def test_recent_events_and_resolved_dates_separate_occurrences(self):
        may = event_spans("Sam had a check-up with his doctor a few days before the conversation.", "1:20 pm on 24 May, 2023")
        october = event_spans("Sam went for a medical check-up on Monday.", "3:09 pm on 8 October, 2023")
        self.assertTrue(dates_differ(may, october))
        rockies = event_spans("Evan went to the Rockies, where they hiked the trails last week.", "1:47 pm on 18 May, 2023")
        jasper = event_spans("Evan took a road trip to Jasper last weekend.", "7:11 pm on 24 May, 2023")
        self.assertTrue(dates_differ(rockies, jasper))
        # "July 2023" is a month, not "July 20".
        self.assertEqual(event_spans("Evan took a road trip last month (July 2023).", "10:18 am on 27 August, 2023")[0][0].day, 1)

    def test_a_lasting_state_or_unknown_date_never_blocks_same(self):
        state = event_spans("Evan loves watercolor painting.", "1:47 pm on 18 May, 2023")
        self.assertEqual(state, [])
        self.assertFalse(dates_differ(state, event_spans("Evan painted a sunset yesterday.", "8 June, 2023")))
        self.assertEqual(event_spans("The user rescheduled the meeting to March 30.", "March-15-2024")[0][0].month, 3)


class RelationStageTests(unittest.TestCase):
    def setUp(self):
        self.store = CollectionStore()
        self.addCleanup(self.store.close)
        self.scope = Scope("alice", "exhibition")

    def add(self, request, facts, *, relation_responses=None):
        model = ScriptedModel(relation_responses=relation_responses)
        engine = CollectionEngine(self.store, model, embedder=Embedder(), floor=0.5)
        engine.add(request, extraction=facts)
        return model, engine

    def test_same_with_different_event_dates_is_saved_as_another_and_runs_once(self):
        first = session("s1", "1:20 pm on 24 May, 2023", ("user", "I had a check-up a few days ago."))
        self.add(first, extraction(("Sam had a check-up with his doctor a few days before the conversation.", 0)))
        second = session("s2", "3:09 pm on 8 October, 2023", ("user", "I went for a check-up on Monday."))
        model, engine = self.add(second, extraction(("Sam went for a medical check-up on Monday.", 0)),
                                 relation_responses=[{"pairs": [{"id": "p0", "relation": "same"}]}])
        self.assertEqual([c["purpose"] for c in model.calls], ["collection_relation_pairs"])
        body = json.loads(model.calls[-1]["messages"][-1]["content"])
        record = body["records"][body["pairs"][0]["A"]]
        self.assertEqual([body["messages"][m] for m in record["messages"]], ["user: I had a check-up a few days ago."])
        self.assertEqual([l["relation"] for l in self.store.fact_relations(self.scope, prompt_hash=PROMPT_HASH)], ["another"])
        self.assertFalse(engine.relations_pending(second))
        again = ScriptedModel()
        CollectionEngine(self.store, again, embedder=Embedder(), floor=0.5).add(second)
        self.assertEqual(again.calls, [])

    def test_a_record_shared_by_pairs_is_sent_once(self):
        first = session("s1", "2024-03-01", ("user", "I bought a lamp and a chair."))
        self.add(first, extraction(("The user bought a lamp.", 0), ("The user bought a chair.", 0)))
        second = session("s2", "2024-03-05", ("user", "I returned the lamp."))
        model, _ = self.add(second, extraction(("The user returned the lamp.", 0)),
                            relation_responses=[{"pairs": [{"id": "p0", "relation": "none"}, {"id": "p1", "relation": "none"}]}])
        body = json.loads(model.calls[-1]["messages"][-1]["content"])
        self.assertEqual(len(body["pairs"]), 2)
        self.assertEqual(len(body["records"]), 3)
        self.assertEqual(body["pairs"][0]["B"], body["pairs"][1]["B"])
        self.assertEqual(list(body["messages"].values()).count("user: I bought a lamp and a chair."), 1)

    def test_assistant_only_facts_are_neither_labeled_nor_candidates(self):
        first = session("s1", "2024-03-01", ("assistant", "A $55 snack plan fits your budget."))
        self.add(first, extraction(("The assistant proposed a $55 snack plan.", 0)))
        second = session("s2", "2024-03-02", ("user", "I raised my snack budget to $65."),
                         ("assistant", "Here is a $60 plan."))
        model, _ = self.add(second, extraction(("The user raised the snack budget to $65.", 0),
                                               ("The assistant proposed a $60 plan.", 1)))
        self.assertNotIn("collection_relation_pairs", [c["purpose"] for c in model.calls])
        self.assertEqual(self.store.fact_relations(self.scope, prompt_hash=PROMPT_HASH), [])
        self.assertTrue(self.store.relations_done(self.scope, "s2", prompt_hash=PROMPT_HASH))

    def test_negation_is_checked_against_neighbours_including_later_facts(self):
        first = session("s1", "2024-03-28", ("user", "I have never invited Mason to any movie event."))
        model, _ = self.add(first, extraction(("The user has never invited Mason to any movie event.", 0)))
        self.assertEqual([c["purpose"] for c in model.calls], [])
        second = session("s2", "2024-04-02", ("user", "I invited Mason to the April 7 session."))
        model, _ = self.add(second, extraction(("The user invited Mason to the April 7 session.", 0)),
                            relation_responses=[{"pairs": [{"id": "p0", "relation": "none"}]}, {"conflicts": ["r0"]}])
        self.assertEqual([c["purpose"] for c in model.calls],
                         ["collection_relation_pairs", "collection_relation_negation"])
        checked = json.loads(model.calls[-1]["messages"][-1]["content"])
        self.assertTrue(checked["S"]["text"].startswith("The user has never invited"))
        self.assertEqual([r["text"] for r in checked["records"]], ["The user invited Mason to the April 7 session."])
        [link] = self.store.fact_relations(self.scope, prompt_hash=PROMPT_HASH)
        self.assertEqual(link["relation"], "conflicts")
        self.assertEqual(self.store.get(self.scope, "point", link["earlier"])["text"],
                         "The user has never invited Mason to any movie event.")

    def test_without_vectors_the_stage_is_skipped_and_left_pending(self):
        model = ScriptedModel()
        engine = CollectionEngine(self.store, model)
        request = session("s1", "2024-03-01", ("user", "I bought a book."))
        engine.add(request, extraction=extraction(("The user bought a book.", 0)))
        self.assertEqual([c["purpose"] for c in model.calls], [])
        self.assertIn({"tool": "fact_relations", "result": {"skipped": "no embedder"}}, engine.trace)
        self.assertFalse(self.store.relations_done(self.scope, "s1", prompt_hash=PROMPT_HASH))

    def test_resumed_write_runs_only_the_missing_relation_stage(self):
        import tempfile
        from pathlib import Path
        from text_memory.collections import write_runner
        first = session("s1", "2024-05-01", ("user", "I adopted a dog named Toby."))
        second = session("s2", "2024-06-01", ("user", "I adopted a second dog named Buddy."))
        for request, text in ((first, "The user adopted a dog named Toby."), (second, "The user adopted a dog named Buddy.")):
            CollectionEngine(self.store, ScriptedModel()).add(request, extraction=extraction((text, 0)))
        self.assertFalse(self.store.relations_done(self.scope, "s2", prompt_hash=PROMPT_HASH))
        models = []
        def factory(history):
            models.append(ScriptedModel(relation_responses=[{"pairs": [{"id": "p0", "relation": "another"}]}]))
            return models[-1]
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "memory.sqlite"
            self.store.db.execute(f"VACUUM INTO '{database}'")
            with patch("builtins.print"):
                write_runner.run([first, second], directory, factory, embedder=Embedder(), floor=0.5)
            with CollectionStore(database) as store:
                self.assertEqual([l["relation"] for l in store.fact_relations(self.scope, prompt_hash=PROMPT_HASH)], ["another"])
                self.assertTrue(all(store.relations_done(self.scope, r, prompt_hash=PROMPT_HASH) for r in ("s1", "s2")))
        # s1 has no earlier fact, so only s2 needed a model, and it made only the pair call.
        self.assertEqual([[c["purpose"] for c in m.calls] for m in models], [[], ["collection_relation_pairs"]])

    def test_answer_material_shows_at_most_two_notes_by_priority_and_can_hide_them(self):
        facts = ["The user set the snack budget to $65.", "The user raised the snack budget to $75.",
                 "The user has never set a snack budget.", "The user bought popcorn for the movie night."]
        for n, text in enumerate(facts):
            self.add(session(f"s{n}", f"March-2{n}-2024", ("user", text)), extraction((text, 0)), relation_responses=[
                {"pairs": [{"id": f"p{k}", "relation": rel} for k, rel in enumerate(
                    [[], ["replaces"], ["conflicts", "conflicts"], ["same", "another", "none"]][n])]}] +
                # The "never" statement is checked when written, and again against the later popcorn fact.
                ([{"conflicts": ["r0"]}] if n == 2 else [{"conflicts": []}] if n == 3 else []))
        ids = {p["text"]: p["id"] for p in self.store.records(self.scope, "point")}
        budget = ids[facts[1]]
        result = {"evidence": [{"id": budget, "text": facts[1], "source_ids": self.store.get(self.scope, "point", budget)["evidence"]}],
                  "matched_fact_ids": [budget]}
        material = render_fact_context(self.store, self.scope, result)
        lines = material.split("\n")
        self.assertTrue(lines[0].startswith('Notes starting with "Link:"'))
        notes = [l for l in lines if l.startswith("    Link: ")]
        self.assertEqual(len(notes), 2)
        self.assertTrue(notes[0].startswith("    Link: Conflicts with the record of March-22-2024"))
        self.assertTrue(notes[1].startswith("    Link: Replaces the earlier record of March-20-2024"))
        # A link whose labeling request is not visible at an earlier revision is not shown.
        early = render_fact_context(self.store, Scope("alice", "exhibition", self.store.receipt(
            session("s1", "March-21-2024", ("user", facts[1])))["revision"]), result)
        self.assertEqual([l for l in early.split("\n") if l.startswith("    Link: ")],
                         ['    Link: Replaces the earlier record of March-20-2024: "The user set the snack budget to $65."'])


if __name__ == "__main__":
    unittest.main()
