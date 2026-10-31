"""Integration boundaries; quality evidence remains in the saved diagnostics."""
import unittest
import json
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import tempfile
from unittest.mock import patch

from tests.fixtures import ScriptedModel, payload, cite
from text_memory.collections import CollectionEngine, CollectionStore
from text_memory.collections.retrieval import CollectionSearcher
from text_memory.errors import ConfigurationError, ModelError, ValidationError
from text_memory.model import ModelConfig
from text_memory.store import Scope


class IdenticalFactVectors:
    identity = {"model": "synthetic-session-context-test"}

    def encode(self, texts):
        # Retrieval in these tests is injected or lexical; these vectors only
        # exercise context assembly on a single unambiguous local segment.
        return [[1., 0.] for _ in texts]


class SessionFactContextTests(unittest.TestCase):
    def setUp(self):
        self.store = CollectionStore()
        self.addCleanup(self.store.close)
        self.scope = Scope("alice", "exhibition")
        self.searcher = CollectionSearcher(self.store, IdenticalFactVectors(), floor=.62)
        request = payload("r1", session_id="s1", messages=[{
            "role": "user", "speaker": "Alice", "content": "RAW TEXT MUST NOT BE RETURNED"}])
        self.store.commit(request, {"points": [
            {"id": "z-hit", "text": "Alice has a notebook.", "evidence": cite(request)},
            {"id": "a-context", "text": "She designed it herself.", "evidence": cite(request)}]}, expected_revision=0)
        request = payload("r2", session_id="s2", messages=[{"role": "user", "content": "other session"}])
        self.store.commit(request, {"points": [
            {"id": "outside", "text": "Alice repaired a bicycle.", "evidence": cite(request)}]},
            expected_revision=1)

    def test_context_uses_receipt_order_not_hash_order_and_does_not_open_another_collection(self):
        packet = self.searcher.complete_fact_context(self.scope, ["z-hit"])
        self.assertEqual(packet["matched_fact_ids"], ["z-hit"])
        self.assertEqual(packet["context_fact_ids"], ["a-context"])
        self.assertEqual(packet["context_completion"]["contexts"][0]["core_fact_ids"], ["z-hit", "a-context"])
        self.assertEqual([f["id"] for f in packet["evidence"]], ["z-hit", "a-context"])
        self.assertEqual(packet["sources"], [])
        self.assertNotIn("RAW TEXT", str(packet))

    def test_no_hits_cannot_restore_the_first_round_or_any_session(self):
        packet = self.searcher.complete_fact_context(self.scope, [])
        self.assertEqual(packet["evidence"], [])
        self.assertEqual(packet["context_completion"]["contexts"], [])

    def test_user_scope_history_and_superseded_facts_are_respected(self):
        for scope in [Scope("bob", "exhibition"), Scope("alice", "other"), Scope("alice", "exhibition", 0)]:
            with self.assertRaises(ValidationError):
                self.searcher.complete_fact_context(scope, ["z-hit"])
        first = self.searcher.complete_fact_context(Scope("alice", "exhibition", 1), ["z-hit"])
        self.assertNotIn("outside", str(first))
        correction = payload("r3", session_id="s1", messages=[{"role": "user", "content": "Correction."}])
        original = self.store.get(self.scope, "point", "a-context")
        point = {k: original[k] for k in ["id", "text", "evidence"]}
        point.update(state="superseded", change={"reason": "corrected", "evidence": cite(correction)})
        self.store.commit(correction, {"points": [point]}, expected_revision=2)
        current = self.searcher.complete_fact_context(self.scope, ["z-hit"])
        self.assertEqual(current["context_fact_ids"], [])
        # A current cached profile must not replace the historical profile.
        old = self.searcher.complete_fact_context(Scope("alice", "exhibition", 1), ["z-hit"])
        self.assertEqual(old["context_fact_ids"], ["a-context"])

    def test_new_write_invalidates_cached_session(self):
        before = self.searcher.complete_fact_context(self.scope, ["z-hit"])
        request = payload("r3", session_id="s1", messages=[{"role": "user", "content": "It has stars."}])
        self.store.commit(request, {"points": [
            {"id": "new", "text": "The notebook has stars.", "evidence": cite(request)}]}, expected_revision=2)
        after = self.searcher.complete_fact_context(self.scope, ["z-hit"])
        self.assertNotIn("new", str(before))
        self.assertIn("new", after["context_fact_ids"])


    def test_missing_or_failed_vectors_cannot_silently_skip_completion(self):
        with self.assertRaises(ConfigurationError):
            CollectionSearcher(self.store).complete_fact_context(self.scope, ["z-hit"])
        class Failed(IdenticalFactVectors):
            identity = {"model": "failed-context-test"}
            def encode(self, texts):
                raise ModelError("vector unavailable")
        with self.assertRaises(ModelError):
            CollectionSearcher(self.store, Failed(), floor=.62).complete_fact_context(self.scope, ["z-hit"])

    def test_query_attaches_context_after_ranking(self):
        engine = CollectionEngine(self.store, ScriptedModel(), embedder=IdenticalFactVectors(), floor=.62)
        engine.searcher.rank_question_facts = lambda *_, **__: {"facts": [{"id": "z-hit"}], "collections": [],
                                                                 "question_time_ranges": []}
        packet = engine.query("notebook", self.scope)
        self.assertEqual(packet["matched_fact_ids"], ["z-hit"])
        self.assertEqual(packet["context_fact_ids"], ["a-context"])
        self.assertEqual(packet["sources"], [])

    def test_cli_and_http_attach_real_context(self):
        from text_memory.cli import main
        from text_memory.collections.service import CollectionApplication
        class QueryVectors(IdenticalFactVectors):
            identity = {"model": "synthetic-lexical-query-semantic-context"}
            def encode(self, texts):
                return [[-1., 0.] if t == "notebook" else [1., 0.] for t in texts]
            def close(self):
                pass
        with tempfile.TemporaryDirectory() as tmp:
            database = str(Path(tmp) / "memory.sqlite")
            with CollectionStore(database) as destination:
                self.store.db.backup(destination.db)
            # Only the lexical match separates the facts; the top one gets its session context.
            app = CollectionApplication(database, ScriptedModel, embedder=QueryVectors(), floor=.62, fact_limit=1)
            body = {"query": "notebook", "user_id": "alice", "scope_id": "exhibition"}
            packet = app.search(body)[0]
            self.assertEqual(packet["matched_fact_ids"], ["z-hit"])
            self.assertEqual(packet["context_fact_ids"], ["a-context"])
            request = Path(tmp) / "query.json"
            request.write_text(json.dumps(body | {"question": "notebook"}))
            output = StringIO()
            with patch("text_memory.cli.load_dotenv"), \
                 patch("text_memory.cli.ModelConfig.from_env", return_value=ModelConfig("https://test.invalid", "k", "m")), \
                 patch("text_memory.cli.TimedModel", side_effect=lambda config, **kwargs: ScriptedModel()), \
                 patch("text_memory.cli.embedder_from_env", return_value=QueryVectors()), \
                 patch("text_memory.cli.embed_floor_from_env", return_value=.62), redirect_stdout(output):
                self.assertEqual(main(["query", "--database", database, "--input", str(request),
                                       "--fact-limit", "1"]), 0)
            cli = json.loads(output.getvalue())
            self.assertEqual(cli["evidence"], packet["evidence"])
            self.assertEqual(cli["context_fact_ids"], ["a-context"])
            self.assertEqual(cli["sources"], [])

if __name__ == "__main__":
    unittest.main()
