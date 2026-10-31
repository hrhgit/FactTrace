"""Check resume boundaries; scripted responses do not measure model quality."""
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx

from tests.fixtures import ScriptedModel, payload
from text_memory.collections import CollectionEngine, CollectionStore
from text_memory.collections import write_runner as runner
from text_memory.errors import ModelError, ValidationError
from text_memory.model import ModelConfig
from text_memory.schema import digest
from text_memory.store import Scope


class WriteRunnerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.output = Path(self.directory.name)
        self.first = payload("first", messages=[{"role": "user", "content": "Alice plans a trip."}])
        self.second = payload("second", messages=[{"role": "user", "content": "Alice bought a brush."}])
        self.extraction = {"facts": [{"text": "Alice plans a trip.", "evidence": ["m0.1"]}]}
        self.other_extraction = {"facts": [{"text": "Alice bought a brush.", "evidence": ["m0.1"]}]}
        self.commit = {"action": "commit", "groups": [{"collection_name": "trip", "facts": [0]}]}
        self.other_commit = {"action": "commit", "groups": [{"collection_name": "brush", "facts": [0]}]}

    def tearDown(self):
        self.directory.cleanup()

    def run_quiet(self, histories, factory):
        with redirect_stdout(StringIO()):
            return runner.run(histories, self.output, factory)

    def test_resume_skips_committed_session_and_reuses_saved_extraction(self):
        models = iter([ScriptedModel(self.extraction), ScriptedModel(self.other_extraction)])
        commit = CollectionStore.commit
        def interrupted(store, history, plan, **kwargs):
            if history['request_id'] == 'second':
                raise OSError('disk unavailable')
            return commit(store, history, plan, **kwargs)
        with patch.object(CollectionStore, 'commit', interrupted), self.assertRaises(OSError):
            self.run_quiet([self.first, self.second], lambda history: next(models))
        first_write = self.output / "writes" / (runner.token(self.first) + ".json")
        original_write = first_write.read_bytes()
        resumed = ScriptedModel(self.other_commit)
        requested = []

        def factory(history):
            requested.append(history["request_id"])
            return resumed

        result = self.run_quiet([self.first, self.second], factory)
        self.assertEqual(requested, ["second"])
        self.assertEqual([c["purpose"] for c in resumed.calls], [])
        self.assertEqual(first_write.read_bytes(), original_write)
        self.assertEqual(result["completed_writes"], 2)
        self.assertEqual(result["answer_calls"], 0)

    def test_commit_remains_authoritative_if_receipt_file_save_was_interrupted(self):
        original = runner.save

        def interrupted(path, value):
            if path.parent.name == "writes":
                raise OSError("simulated disk interruption after database commit")
            original(path, value)

        with patch.object(runner, "save", side_effect=interrupted), self.assertRaises(OSError):
            self.run_quiet([self.first], lambda history: ScriptedModel(self.extraction, self.commit))
        result = self.run_quiet([self.first], lambda history: self.fail("Committed data must not call a model"))
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["completed_writes"], 1)

    def test_completed_extraction_response_can_restore_missing_checkpoint(self):
        path = self.output / "attempts" / runner.token(self.first) / "000000.json"
        runner.save(path, {"purpose": "collection_extract", "status": "completed",
            "messages": CollectionEngine.extraction_messages(self.first),
            "response": {"choices": [{"message": {"content": json.dumps(self.extraction)}}]}})
        model = ScriptedModel(self.commit)
        self.run_quiet([self.first], lambda history: model)
        self.assertEqual([c["purpose"] for c in model.calls], [])
        saved = json.loads((self.output / "extractions" / (runner.token(self.first) + ".json")).read_text())
        self.assertEqual(saved["recovered_from_attempt"], str(path))

    def test_cached_extraction_with_wrong_source_hash_fails_before_a_paid_call(self):
        runner.save(self.output / "extractions" / (runner.token(self.first) + ".json"),
                    {"payload_sha256": "wrong", "extraction": self.extraction})
        model = ScriptedModel()
        with self.assertRaises(ValidationError):
            self.run_quiet([self.first], lambda history: model)
        self.assertEqual(model.calls, [])
        with CollectionStore(self.output / "memory.sqlite") as store:
            self.assertEqual(store.revision(Scope("alice", "exhibition")), 0)

    def test_response_journal_records_failure_usage_before_model_correction(self):
        directory = self.output / "attempts" / runner.token(self.first)
        replies = iter(["not JSON", json.dumps(self.extraction)])

        def handler(request):
            return httpx.Response(200, json={"model": "same", "usage": {"total_tokens": 10}, "choices": [
                {"finish_reason": "stop", "message": {"content": next(replies)}}]})

        model = runner.JournaledWriteModel(ModelConfig("https://example.invalid/v1", "key", "same"), directory,
                                           transport=httpx.MockTransport(handler))
        with CollectionStore() as store:
            CollectionEngine(store, model).extract(self.first)
        calls = runner.recorded_calls(self.output)
        self.assertEqual([c["status"] for c in calls], ["invalid_output", "completed"])
        self.assertEqual(runner.usage(calls)["total_tokens"], 20)
        self.assertIsNone(runner.usage(calls)["prompt_tokens"])

    def test_response_journal_records_channel_failure_and_backup_success(self):
        directory = self.output / "attempts" / runner.token(self.first)

        def handler(request):
            if request.url.host == "one.invalid":
                return httpx.Response(401, json={"error": "unauthorized"})
            return httpx.Response(200, json={"model": "two", "usage": {"total_tokens": 10}, "choices": [
                {"finish_reason": "stop", "message": {"content": json.dumps(self.extraction)}}]})

        config = ModelConfig("https://one.invalid/v1", "one-key", "one", fallbacks=(
            ModelConfig("https://two.invalid/v1", "two-key", "two"),))
        model = runner.JournaledWriteModel(config, directory, transport=httpx.MockTransport(handler))
        result = model.complete(CollectionEngine.extraction_messages(self.first), purpose="collection_extract")
        self.assertEqual(result, self.extraction)
        calls = runner.recorded_calls(self.output)
        self.assertEqual([call["status"] for call in calls], ["http_error", "completed"])
        self.assertEqual([call["channel_index"] for call in calls], [1, 2])
        self.assertEqual([call["failover_index"] for call in calls], [0, 1])
        self.assertTrue(calls[0]["failover_scheduled"])
        self.assertTrue(all(call["new_attempt"] for call in calls))
        self.assertEqual(runner.usage(calls)["total_tokens"], None)


if __name__ == "__main__":
    unittest.main()
