"""Write/search entry for mixed sets; a mock transport stands in for the model."""
import json
from pathlib import Path
import tempfile
from threading import Barrier
import unittest
from unittest.mock import patch

import httpx

from scripts import run_mixed_memory
from text_memory.collections import write_runner
from text_memory.collections.prompts import QUESTION_FLAGS, RELATION_NEGATION, RELATION_PAIRS
from text_memory.model import HttpJsonModel, ModelConfig


class Embedder:
    identity = {"model": "synthetic-constant"}

    def encode(self, texts):
        return [[1.0, 0.0] for _ in texts]

    def close(self):
        pass


class MixedMemoryTests(unittest.TestCase):
    def write_set(self, root, users):
        (root / "inputs").mkdir(parents=True)
        histories, queries, tasks = [], [], []
        for user, sentences in users.items():
            for n, sentence in enumerate(sentences):
                histories.append({"user_id": user, "scope_id": "history", "session_id": f"s{n}", "request_id": f"{user}:{n}",
                                  "messages": [{"role": "user", "content": sentence, "session_date": "2024-05-0" + str(n + 1)}]})
            queries.append({"id": user + "#q", "user_id": user, "scope_id": "history", "question": "What did Alice buy?"})
            tasks.append({"id": user + "#q", "dataset": "locomo-refined", "category": "single_hop"})
        for name, rows in (("histories", histories), ("queries", queries), ("tasks", tasks)):
            (root / "inputs" / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))

    def handler(self, request):
        body = json.loads(request.content)
        messages = body["messages"]
        last = json.loads(messages[-1]["content"]) if messages[-1]["content"].startswith(("{", "[")) else None
        if messages[0]["content"] == QUESTION_FLAGS:
            self.purposes.append("flags")
            # flag_failures > 0: that many judgments get an invalid reply (each judgment makes MAX_ATTEMPTS calls).
            failing = getattr(self, "flag_failures", 0) > 0
            if failing and self.purposes.count("flags") % 3 == 0:
                self.flag_failures -= 1
            reply = {"combine": "yes"} if failing else {"combine": False, "changed": False, "ever": False}
        elif messages[0]["content"] == RELATION_PAIRS:
            self.purposes.append("relation_pairs")
            reply = {"pairs": [{"id": p["id"], "relation": "none"} for p in last["pairs"]]}
        elif messages[0]["content"] == RELATION_NEGATION:
            self.purposes.append("relation_negation")
            reply = {"conflicts": []}
        else:
            self.purposes.append("extract")
            ref, text = json.loads(messages[-1]["content"])["sessions"][0]["turns"][0]["paragraphs"][0]
            reply = {"facts": [{"text": text, "evidence": [ref]}]}
        return httpx.Response(200, json={"model": "m", "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                                         "choices": [{"message": {"role": "assistant", "content": json.dumps(reply)},
                                                      "finish_reason": "stop"}]})

    def run_command(self, *argv):
        original = write_runner.JournaledWriteModel
        transport = httpx.MockTransport(self.handler)
        with patch.object(run_mixed_memory.ModelConfig, "from_env", return_value=ModelConfig("https://test.invalid", "k", "m")), \
             patch.object(run_mixed_memory, "embedder_from_env", return_value=Embedder()), \
             patch.object(run_mixed_memory, "embed_floor_from_env", return_value=0.5), \
             patch.object(write_runner, "JournaledWriteModel",
                          lambda config, directory, channel_pool=None: original(
                              config, directory, transport=transport, channel_pool=channel_pool)), \
             patch.object(run_mixed_memory, "HttpJsonModel",
                          lambda config, channel_pool=None: HttpJsonModel(
                              config, transport=transport, channel_pool=channel_pool)), \
             patch("sys.argv", ["run", *argv]), patch.object(run_mixed_memory, "print", create=True):
            return run_mixed_memory.main()

    def test_users_commit_concurrently_with_separate_ordered_logs(self):
        self.purposes = []
        barrier = Barrier(2)
        original_handler = self.handler

        def concurrent_handler(request):
            body = json.loads(request.content)
            if body["messages"][0]["content"] == RELATION_PAIRS:
                # Both users must reach their second write together. A serial
                # commit runner cannot pass this barrier.
                barrier.wait(timeout=10)
            return original_handler(request)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_set(root / "set", {
                "alice": ["Alice bought a red brush.", "Alice bought a blue easel."],
                "bob": ["Bob adopted a cat.", "Bob adopted a dog."]})
            with patch.object(self, "handler", side_effect=concurrent_handler):
                self.assertEqual(self.run_command("write", "--set", str(root / "set"),
                    "--output", str(root / "run"), "--workers", "2", "--commit-workers", "2"), 0)
            for user in ("alice", "bob"):
                directory = root / "run/write" / run_mixed_memory.directory_name(user)
                with (directory / "run.log").open() as stream:
                    records = [json.loads(line) for line in stream]
                written = [r["request_id"] for r in records if r.get("phase") == "write" and r.get("status") == "ok"]
                self.assertEqual(written, [f"{user}:0", f"{user}:1"])
                self.assertTrue(all(r.get("request_id", user + ":").startswith(user + ":") for r in records))

    def test_parallel_calls_are_reused_by_ordered_commits_and_search_exports_contexts(self):
        self.purposes = []
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_set(root / "set", {"alice": ["Alice bought a red brush.", "Alice bought a blue easel."],
                                          "bob": ["Bob adopted a cat."]})
            out = root / "run"
            self.assertEqual(self.run_command("write", "--set", str(root / "set"), "--output", str(out)), 0)
            # Each session: one extraction, with no default keyword call; the relation stage
            # then pairs Alice's second fact with her first (one call), Bob has no earlier fact.
            self.assertEqual(sorted(self.purposes), ["extract"] * 3 + ["relation_pairs"])
            for user in ("alice", "bob"):
                directory = out / "write" / run_mixed_memory.directory_name(user)
                summary = json.loads((directory / "summary.json").read_text())
                self.assertEqual((summary["status"], summary["completed_writes"]), ("complete", 2 if user == "alice" else 1))
                commits = [json.loads(p.read_text()) for p in (directory / "writes").glob("*.json")]
                self.assertTrue(all(c["purpose"] == "collection_relation_pairs" for commit in commits for c in commit["calls"]),
                                "commit made a model call other than the relation stage")
            self.purposes = []
            self.assertEqual(self.run_command("search", "--set", str(root / "set"), "--output", str(out)), 0)
            # Search ranks by the full question; the only model call per question marks its attributes.
            self.assertEqual(self.purposes, ["flags", "flags"])
            contexts = {r["id"]: r["context"] for r in map(json.loads, (out / "search-top16" / "contexts.jsonl").read_text().splitlines())}
            self.assertEqual(set(contexts), {"alice#q", "bob#q"})
            self.assertIn("Alice bought", contexts["alice#q"])
            self.assertIn("2024-05", contexts["alice#q"])

            # A larger set reuses finished users without any model call.
            self.purposes = []
            self.write_set(root / "big", {"alice": ["Alice bought a red brush.", "Alice bought a blue easel."],
                                          "bob": ["Bob adopted a cat."], "carol": ["Carol moved to Lyon."]})
            self.assertEqual(self.run_command("write", "--set", str(root / "big"), "--output", str(root / "big-run"),
                                              "--reuse-writes", str(out)), 0)
            self.assertEqual(sorted(self.purposes), ["extract"])

    def test_failed_question_judgment_is_searched_again_and_never_exported_silently(self):
        self.purposes = []
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_set(root / "set", {"alice": ["Alice bought a red brush."]})
            out = root / "run"
            self.assertEqual(self.run_command("write", "--set", str(root / "set"), "--output", str(out)), 0)
            with patch.object(run_mixed_memory, "SEARCH_RETRY_DELAY", 0):
                # The first judgment fails, the repeated search succeeds.
                self.purposes, self.flag_failures = [], 1
                self.assertEqual(self.run_command("search", "--set", str(root / "set"), "--output", str(out)), 0)
                self.assertEqual(self.purposes, ["flags"] * 4)
                saved = json.loads((out / "search-top16/search/alice_q.json").read_text())
                self.assertEqual((len(saved["calls"]), saved["question_flags_error"]), (4, None))
                self.assertTrue((out / "search-top16/contexts.jsonl").exists())
                # Every repeat fails: the run fails and writes no new contexts.
                (out / "search-top16/contexts.jsonl").unlink()
                self.purposes, self.flag_failures = [], run_mixed_memory.SEARCH_ATTEMPTS
                self.assertEqual(self.run_command("search", "--set", str(root / "set"), "--output", str(out)), 1)
                self.assertEqual(len(self.purposes), 3 * run_mixed_memory.SEARCH_ATTEMPTS)
                self.assertFalse((out / "search-top16/contexts.jsonl").exists())
                failures = json.loads((out / "search-top16/search-failures.json").read_text())
                self.assertIn("invalid reply", failures["alice#q"])

    def test_question_without_history_gets_empty_material(self):
        self.purposes = []
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_set(root / "set", {"alice": ["Alice bought a red brush."]})
            extra = {"id": "clbench-single:t1", "user_id": "clbench-single:t1", "scope_id": "history",
                     "question": "DOCUMENT ... What does section 2 require?"}
            with (root / "set/inputs/queries.jsonl").open("a") as stream:
                stream.write(json.dumps(extra) + "\n")
            with (root / "set/inputs/tasks.jsonl").open("a") as stream:
                stream.write(json.dumps({"id": extra["id"], "dataset": "clbench", "category": "single-turn: x"}) + "\n")
            out = root / "run"
            self.assertEqual(self.run_command("write", "--set", str(root / "set"), "--output", str(out)), 0)
            self.assertEqual(self.run_command("search", "--set", str(root / "set"), "--output", str(out)), 0)
            contexts = {r["id"]: r["context"] for r in map(json.loads, (out / "search-top16" / "contexts.jsonl").read_text().splitlines())}
            self.assertEqual(contexts["clbench-single:t1"], "")
            self.assertTrue(contexts["alice#q"])
            self.assertFalse((out / "write" / run_mixed_memory.directory_name("clbench-single:t1")).exists())

    def test_changed_inputs_refuse_to_resume(self):
        self.purposes = []
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_set(root / "set", {"alice": ["Alice bought a red brush."]})
            out = root / "run"
            self.run_command("write", "--set", str(root / "set"), "--output", str(out))
            (root / "set" / "inputs" / "histories.jsonl").write_text(
                (root / "set" / "inputs" / "histories.jsonl").read_text().replace("red", "green"))
            with self.assertRaises(Exception):
                self.run_command("write", "--set", str(root / "set"), "--output", str(out))


if __name__ == "__main__":
    unittest.main()
