"""Contract verification with explicit test doubles; not a live-model or AML platform run."""
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import httpx

from tests.fixtures import ScriptedModel
from text_memory.collections.service import CollectionApplication
from text_memory.errors import ModelError, ConfigurationError
from text_memory.http_embeddings import HttpEmbedder
from text_memory.model import HttpJsonModel, ModelChannelPool, ModelConfig
from text_memory.retrieval import embedder_from_env
from text_memory.service import make_server


class AMLContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.models = []
        def factory():
            model = ScriptedModel({"facts": [{"text": "Alice keeps the copper beacon in the attic.",
                                              "evidence": ["m0.1"]}]})
            self.models.append(model)
            return model
        class Vectors:
            identity = {"model": "synthetic-contract"}
            def encode(self, texts):
                return [[1.0, 0.0] for _ in texts]
        app = CollectionApplication(Path(self.tmp.name) / "memory.sqlite", factory, embedder=Vectors(), floor=.62)
        self.server = make_server(app, "127.0.0.1", 0, "unit-test-token")
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.stop)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        self.client = httpx.Client(base_url=self.url, timeout=10,
                                   headers={"Authorization": "Bearer unit-test-token"})
        self.addCleanup(self.client.close)
        self.payload = {"request_id": "contract:request:1", "user_id": "contract:user:1",
                        "session_id": "contract:session:1", "messages": [{"role": "user",
                        "timestamp": 1704067200000, "content": "I keep the copper beacon in the attic."}]}

    def stop(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def test_synchronous_add_retry_search_and_user_isolation(self):
        r = self.client.post("/add", json=self.payload)
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json(), {"success": True, **{k: self.payload[k] for k in
                                                      ("request_id", "user_id", "session_id")}})
        retry = self.client.post("/add", json=self.payload)
        self.assertEqual(retry.json(), r.json())
        self.assertEqual(sum(len(m.calls) for m in self.models), 1)
        query = {"query": "copper beacon attic", "user_id": self.payload["user_id"], "top_k": 1,
                 "options": ["A. Attic", "B. Garden"]}
        found = self.client.post("/search", json=query)
        self.assertEqual(found.status_code, 200, found.text)
        data = found.json()["data"]
        self.assertEqual(len(data), 1)
        self.assertIn("Alice keeps the copper beacon in the attic.", data[0]["content"])
        self.assertIn("2024-01-01", data[0]["content"])
        for forbidden in ("cf_", "s_", "context_completion", "matched_fact_ids", "I keep"):
            self.assertNotIn(forbidden, data[0]["content"])
        self.assertEqual(self.client.post("/search", json=query).json(), found.json())
        isolated = self.client.post("/search", json=query | {"user_id": "different-user"})
        self.assertEqual(isolated.json(), {"data": []})
        changed = self.payload | {"messages": [{"role": "user", "content": "Changed data"}]}
        self.assertEqual(self.client.post("/add", json=changed).status_code, 409)

    def test_authentication_health_and_invalid_requests(self):
        self.assertEqual(httpx.get(self.url + "/health").status_code, 200)
        for auth in ({}, {"Authorization": "Bearer invalid"}):
            self.assertEqual(httpx.post(self.url + "/search", json={}, headers=auth).status_code, 401)
        query = {"query": "beacon", "user_id": "empty-user", "top_k": 100}
        for auth in ({"Authorization": "Token unit-test-token"}, {"X-Api-Key": "unit-test-token"}):
            self.assertEqual(httpx.post(self.url + "/search", json=query, headers=auth).json(), {"data": []})
        for value in (None, 0, -1, True, "100"):
            self.assertEqual(self.client.post("/search", json=query | {"top_k": value}).status_code, 400)
        self.assertEqual(self.client.post("/search", json={"query": "beacon", "top_k": 1}).status_code, 400)
        self.assertEqual(self.client.post("/missing", json={}).status_code, 404)
        self.assertEqual(self.client.post("/add", content="broken json").status_code, 400)

    def test_exhausted_model_output_returns_502_with_stage_and_reason(self):
        model = ScriptedModel(*({"facts": f"bad-{i}"} for i in range(3)))
        with patch.object(ScriptedModel, "complete", side_effect=model.complete):
            response = self.client.post("/add", json=self.payload)
        self.assertEqual(response.status_code, 502, response.text)
        error = response.json()
        self.assertEqual(error["type"], "ModelOutputError")
        self.assertIn("collection_extract", error["error"])
        self.assertIn("3 次输出尝试", error["error"])
        self.assertIn("facts", error["error"])
        self.assertNotIn("data", error)


class ChannelPoolApplicationTests(unittest.TestCase):
    def test_round_robin_is_shared_across_application_requests(self):
        requested = []

        def handler(request):
            model = json.loads(request.content)["model"]
            requested.append(model)
            return httpx.Response(200, json={"model": model, "choices": [
                {"finish_reason": "stop", "message": {"content": json.dumps(
                    {"combine": False, "changed": False, "ever": False})}}]})

        config = ModelConfig("https://one.invalid/v1", "one-key", "one-model", fallbacks=(
            ModelConfig("https://two.invalid/v1", "two-key", "two-model"),), strategy="round_robin")
        channel_pool = ModelChannelPool(config)
        transport = httpx.MockTransport(handler)
        class Vectors:
            identity = {"model": "synthetic-channels"}
            def encode(self, texts):
                return [[1.0, 0.0] for _ in texts]
        with tempfile.TemporaryDirectory() as directory:
            app = CollectionApplication(Path(directory) / "memory.sqlite",
                lambda: HttpJsonModel(config, transport=transport, channel_pool=channel_pool),
                embedder=Vectors(), floor=.62)
            body = {"query": "anything", "user_id": "channel-test", "top_k": 1}
            self.assertEqual(app.handle("/search", body), {"data": []})
            self.assertEqual(app.handle("/search", body), {"data": []})
        self.assertEqual(requested, ["one-model", "two-model"])


class EmbeddingHTTPTests(unittest.TestCase):
    def test_batches_preserve_order_all_text_and_unit_vectors(self):
        requests = []
        def handle(request):
            body = json.loads(request.content)
            requests.append(body)
            return httpx.Response(200, json={"data": list(reversed([
                {"index": i, "embedding": [3., 4.]} for i, _ in enumerate(body["input"])]))})
        embedder = HttpEmbedder("https://provider.invalid/v1", "secret", "text-embedding-v4",
                                batch_size=2, transport=httpx.MockTransport(handle))
        texts = ["one", "two" * 10000, "three"]
        self.assertEqual(embedder.encode(texts), [[.6, .8]] * 3)
        self.assertEqual([s for req in requests for s in req["input"]], texts)
        self.assertEqual(len(requests), 2)

    def test_bad_vectors_and_provider_failures_do_not_leak_or_fallback(self):
        cases = [httpx.Response(401, text="secret"), httpx.Response(200, json={"data": []}),
                 httpx.Response(200, json={"data": [{"index": 0, "embedding": [0., 0.]}]}),
                 httpx.Response(200, json={"data": [{"index": True, "embedding": [1., 0.]}]}),
                 httpx.Response(200, json={"data": [{"index": 0, "embedding": [True, 1.]}]})]
        for response in cases:
            with self.subTest(response=response):
                embedder = HttpEmbedder("https://provider.invalid/v1", "secret", "text-embedding-v4",
                                        transport=httpx.MockTransport(lambda _: response))
                with self.assertRaises(ModelError) as error:
                    embedder.encode(["fact"])
                self.assertNotIn("secret", str(error.exception))

    def test_remote_and_local_configuration_cannot_mix(self):
        with patch.dict(os.environ, {"MEMORY_EMBED_PYTHON": "/python", "MEMORY_EMBED_API_BASE": "https://provider.invalid/v1",
                                    "MEMORY_EMBED_API_KEY": "secret"}, clear=True):
            with self.assertRaises(ConfigurationError):
                embedder_from_env()


if __name__ == "__main__":
    unittest.main()
