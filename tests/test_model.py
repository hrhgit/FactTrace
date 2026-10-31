import json
import os
import unittest
from dataclasses import replace
from unittest.mock import patch

import httpx

from text_memory.errors import ConfigurationError, InvalidOutputError, ModelError
from text_memory.model import HttpJsonModel, ModelChannelPool, ModelConfig


def reply(model="test-model", content="ok", finish_reason="stop"):
    return httpx.Response(200, json={"model": model, "usage": {"total_tokens": 3}, "choices": [
        {"finish_reason": finish_reason, "message": {"content": content}}]})


def channels(strategy="round_robin"):
    return ModelConfig("https://one.invalid/v1", "one-secret", "one-model", "low", fallbacks=(
        ModelConfig("https://two.invalid/v1", "two-secret", "two-model"),
        ModelConfig("https://three.invalid/v1", "three-secret", "three-model", "high"),
    ), strategy=strategy)


class ModelConfigTests(unittest.TestCase):
    def test_single_and_numbered_channels_from_env(self):
        env = {
            "MEMORY_API_BASE": "https://one.invalid/v1",
            "MEMORY_API_KEY": "one-secret",
            "MEMORY_MODEL": "one-model",
            "MEMORY_REASONING_EFFORT": "low",
            "MEMORY_API_BASE_2": "https://two.invalid/v1",
            "MEMORY_API_KEY_2": "two-secret",
            "MEMORY_MODEL_2": "two-model",
            "MEMORY_API_BASE_4": "https://four.invalid/v1",
            "MEMORY_API_KEY_4": "four-secret",
            "MEMORY_MODEL_4": "four-model",
            "MEMORY_REASONING_EFFORT_4": "high",
        }
        with patch.dict(os.environ, env, clear=True):
            config = ModelConfig.from_env()
        self.assertEqual([channel.model for channel in config.channels],
                         ["one-model", "two-model", "four-model"])
        self.assertEqual([channel.reasoning_effort for channel in config.channels],
                         ["low", None, "high"])
        self.assertEqual(config.model, "one-model")
        self.assertEqual(len(config.fallbacks), 2)

        with patch.dict(os.environ, {"MEMORY_API_BASE": "https://one.invalid", "MEMORY_API_KEY": "key",
                                     "MEMORY_MODEL": "model"}, clear=True):
            single = ModelConfig.from_env()
        self.assertEqual(single.channels, (single,))
        self.assertIsNone(single.reasoning_effort)
        self.assertEqual(single.strategy, "failover")

    def test_channel_strategy_from_env_and_validation(self):
        base = {"MEMORY_API_BASE": "https://one.invalid", "MEMORY_API_KEY": "key",
                "MEMORY_MODEL": "model"}
        for value, expected in ((None, "failover"), ("failover", "failover"),
                                ("round_robin", "round_robin")):
            with self.subTest(value=value):
                env = dict(base)
                if value is not None:
                    env["MEMORY_CHANNEL_STRATEGY"] = value
                with patch.dict(os.environ, env, clear=True):
                    self.assertEqual(ModelConfig.from_env().strategy, expected)
        with patch.dict(os.environ, base | {"MEMORY_CHANNEL_STRATEGY": "random"}, clear=True):
            with self.assertRaises(ConfigurationError):
                ModelConfig.from_env()

    def test_prefix_and_missing_numbered_channel_settings(self):
        env = {
            "ANSWER_API_BASE": "https://one.invalid/v1",
            "ANSWER_API_KEY": "one-secret",
            "ANSWER_MODEL": "one-model",
            "ANSWER_API_BASE_2": "https://two.invalid/v1",
            "ANSWER_API_KEY_2": "",
            "ANSWER_MODEL_2": "two-model",
        }
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaises(ConfigurationError) as caught:
                ModelConfig.from_env("ANSWER")
        self.assertIn("ANSWER_API_KEY_2", str(caught.exception))
        self.assertNotIn("one-secret", str(caught.exception))

    def test_backup_channel_numbers_start_at_two(self):
        env = {
            "MEMORY_API_BASE": "https://one.invalid/v1",
            "MEMORY_API_KEY": "one-secret",
            "MEMORY_MODEL": "one-model",
            "MEMORY_API_BASE_1": "https://duplicate.invalid/v1",
        }
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaises(ConfigurationError):
                ModelConfig.from_env()

    def test_exhausted_fallback_channel_is_loaded_and_validated(self):
        env = {
            "MEMORY_API_BASE": "https://one.invalid/v1",
            "MEMORY_API_KEY": "one-secret",
            "MEMORY_MODEL": "one-model",
            "MEMORY_API_BASE_2": "https://two.invalid/v1",
            "MEMORY_API_KEY_2": "two-secret",
            "MEMORY_MODEL_2": "two-model",
            "MEMORY_EXHAUSTED_FALLBACK": "1",
        }
        with patch.dict(os.environ, env, clear=True):
            self.assertEqual(ModelConfig.from_env().exhausted_fallback, 1)
        with patch.dict(os.environ, env | {"MEMORY_EXHAUSTED_FALLBACK": "3"}, clear=True):
            with self.assertRaises(ConfigurationError):
                ModelConfig.from_env()


class HttpJsonModelChannelTests(unittest.TestCase):
    def setUp(self):
        self.messages = [{"role": "user", "content": "question"}]

    def test_healthy_calls_rotate_channels_and_wrap(self):
        requests = []

        def handler(request):
            body = json.loads(request.content)
            requests.append((request, body))
            return reply(body["model"], body["model"])

        model = HttpJsonModel(channels(), transport=httpx.MockTransport(handler))
        answers = [model.complete_text(self.messages, purpose="answer", temperature=0)
                   for _ in range(4)]
        self.assertEqual(answers, ["one-model", "two-model", "three-model", "one-model"])
        self.assertEqual([str(request.url) for request, _ in requests], [
            "https://one.invalid/v1/chat/completions",
            "https://two.invalid/v1/chat/completions",
            "https://three.invalid/v1/chat/completions",
            "https://one.invalid/v1/chat/completions",
        ])
        self.assertEqual([request.headers["Authorization"] for request, _ in requests], [
            "Bearer one-secret", "Bearer two-secret", "Bearer three-secret", "Bearer one-secret"])
        self.assertEqual([body["model"] for _, body in requests],
                         ["one-model", "two-model", "three-model", "one-model"])
        self.assertEqual([body.get("reasoning_effort") for _, body in requests],
                         ["low", None, "high", "low"])
        self.assertNotIn("reasoning_effort", requests[1][1])
        self.assertTrue(all(body["temperature"] == 0 for _, body in requests))
        self.assertEqual([call["channel_index"] for call in model.calls], [1, 2, 3, 1])
        self.assertEqual([call["failover_index"] for call in model.calls], [0, 0, 0, 0])

    def test_exhausted_fallback_reuses_selected_channel_after_all_caps(self):
        config = replace(channels(), fallbacks=(channels().fallbacks[0],),
                         exhausted_fallback=1)
        pool = ModelChannelPool(config)
        self.addCleanup(pool.usage.close)
        pool.update_limits(1, max_total_tokens=3)
        pool.update_limits(2, max_total_tokens=3)
        def handler(request):
            model_name = json.loads(request.content)["model"]
            return reply(model_name, model_name)
        model = HttpJsonModel(config, channel_pool=pool,
                              transport=httpx.MockTransport(handler), max_retries=0)
        self.assertEqual([model.complete_text(self.messages, purpose="answer") for _ in range(3)],
                         ["one-model", "two-model", "one-model"])
        self.assertEqual([call["channel_index"] for call in model.calls], [1, 2, 1])
        self.assertTrue(model.calls[-1]["exhausted_fallback"])
        self.assertEqual(pool.snapshot()["channels"][0]["usage"]["total_tokens"], 6)

    def test_shared_pool_applies_strategy_across_model_instances(self):
        for strategy, expected in (("round_robin", ["one-model", "two-model", "three-model"]),
                                   ("failover", ["one-model", "one-model", "one-model"])):
            with self.subTest(strategy=strategy):
                config = channels(strategy)
                channel_pool = ModelChannelPool(config)
                transport = httpx.MockTransport(lambda request: reply(
                    json.loads(request.content)["model"], json.loads(request.content)["model"]))
                models = [HttpJsonModel(config, transport=transport, channel_pool=channel_pool)
                          for _ in range(3)]
                answers = [model.complete_text(self.messages, purpose="answer") for model in models]
                self.assertEqual(answers, expected)
                for model in models:
                    self.assertEqual(model.calls[0]["channel_strategy"], strategy)

    def test_failover_strategy_switches_only_after_failure(self):
        requested = []

        def handler(request):
            model = json.loads(request.content)["model"]
            requested.append(model)
            if model == "one-model" and requested.count("one-model") == 2:
                return httpx.Response(503, json={"error": "temporary"})
            return reply(model, model)

        model = HttpJsonModel(channels("failover"), transport=httpx.MockTransport(handler), max_retries=0)
        answers = [model.complete_text(self.messages, purpose="answer") for _ in range(3)]
        self.assertEqual(answers, ["one-model", "two-model", "one-model"])
        self.assertEqual(requested, ["one-model", "one-model", "two-model", "one-model"])

    def test_transient_http_errors_retry_then_fail_over_with_audit(self):
        def handler(request):
            if request.url.host == "one.invalid":
                return httpx.Response(503, json={"error": "temporarily unavailable"})
            return reply("two-model", '{"answer": 42}')

        model = HttpJsonModel(channels(), transport=httpx.MockTransport(handler), max_retries=1,
                              retry_base_delay_s=0, retry_jitter_s=0, sleep=lambda _: None)
        self.assertEqual(model.complete(self.messages, purpose="answer"), {"answer": 42})
        self.assertEqual([call["status"] for call in model.calls],
                         ["http_error", "http_error", "completed"])
        self.assertEqual([call["channel_index"] for call in model.calls], [1, 1, 2])
        self.assertEqual([call["retry_index"] for call in model.calls], [0, 1, 0])
        self.assertEqual([call["failover_index"] for call in model.calls], [0, 0, 1])
        self.assertTrue(model.calls[0]["retry_scheduled"])
        self.assertFalse(model.calls[1]["retry_scheduled"])
        self.assertTrue(model.calls[1]["failover_scheduled"])
        self.assertEqual(model.calls[-1]["requested_model"], "two-model")
        self.assertEqual(model.calls[-1]["actual_model"], "two-model")

    def test_transport_failure_switches_channel(self):
        requests = []

        def handler(request):
            requests.append(request.url.host)
            if request.url.host == "one.invalid":
                raise httpx.ConnectError("offline", request=request)
            return reply("two-model")

        model = HttpJsonModel(channels(), transport=httpx.MockTransport(handler), max_retries=0)
        self.assertEqual(model.complete_text(self.messages, purpose="answer"), "ok")
        self.assertEqual(requests, ["one.invalid", "two.invalid"])
        self.assertEqual([call["status"] for call in model.calls], ["transport_error", "completed"])
        self.assertTrue(model.calls[0]["failover_scheduled"])

    def test_auth_and_quota_failures_do_not_retry_in_place_but_fail_over(self):
        for status, detail in ((401, "unauthorized"), (429, "insufficient quota")):
            with self.subTest(status=status):
                hosts = []

                def handler(request):
                    hosts.append(request.url.host)
                    if request.url.host == "one.invalid":
                        return httpx.Response(status, text=detail + " one-secret")
                    return reply("two-model")

                model = HttpJsonModel(channels(), transport=httpx.MockTransport(handler), max_retries=2,
                                      retry_base_delay_s=0, retry_jitter_s=0, sleep=lambda _: None)
                self.assertEqual(model.complete_text(self.messages, purpose="answer"), "ok")
                self.assertEqual(hosts, ["one.invalid", "two.invalid"])
                self.assertEqual(len(model.calls), 2)
                self.assertFalse(model.calls[0]["retryable"])
                self.assertFalse(model.calls[0]["retry_scheduled"])
                self.assertTrue(model.calls[0]["failover_scheduled"])
                self.assertNotIn("one-secret", model.calls[0]["http_error_body"])

    def test_failover_wraps_from_the_rotated_start_channel(self):
        hosts = []

        def handler(request):
            hosts.append(request.url.host)
            if len(hosts) > 1 and request.url.host == "two.invalid":
                return httpx.Response(500, json={"error": "offline"})
            return reply(json.loads(request.content)["model"])

        model = HttpJsonModel(channels(), transport=httpx.MockTransport(handler), max_retries=0)
        model.complete_text(self.messages, purpose="first")
        self.assertEqual(model.complete_text(self.messages, purpose="second"), "ok")
        self.assertEqual(hosts, ["one.invalid", "two.invalid", "three.invalid"])
        self.assertEqual([call["channel_index"] for call in model.calls[1:]], [2, 3])

    def test_all_channels_fail_once_per_round_without_leaking_secrets(self):
        hosts = []

        def handler(request):
            hosts.append(request.url.host)
            return httpx.Response(500 if request.url.host == "one.invalid" else 401,
                                  text="failure " + request.headers["Authorization"])

        model = HttpJsonModel(channels(), transport=httpx.MockTransport(handler), max_retries=0)
        with self.assertRaises(ModelError) as caught:
            model.complete_text(self.messages, purpose="answer")
        self.assertEqual(len(hosts), 3)
        self.assertEqual(set(hosts), {"one.invalid", "two.invalid", "three.invalid"})
        self.assertEqual(len(model.calls), 3)
        self.assertEqual([call["failover_scheduled"] for call in model.calls], [True, True, False])
        message = str(caught.exception)
        self.assertIn("3 个模型渠道均请求失败", message)
        for secret in ("one-secret", "two-secret", "three-secret", "https://"):
            self.assertNotIn(secret, message)

    def test_successful_http_response_is_not_replayed_after_output_errors(self):
        cases = (
            ("complete", reply("one-model", "{broken"), InvalidOutputError),
            ("complete_text", reply("one-model", "truncated", "length"), ModelError),
            ("complete_text", httpx.Response(200, json={"unexpected": []}), ModelError),
        )
        for method, response, error_type in cases:
            with self.subTest(method=method, error_type=error_type):
                hosts = []

                def handler(request):
                    hosts.append(request.url.host)
                    return response if request.url.host == "one.invalid" else reply("two-model")

                model = HttpJsonModel(channels(), transport=httpx.MockTransport(handler), max_retries=0)
                with self.assertRaises(error_type):
                    getattr(model, method)(self.messages, purpose="answer")
                self.assertEqual(hosts, ["one.invalid"])
                self.assertEqual(len(model.calls), 1)
                self.assertNotEqual(model.calls[0]["status"], "completed")


if __name__ == "__main__":
    unittest.main()
