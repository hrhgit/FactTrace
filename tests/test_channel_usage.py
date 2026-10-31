"""Accounting and live control against mock HTTP; no paid model calls."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

import httpx

from text_memory import cli
from text_memory.errors import ConfigurationError, InvalidOutputError, ModelError
from text_memory.http_embeddings import HttpEmbedder
from text_memory.model import HttpJsonModel, ModelChannelPool, ModelConfig
from text_memory.usage import ChannelConcurrencyError, RuntimeConcurrency, UsageLimitError, normalize_usage

ROOT = Path(__file__).resolve().parents[1]
USAGE = {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120,
         "prompt_tokens_details": {"cached_tokens": 80},
         "completion_tokens_details": {"reasoning_tokens": 5}}


def reply(usage=USAGE, *, content="ok"):
    return httpx.Response(200, json={"usage": usage, "choices": [
        {"finish_reason": "stop", "message": {"content": content}}]})


class ChannelUsageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.database = Path(self.tmp.name) / "usage.sqlite"
        self.config = ModelConfig("https://one.invalid/v1", "one-secret", "codex/gpt-6-luna",
                                  source="OpenAI source", usage_path=str(self.database), fallbacks=(
            ModelConfig("https://two.invalid/v1", "two-secret", "cn:deepseek-v4.1-flash", source="other source"),))
        self.messages = [{"role": "user", "content": "a private prompt"}]

    def pool(self, config=None):
        pool = ModelChannelPool(config or self.config)
        self.addCleanup(pool.usage.close)
        return pool

    def model(self, pool, handler=lambda _: reply(), **kwargs):
        return HttpJsonModel(pool.config, channel_pool=pool, transport=httpx.MockTransport(handler),
                             retry_base_delay_s=0, retry_jitter_s=0, sleep=lambda _: None, **kwargs)

    def ask(self, model):
        return model.complete_text(self.messages, purpose="answer")

    def test_official_fields_and_reference_amount_do_not_duplicate_cache_or_reasoning(self):
        pool = self.pool()
        model = self.model(pool)
        self.assertEqual(self.ask(model), "ok")
        usage = pool.snapshot()["channels"][0]["usage"]
        self.assertEqual({k: usage[k] for k in USAGE if not k.endswith("details")},
                         {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120})
        self.assertEqual((usage["cached_input_tokens"], usage["reasoning_tokens"]), (80, 5))
        self.assertEqual(usage["estimated_cost_usd"], 0.0000128)
        self.assertEqual((usage["requests"], usage["completed"], usage["errors"]), (1, 1, 0))
        self.assertEqual(usage["unfinished_requests"], 0)

    def test_invalid_output_is_billed_and_does_not_trigger_channel_failover(self):
        pool = self.pool()
        model = self.model(pool, lambda _: reply(content="{broken"))
        with self.assertRaises(InvalidOutputError):
            model.complete(self.messages, purpose="extract")
        self.assertEqual(len(model.calls), 1)
        usage = pool.snapshot()["channels"][0]["usage"]
        self.assertEqual((usage["total_tokens"], usage["errors"], usage["completed"]), (120, 1, 0))
        self.assertEqual(usage["estimated_cost_usd"], 0.0000128)

    def test_usage_survives_malformed_choices_and_missing_measurements_stay_visible(self):
        pool = self.pool()
        model = self.model(pool, lambda _: httpx.Response(200, json={"usage": USAGE, "choices": []}))
        with self.assertRaises(ModelError):
            self.ask(model)
        self.assertEqual(pool.snapshot()["channels"][0]["usage"]["total_tokens"], 120)
        self.ask(self.model(pool, lambda _: reply(None)))
        usage = pool.snapshot()["channels"][0]["usage"]
        self.assertEqual((usage["total_tokens"], usage["unmeasured_requests"], usage["unpriced_requests"]), (120, 1, 1))
        self.assertIsNone(normalize_usage({"total_tokens": -1})["total_tokens"])
        self.assertIsNone(normalize_usage({"prompt_tokens": True})["prompt_tokens"])
        self.assertEqual(normalize_usage({"input_tokens": 6, "output_tokens": 4})["total_tokens"], 10)

    def test_same_key_is_shared_across_roles_models_and_new_pool_instances(self):
        memory = self.pool()
        answer = self.pool(replace(self.config, channel_group="ANSWER", model="gpt-6-luna"))
        self.ask(self.model(memory))
        self.ask(self.model(answer))
        restored = self.pool()
        usage = restored.snapshot()["channels"][0]["usage"]
        self.assertEqual((usage["requests"], usage["total_tokens"]), (2, 240))
        self.assertEqual(usage["estimated_cost_usd"], 0.0000256)
        # A rotated key has separate accounting even if its friendly label stays the same.
        rotated = self.pool(replace(self.config, api_key="replacement-secret"))
        self.assertEqual(rotated.snapshot()["channels"][0]["usage"]["requests"], 0)

    def test_source_labels_remain_stable_when_shared_keys_load_other_roles(self):
        named = self.pool()
        unnamed = self.pool(replace(self.config, source=None, channel_group="ANSWER"))
        self.assertEqual(named.snapshot()["channels"][0]["source"], "OpenAI source")
        self.assertEqual(unnamed.channel_source(1), "OpenAI source")
        alias = self.pool(replace(self.config, source="answer alias", channel_group="ANSWER"))
        self.assertEqual(alias.snapshot()["channels"][0]["source"], "answer alias")
        self.assertEqual(named.snapshot()["channels"][0]["source"], named.channel_source(1))
        self.assertEqual(named.snapshot()["channels"][0]["key_id"], alias.snapshot()["channels"][0]["key_id"])

    def test_total_token_cap_skips_a_source_and_all_exhausted_sources_fail_visibly(self):
        pool = self.pool()
        pool.update_limits(1, max_total_tokens=120)
        hosts = []
        def handler(request):
            hosts.append(request.url.host)
            return reply()
        model = self.model(pool, handler)
        self.ask(model)
        self.ask(model)
        self.assertEqual(hosts, ["one.invalid", "two.invalid"])
        self.assertEqual([c["status"] for c in model.calls], ["completed", "usage_limit", "completed"])
        self.assertEqual(pool.snapshot()["channels"][0]["usage"]["requests"], 1)
        pool.update_limits(2, max_requests=1)
        with self.assertRaisesRegex(ModelError, "用量上限"):
            self.ask(model)
        self.assertEqual(hosts, ["one.invalid", "two.invalid"])

    def test_retries_count_and_recheck_the_request_cap(self):
        pool = self.pool()
        pool.update_limits(1, max_requests=1)
        hosts = []
        def handler(request):
            hosts.append(request.url.host)
            return httpx.Response(503) if request.url.host == "one.invalid" else reply()
        model = self.model(pool, handler, max_retries=2)
        self.assertEqual(self.ask(model), "ok")
        self.assertEqual(hosts, ["one.invalid", "two.invalid"])
        self.assertEqual([c["status"] for c in model.calls], ["http_error", "usage_limit", "completed"])
        usage = pool.snapshot()["channels"][0]["usage"]
        self.assertEqual((usage["requests"], usage["errors"], usage["unmeasured_requests"]), (1, 1, 1))

    def test_usage_on_an_http_error_is_committed_before_retry_waits(self):
        pool = self.pool()
        pool.update_limits(1, max_total_tokens=120)
        model = self.model(pool, lambda _: httpx.Response(503, json={"usage": USAGE}))
        snapshots = []
        model._sleep = lambda _: snapshots.append(pool.snapshot()["channels"][0]["usage"]["total_tokens"])
        with self.assertRaises(ModelError):
            self.ask(model)
        self.assertEqual(snapshots[0], 120)
        self.assertEqual(pool.snapshot()["channels"][0]["usage"]["requests"], 1)

    def test_reference_dollar_cap_and_explicit_reference_for_other_provider(self):
        pool = self.pool()
        pool.update_limits(1, max_cost_usd=0.0000128)
        self.ask(self.model(pool))
        self.assertTrue(pool.snapshot()["channels"][0]["exhausted"])
        with self.assertRaisesRegex(ConfigurationError, "参考单价"):
            pool.update_limits(2, max_cost_usd=1)
        pool.update_limits(2, max_cost_usd=1, reference_model="gpt-6-luna")
        pool.update_priority([2, 1])
        model = self.model(pool)
        self.ask(model)
        self.assertEqual(model.calls[0]["requested_model"], "cn:deepseek-v4.1-flash")
        self.assertEqual(pool.snapshot()["channels"][1]["usage"]["estimated_cost_usd"], 0.0000128)
        pool.update_limits(1, max_cost_usd=None)
        self.assertFalse(pool.snapshot()["channels"][0]["exhausted"])

    def test_shared_dollar_cap_blocks_unpriced_models_before_sending(self):
        priced = self.pool(replace(self.config, fallbacks=()))
        priced.update_limits(1, max_cost_usd=1)
        unpriced = self.pool(replace(priced.config, model="unpriced-model", channel_group="ANSWER"))
        sent = []
        model = self.model(unpriced, lambda request: sent.append(request) or reply())
        with self.assertRaisesRegex(ModelError, "参考单价"):
            self.ask(model)
        self.assertEqual(sent, [])
        self.assertEqual(priced.snapshot()["channels"][0]["usage"]["requests"], 0)
        priced.update_limits(1, reference_model="gpt-6-luna")
        self.assertEqual(self.ask(model), "ok")
        self.assertEqual(len(sent), 1)
        self.assertEqual(priced.snapshot()["channels"][0]["usage"]["estimated_cost_usd"], 0.0000128)

    def test_all_dollar_caps_switch_to_original_key_and_keep_accounting_after_restart(self):
        config = replace(self.config, strategy="round_robin", exhausted_fallback=1)
        pool = self.pool(config)
        pool.update_limits(1, max_cost_usd=0.0000128)
        pool.update_limits(2, max_cost_usd=0.0000256, reference_model="gpt-6-luna")
        hosts = []
        model = self.model(pool, lambda request: hosts.append(request.url.host) or reply())
        for _ in range(5):
            self.assertEqual(self.ask(model), "ok")
        self.assertEqual(hosts, ["one.invalid", "two.invalid", "two.invalid", "one.invalid", "one.invalid"])
        restored = self.pool(config)
        report = restored.snapshot()
        self.assertEqual(report["exhausted_fallback"], 1)
        self.assertTrue(report["exhausted_fallback_active"])
        self.assertEqual(report["channels"][0]["limits"]["max_cost_usd"], 0.0000128)
        self.assertEqual(report["channels"][0]["usage"]["exhausted_fallback_requests"], 2)
        self.assertEqual(report["channels"][0]["usage"]["estimated_cost_usd"], 0.0000384)
        self.assertEqual(report["channels"][1]["usage"]["requests"], 2)
        rows = restored.usage.db.execute(
            "SELECT exhausted_fallback, estimated_cost_nusd FROM model_attempts WHERE exhausted_fallback=1"
        ).fetchall()
        self.assertEqual([tuple(row) for row in rows], [(1, 12800), (1, 12800)])
        self.assertEqual(self.ask(self.model(restored, lambda request: hosts.append(request.url.host) or reply())), "ok")
        self.assertEqual(hosts[-1], "one.invalid")
        self.assertEqual(restored.snapshot()["channels"][0]["usage"]["exhausted_fallback_requests"], 3)
        # The exception is local to this configured pool, not a global budget removal.
        strict = self.pool(replace(config, exhausted_fallback=None))
        with self.assertRaises(ModelError):
            self.ask(self.model(strict, lambda request: self.fail("strict pool sent an over-limit request")))

    def test_one_capped_key_and_one_service_error_do_not_enable_over_limit_fallback(self):
        pool = self.pool(replace(self.config, exhausted_fallback=1))
        pool.update_limits(1, max_requests=0)
        pool.update_limits(2, max_requests=10)
        hosts = []
        model = self.model(pool, lambda request: hosts.append(request.url.host) or httpx.Response(503),
                           max_retries=0)
        with self.assertRaises(ModelError):
            self.ask(model)
        self.assertEqual(hosts, ["two.invalid"])
        self.assertFalse(pool.snapshot()["exhausted_fallback_active"])
        self.assertTrue(all(not call.get("exhausted_fallback") for call in model.calls))

    def test_error_spending_last_allowance_can_use_fallback_in_the_same_request(self):
        pool = self.pool(replace(self.config, exhausted_fallback=1))
        pool.update_limits(1, max_requests=0)
        pool.update_limits(2, max_total_tokens=120)
        hosts = []
        def handler(request):
            hosts.append(request.url.host)
            return httpx.Response(503, json={"usage": USAGE}) if request.url.host == "two.invalid" else reply()
        model = self.model(pool, handler, max_retries=0)
        self.assertEqual(self.ask(model), "ok")
        self.assertEqual(hosts, ["two.invalid", "one.invalid"])
        self.assertEqual([call["status"] for call in model.calls], ["usage_limit", "http_error", "completed"])
        self.assertTrue(model.calls[-1]["exhausted_fallback"])
        self.assertEqual(pool.snapshot()["channels"][0]["usage"]["exhausted_fallback_requests"], 1)

    def test_fallback_rechecks_live_limits_and_only_allows_the_configured_key(self):
        config = replace(self.config, exhausted_fallback=1)
        running, controller = self.pool(config), self.pool(config)
        running.update_limits(1, max_requests=0)
        running.update_limits(2, max_requests=0)
        self.assertTrue(running.selection()["terminal_fallback"])
        controller.update_limits(2, max_requests=1)
        with self.assertRaisesRegex(UsageLimitError, "尚有渠道"):
            running.begin(1, "answer", exhausted_fallback=True)
        with self.assertRaises(ConfigurationError):
            running.begin(2, "answer", exhausted_fallback=True)
        self.assertEqual(running.snapshot()["channels"][0]["usage"]["requests"], 0)
        self.assertEqual(self.ask(self.model(running)), "ok")
        self.assertTrue(running.snapshot()["exhausted_fallback_active"])

    def test_fallback_does_not_bypass_missing_dollar_prices_or_unconfigured_limits(self):
        pool = self.pool(replace(self.config, exhausted_fallback=1))
        pool.update_limits(1, max_cost_usd=0)
        pool.update_limits(2, max_requests=0)
        unpriced = self.pool(replace(pool.config, model="unpriced-model"))
        with self.assertRaisesRegex(ModelError, "参考单价"):
            self.ask(self.model(unpriced, lambda request: self.fail("unpriced over-limit request was sent")))
        pool.update_limits(2, max_requests=None)
        self.assertFalse(pool.exhausted_fallback_ready())
        with self.assertRaises(UsageLimitError):
            pool.begin(1, "answer", exhausted_fallback=True)

    def test_legacy_usage_migration_preserves_counters_and_marks_old_attempts_as_regular(self):
        pool = self.pool()
        self.ask(self.model(pool))
        pool.usage.db.execute("ALTER TABLE model_attempts DROP COLUMN exhausted_fallback")
        pool.usage.close()
        restored = self.pool()
        report = restored.snapshot()["channels"][0]["usage"]
        self.assertEqual((report["requests"], report["total_tokens"]), (1, 120))
        self.assertEqual(report["estimated_cost_usd"], 0.0000128)
        self.assertEqual(report["exhausted_fallback_requests"], 0)
        self.assertEqual(self.ask(self.model(restored)), "ok")

    def test_invalid_or_partial_totals_do_not_discard_known_tokens(self):
        pool = self.pool(replace(self.config, fallbacks=()))
        for measured, expected_total, missing in (
            ({"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 0}, 120, 0),
            ({"prompt_tokens": 100}, 220, 1),
            ({"completion_tokens": 20, "total_tokens": 0}, 240, 2),
        ):
            with self.subTest(measured=measured):
                self.ask(self.model(pool, lambda _, measured=measured: reply(measured)))
                usage = pool.snapshot()["channels"][0]["usage"]
                self.assertEqual(usage["total_tokens"], expected_total)
                self.assertEqual(usage["unmeasured_requests"], missing)
        pool.update_limits(1, max_total_tokens=240)
        with self.assertRaisesRegex(ModelError, "用量上限"):
            self.ask(self.model(pool))

    def test_two_pools_cannot_both_spend_the_last_request(self):
        config = replace(self.config, fallbacks=())
        first, second = self.pool(config), self.pool(config)
        first.update_limits(1, max_requests=1)
        barrier = threading.Barrier(2)
        def work(pool):
            barrier.wait(timeout=5)
            try:
                return self.ask(self.model(pool))
            except ModelError:
                return "limited"
        with ThreadPoolExecutor(max_workers=2) as workers:
            answers = list(workers.map(work, (first, second)))
        self.assertCountEqual(answers, ["ok", "limited"])
        self.assertEqual(first.snapshot()["channels"][0]["usage"]["requests"], 1)

    def test_running_pool_observes_another_pool_changing_limits(self):
        running, controller = self.pool(), self.pool()
        model = self.model(running)
        self.ask(model)
        controller.update_limits(1, max_requests=1)
        self.ask(model)
        self.assertEqual([c["channel_index"] for c in model.calls], [1, 1, 2])
        self.assertEqual(model.calls[1]["status"], "usage_limit")

    def test_default_strategy_can_change_until_manually_overridden(self):
        self.pool()
        rotated = self.pool(replace(self.config, strategy="round_robin"))
        self.assertEqual([rotated.order()[0][0] for _ in range(2)], [1, 2])
        rotated.update_priority([2], strategy="failover")
        restored = self.pool()
        self.assertEqual([restored.order()[0][0] for _ in range(2)], [2, 2])
        self.assertEqual(restored.snapshot()["strategy"], "failover")

    def test_accounting_failure_stops_without_sending_to_another_provider(self):
        pool = self.pool()
        model = self.model(pool)
        with patch.object(pool.usage, "finish", side_effect=ConfigurationError("用量无法保存")):
            with self.assertRaisesRegex(ConfigurationError, "用量无法保存"):
                self.ask(model)
        self.assertEqual(len(model.calls), 1)
        self.assertTrue(model.calls[0]["accounting_error"])
        self.assertEqual(pool.snapshot()["channels"][0]["usage"]["unfinished_requests"], 1)

    def test_remote_embeddings_share_key_budget_with_generation_calls(self):
        pool = self.pool()
        pool.update_limits(1, max_total_tokens=5)
        requested = []
        def handler(request):
            requested.append(json.loads(request.content))
            return httpx.Response(200, json={"usage": {"prompt_tokens": 5, "total_tokens": 5},
                "data": [{"index": 0, "embedding": [3., 4.]}]})
        embedder = HttpEmbedder(self.config.base_url, self.config.api_key, "embedding-test", batch_size=1,
                                usage_path=self.database, transport=httpx.MockTransport(handler))
        self.addCleanup(embedder.close)
        with self.assertRaisesRegex(ModelError, "用量上限"):
            embedder.encode(["first", "second"])
        self.assertEqual(len(requested), 1)
        usage = pool.snapshot()["channels"][0]["usage"]
        self.assertEqual((usage["requests"], usage["total_tokens"], usage["unmeasured_requests"]), (1, 5, 0))
        model = self.model(pool)
        self.ask(model)
        self.assertEqual([c["status"] for c in model.calls], ["usage_limit", "completed"])

    def test_remote_embedding_error_usage_is_preserved_and_blocks_the_next_send(self):
        pool = self.pool(replace(self.config, fallbacks=()))
        pool.update_limits(1, max_total_tokens=9)
        sent = []
        def handler(request):
            sent.append(request)
            return httpx.Response(503, json={"usage": {"prompt_tokens": 9, "total_tokens": 9}})
        embedder = HttpEmbedder(self.config.base_url, self.config.api_key, "embedding-test",
                                usage_path=self.database, transport=httpx.MockTransport(handler))
        self.addCleanup(embedder.close)
        with self.assertRaisesRegex(ModelError, "HTTP 503"):
            embedder.encode(["review"])
        with self.assertRaisesRegex(ModelError, "用量上限"):
            embedder.encode(["review"])
        self.assertEqual(len(sent), 1)
        usage = pool.snapshot()["channels"][0]["usage"]
        self.assertEqual((usage["requests"], usage["total_tokens"], usage["unmeasured_requests"]), (1, 9, 0))
        self.assertEqual(pool.snapshot()["channels"][0]["source"], "OpenAI source")

    def test_separate_cli_process_changes_next_request_without_interrupting_the_current_request(self):
        third = ModelConfig("https://three.invalid/v1", "three-secret", "gpt-6-luna", source="third source")
        config = replace(self.config, fallbacks=(*self.config.fallbacks, third))
        pool = self.pool(config)
        entered, release = threading.Event(), threading.Event()
        self.addCleanup(release.set)
        hosts = []
        def handler(request):
            hosts.append(request.url.host)
            if request.url.host == "one.invalid":
                entered.set()
                if not release.wait(timeout=10):
                    raise AssertionError("test did not release the current request")
                return httpx.Response(503)
            return reply()
        env = dict(os.environ, MODEL_USAGE_DB=str(self.database), MEMORY_API_BASE=config.base_url,
                   MEMORY_API_KEY=config.api_key, MEMORY_MODEL=config.model, MEMORY_CHANNEL_STRATEGY="failover")
        for number, channel in enumerate(config.fallbacks, start=2):
            env.update({f"MEMORY_API_BASE_{number}": channel.base_url, f"MEMORY_API_KEY_{number}": channel.api_key,
                        f"MEMORY_MODEL_{number}": channel.model})
        empty_env = Path(self.tmp.name) / "empty.env"
        empty_env.touch()
        model = self.model(pool, handler, max_retries=0)
        with ThreadPoolExecutor(max_workers=1) as workers:
            future = workers.submit(self.ask, model)
            try:
                self.assertTrue(entered.wait(timeout=5))
                command = [sys.executable, "-m", "text_memory", "channels", "priority", "3", "2", "1",
                           "--env-file", str(empty_env), "--json"]
                process = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=10)
                self.assertEqual(process.returncode, 0, process.stderr)
                self.assertEqual(json.loads(process.stdout)["priority"], [3, 2, 1])
                self.assertFalse(future.done())
            finally:
                release.set()
            self.assertEqual(future.result(timeout=5), "ok")
        self.assertEqual(self.ask(model), "ok")
        self.assertEqual(hosts, ["one.invalid", "two.invalid", "three.invalid"])
        self.assertEqual([c["channel_generation"] for c in model.calls], [0, 0, 1])

    def test_invalid_priority_is_atomic_and_reports_contain_no_credentials_or_prompts(self):
        pool = self.pool()
        for priority in ([2, 2], [7], [True], "2,1", []):
            with self.assertRaises(ConfigurationError):
                pool.update_priority(priority)
            self.assertEqual(pool.priority(), [1, 2])
        self.ask(self.model(pool))
        report = json.dumps(pool.snapshot())
        for forbidden in ("one-secret", "two-secret", "a private prompt", "https://one.invalid"):
            self.assertNotIn(forbidden, report)
            rows = pool.usage.db.execute("SELECT * FROM model_attempts").fetchall()
            self.assertNotIn(forbidden, str([dict(row) for row in rows]))

    def test_cli_sets_limits_without_loading_embeddings_and_none_removes_a_cap(self):
        config = replace(self.config, fallbacks=())
        with patch.object(cli, "load_dotenv"), patch.object(cli.ModelConfig, "from_env", return_value=config), \
             patch.object(cli, "embedder_from_env", side_effect=AssertionError("must not load embeddings")), \
             redirect_stdout(StringIO()) as output:
            self.assertEqual(cli.main(["channels", "limit", "--channel", "1", "--max-total-tokens", "120", "--json"]), 0)
            self.assertEqual(json.loads(output.getvalue())["channels"][0]["limits"]["max_total_tokens"], 120)
            output.seek(0); output.truncate()
            self.assertEqual(cli.main(["channels", "limit", "--channel", "1", "--max-total-tokens", "none", "--json"]), 0)
            self.assertIsNone(json.loads(output.getvalue())["channels"][0]["limits"]["max_total_tokens"])

    def test_channel_concurrency_is_live_and_zero_disables_new_requests(self):
        pool = self.pool(replace(self.config, fallbacks=()))
        pool.update_concurrency(1, 1)
        attempt = pool.begin(1, "answer")
        with self.assertRaises(ChannelConcurrencyError):
            pool.begin(1, "answer")
        pool.update_concurrency(1, 0)
        pool.usage.finish(attempt, USAGE, "completed")
        with self.assertRaises(UsageLimitError):
            pool.begin(1, "answer")
        self.assertEqual(pool.snapshot()["channels"][0]["limits"]["max_concurrency"], 0)

    def test_total_concurrency_controller_wakes_waiting_tasks(self):
        pool = self.pool(replace(self.config, fallbacks=()))
        limiter = RuntimeConcurrency(pool.usage, "MEMORY", 1)
        limiter.acquire()
        entered = threading.Event()
        def wait_for_slot():
            limiter.acquire()
            entered.set()
        with ThreadPoolExecutor(max_workers=1) as workers:
            future = workers.submit(wait_for_slot)
            self.assertFalse(entered.wait(timeout=0.2))
            pool.update_total_concurrency(2)
            self.assertTrue(entered.wait(timeout=2))
            limiter.release()
            limiter.release()
            future.result(timeout=2)


if __name__ == "__main__":
    unittest.main()
