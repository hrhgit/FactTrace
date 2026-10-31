"""The formal path exposes model failures instead of selecting substitutes."""
import unittest
import json

from tests.fixtures import ScriptedModel, payload
from text_memory.agent_loop import AgentLoop
from text_memory.collections import CollectionEngine, CollectionStore
from text_memory.collections.question_flags import check_question
from text_memory.errors import InvalidOutputError, ModelError, ModelOutputError, SearchLoopError
from text_memory.model import (ModelChannelPool, ModelConfig,
                               decode_json_object, decode_json_object_with_meta)


class ModelFailurePolicyTests(unittest.TestCase):
    class Replies:
        def __init__(self, *values):
            self.values = iter(values)
            self.calls = []

        def complete(self, messages, *, purpose):
            self.calls.append((messages, purpose))
            value = next(self.values)
            if isinstance(value, BaseException):
                raise value
            return value

    def test_question_flags_never_turn_a_failed_judgment_into_false(self):
        model = self.Replies({"combine": "yes"}, {"combine": "yes"}, {"combine": "yes"})
        with self.assertRaises(ModelError):
            check_question(model, "How many?", sleep=lambda _: None)
        self.assertEqual(len(model.calls), 3)

    def test_question_flags_require_all_declared_fields(self):
        model = self.Replies({"ever": True}, {"ever": True}, {"ever": True})
        with self.assertRaises(ModelError):
            check_question(model, "Have I ever?", sleep=lambda _: None)

    def test_question_flags_send_validation_feedback_and_keep_the_original_question(self):
        valid = {"combine": True, "changed": False, "ever": False}
        model = ScriptedModel(flag_responses=[InvalidOutputError("{bad", "bad JSON"), {"combine": "yes"}, valid])
        self.assertEqual(check_question(model, "How many?", sleep=lambda _: None), (valid, None))
        for call in model.calls[1:]:
            self.assertEqual(call["messages"][:2], model.calls[0]["messages"])
            self.assertIn("tool_error", json.loads(call["messages"][-1]["content"]))
        self.assertIn("三个布尔值", model.calls[1]["output_validation"]["message"])
        feedback = json.loads(model.calls[2]["messages"][-1]["content"])["tool_error"]
        self.assertEqual(feedback["message"], "Return exactly the three boolean fields combine, changed and ever and no other fields.")
        self.assertEqual(feedback["path"], "$")
        self.assertEqual(feedback["expected"], "an object with exactly boolean combine, changed and ever fields")

    def test_json_parse_feedback_keeps_exact_parser_location(self):
        with self.assertRaises(InvalidOutputError) as caught:
            decode_json_object('{"ok": }')
        error = caught.exception
        self.assertEqual(error.diagnostic()["path"], "$")
        self.assertEqual(error.diagnostic()["line"], 1)
        self.assertEqual(error.diagnostic()["column"], 8)
        self.assertEqual(error.diagnostic()["position"], 7)
        self.assertIn("The response is not valid JSON", error.model_message)

    def test_extraction_feedback_identifies_invalid_reference_and_keeps_original_reply(self):
        request = payload(messages=[{"role": "user", "content": "Alice owns a red bicycle."}])
        bad = {"facts": [{"text": "Alice owns a red bicycle.", "evidence": ["m0.1", "m99.9"]}]}
        good = {"facts": [{"text": "Alice owns a red bicycle.", "evidence": ["m0.1"]}]}
        model = ScriptedModel(bad, good)
        with CollectionStore() as store:
            extracted = CollectionEngine(store, model).extract(request)
        self.assertEqual(extracted, good)
        retry = model.calls[1]["messages"]
        self.assertEqual(retry[-2], {"role": "assistant", "content": json.dumps(bad, ensure_ascii=False, sort_keys=True, separators=(",", ":"))})
        feedback = json.loads(retry[-1]["content"])["tool_error"]
        self.assertEqual(feedback["path"], "facts[0].evidence[1]")
        self.assertEqual(feedback["actual"], "m99.9")
        self.assertIn("existing paragraph reference", feedback["expected"])
        self.assertIn("Alice owns a red bicycle", feedback["source"])

    def test_changing_invalid_json_stops_after_three_attempts_and_records_the_last_error(self):
        model = ScriptedModel(*(InvalidOutputError(f"bad-{i}", f"parse failure {i}") for i in range(3)))
        trace = []
        loop = AgentLoop(model, [{"role": "user", "content": "original"}], purpose="extract", trace=trace)
        with self.assertRaisesRegex(ModelOutputError, "extract.*3 次输出尝试.*parse failure 2"):
            loop.complete()
        self.assertEqual(len(model.calls), 3)
        self.assertTrue(all("error" in entry and "corrections" in entry for entry in trace))
        self.assertFalse(trace[-1]["retry_scheduled"])
        self.assertFalse(model.calls[-1]["output_validation"]["retry_scheduled"])

    def test_repeated_invalid_reply_stops_early_with_its_reason(self):
        model = ScriptedModel(InvalidOutputError("bad", "not JSON"), InvalidOutputError("bad", "not JSON"))
        with self.assertRaisesRegex(SearchLoopError, "extract.*2 次输出尝试.*not JSON"):
            AgentLoop(model, [], purpose="extract", trace=[]).complete()
        self.assertEqual(len(model.calls), 2)

    def test_invalid_extraction_never_commits_facts_sources_or_receipt(self):
        model = ScriptedModel(*({"facts": f"bad-{i}"} for i in range(3)))
        request = payload(messages=[{"role": "user", "content": "Alice owns a red bicycle."}])
        with CollectionStore() as store:
            engine = CollectionEngine(store, model)
            with self.assertRaisesRegex(ModelError, "collection_extract.*3 次输出尝试.*facts"):
                engine.add(request)
            self.assertIsNone(store.receipt(request))
            self.assertEqual(store.db.execute("SELECT count(*) FROM sources").fetchone()[0], 0)
            self.assertEqual(store.db.execute("SELECT count(*) FROM objects").fetchone()[0], 0)
            self.assertIn("error", engine.trace[-1])

    def test_exhausted_service_error_is_recorded_without_an_extra_correction_round(self):
        model = ScriptedModel(ModelError("all channels failed"))
        trace = []
        with self.assertRaisesRegex(ModelError, "all channels failed"):
            AgentLoop(model, [], purpose="extract", trace=trace).complete()
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(trace[0]["error"]["message"], "all channels failed")

    def test_json_output_allows_only_syntax_repairs(self):
        self.assertEqual(decode_json_object("```json\n{}\n```"), {})
        self.assertEqual(decode_json_object('{"ok":true'), {"ok": True})
        with self.assertRaises(InvalidOutputError):
            decode_json_object('{"ok": }')

        accepted = (
            ('{"a":["x"]}]}', {"a": ["x"]}),
            ('{"a":1}}', {"a": 1}),
            ('{"a":"b"}"}', {"a": "b"}),
            ('{"a":1}</|assistant|>', {"a": 1}),
            ('{"a":1}]}</|assistant|>', {"a": 1}),
            ('{"a":1}</|im_end|>', {"a": 1}),
            ('{"a":1}</assistant>', {"a": 1}),
            ('{"a":1}</final>', {"a": 1}),
            ('{"a":1}</｜｜DSML｜｜ parameter>\n</｜｜DSML｜｜ invoke>\n</｜｜DSML｜｜ calls>', {"a": 1}),
        )
        for content, expected in accepted:
            result, repaired = decode_json_object_with_meta(content)
            self.assertEqual(result, expected)
            self.assertTrue(repaired)

        for content in (
            '{"a":1} trailing',
            '{"a":1}{"b":2}',
            '{"a":1},',
            '[1]]',
            '{"a":1}' + '}' * 9,
            '{"a":1]}}',
            '{"a":1}</unknown>',
            '{"a":NaN}',
            '{"a":Infinity}',
            '{"a":1e999}',
        ):
            with self.assertRaises(InvalidOutputError):
                decode_json_object(content)

    def test_secondary_model_channels_remain_available_for_channel_failover(self):
        config = ModelConfig("https://primary", "key", "model",
                             fallbacks=(ModelConfig("https://secondary", "key2", "model2"),))
        self.assertEqual([channel.model for channel in ModelChannelPool(config).channels],
                         ["model", "model2"])


if __name__ == "__main__":
    unittest.main()
