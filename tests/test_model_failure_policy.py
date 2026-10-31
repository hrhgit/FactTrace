"""The formal path exposes model failures instead of selecting substitutes."""
import unittest

from text_memory.collections.question_flags import check_question
from text_memory.errors import InvalidOutputError, ModelError
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
