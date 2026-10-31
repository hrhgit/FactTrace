import unittest

from scripts.prepare_longmemeval import preserve_empty_messages
from text_memory.errors import ValidationError


class EmptyMessagePreparationTests(unittest.TestCase):
    def test_explicit_empty_turn_keeps_position_without_changing_source(self):
        rows = [{"question_id": "q", "haystack_session_ids": ["s"], "haystack_sessions": [[
            {"role": "user", "content": ""}, {"role": "assistant", "content": "real text", "has_answer": True}]]}]
        with self.assertRaises(ValidationError):
            preserve_empty_messages(rows, {"q"}, policy="error")
        adapted, empty = preserve_empty_messages(rows, {"q"}, policy="preserve")
        self.assertEqual(empty, [{"id": "q", "session_id": "s", "position": 0}])
        self.assertEqual(len(adapted[0]["haystack_sessions"][0]), 2)
        self.assertEqual(adapted[0]["haystack_sessions"][0][1], rows[0]["haystack_sessions"][0][1])
        self.assertEqual(rows[0]["haystack_sessions"][0][0]["content"], "")
        self.assertEqual(preserve_empty_messages(rows, {"other"}, policy="error"), (rows, []))
