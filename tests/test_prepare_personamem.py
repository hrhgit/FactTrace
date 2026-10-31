import csv
import json
from pathlib import Path
import tempfile
import unittest

from scripts.prepare_personamem import prepare


class PersonaMemPreparationTests(unittest.TestCase):
    def source(self, root):
        root.mkdir()
        fields = ['shared_context_id', 'end_index_in_shared_context', 'question_id',
                  'user_question_or_message', 'all_options', 'correct_answer',
                  'question_type', 'topic', 'context_length_in_tokens']
        with (root / 'questions_32k.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fields)
            writer.writeheader()
            for context in ['a', 'b', 'c']:
                for suffix, boundary in [('earlier', 1), ('latest-1', 2), ('latest-2', 2)]:
                    writer.writerow(dict(zip(fields, [context, boundary, context + suffix, 'What do I like?',
                        "['(a) tea', '(b) coffee']", '(a)', 'recall', 'drinks', '20'])))
        histories = [{key: [{'role': 'user', 'content': 'I like tea.'},
                           {'role': 'assistant', 'content': 'Tea suits you.'},
                           {'role': 'user', 'content': 'FUTURE_SECRET: now I prefer coffee.'}]}
                     for key in ['a', 'b', 'c']]
        (root / 'shared_contexts_32k.jsonl').write_text(''.join(json.dumps(h) + '\n' for h in histories))
        return root

    def test_future_and_earlier_questions_are_excluded(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = prepare(self.source(root / 'source'), root / 'run', contexts=2)
            histories = (root / 'run/inputs/histories.jsonl').read_text()
            queries = (root / 'run/inputs/queries.jsonl').read_text()
            self.assertNotIn('FUTURE_SECRET', histories)
            self.assertTrue(all(len(json.loads(line)['messages']) == 2 for line in histories.splitlines()))
            self.assertNotIn('earlier', queries)
            self.assertEqual(manifest['questions'], 4)
            self.assertEqual(manifest['contexts'], 2)

    def test_answers_are_separate_and_options_unchanged(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            prepare(self.source(root / 'source'), root / 'run', contexts=1)
            query = json.loads((root / 'run/inputs/queries.jsonl').read_text().splitlines()[0])
            self.assertNotIn('correct_answer', query)
            self.assertEqual(query['all_options'], "['(a) tea', '(b) coffee']")
            label = json.loads((root / 'run/scoring/labels.jsonl').read_text().splitlines()[0])
            self.assertEqual(label['correct_answer'], '(a)')

    def test_seed_is_reproducible_and_contexts_do_not_repeat(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = self.source(root / 'source')
            a = prepare(source, root / 'one', contexts=2)
            b = prepare(source, root / 'two', contexts=2)
            self.assertEqual(a['input_sha256'], b['input_sha256'])
            self.assertEqual(len({r['shared_context_id'] for r in a['prefixes']}), 2)


if __name__ == '__main__':
    unittest.main()
