import unittest

from text_memory.collections.answer_context import chronological, session_fact_blocks, render_blocks


class AnswerContextTests(unittest.TestCase):
    def test_ranked_blocks_keep_chronology_and_deduplicate_shared_facts(self):
        sources = [
            {'id': 'sA', 'session_id': 'A', 'position': 0,
             'message': {'session_date': '2026-01-01', 'content': 'RAW SECRET'}},
            {'id': 'sB1', 'session_id': 'B', 'position': 1,
             'message': {'session_date': '2026-02-01', 'content': 'RAW SECRET'}},
            {'id': 'sB2', 'session_id': 'B', 'position': 2,
             'message': {'session_date': '2026-02-01', 'content': 'RAW SECRET'}}]
        result = {'matched_fact_ids': ['high', 'low'], 'evidence': [
            {'id': 'low', 'text': 'Lower match', 'source_ids': ['sA.1']},
            {'id': 'high', 'text': 'Highest match', 'source_ids': ['sB2.1']},
            {'id': 'context', 'text': 'Earlier context', 'source_ids': ['sB1.1']},
            {'id': 'shared', 'text': 'Shared fact', 'source_ids': ['sA.1', 'sB1.1']}],
            'diagnostics': 'INTERNAL'}
        blocks = session_fact_blocks(result, sources, ['low', 'context', 'shared', 'high'])
        self.assertEqual([b['session_id'] for b in blocks], ['B', 'A'])
        self.assertEqual([f['id'] for f in blocks[0]['facts']], ['context', 'shared', 'high'])
        text = render_blocks(blocks)
        self.assertEqual(text.count('Shared fact'), 1)
        self.assertIn('Source conversation dates: 2026-01-01; 2026-02-01', text)
        self.assertLess(text.index('Earlier context'), text.index('Highest match'))
        self.assertLess(text.index('Highest match'), text.index('Lower match'))
        for forbidden in ['RAW SECRET', 'INTERNAL', 'sB1', 'matched_fact_ids']:
            self.assertNotIn(forbidden, text)

    def test_chronological_blocks_and_notes_for_flagged_questions(self):
        dates = {'A': 'April-10-2024', 'B': '2024/03/03 (Sun) 09:00', 'C': None, 'D': '1:56 pm on 8 May, 2023'}
        sources = [{'id': 's' + k, 'session_id': k, 'position': n, 'message': {'session_date': d} if d else {}}
                   for n, (k, d) in enumerate(dates.items())]
        result = {'matched_fact_ids': ['a', 'b', 'c', 'd'],
                  'evidence': [{'id': k.lower(), 'text': 'Fact ' + k, 'source_ids': ['s' + k + '.1']} for k in dates]}
        blocks = chronological(session_fact_blocks(result, sources), sources)
        # Readable dates in any of the stored formats first, earliest first; undated sessions last.
        self.assertEqual([b['session_id'] for b in blocks], ['D', 'B', 'A', 'C'])
        text = render_blocks(blocks, question_date='2024-05-01', notes=['Sessions below are in chronological order.'])
        self.assertTrue(text.startswith('Sessions below are in chronological order.\n\nQuestion date: 2024-05-01'))

    def test_unknown_date_is_not_invented_and_raw_units_are_rejected(self):
        source = {'id': 's', 'session_id': 'A', 'position': 0, 'message': {}}
        result = {'evidence': [{'id': 'f', 'text': 'A fact', 'source_ids': ['s.1']}]}
        text = render_blocks(session_fact_blocks(result, [source]))
        self.assertIn('not recorded', text)
        with self.assertRaises(ValueError):
            session_fact_blocks({'evidence': [{'id': 's.1', 'text': 'Raw'}]}, [source])

    def test_fact_lines_carry_source_date_and_resolved_relative_time(self):
        source = {'id': 's', 'session_id': 'A', 'position': 0,
                  'message': {'session_date': '1:56 pm on 20 January, 2023'}}
        result = {'evidence': [{'id': 'f', 'text': 'Jon lost his job yesterday', 'source_ids': ['s.1']}]}
        text = render_blocks(session_fact_blocks(result, [source]))
        self.assertIn('- [20 January 2023] Jon lost his job yesterday (19 January 2023)', text)
        self.assertIn('Source conversation date: 1:56 pm on 20 January, 2023', text)


class RelativeDateTests(unittest.TestCase):
    def test_resolves_common_expressions_and_keeps_wording(self):
        from text_memory.collections.relative_dates import annotate, date_label
        self.assertEqual(annotate('went last week', '2023-06-15')[0], 'went last week (week of 5 June 2023)')
        self.assertEqual(annotate('two months ago', '15 June, 2023')[0], 'two months ago (April 2023)')
        self.assertEqual(annotate('last summer', '15 June, 2023')[0], 'last summer (summer 2022)')
        self.assertEqual(annotate('no time words', '15 June, 2023'), ('no time words', []))
        self.assertEqual(annotate('yesterday', 'unknown'), ('yesterday', []))
        self.assertEqual(date_label('2026-01-01T00:00:00+00:00'), '1 January 2026')
        self.assertEqual(date_label('sometime'), 'sometime')
