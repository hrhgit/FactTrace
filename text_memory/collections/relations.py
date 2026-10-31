"""Write-time relation notes between a user's facts: an independent stage after each committed Add.

Validated in outputs/relation-new-20261030 (51 aggregation questions no earlier relation experiment used:
20.81 -> 25.33 over three same-window runs). Only user-side facts take part; a fact taken only from assistant
messages is neither labeled nor a candidate. Each new fact is paired with its NEIGHBORS most similar earlier
user-side facts, and every pair is judged on its own with both facts' source messages. A program rule turns
"same" into "another" when the two facts' event dates do not overlap. A statement that something never / no
longer happened is checked against its NEGATION_NEIGHBORS nearest user-side facts for records that show it false.
Candidates come only from the revision of the request being labeled, so a resumed or backfilled run sees what a
live run saw. Counts and order are never written by the model; answer material states each pair on its own.
"""
import re
from datetime import datetime, timedelta

from ..model import encode
from ..schema import digest, require
from ..store import Scope
from . import prompts
from . import contracts
from .answer_context import session_day, source_date
from .relative_dates import annotate

RELATIONS = ('same', 'another', 'replaces', 'conflicts', 'outcome')
PAIRS_PURPOSE, NEGATION_PURPOSE = 'collection_relation_pairs', 'collection_relation_negation'
PROMPT_HASH = digest([prompts.RELATION_PAIRS, prompts.RELATION_NEGATION])
NEIGHBORS, BATCH, NEGATION_NEIGHBORS, SNIPPET = 5, 20, 30, 600
NEGATION = re.compile(r"\bnever\b|\bno longer\b|\b(?:has|have|had|did|does|do)(?:n't| not) (?:ever |yet |previously )?\w+", re.I)
# A recent past event with no resolvable date is placed in the 10 days before its conversation.
RECENT = re.compile(r"\b(?:recent|recently|just|a few days|yesterday|this week|tough week|last (?:night|week|weekend)"
                    r"|on (?:mon|tues|wednes|thurs|fri|satur|sun)day)\b", re.I)
MONTHS = 'January|February|March|April|May|June|July|August|September|October|November|December'


def event_spans(text, session):
    """(start, end) dates a fact talks about: program-resolved relative dates and explicit dates in the text.
    Empty when the fact names no time; the conversation date alone is not an event date."""
    d0 = session_day(session)
    if d0 is None:
        return []
    annotated, _ = annotate(text, d0.strftime('%d %B %Y'))
    spans = []
    for m in re.finditer(rf'(?:(\d{{1,2}}) ({MONTHS})|({MONTHS}) (\d{{1,2}})(?!\d)(?:st|nd|rd|th)?)(?:,? (\d{{4}}))?', annotated):
        day, month = (m[1], m[2]) if m[1] else (m[4], m[3])
        try:
            d = datetime.strptime(f'{day} {month} {m[5] or d0.year}', '%d %B %Y').date()
        except ValueError:
            continue
        width = 6 if 'week' in annotated[max(0, m.start() - 12):m.start()] else 2
        spans.append((d - timedelta(days=1), d + timedelta(days=width)))
    for m in re.finditer(rf'\((?:around |month of )?({MONTHS}) (\d{{4}})\)', annotated):
        start = datetime.strptime(f'1 {m[1]} {m[2]}', '%d %B %Y').date()
        spans.append((start, (start.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)))
    if not spans and RECENT.search(text):
        spans.append((d0 - timedelta(days=10), d0))
    return spans


def dates_differ(a, b):
    return bool(a and b) and not any(s1 <= e2 and s2 <= e1 for s1, e1 in a for s2, e2 in b)


def shared_messages(records):
    """Each distinct message once under an id; the records point to it."""
    messages, rows = {}, []
    for record in records:
        rows.append({**record, 'messages': [messages.setdefault(m, f'm{len(messages)}') for m in record['messages']]})
    return {name: m for m, name in messages.items()}, rows


def unit(vector):
    norm = sum(x * x for x in vector) ** .5 or 1.0
    return [x / norm for x in vector]


class RelationAnnotator:
    """The relation stage of one committed request."""

    def __init__(self, engine):
        self.engine, self.store = engine, engine.store

    def run(self, scope, receipt):
        store = self.store
        receipts = sorted((r for r in store.request_receipts(scope) if r['revision'] <= receipt['revision']),
                          key=lambda r: r['revision'])
        view = Scope(scope.user_id, scope.scope_id, receipt['revision'])
        facts = {p['id']: p for p in store.records(view, 'point')}
        sources = {s['id']: s for s in store.records(view, 'source')}
        order = [i for i in dict.fromkeys(i for r in receipts for i in r['point_ids']) if i in facts]
        messages = {i: [sources[s]['message'] for s in dict.fromkeys(ref.rsplit('.', 1)[0] for ref in facts[i]['evidence'])]
                    for i in order}
        users = [i for i in order if any(m['role'] != 'assistant' for m in messages[i])]
        new = [i for i in users if i in set(receipt['point_ids'])]
        links = []
        if new:
            vectors = dict(zip(users, map(unit, self.engine.searcher._vectors([facts[i]['text'] for i in users]))))
            cos = lambda a, b: sum(x * y for x, y in zip(vectors[a], vectors[b]))
            dates = {i: source_date(messages[i][0]) for i in users}
            record = lambda i: {'date': dates[i], 'text': facts[i]['text'], 'messages': [
                f"{m.get('speaker') or m['role']}: {m['content'][:SNIPPET]}" for m in messages[i] if m['role'] != 'assistant'][:2]}
            position = {i: n for n, i in enumerate(users)}
            pairs = [(j, i) for i in new for j in sorted(users[:position[i]], key=lambda j: -cos(i, j))[:NEIGHBORS]]
            for start in range(0, len(pairs), BATCH):
                for earlier, later, relation in self.judge(pairs[start:start + BATCH], record):
                    if relation == 'same' and dates_differ(event_spans(facts[earlier]['text'], dates[earlier]),
                                                           event_spans(facts[later]['text'], dates[later])):
                        relation = 'another'
                    links.append({'earlier': earlier, 'later': later, 'relation': relation})
            fresh = set(new)
            for i in users:
                if not NEGATION.search(facts[i]['text']):
                    continue
                near = sorted((j for j in users if j != i), key=lambda j: -cos(i, j))[:NEGATION_NEIGHBORS]
                checked = near if i in fresh else [j for j in near if j in fresh]
                for j in self.negation(i, checked, record) if checked else ():
                    earlier, later = sorted((i, j), key=position.get)
                    links.append({'earlier': earlier, 'later': later, 'relation': 'conflicts'})
        store.save_relations(scope, receipt['request_id'], links, prompt_hash=PROMPT_HASH)
        self.engine.trace.append({'tool': 'fact_relations', 'result': {
            'new_user_facts': len(new), 'links': len(links)}})
        return links

    def judge(self, batch, record):
        ids = [f'p{n}' for n in range(len(batch))]
        def validate(reply):
            require(isinstance(reply, dict) and isinstance(reply.get('pairs'), list), 'pairs 必须是数组')
            answered = {}
            for item in reply['pairs']:
                require(isinstance(item, dict) and item.get('id') in ids and item['id'] not in answered,
                        '每个配对编号必须出现且只出现一次')
                require(item.get('relation') in (*RELATIONS, 'none'), '关系必须是 ' + ', '.join((*RELATIONS, 'none')))
                answered[item['id']] = item['relation']
            require(len(answered) == len(ids), '每个配对都需要一个关系')
        facts = list(dict.fromkeys(i for pair in batch for i in pair))
        names = {i: f'r{n}' for n, i in enumerate(facts)}
        messages, rows = shared_messages(map(record, facts))
        reply = self.engine._complete([{'role': 'system', 'content': prompts.RELATION_PAIRS}, {'role': 'user', 'content': encode(
            {'instructions': contracts.relation_pairs(ids),
             'messages': messages, 'records': dict(zip(names.values(), rows)),
             'pairs': [{'id': p, 'A': names[a], 'B': names[b]} for p, (a, b) in zip(ids, batch)]})}],
            PAIRS_PURPOSE, validate=validate)
        got = {item['id']: item['relation'] for item in reply['pairs']}
        return [(a, b, got[p]) for p, (a, b) in zip(ids, batch) if got[p] != 'none']

    def negation(self, statement, near, record):
        ids = [f'r{n}' for n in range(len(near))]
        def validate(reply):
            require(isinstance(reply, dict) and isinstance(reply.get('conflicts'), list)
                    and all(x in ids for x in reply['conflicts']), 'conflicts 必须是给定记录编号的数组')
        messages, rows = shared_messages(map(record, [statement, *near]))
        reply = self.engine._complete([{'role': 'system', 'content': prompts.RELATION_NEGATION}, {'role': 'user', 'content': encode(
            {'instructions': contracts.relation_negation(ids),
             'messages': messages, 'S': rows[0], 'records': [{'id': r, **row} for r, row in zip(ids, rows[1:])]})}],
            NEGATION_PURPOSE, validate=validate)
        return [near[ids.index(x)] for x in dict.fromkeys(reply['conflicts'])]
