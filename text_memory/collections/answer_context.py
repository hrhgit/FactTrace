"""Facts-only answer presentation; retrieval and completion remain unchanged."""
import json
from collections import defaultdict
from datetime import date, datetime, timezone

from .relative_dates import annotate, date_label, parse_session

POLICY = "ranked-session-facts-dated-v5-relations"
RELATION_HEAD = ("Notes starting with \"Link:\" describe how records relate: which describe the same event or thing, "
                 "which are separate occurrences, which later statements replace earlier ones, whether a plan's outcome "
                 "is recorded, and which statements conflict.")
# At most this many notes under one fact line, kept in this order; each pair is stated on its own.
MAX_RELATION_NOTES = 2
RELATION_PRIORITY = ('conflicts', 'replaces', 'outcome', 'another', 'same')


def source_date(message):
    if message.get('session_date'):
        return message['session_date']
    timestamp = message.get('timestamp')
    if type(timestamp) in (int, float):
        try:
            return datetime.fromtimestamp(timestamp / 1000, timezone.utc).isoformat()
        except (ValueError, OverflowError, OSError):
            return None
    return timestamp if isinstance(timestamp, str) and timestamp.strip() else None


def session_fact_blocks(result, sources, fact_order=()):
    """Return auditable blocks, ordered by best selected-fact rank per session.

    matched_fact_ids must be in descending retrieval-score order. A fact that
    cites several sessions is shown once, in the highest-ranked applicable
    block, with all of its source dates retained. IDs stay in these audit
    blocks, never in the rendered answer text.
    """
    sources = {s['id']: s for s in sources}
    facts = {f['id']: f for f in result['evidence']}
    if any('source_ids' not in f for f in facts.values()):
        raise ValueError('Answer context requires stored facts, not source paragraphs')
    hits = result.get('matched_fact_ids', list(facts))
    if not set(hits) <= facts.keys():
        raise ValueError('Selected facts are missing from answer evidence')
    order = {ident: n for n, ident in enumerate(fact_order)}
    fallback_order = {ident: n for n, ident in enumerate(facts)}
    members, refs = defaultdict(list), {}
    for ident, fact in facts.items():
        refs[ident] = [sources[ref.rsplit('.', 1)[0]] for ref in fact['source_ids']]
        if not refs[ident]:
            raise ValueError('Fact has no source conversation')
        for sid in dict.fromkeys(s['session_id'] for s in refs[ident]):
            members[sid].append(ident)
    rank = {ident: n for n, ident in enumerate(hits)}
    def priority(sid):
        return min((rank[i] for i in members[sid] if i in rank), default=len(hits))
    # Insertion order breaks equal scores without exposing hashed IDs.
    sessions = sorted(members, key=priority)
    seen, blocks = set(), []
    for sid in sessions:
        ids = [i for i in members[sid] if i not in seen]
        ids.sort(key=lambda i: (
            min(s['position'] for s in refs[i] if s['session_id'] == sid),
            order.get(i, len(order) + fallback_order[i])))
        if not ids:
            continue
        seen.update(ids)
        dates = list(dict.fromkeys(source_date(s['message'])
                    for s in sources.values() if s['session_id'] == sid
                    and source_date(s['message'])))
        blocks.append({'session_id': sid, 'best_match_rank': priority(sid) + 1,
                       'dates': dates, 'facts': [
                           {'id': i, 'text': facts[i]['text'],
                            'source_dates': list(dict.fromkeys(
                                source_date(s['message']) for s in refs[i]
                                if source_date(s['message'])))} for i in ids]})
    return blocks


def session_day(value):
    """Calendar date of a source date for chronological ordering; None when it cannot be read."""
    found = parse_session(value)
    if found:
        return found
    head = (value or '').split(' ')[0]
    for form in ('%B-%d-%Y', '%Y/%m/%d'):
        try:
            return datetime.strptime(head, form).date()
        except ValueError:
            pass
    return None


def chronological(blocks, sources):
    """Session blocks earliest first: by readable session date, then by the order the sessions were written."""
    first = {}
    for n, source in enumerate(sources):
        first.setdefault(source['session_id'], n)
    def key(block):
        # Sessions without a readable date go last.
        days = [d for d in map(session_day, block['dates']) if d]
        return min(days, default=date.max), first.get(block['session_id'], len(first))
    return sorted(blocks, key=key)


def relation_notes(store, scope, sources, order, shown):
    """Write-time relation links touching the shown facts, as note texts per shown fact id."""
    from .relations import PROMPT_HASH
    links = [l for l in store.fact_relations(scope, prompt_hash=PROMPT_HASH) if {l['earlier'], l['later']} & shown]
    if not links:
        return {}
    by_source, position = {s['id']: s for s in sources}, {i: n for n, i in enumerate(order)}
    facts = {}
    def fact(ident):
        if ident not in facts:
            point = store.maybe_get(scope, 'point', ident)
            when = point and source_date(by_source[point['evidence'][0].rsplit('.', 1)[0]]['message'])
            facts[ident] = point and {'text': point['text'], 'date': date_label(when) if when else 'date not recorded'}
        return facts[ident]
    def cite(ident):
        text = fact(ident)['text']
        return f"{fact(ident)['date']}: \"{text[:100]}{'…' if len(text) > 100 else ''}\""
    phrases = {'conflicts': ('Conflicts with the record of ', 'Conflicts with the record of '),
               'replaces': ('Replaced later by the record of ', 'Replaces the earlier record of '),
               'outcome': ('A plan; its outcome is recorded in the record of ', 'Reports the outcome of the plan in the record of '),
               'another': ('A separate occurrence of the same kind as the record of ',) * 2,
               'same': ('Same event or thing as the record of ',) * 2}
    notes, seen = {}, set()
    for link in links:
        a, b, relation = link['earlier'], link['later'], link['relation']
        if a == b or (a, b, relation) in seen or not fact(a) or not fact(b):
            continue
        seen.add((a, b, relation))
        if position.get(a, -1) > position.get(b, -1):
            a, b = b, a
        rank = RELATION_PRIORITY.index(relation)
        notes.setdefault(a, []).append((rank, phrases[relation][0] + cite(b)))
        notes.setdefault(b, []).append((rank, phrases[relation][1] + cite(a)))
    return {i: [t for _, t in sorted(v)[:MAX_RELATION_NOTES]] for i, v in notes.items() if i in shown}


def render_blocks(blocks, *, question_date=None, notes=(), links=None):
    # Notes describe how the material is arranged; they come first, before the question date.
    parts = ['\n'.join(notes)] if notes else []
    parts += [f'Question date: {question_date}'] if question_date else []
    for block in blocks:
        dates = block['dates']
        header = 'Source conversation date: ' + ('; '.join(dates) if dates else 'not recorded')
        lines = [header]
        for fact in block['facts']:
            # Each line carries its source date; relative time expressions get a program-resolved absolute date.
            when = (fact['source_dates'] or dates or [None])[0]
            body = annotate(fact['text'], when)[0] if when else fact['text']
            line = f"- [{date_label(when) if when else 'date not recorded'}] {body}"
            if set(fact['source_dates']) - set(dates):
                line += ' (Source conversation dates: ' + '; '.join(fact['source_dates']) + ')'
            lines.append(line)
            lines.extend('    Link: ' + note for note in (links or {}).get(fact['id'], ()))
        parts.append('\n'.join(lines))
    return '\n\n'.join(parts)


def render_fact_context(store, scope, result):
    scope = store.snapshot(scope)
    receipts = [json.loads(r[0]) for r in store.db.execute(
        'SELECT result FROM requests WHERE user_id=? AND scope_id=?',
        (scope.user_id, scope.scope_id))]
    receipts = sorted((r for r in receipts if r['revision'] <= scope.as_of),
                      key=lambda r: r['revision'])
    order = list(dict.fromkeys(i for r in receipts for i in r['point_ids']))
    sources = list(store.records(scope, 'source'))
    blocks = session_fact_blocks(result, sources, order)
    if result.get('chronological'):
        blocks = chronological(blocks, sources)
    notes = tuple(result.get('notes', ()))
    links = relation_notes(store, scope, sources, order, {f['id'] for b in blocks for f in b['facts']})
    if links:
        notes = (RELATION_HEAD, *notes)
    return render_blocks(blocks, question_date=result.get('question_date'), notes=notes, links=links)
