"""Prepare published PersonaMem v1 questions without future-history leakage."""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from text_memory.schema import normalize_add, require


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(source, output, *, contexts=2, seed=20261028):
    source, output = Path(source), Path(output)
    require(contexts > 0, 'contexts 必须大于零')
    with (source / 'questions_32k.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['shared_context_id']].append(row)
    require(contexts <= len(grouped), '抽样数量超过历史总数')
    chosen = sorted(grouped, key=lambda key: hashlib.sha256(f'{seed}:{key}'.encode()).hexdigest())[:contexts]
    histories, queries, labels, audit = [], [], [], []
    originals = {}
    with (source / 'shared_contexts_32k.jsonl').open() as stream:
        for line in stream:
            for key, messages in json.loads(line).items():
                if key in chosen:
                    require(key not in originals, '重复的共享历史')
                    originals[key] = messages
    for key in chosen:
        # One independently selected history per user. Use the latest published
        # question boundary, then all questions AT that boundary (never earlier).
        end = max(int(row['end_index_in_shared_context']) for row in grouped[key])
        selected = [row for row in grouped[key] if int(row['end_index_in_shared_context']) == end]
        original = originals[key]
        require(0 < end <= len(original), '历史边界超出范围')
        # The publisher explicitly defines context[:int(end_index)], excluding
        # the end item. In particular, do not add the following assistant turn.
        messages = original[:end]
        require(all(set(m) >= {'role', 'content'} for m in messages), '历史格式无效')
        uid = 'personamem-v1:' + key
        history = normalize_add({'user_id': uid, 'scope_id': 'history', 'session_id': 'published-prefix',
                                 'request_id': key + ':' + str(end), 'messages': messages})
        histories.append(history)
        for row in sorted(selected, key=lambda r: r['question_id']):
            options = ast.literal_eval(row['all_options'])
            require(isinstance(options, list) and all(isinstance(v, str) for v in options), '选项格式无效')
            queries.append({'id': row['question_id'], 'user_id': uid, 'scope_id': 'history',
                            'sample_id': key, 'question': row['user_question_or_message'],
                            'all_options': row['all_options'], 'question_type': row['question_type'],
                            'topic': row['topic']})
            labels.append({'id': row['question_id'], 'correct_answer': row['correct_answer']})
        audit.append({'shared_context_id': key, 'end_index_exclusive': end, 'messages': len(messages),
                      'original_messages': len(original), 'questions': len(selected),
                      'estimated_source_tokens_publisher': selected[0]['context_length_in_tokens']})
    require(len({q['id'] for q in queries}) == len(queries), '重复问题')
    require(all('correct_answer' not in q for q in queries), '答案不能进入问题输入')
    output.mkdir(parents=True, exist_ok=False)
    for name, data in [('histories', histories), ('queries', queries), ('labels', labels)]:
        directory = output / ('scoring' if name == 'labels' else 'inputs')
        directory.mkdir(exist_ok=True)
        (directory / (name + '.jsonl')).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in data))
    manifest = {'dataset': 'personamem-v1-32k', 'seed': seed, 'contexts': len(histories), 'questions': len(queries),
                'question_types': dict(Counter(q['question_type'] for q in queries)), 'prefixes': audit,
                'sampling': 'seeded distinct shared contexts; latest published boundary per context; all questions at that exact boundary; no selection by answers or scores',
                'source_sha256': {name: sha(source / name) for name in ['questions_32k.csv', 'shared_contexts_32k.jsonl']},
                'input_sha256': {str(p.relative_to(output)): sha(p) for p in sorted(output.rglob('*.jsonl'))},
                'scope': 'public upstream pilot, not the private AML suite or an overall accuracy estimate'}
    (output / 'preparation.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'data/public/personamem-v1')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--contexts', type=int, default=2)
    parser.add_argument('--seed', type=int, default=20261028)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.output, contexts=args.contexts, seed=args.seed), ensure_ascii=False))


if __name__ == '__main__':
    main()
