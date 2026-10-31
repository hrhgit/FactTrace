"""Separate public LongMemEval histories, questions and grading labels."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from text_memory.datasets import prepare_longmemeval
from text_memory.errors import MemoryErrorBase
from text_memory.evaluation import write_json
from text_memory.schema import require


def preserve_empty_messages(rows: list[dict], ids: set[str] | None, *, policy: str) -> tuple[list[dict], list[dict]]:
    """Represent an empty source turn explicitly without dropping its position.

    This input-format adapter never invents dialogue or changes the memory
    writer. The original file stays unchanged; the report records its hash and
    the exact positions of represented empty turns.
    """
    adapted, empty = [], []
    for row in rows:
        if ids is not None and str(row["question_id"]) not in ids:
            adapted.append(row)
            continue
        sessions = []
        for sid, messages in zip(row["haystack_session_ids"], row["haystack_sessions"]):
            session = []
            for position, message in enumerate(messages):
                content = message.get("content")
                require(isinstance(content, str), "原消息 content 不是文字")
                if not content.strip():
                    require(policy == "preserve", "原数据含空消息；显式 --empty-messages preserve 可保留空轮次和位置")
                    empty.append({"id": str(row["question_id"]), "session_id": str(sid), "position": position})
                    message = message | {"content": "[Empty message in the original conversation.]"}
                session.append(message)
            sessions.append(session)
        adapted.append(row | {"haystack_sessions": sessions})
    return adapted, empty


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/public/longmemeval/longmemeval_s_cleaned.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--ids", nargs="+", help="显式选择题目；不指定时处理全部题目")
    parser.add_argument("--ids-file", help="每行一个题号，与 --ids 互斥")
    parser.add_argument("--inputs-only", action="store_true", help="留出集：不读取答案标注，不生成 labels")
    parser.add_argument("--empty-messages", choices=["error", "preserve"], default="error",
                        help="preserve 将原始空消息显式标为空轮次并保留位置；默认报错，不静默删除")
    parser.add_argument("--filler", type=int, help="只保留证据会话加 N 个按固定哈希选出的其他会话（保持原顺序），用于写入实验")
    args = parser.parse_args()
    if args.ids and args.ids_file:
        parser.error("--ids 与 --ids-file 不能同时使用")
    try:
        ids = set(Path(args.ids_file).read_text().split()) if args.ids_file else set(args.ids) if args.ids else None
        raw = Path(args.input).read_bytes()
        rows, empty = preserve_empty_messages(json.loads(raw), ids, policy=args.empty_messages)
        result = prepare_longmemeval(rows, Path(args.output),
                                     ids=ids, filler=args.filler, include_labels=not args.inputs_only)
        result.update(source_file_sha256=hashlib.sha256(raw).hexdigest(), empty_message_policy=args.empty_messages,
                      empty_message_locations=empty)
        write_json(Path(args.output) / "preparation.json", result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (MemoryErrorBase, OSError, ValueError, KeyError) as error:
        print(str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
