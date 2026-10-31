"""Prepare whole LoCoMo-Refined conversations with labels kept separate."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from text_memory.datasets import prepare_locomo
from text_memory.errors import MemoryErrorBase


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/public/locomo-refined/data/raw/locomo_refined.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--ids-file", required=True, help="每行一个对话 sample_id")
    parser.add_argument("--media-text", choices=["captions", "none"], required=True,
                        help="captions 保留原有图片说明和图片搜索文字；两者都不下载图片")
    parser.add_argument("--inputs-only", action="store_true", help="留出集：不读取答案标注，不生成 labels")
    args = parser.parse_args()
    try:
        result = prepare_locomo(json.loads(Path(args.input).read_text(encoding="utf-8")), Path(args.output),
                                ids=set(Path(args.ids_file).read_text().split()), media_text=args.media_text,
                                include_labels=not args.inputs_only)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (MemoryErrorBase, OSError, ValueError, KeyError) as error:
        print(str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
