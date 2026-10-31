"""Choose reproducible development and untouched holdout IDs without answers."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from text_memory.datasets import sample_longmemeval, split_locomo
from text_memory.evaluation import write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--longmemeval", default="data/public/longmemeval/longmemeval_s_cleaned.json")
    parser.add_argument("--locomo", default="data/public/locomo-refined/data/raw/locomo_refined.json")
    parser.add_argument("--mandatory-ids", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--dev-per-type", required=True, type=int)
    parser.add_argument("--holdout-per-type", required=True, type=int)
    parser.add_argument("--unanswerable", required=True, type=int)
    parser.add_argument("--locomo-dev-conversations", required=True, type=int)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        parser.error("抽样目录已有内容；请使用空目录")
    reports = {
        "longmemeval": sample_longmemeval(json.loads(Path(args.longmemeval).read_text()),
            set(Path(args.mandatory_ids).read_text().split()), seed=args.seed, dev_per_type=args.dev_per_type,
            holdout_per_type=args.holdout_per_type, unanswerable=args.unanswerable),
        "locomo-refined": split_locomo(json.loads(Path(args.locomo).read_text()), seed=args.seed,
                                       dev_conversations=args.locomo_dev_conversations)}
    for dataset, report in reports.items():
        directory = output / dataset
        directory.mkdir(parents=True)
        for split in ("dev", "holdout"):
            (directory / f"ids-{split}.txt").write_text("\n".join(report[split]) + "\n")
        write_json(directory / "sampling.json", report)
    (output / "README.md").write_text(
        "# 评测题集\n\n固定种子：" + str(args.seed) + "。按题号排序后用 Python random.Random 抽样。\n\n"
        "LongMemEval 保留原题，每题型补至开发集指定数量；先补无答案题，再补有答案题，最后取留出题。"
        "同题的 _abs 与非 _abs 版本也不跨集。LoCoMo 按完整对话划分，每段全部问题可用。"
        "精确数量与 id 见各目录 sampling.json。抽样只用 id 和题型，不看答案或分数。\n\n"
        "留出集只准备 histories/queries，不运行、不查看答案或错题；不生成 labels。"
        "仅在以后一次性验收时重新准备标签并评分。该规则为工作流程约束，不是操作系统访问隔离。\n")
    print(json.dumps(reports, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
