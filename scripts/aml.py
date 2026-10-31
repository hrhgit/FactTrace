"""Load .env and launch the pinned official pipeline with its recorded compatibility patch."""
from pathlib import Path
import os
import subprocess
import sys

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
PIPELINES = {
    "personamem-v1": "personamem/pipeline_v1.py",
    "personamem-v2": "personamem/pipeline_v2.py",
    "locomo-refined": "locomo-refined/pipeline.py",
    "clbench": "clbench/pipeline.py",
    "beam": "beam/pipeline.py",
    "longmemeval": "longmemeval-s/pipeline.py",
    "scriptmem": "scriptmem/pipeline.py",
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in PIPELINES:
        print("Usage: python scripts/aml.py DATASET [official pipeline arguments]")
        print("Datasets: " + ", ".join(PIPELINES))
        return 0 if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"} else 2
    load_dotenv(ROOT / ".env", override=False)
    dataset, *args = sys.argv[1:]
    if args and not any(a in args for a in ("-h", "--help")):
        stage = args[0]
        needs_answer = stage == "answer"
        needs_judge = stage == "evaluate" and dataset not in {"personamem-v1", "personamem-v2", "scriptmem"}
        needs_judge |= stage == "evaluate-narrow"
        prefix = "ANSWER" if needs_answer else "JUDGE" if needs_judge else None
        if prefix:
            missing = [f"{prefix}_{k}" for k in ("API_BASE", "API_KEY", "MODEL") if not os.environ.get(f"{prefix}_{k}")]
            if missing:
                print("请先在 .env 填写：" + ", ".join(missing), file=sys.stderr)
                return 2
    script = ROOT / "vendor/aml/data" / PIPELINES[dataset]
    return subprocess.call([sys.executable, str(script), *args], cwd=ROOT, env=os.environ | {"PYTHONDONTWRITEBYTECODE": "1"})


if __name__ == "__main__":
    raise SystemExit(main())
