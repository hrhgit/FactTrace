"""Offline integrity and reference checks; never calls a model or scores answers."""
from pathlib import Path
import argparse
import ast
import csv
import hashlib
import json
import subprocess
import sys

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "data/public"


def jsonl(path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def csv_rows(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def inventory():
    result = {}
    for size in ("32k", "128k", "1M"):
        base = PUBLIC / "personamem-v1"
        contexts = {}
        for row in jsonl(base / f"shared_contexts_{size}.jsonl"):
            contexts.update(row)
        rows = csv_rows(base / f"questions_{size}.csv")
        for row in rows:
            history = contexts[row["shared_context_id"]]
            assert isinstance(history, list) and history
            end = int(row["end_index_in_shared_context"])
            # Upstream explicitly uses Python context[:end]; -1 is valid.
            assert -len(history) <= end <= len(history), (size, row["question_id"], end, len(history))
        result[f"personamem-v1-{size}"] = {"questions": len(rows), "shared_contexts": len(contexts)}
    base = PUBLIC / "personamem-v2"
    result["personamem-v2"] = {}
    for split in ("benchmark", "train", "val"):
        rows = csv_rows(base / f"benchmark/text/{split}.csv")
        entry = {"questions": len(rows)}
        for size in ("32k", "128k"):
            paths = {row[f"chat_history_{size}_link"] for row in rows}
            for path in paths:
                history = json.loads((base / path).read_text())["chat_history"]
                assert isinstance(history, list) and history, path
            entry[f"contexts_{size}"] = len(paths)
        result["personamem-v2"][split] = entry
    for name in ("locomo-refined", "scriptmem"):
        base = PUBLIC / name / "data/public"
        questions = list(jsonl(base / "questions.jsonl"))
        conversations = list(jsonl(base / "conversations.jsonl"))
        manifest = json.loads((base / "manifest.json").read_text())
        assert len(questions) == manifest["question_count"]
        assert len(conversations) == manifest["conversation_count"]
        key = "sample_id" if name == "locomo-refined" else "conversation_id"
        ids = {row[key] for row in conversations}
        assert all(row[key] in ids for row in questions)
        result[name] = {"questions": len(questions), "conversations": len(conversations)}
    result["clbench"] = {"questions": sum(1 for _ in jsonl(PUBLIC / "clbench/CL-bench.jsonl"))}
    longmem = json.loads((PUBLIC / "longmemeval/longmemeval_s_cleaned.json").read_text())
    for row in longmem:
        assert len(row["haystack_sessions"]) == len(row["haystack_dates"]) == len(row["haystack_session_ids"])
        assert set(row["answer_session_ids"]) <= set(row["haystack_session_ids"])
    result["longmemeval"] = {"questions": len(longmem)}
    result["beam"] = {}
    for file in sorted((PUBLIC / "beam/data").glob("*.parquet")):
        parquet = pq.ParquetFile(file)
        count = 0
        questions = 0
        for batch in parquet.iter_batches(batch_size=1):
            count += batch.num_rows
            for row in batch.to_pylist():
                probes = row["probing_questions"]
                if isinstance(probes, str):
                    probes = ast.literal_eval(probes)
                assert isinstance(probes, dict)
                for category in probes.values():
                    assert isinstance(category, list)
                    questions += len(category)
        assert count == parquet.metadata.num_rows
        result["beam"][file.name] = {"conversations": count, "questions": questions, "columns": parquet.schema_arrow.names}
    assert len(result["beam"]) == 5
    return result


def official_patch_status():
    patch = ROOT / "patches/aml-async-file-context.patch"
    command = ["git", "-C", str(ROOT / "vendor/aml"), "apply", "--check"]
    applied = subprocess.run(command + ["--reverse", str(patch)], capture_output=True).returncode == 0
    available = subprocess.run(command + [str(patch)], capture_output=True).returncode == 0
    return {"patch": str(patch.relative_to(ROOT)),
            "status": "applied" if applied else "not_applied" if available else "mismatch"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", action="store_true", help="Create a new local file hash manifest after successful checks")
    parser.add_argument("--patch-only", action="store_true", help="Only check the recorded official compatibility patch")
    args = parser.parse_args()
    patch_status = official_patch_status()
    if args.patch_only:
        print(json.dumps(patch_status, indent=2))
        return 0 if patch_status["status"] == "applied" else 1
    assert patch_status["status"] == "applied", patch_status
    lock = ROOT / "data/files.sha256.json"
    files = sorted(p for p in PUBLIC.rglob("*") if p.is_file() and ".cache" not in p.parts)
    for path in files:
        with path.open("rb") as handle:
            assert not handle.read(100).startswith(b"version https://git-lfs.github.com/spec/v1"), f"Unexpanded LFS pointer: {path}"
    result = inventory()
    actual = {str(p.relative_to(ROOT)): {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in files}
    if args.record:
        lock.write_text(json.dumps(actual, indent=2, ensure_ascii=False) + "\n")
    else:
        expected = json.loads(lock.read_text())
        assert actual == expected, "File list, size or SHA-256 mismatch; inspect source before replacing the manifest"
    for name in ("personamem-v1", "personamem-v2", "locomo-refined", "clbench", "beam", "longmemeval", "scriptmem"):
        subprocess.run([sys.executable, str(ROOT / "scripts/aml.py"), name, "--help"], check=True, stdout=subprocess.DEVNULL)
    report = {"status": "passed", "kind": "offline_data_and_environment_check_only", "files": len(files), "bytes": sum(v["bytes"] for v in actual.values()), "datasets": result, "official_pipeline_help_checks": 7, "official_patch": patch_status}
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports/setup-check.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    raise SystemExit(main())
