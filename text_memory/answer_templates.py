"""Answer-only adapters for the pinned AML pipelines; no copied templates."""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys

from .schema import require

ROOT = Path(__file__).resolve().parents[1]
PIPELINES = {"longmemeval": "longmemeval-s", "locomo-refined": "locomo-refined"}


def load_pipeline(dataset: str):
    require(dataset in PIPELINES, "官方答题只支持 " + "、".join(PIPELINES))
    return load_pipeline_file("vendor/aml/data/" + PIPELINES[dataset] + "/pipeline.py")


def load_pipeline_file(relative: str):
    """Import one pinned AML pipeline by its path under the project root."""
    path = ROOT / relative
    require(path.is_file() and path.is_relative_to(ROOT / "vendor/aml/data"), "不是官方 pipeline 文件：" + relative)
    name = "aml_" + "_".join(path.relative_to(ROOT / "vendor/aml/data").with_suffix("").parts).replace("-", "_")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    # The upstream module imports api_config and changes sys.path. Neither
    # importing it nor rendering a prompt should write into the vendor tree.
    previous_path, previous_bytecode = sys.path[:], sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = previous_path
        sys.dont_write_bytecode = previous_bytecode
    return module


def configuration(template: str, dataset: str) -> dict:
    """Manifest entry for the pinned official answer template (the only template kept)."""
    require(template == "official", "answer_template 只能是 official")
    require(dataset in PIPELINES, "题库必须是 " + "、".join(PIPELINES))
    prompt = load_pipeline(dataset).OPEN_ENDED_ANSWER_TEMPLATE
    sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    path = ROOT / "vendor/aml/data" / PIPELINES[dataset] / "pipeline.py"
    return {"answer_template": template, "answer_template_sha256": sha, "answer_prompt_sha256": sha,
            "dataset": dataset, "answer_parameters": {"temperature": 0},
            "official_pipeline": {"path": str(path.relative_to(ROOT)),
                                  "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}}


def official_messages(question: str, context: str, dataset: str) -> list[dict]:
    # context is the ordered concatenation of AML Search data[*].content.
    # Use the shared-evidence fallback; never label mixed evidence as belonging
    # to one speaker, or turn local sufficiency judgements into fake memories.
    prompt = load_pipeline(dataset).render_answer_prompt({"question": question, "retrieved_context": context})
    return [{"role": "user", "content": prompt}]
