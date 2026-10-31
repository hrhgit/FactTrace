"""Answer and grade a mixed local set with each benchmark's pinned AML pipeline.

Answer prompts and grading parsers come from vendor/aml/data; this module only
routes a question to its pipeline and makes the calls through the project model
client. Search output enters as one context string per question. Labels are
used only when grading, never in answer prompts.
"""
from __future__ import annotations

import json

from .answer_templates import load_pipeline_file
from .schema import require

PIPELINES = {
    "locomo-refined": "vendor/aml/data/locomo-refined/pipeline.py",
    "longmemeval": "vendor/aml/data/longmemeval-s/pipeline.py",
    "beam": "vendor/aml/data/beam/pipeline.py",
    "clbench": "vendor/aml/data/clbench/pipeline.py",
    "personamem-v1": "vendor/aml/data/personamem/pipeline_v1.py",
    "personamem-v2": "vendor/aml/data/personamem/pipeline_v2.py",
}
_modules: dict[str, object] = {}


def pipeline(dataset: str):
    require(dataset in PIPELINES, "未知题库：" + dataset)
    if dataset not in _modules:
        _modules[dataset] = load_pipeline_file(PIPELINES[dataset])
    return _modules[dataset]


def answer_messages(task: dict, question: str, context: str, model: str) -> list[dict]:
    """The official answer request, with Search output in each pipeline's memory slot."""
    dataset = task["dataset"]
    module = pipeline(dataset)
    if dataset in {"locomo-refined", "longmemeval"}:
        # Shared-evidence fallback; mixed evidence is not attributed to one speaker.
        return [{"role": "user", "content": module.render_answer_prompt({"question": question, "retrieved_context": context})}]
    if dataset == "beam":
        return [{"role": "user", "content": module.render_answer_prompt({"question": question, "context": context})}]
    if dataset == "clbench":
        # Empty memories render as "(no memories)", as in the official template.
        return [{"role": "user", "content": module.render_answer_prompt_clbench(
            system_prompt=task["system_prompt"], memories=context, question=question)}]
    # PersonaMem pipelines read the full history as chat messages. Search output
    # takes that place as one user message, as in the 2026-10-28 PersonaMem pilot.
    history = [{"role": "user", "content": context}] if context else []
    if dataset == "personamem-v1":
        return module.official_messages({"question": question, "all_options": task["all_options"],
                                         "context_messages": history}, model)
    # official_mcq_messages shuffles options with Python hash(), which differs
    # per process; the set fixes one order, rendered with the official template.
    messages = module.official_generative_messages({"user_query": question, "chat_history": history})
    options = "\n".join(f"{chr(65 + i)}. {option}" for i, option in enumerate(task["options"]))
    return messages + [{"role": "system", "content": module.MCQ_PROMPT_TEMPLATE.format(options=options)}]


def _judge(judge, prompt: str, *, system: str | None = None) -> str:
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    return judge.complete_text(messages, purpose="judge", temperature=0)


def grade(task: dict, question: str, label: dict, answer: str, judge) -> dict:
    """{"score": 0..1, ...details}; parse failures raise instead of scoring zero."""
    dataset = task["dataset"]
    module = pipeline(dataset)
    if dataset in {"locomo-refined", "longmemeval"}:
        response = _judge(judge, module.render_accuracy_prompt(label | {"question": question}, answer))
        correct = module.parse_judge_label(response) == "CORRECT"
        return {"score": float(correct), "correct": correct, "judge_response": response}
    if dataset == "beam":
        rubrics = module.rubric_items(label)
        response = _judge(judge, module.render_batch_judge_prompt(question, answer, rubrics))
        scores = module.parse_rubric_scores(response, len(rubrics))
        result = {"score": sum(s["score"] for s in scores) / len(scores),
                  "rubric_scores": [{"rubric": r, "score": s["score"], "reason": s["reason"]} for r, s in zip(rubrics, scores)],
                  "judge_response": response}
        if label["question_type"] == "event_ordering":
            # Same greedy alignment as align_with_llm, then the official metrics.
            used, system = set(), []
            for candidate in answer.split("\n"):
                match = None
                for index, expected in enumerate(rubrics):
                    if index in used:
                        continue
                    reply = _judge(judge, f"First snippet: {expected}\n\nSecond snippet: {candidate}",
                                   system=module.EQUIVALENCE_SYSTEM_PROMPT)
                    if "yes" in reply.casefold():
                        match = index
                        break
                system.append(candidate if match is None else rubrics[match])
                if match is not None:
                    used.add(match)
            result["event_ordering"] = module.event_ordering_metrics(rubrics, system)
        return result
    if dataset == "clbench":
        normalized = module._normalize_model_output(answer)
        rubrics = module._normalize_rubrics(label["rubrics"])
        require(bool(rubrics), "CL-Bench 题目没有评分要求")
        if not normalized:
            # The official grader counts an empty answer as 0 without a judge call.
            return {"score": 0.0, "requirement_ratio": 0.0, "rationale": "No model output (counted as score 0)"}
        response = _judge(judge, module.rubric_judge_prompt(rubrics_text=module._build_rubrics_text(rubrics),
                                                            model_output=normalized))
        text = response.strip()
        for prefix in ("```json", "```"):
            text = text.removeprefix(prefix)
        payload = json.loads(text.removesuffix("```").strip())
        require(isinstance(payload, dict) and "Overall Score" in payload, "CL-Bench 评分结果缺少 Overall Score")
        statuses = module._coerce_status_list(payload.get("List of Requirement Satisfaction Status", []))
        return {"score": float(module._coerce_score(payload)), "requirement_status": statuses,
                "requirement_ratio": module._compute_requirement_ratio(statuses),
                "rationale": str(payload.get("Grading Rationale", "") or ""), "judge_response": response}
    if dataset == "personamem-v1":
        correct, predicted = module.official_extract_answer(answer, label["correct_answer"])
        return {"score": float(correct), "correct": correct, "predicted": predicted,
                "gold_option": module.extract_gold_option(label["correct_answer"])}
    letter = module.extract_final_letter(answer)
    correct = letter == label["correct_letter"]
    return {"score": float(correct), "correct": correct, "predicted_letter": letter, "gold_letter": label["correct_letter"]}
