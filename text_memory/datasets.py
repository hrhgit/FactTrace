"""Public-data preparation; labels never enter the memory input files."""
from __future__ import annotations

from copy import deepcopy
from collections import Counter
from pathlib import Path
import random
import re

from .errors import ConflictError
from .evaluation import append_jsonl, write_json
from .schema import digest, require


def prepare_longmemeval(rows: list[dict], output: Path, *, ids: set[str] | None = None, filler: int | None = None,
                       include_labels: bool = True) -> dict:
    """Preserve each supplied history session and its ordering without truncation.

    filler, when given, builds a reduced history for quick write experiments:
    every evidence session plus that many other sessions chosen by a fixed hash,
    all in their original order. Session contents are never shortened.
    """
    require(filler is None or (type(filler) is int and filler >= 0), "filler 必须是非负整数")
    require(include_labels or filler is None, "留出输入不能按答案位置缩减历史")
    require(isinstance(rows, list) and all(isinstance(row, dict) for row in rows), "公开题必须是对象数组")
    available = {str(row["question_id"]) for row in rows}
    require(ids is None or ids <= available, "指定的题目 id 不在题库中")
    selected = [row for row in rows if ids is None or str(row["question_id"]) in ids]
    require(bool(selected), "没有选择题目")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ConflictError("准备目录已有内容；请使用空目录")
    output.mkdir(parents=True, exist_ok=True)
    sessions = 0
    seen = set()
    for row in selected:
        ident = str(row["question_id"])
        require(ident not in seen, "公开题 question_id 重复")
        seen.add(ident)
        histories, session_ids, dates = row["haystack_sessions"], row["haystack_session_ids"], row["haystack_dates"]
        require(len(histories) == len(session_ids) == len(dates), "会话、来源 id 与日期没有对齐")
        if filler is not None:
            evidence = set(map(str, row.get("answer_session_ids", [])))
            others = sorted((str(sid) for sid in session_ids if str(sid) not in evidence),
                            key=lambda sid: digest([ident, sid]))[:filler]
            keep = evidence | set(others)
            chosen = [i for i, sid in enumerate(session_ids) if str(sid) in keep]
            histories = [histories[i] for i in chosen]
            session_ids = [session_ids[i] for i in chosen]
            dates = [dates[i] for i in chosen]
        user = "longmemeval:" + ident
        answer_turns = []
        for index, (messages, session_id, date) in enumerate(zip(histories, session_ids, dates)):
            # Positions match source_records, which enumerate messages in order.
            if include_labels:
                answer_turns.extend({"session_id": str(session_id), "position": position}
                                    for position, message in enumerate(messages) if message.get("has_answer") is True)
            # Public releases annotate some messages with has_answer. This is
            # gold location information, not conversation history. Preserve only
            # actual message fields, not arbitrary evaluation annotations.
            original = [{key: deepcopy(message[key]) for key in ("role", "content", "timestamp", "speaker", "name") if key in message}
                        for message in messages]
            for message in original:
                # A session date is source context, not an invented per-message
                # event timestamp. Keep the original content and any timestamp.
                message["session_date"] = date
            append_jsonl(output / "histories.jsonl", {"user_id": user, "scope_id": "history",
                         "session_id": str(session_id), "request_id": f"{ident}:session:{index}", "messages": original})
            sessions += 1
        append_jsonl(output / "queries.jsonl", {"id": ident, "user_id": user, "scope_id": "history",
                     "question": row["question"], "question_date": row.get("question_date"),
                     "category": row.get("question_type")})
        # answer_session_ids and has_answer are gold locations: absent from
        # histories and queries, kept only for the separate scorer's diagnostics.
        if include_labels:
            append_jsonl(output / "labels.jsonl", {"id": ident, "gold_answer": row["answer"],
                         "accepted_answers": [str(row["answer"])],
                         "answer_session_ids": [str(value) for value in row.get("answer_session_ids", [])],
                         "answer_turns": answer_turns})
    report = {"dataset": "LongMemEval-S cleaned public", "cases": len(selected), "sessions": sessions,
              "source_hash": digest(rows), "selected_ids": sorted(seen),
              "history_truncation": False, "reduced_history_filler": filler,
              "labels_included": include_labels, "model_calls": 0, "formal_AML_evaluation": "not_run"}
    write_json(output / "preparation.json", report)
    return report


def sample_longmemeval(rows: list[dict], mandatory: set[str], *, seed: int, dev_per_type: int,
                      holdout_per_type: int, unanswerable: int) -> dict:
    """Stratify using only IDs/types, never answers or observed model scores."""
    metadata = {str(row["question_id"]): row["question_type"] for row in rows}
    require(len(metadata) == len(rows), "question_id 重复")
    require(mandatory <= set(metadata), "原题 id 不在题库中")
    require(dev_per_type > 0 and holdout_per_type > 0 and unanswerable >= 0, "抽样数量无效")
    rng = random.Random(seed)
    categories = sorted(set(metadata.values()))
    dev = set(mandatory)
    counts = Counter(metadata[ident] for ident in dev)
    require(all(count <= dev_per_type for count in counts.values()), "原题超过某题型的开发集数量")
    base = lambda ident: ident.removesuffix("_abs")
    require(len({base(i) for i in dev}) == len(dev), "原题包含同一道题的有答案和无答案版本")
    needed = unanswerable - sum(i.endswith("_abs") for i in dev)
    require(needed >= 0, "原题中的无答案题超过目标数")
    candidates = sorted(i for i in metadata if i.endswith("_abs") and i not in dev)
    rng.shuffle(candidates)
    for ident in candidates:
        if needed and counts[metadata[ident]] < dev_per_type and base(ident) not in {base(i) for i in dev}:
            dev.add(ident)
            counts[metadata[ident]] += 1
            needed -= 1
    require(needed == 0, "无法补齐指定数量的无答案题")
    for category in categories:
        candidates = sorted(i for i, c in metadata.items() if c == category and not i.endswith("_abs")
                            and base(i) not in {base(d) for d in dev})
        rng.shuffle(candidates)
        needed = dev_per_type - counts[category]
        require(len(candidates) >= needed, "开发题型数量不足：" + category)
        dev.update(candidates[:needed])
    holdout = set()
    used = {base(i) for i in dev}
    for category in categories:
        candidates = sorted(i for i, c in metadata.items() if c == category and base(i) not in used)
        rng.shuffle(candidates)
        count = 0
        for ident in candidates:
            if base(ident) not in used:
                holdout.add(ident)
                used.add(base(ident))
                count += 1
                if count == holdout_per_type:
                    break
        require(count == holdout_per_type, "留出题型数量不足：" + category)
    return {"seed": seed, "dev": sorted(dev), "holdout": sorted(holdout), "mandatory": sorted(mandatory),
            "dev_counts": dict(sorted(Counter(metadata[i] for i in dev).items())),
            "holdout_counts": dict(sorted(Counter(metadata[i] for i in holdout).items())),
            "dev_unanswerable": sum(i.endswith("_abs") for i in dev),
            "holdout_unanswerable": sum(i.endswith("_abs") for i in holdout),
            "selection_fields": ["question_id", "question_type"], "base_question_overlap": False}


def split_locomo(rows: list[dict], *, seed: int, dev_conversations: int) -> dict:
    ids = sorted(str(row["sample_id"]) for row in rows)
    require(len(set(ids)) == len(ids), "sample_id 重复")
    require(0 < dev_conversations < len(ids), "开发对话数必须保留独立的留出对话")
    random.Random(seed).shuffle(ids)
    return {"seed": seed, "dev": sorted(ids[:dev_conversations]), "holdout": sorted(ids[dev_conversations:]),
            "selection_fields": ["sample_id"], "unit": "whole conversation, all questions"}


def prepare_locomo(rows: list[dict], output: Path, *, ids: set[str], media_text: str,
                   include_labels: bool = True) -> dict:
    """Keep human speakers, dates and complete sessions; gold stays in labels."""
    require(media_text in {"captions", "none"}, "media_text 只能是 captions 或 none")
    available = {str(row["sample_id"]) for row in rows}
    require(bool(ids) and ids <= available and len(available) == len(rows), "对话 id 不存在或重复")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ConflictError("准备目录已有内容；请使用空目录")
    histories, queries, labels = [], [], []
    categories, multimodal = Counter(), Counter()
    normalized_evidence_questions = []
    for row in rows:
        ident = str(row["sample_id"])
        if ident not in ids:
            continue
        conversation = row["conversation"]
        user = "locomo-refined:" + ident
        lookup = {}
        sessions = sorted((key for key in conversation if re.fullmatch(r"session_\d+", key)),
                          key=lambda key: int(key.split("_")[1]))
        require(bool(sessions), "对话没有会话：" + ident)
        for session in sessions:
            messages = []
            date = conversation[session + "_date_time"]
            for position, turn in enumerate(conversation[session]):
                content = turn["text"]
                if media_text == "captions":
                    for key, name in (("blip_caption", "image caption"), ("query", "image search query")):
                        if turn.get(key):
                            content += "\n[" + name + "] " + turn[key]
                require(isinstance(content, str) and bool(content.strip()), "对话含空消息")
                # Both participants are people, not an assistant; speaker keeps
                # attribution explicit without privileging one participant.
                messages.append({"role": "user", "speaker": turn["speaker"], "content": content,
                                 "session_date": date})
                dia_id = turn["dia_id"]
                require(dia_id not in lookup, "对话中的 dia_id 重复")
                lookup[dia_id] = {"session_id": session, "position": position}
            require(bool(messages), "对话含空会话")
            histories.append({"user_id": user, "scope_id": "history", "session_id": session,
                              "request_id": ident + ":" + session, "messages": messages})
        for index, qa in enumerate(row["qa"]):
            qid = f"{ident}#q{index:04d}"
            category = str(qa["category"])
            categories[category] += 1
            multimodal[str(bool(qa.get("is_multi_modality"))).lower()] += 1
            queries.append({"id": qid, "user_id": user, "scope_id": "history", "question": qa["question"],
                            "category": category})
            if include_labels:
                evidence = qa["evidence"]
                # Some published annotations put two IDs in one string, e.g.
                # "D8:6; D9:17". Split this explicit separator, retain the raw
                # annotation, and still reject genuinely missing references.
                references = [part.strip() for ref in evidence for part in ref.split(";")]
                require(all(ref in lookup for ref in references), "证据标注引用不存在的消息：" + qid)
                if references != evidence:
                    normalized_evidence_questions.append(qid)
                turns = [lookup[ref] for ref in references]
                answer = qa["answer"]
                rendered = "\n".join(map(str, answer)) if isinstance(answer, list) else str(answer)
                labels.append({"id": qid, "user_id": user, "scope_id": "history", "gold_answer": answer,
                               "accepted_answers": [rendered], "evidence": evidence, "resolved_evidence": references,
                               "answer_session_ids": sorted({turn["session_id"] for turn in turns}),
                               "answer_turns": turns, "is_unanswerable": category == "5",
                               "is_multi_modality": bool(qa.get("is_multi_modality"))})
    output.mkdir(parents=True, exist_ok=True)
    for filename, values in (("histories", histories), ("queries", queries), ("labels", labels)):
        for value in values:
            append_jsonl(output / (filename + ".jsonl"), value)
    report = {"dataset": "LoCoMo-Refined public", "source_hash": digest(rows), "selected_ids": sorted(ids),
              "conversations": len(ids), "sessions": len(histories), "cases": len(queries),
              "category_counts": dict(categories), "multimodal_counts": dict(multimodal),
              "media_text": media_text, "images_fetched": False, "history_truncation": False,
              "semicolon_evidence_questions": normalized_evidence_questions,
              "labels_included": include_labels, "model_calls": 0, "formal_AML_evaluation": "not_run"}
    write_json(output / "preparation.json", report)
    return report
