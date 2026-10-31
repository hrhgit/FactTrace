"""Program-flagged extraction gaps, filled through the same model client.

The program only points at paragraphs whose details may be missing: image
captions not reflected in the citing facts, and uncited statements the speaker
makes about themself. It never writes or edits a fact. Claiming "already saved"
requires a quote the program can check against the fact and the paragraph.
Each answer is checked item by item: accepted items stay, and only the flagged
paragraphs left unsettled are asked again.
"""
from __future__ import annotations

import re

from ..errors import ValidationError
from ..model import encode
from ..schema import require, text
from . import prompts
from . import contracts

PURPOSE = "collection_gap_fill"
# Flagged paragraphs per request, and how far around a flagged paragraph a long session's facts are shown.
BATCH = 8
FACT_REACH = 3
# A substantive paragraph is long enough to carry a statement rather than a
# greeting; a caption counts as reflected when most of its content words are.
UNCITED_MIN_CHARS = 60
CAPTION_SHARE = 0.5
CAPTION = re.compile(r"\[image caption\]\s*(.*)")
ATTACHMENT = re.compile(r"\[image (?:caption|search query)\]")
GENERIC = frozenset("photo photography picture image with that this some there their from into onto over "
                    "sitting standing holding looking front about next near what".split())
# An uncited paragraph is worth a second look only when the speaker tells about themself;
# listeners' replies ("Sounds great, keep it up!") are reactions, not facts.
SELF = re.compile(r"\b(?:i|i'm|i've|i'd|i'll|my|me|mine)\b", re.I)
# Stock reactions to the other person's news carry "I" without telling anything about the speaker.
# A no_fact reason that says the content is already saved is a covered claim in disguise.
SAVED_CLAIM = re.compile(r"\b(?:already|recorded|covered|captured|saved|merged|included)\b|\bfacts? \d", re.I)
REACTION = re.compile(r"\bi(?:'m| am)? (?:so |really )?(?:sorry|glad|happy for you|here for you|proud of you)\b"
                      r"|\bi (?:hope|bet|guess|love that|love how|can't wait)\b", re.I)


def stem(word):
    # Drop a plain inflection, then compare five-letter prefixes, so "gives"/"give",
    # "sneakers"/"sneaker" and "painting"/"painted" match.
    return (re.sub(r"(?:ing|ed|es|s)$", "", word) if len(word) > 5 else word.rstrip("s"))[:5]


def content_words(value):
    return [stem(w) for w in re.findall(r"[a-z0-9]+", value.lower()) if len(w) > 3 and w not in GENERIC]


def told(statement):
    """The statement without its questions, which only supply context for a following answer."""
    return " ".join(p for p in re.split(r"(?<=[.!?])\s+", statement) if not p.rstrip().endswith("?"))


def find_gaps(sources, extracted):
    """Paragraph refs whose caption or self-told statement is absent from the cited facts."""
    citing = {}
    for fact in extracted["facts"]:
        for ref in fact["evidence"]:
            citing.setdefault(ref, []).append(fact["text"])
    gaps = []
    for source in sources:
        ref, body = source["ref"], source["text"]
        saved = set(content_words(" ".join(citing.get(ref, []))))
        statement = told(ATTACHMENT.split(body)[0].strip())
        captions = [content_words(c) for c in CAPTION.findall(body)]
        if any(words and sum(w in saved for w in words) / len(words) < CAPTION_SHARE for words in captions):
            gaps.append({"ref": ref, "check": "image caption content is not reflected in the facts citing this paragraph"})
        # Assistant replies are explanatory/procedural text, not the user's own
        # uncited memories. Keep caption checks above active for every role, but
        # only ask the model to review an uncited self-statement from a
        # conversation participant.
        elif (source.get("role") != "assistant" and ref not in citing
              and len(statement) >= UNCITED_MIN_CHARS
              and SELF.search(REACTION.sub("", statement))):
            gaps.append({"ref": ref, "check": "no fact cites this paragraph"})
    return gaps


def surroundings(sources, facts, gaps):
    """Flagged paragraphs with their neighbours, and the indexes of facts citing paragraphs near them."""
    flagged = {g["ref"] for g in gaps}
    places = [i for i, s in enumerate(sources) if s["ref"] in flagged]
    shown = [s for i, s in enumerate(sources) if any(abs(i - p) <= 1 for p in places)]
    near = {s["ref"] for i, s in enumerate(sources) if any(abs(i - p) <= FACT_REACH for p in places)}
    return shown, [i for i, f in enumerate(facts) if near & set(f["evidence"])]


def gap_messages(conversation, facts, gaps, indexes, allowed_refs, *, paragraphs=None, errors=None):
    """One request about these flagged paragraphs, following the given conversation prefix."""
    notes = [prompts.GAP_FILL]
    body = {"facts": [{"index": i, "text": facts[i]["text"], "evidence": facts[i]["evidence"]} for i in indexes],
            "flagged": gaps}
    if paragraphs is not None:
        notes.append(prompts.GAP_FILL_PARTS)
        body["paragraphs"] = paragraphs
    if errors:
        notes.append(prompts.GAP_FILL_RETRY)
        body["rejected"] = errors
    return [*conversation, {"role": "user", "content": encode({
        "instructions": "\n".join(notes) + "\n" + contracts.gap_fill(allowed_refs, indexes), **body})}]


def normalized(value):
    return " ".join(re.findall(r"[a-z0-9]+", value.lower()))


def check_shape(reply):
    """Malformed answers are corrected in place; item-level problems are handled by apply_fill."""
    keys = ("revised", "facts", "covered", "no_fact")
    require(isinstance(reply, dict) and set(reply) <= set(keys), "补漏结果只能包含 " + "/".join(keys))
    require(all(isinstance(reply.get(k, []), list) for k in keys), "/".join(keys) + " 必须是数组")


def _revise(facts, item, aliases, seen):
    require(isinstance(item, dict) and set(item) == {"index", "text", "evidence"}, "修订需要 index/text/evidence")
    index = item["index"]
    require(type(index) is int and 0 <= index < len(facts) and index not in seen, "修订的事实编号无效或重复")
    text(item["text"], "修订事实")
    refs = item["evidence"]
    require(isinstance(refs, list) and set(refs) <= set(aliases), "修订事实引用了不存在的原文")
    # A revision only adds detail; it must keep the paragraphs already cited.
    lost = [r for r in facts[index]["evidence"] if r not in refs]
    require(not lost, f"修订事实 {index} 必须保留原有依据，evidence 缺少：{', '.join(lost)}")
    seen.add(index)
    facts[index] = {"text": item["text"], "evidence": list(dict.fromkeys(refs))}
    return refs


def _add(facts, item, aliases):
    require(isinstance(item, dict) and set(item) == {"text", "evidence"}, "新事实需要 text/evidence")
    text(item["text"], "新事实")
    refs = item["evidence"]
    require(isinstance(refs, list) and refs and set(refs) <= set(aliases), "新事实引用了不存在的原文")
    facts.append({"text": item["text"], "evidence": list(dict.fromkeys(refs))})
    return refs


def _covered(facts, item, aliases, texts):
    require(isinstance(item, dict) and set(item) == {"ref", "index", "quote"} and item["ref"] in aliases,
            "covered 需要有效的 ref/index/quote")
    index, quote = item["index"], text(item["quote"], "覆盖引文")
    require(type(index) is int and 0 <= index < len(facts), "covered 的事实编号无效")
    # Case, punctuation and spacing are ignored; the words must be the fact's own.
    require(normalized(quote) in normalized(facts[index]["text"]),
            f"{item['ref']} 的引文不在事实 {index} 中；引文须摘自该事实修订后的文字：{facts[index]['text']}")
    # "Already covered" must be the paragraph's own wording, not a looser synonym.
    words, source = content_words(quote), set(content_words(texts[item["ref"]]))
    require(words and sum(w in source for w in words) / len(words) >= 0.5,
            f"{item['ref']} 的引文与该段原文用词不一致；请补回原话，或引用确实使用原话的事实片段；"
            "若该段只是提问、寒暄或其他事实的背景，请改列入 no_fact")
    return [item["ref"]]


def _no_fact(item, aliases):
    require(isinstance(item, dict) and set(item) == {"ref", "reason"} and item["ref"] in aliases,
            "no_fact 需要有效的 ref/reason")
    text(item["reason"], "无事实原因")
    require(not SAVED_CLAIM.search(item["reason"]),
            f"{item['ref']} 的理由称内容已保存；已保存请列入 covered 并逐字引用事实原文，"
            "未按原话保存请修订该事实；no_fact 只用于没有可保存内容的段落")
    return [item["ref"]]


def apply_fill(facts, reply, gaps, aliases, texts):
    """Keep every valid item; return the facts and the reason each flagged paragraph is still unsettled."""
    check_shape(reply)
    facts, seen, handled, errors = [dict(f) for f in facts], set(), set(), {}
    steps = [(item, lambda i: _revise(facts, i, aliases, seen)) for item in reply.get("revised", [])]
    steps += [(item, lambda i: _add(facts, i, aliases)) for item in reply.get("facts", [])]
    steps += [(item, lambda i: _covered(facts, i, aliases, texts)) for item in reply.get("covered", [])]
    steps += [(item, lambda i: _no_fact(i, aliases)) for item in reply.get("no_fact", [])]
    flagged = {g["ref"] for g in gaps}
    for item, step in steps:
        try:
            handled.update(step(item))
        except ValidationError as error:
            refs = item.get("evidence", [item.get("ref")]) if isinstance(item, dict) else []
            for ref in (refs if isinstance(refs, list) else []):
                if ref in flagged:
                    errors.setdefault(ref, str(error))
    unsettled = {r: errors.get(r, "该标记段落既未补入事实也未处理") for r in flagged - handled}
    return facts, unsettled
