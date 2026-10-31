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
# Match an actual saved-claim phrase, rather than ordinary semantic uses such as
# "covered in caramel" or "included a list of songs".
SAVED_CLAIM = re.compile(
    r"\balready\s+(?:been\s+)?(?:recorded|saved|included|captured|merged|covered)\b"
    r"|\b(?:recorded|saved|included|captured|merged|covered)\s+(?:in|by)\s+"
    r"(?:the\s+)?facts?\s*\d+\b"
    r"|\bfacts?\s*\d+\s+(?:already\s+)?(?:records?|saves?|includes?|captures?|merges?|covers?)\b",
    re.I,
)
REACTION = re.compile(r"\bi(?:'m| am)? (?:so |really )?(?:sorry|glad|happy for you|here for you|proud of you)\b"
                      r"|\bi (?:hope|bet|guess|love that|love how|can't wait)\b", re.I)


class ErrorMap(dict):
    """String-compatible gap errors with structured diagnostics for the retry."""
    def __init__(self, *args, error_details=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.error_details = dict(error_details or {})


def _source_text(texts, refs):
    if not texts or not isinstance(refs, (list, tuple, set)):
        return None
    return "\n".join(f"{ref}: {texts[ref]}" for ref in refs
                       if isinstance(ref, str) and ref in texts) or None


def _item_refs(item):
    """Collect only explicit string refs; never infer a paragraph from bad data."""
    if not isinstance(item, dict):
        return []
    refs = []
    ref = item.get("ref")
    if isinstance(ref, str):
        refs.append(ref)
    evidence = item.get("evidence")
    if isinstance(evidence, list):
        refs.extend(value for value in evidence if isinstance(value, str))
    return list(dict.fromkeys(refs))


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


def gap_messages(conversation, facts, gaps, indexes, allowed_refs, *, paragraphs=None, errors=None,
                 error_details=None):
    """One request about these flagged paragraphs, following the given conversation prefix."""
    notes = [prompts.GAP_FILL]
    body = {"facts": [{"index": i, "text": facts[i]["text"], "evidence": facts[i]["evidence"]} for i in indexes],
            "flagged": gaps}
    if paragraphs is not None:
        notes.append(prompts.GAP_FILL_PARTS)
        body["paragraphs"] = paragraphs
    if errors:
        notes.append(prompts.GAP_FILL_RETRY)
        if error_details:
            body["rejected"] = {
                ref: (error_details.get(ref, {}).get("message", message)
                      if isinstance(error_details.get(ref), dict) else message)
                for ref, message in errors.items()
            }
        else:
            body["rejected"] = errors
    if error_details:
        body["diagnostics"] = error_details
    return [*conversation, {"role": "user", "content": encode({
        "instructions": "\n".join(notes) + "\n" + contracts.gap_fill(allowed_refs, indexes), **body})}]


def normalized(value):
    return " ".join(re.findall(r"[a-z0-9]+", value.lower()))


def check_shape(reply):
    """Malformed answers are corrected in place; item-level problems are handled by apply_fill."""
    keys = ("revised", "facts", "covered", "no_fact")
    require(isinstance(reply, dict) and set(reply) <= set(keys), "补漏结果只能包含 " + "/".join(keys),
            path="$", actual=reply,
            expected="an object containing only revised, facts, covered and no_fact arrays",
            model_message=("The gap-fill response must be an object containing only revised, facts, "
                           "covered and no_fact arrays."))
    for key in keys:
        require(isinstance(reply.get(key, []), list), "/".join(keys) + " 必须是数组",
                path=key, actual=reply.get(key), expected="an array",
                model_message=f"{key} must be an array.")


def _revise(facts, item, aliases, seen, *, texts=None, path="revised"):
    require(isinstance(item, dict) and set(item) == {"index", "text", "evidence"}, "修订需要 index/text/evidence",
            path=path, actual=item, expected="an object with exactly index, text and evidence keys",
            source=_source_text(texts, _item_refs(item)),
            model_message=f"{path} must be an object with exactly index, text and evidence keys.")
    index = item["index"]
    require(type(index) is int and 0 <= index < len(facts) and index not in seen, "修订的事实编号无效或重复",
            path=f"{path}.index", actual=index,
            expected="an existing, not-yet-revised fact index",
            model_message=f"{path}.index must identify an existing fact that is not already revised.")
    text(item["text"], "修订事实", path=f"{path}.text",
         source=_source_text(texts, item.get("evidence", [])),
         model_message=f"{path}.text must be a non-empty revised fact string.")
    refs = item["evidence"]
    require(isinstance(refs, list), "修订事实引用了不存在的原文", path=f"{path}.evidence",
            actual=refs, expected="an array of existing paragraph references",
            source=_source_text(texts, refs),
            model_message=f"{path}.evidence must be an array of existing paragraph references.")
    for ref_index, ref in enumerate(refs):
        require(isinstance(ref, str) and ref in aliases, "修订事实引用了不存在的原文",
                path=f"{path}.evidence[{ref_index}]", actual=ref,
                expected="an existing paragraph reference", source=_source_text(texts, refs),
                model_message=f"{path}.evidence[{ref_index}] must be an existing paragraph reference.")
    # A revision only adds detail; it must keep the paragraphs already cited.
    lost = [r for r in facts[index]["evidence"] if r not in refs]
    require(not lost, f"修订事实 {index} 必须保留原有依据，evidence 缺少：{', '.join(lost)}",
            path=f"{path}.evidence", actual=refs,
            expected=f"an array that still contains the original references {facts[index]['evidence']}",
            source=_source_text(texts, lost),
            model_message=(f"{path}.evidence must preserve every paragraph reference already attached "
                           f"to fact {index}: {facts[index]['evidence']}"))
    seen.add(index)
    facts[index] = {"text": item["text"], "evidence": list(dict.fromkeys(refs))}
    return refs


def _add(facts, item, aliases, *, texts=None, path="facts"):
    require(isinstance(item, dict) and set(item) == {"text", "evidence"}, "新事实需要 text/evidence",
            path=path, actual=item, expected="an object with exactly text and evidence keys",
            source=_source_text(texts, _item_refs(item)),
            model_message=f"{path} must be an object with exactly text and evidence keys.")
    text(item["text"], "新事实", path=f"{path}.text",
         source=_source_text(texts, item.get("evidence", [])),
         model_message=f"{path}.text must be a non-empty fact string.")
    refs = item["evidence"]
    require(isinstance(refs, list) and bool(refs), "新事实引用了不存在的原文", path=f"{path}.evidence",
            actual=refs, expected="a non-empty array of existing paragraph references",
            source=_source_text(texts, refs),
            model_message=f"{path}.evidence must be a non-empty array of existing paragraph references.")
    for ref_index, ref in enumerate(refs):
        require(isinstance(ref, str) and ref in aliases, "新事实引用了不存在的原文",
                path=f"{path}.evidence[{ref_index}]", actual=ref,
                expected="an existing paragraph reference", source=_source_text(texts, refs),
                model_message=f"{path}.evidence[{ref_index}] must be an existing paragraph reference.")
    facts.append({"text": item["text"], "evidence": list(dict.fromkeys(refs))})
    return refs


def _covered(facts, item, aliases, texts, *, path="covered"):
    texts = texts or {}
    require(isinstance(item, dict) and set(item) == {"ref", "index", "quote"},
            "covered 需要有效的 ref/index/quote", path=path, actual=item,
            expected="an object with exactly ref, index and quote keys",
            source=_source_text(texts, _item_refs(item)),
            model_message=f"{path} must be an object with exactly ref, index and quote keys.")
    ref = item.get("ref")
    require(isinstance(ref, str) and ref in aliases, "covered 需要有效的 ref/index/quote",
            path=f"{path}.ref", actual=ref, expected="an existing paragraph reference",
            source=texts.get(ref) if isinstance(ref, str) else None,
            model_message=f"{path}.ref must be an existing paragraph reference.")
    index = item["index"]
    quote = text(item["quote"], "覆盖引文", path=f"{path}.quote", source=texts.get(ref),
                 model_message=f"{path}.quote must be a non-empty quote.")
    require(type(index) is int and 0 <= index < len(facts), "covered 的事实编号无效",
            path=f"{path}.index", actual=index, expected="an existing fact index",
            source=texts.get(ref), model_message=f"{path}.index must identify an existing fact.")
    # Case, punctuation and spacing are ignored; the words must be the fact's own.
    require(normalized(quote) in normalized(facts[index]["text"]),
            f"{ref} 的引文不在事实 {index} 中；引文须摘自该事实修订后的文字：{facts[index]['text']}",
            path=f"{path}.quote", actual=quote,
            expected=f"a quote contained in facts[{index}].text", source=texts.get(ref),
            model_message=(f"{path}.quote must be copied from facts[{index}].text; "
                           "do not paraphrase the quote."))
    # "Already covered" must be the paragraph's own wording, not a looser synonym.
    words, source = content_words(quote), set(content_words(texts.get(ref, "")))
    require(words and sum(w in source for w in words) / len(words) >= 0.5,
            f"{ref} 的引文与该段原文用词不一致；请补回原话，或引用确实使用原话的事实片段；"
            "若该段只是提问、寒暄或其他事实的背景，请改列入 no_fact",
            path=f"{path}.quote", actual=quote,
            expected="a quote using the paragraph's own wording", source=texts.get(ref),
            model_message=(f"{path}.quote must use wording from the flagged paragraph. "
                           "Use no_fact if the paragraph has no saveable fact."))
    return [ref]


def _no_fact(item, aliases, texts=None, *, path="no_fact"):
    texts = texts or {}
    require(isinstance(item, dict) and set(item) == {"ref", "reason"},
            "no_fact 需要有效的 ref/reason", path=path, actual=item,
            expected="an object with exactly ref and reason keys",
            source=_source_text(texts, _item_refs(item)),
            model_message=f"{path} must be an object with exactly ref and reason keys.")
    ref = item.get("ref")
    require(isinstance(ref, str) and ref in aliases, "no_fact 需要有效的 ref/reason",
            path=f"{path}.ref", actual=ref, expected="an existing paragraph reference",
            source=texts.get(ref) if isinstance(ref, str) else None,
            model_message=f"{path}.ref must be an existing paragraph reference.")
    text(item["reason"], "无事实原因", path=f"{path}.reason", source=texts.get(ref),
         model_message=f"{path}.reason must explain why the paragraph has no saveable fact.")
    require(not SAVED_CLAIM.search(item["reason"]),
            f"{ref} 的理由称内容已保存；已保存请列入 covered 并逐字引用事实原文，"
            "未按原话保存请修订该事实；no_fact 只用于没有可保存内容的段落",
            path=f"{path}.reason", actual=item["reason"],
            expected="a reason that does not claim the content is already saved",
            source=texts.get(ref),
            model_message=(f"{path}.reason must not claim that the paragraph is already saved. "
                           "Use covered with an exact quote, revise the fact, or explain why no fact is present."))
    return [ref]


def apply_fill(facts, reply, gaps, aliases, texts):
    """Keep every valid item; return the facts and the reason each flagged paragraph is still unsettled."""
    check_shape(reply)
    facts, seen, handled, errors = [dict(f) for f in facts], set(), set(), ErrorMap()
    steps = [(item, f"revised[{index}]", lambda i, path=f"revised[{index}]":
              _revise(facts, i, aliases, seen, texts=texts, path=path))
             for index, item in enumerate(reply.get("revised", []))]
    steps += [(item, f"facts[{index}]", lambda i, path=f"facts[{index}]":
               _add(facts, i, aliases, texts=texts, path=path))
              for index, item in enumerate(reply.get("facts", []))]
    steps += [(item, f"covered[{index}]", lambda i, path=f"covered[{index}]":
               _covered(facts, i, aliases, texts, path=path))
              for index, item in enumerate(reply.get("covered", []))]
    steps += [(item, f"no_fact[{index}]", lambda i, path=f"no_fact[{index}]":
               _no_fact(i, aliases, texts, path=path))
              for index, item in enumerate(reply.get("no_fact", []))]
    flagged = {g["ref"] for g in gaps}
    for item, path, step in steps:
        try:
            handled.update(step(item))
        except ValidationError as error:
            refs = _item_refs(item)
            error_details = error.diagnostic()
            error_details.update({"message": error.model_message or "The gap-fill item failed validation.",
                                  "instruction": "Correct this item using the supplied paragraph and return the full response."})
            for ref in refs:
                if ref in flagged:
                    errors.setdefault(ref, str(error))
                    errors.error_details.setdefault(ref, {"path": error_details.get("path", path),
                                                          "actual": error_details.get("actual", item),
                                                          "expected": error_details.get("expected",
                                                                                         "a valid gap-fill item"),
                                                          "source": error_details.get("source", texts.get(ref)),
                                                          "message": error_details["message"],
                                                          "instruction": error_details["instruction"]})
    unsettled = {r: errors.get(r, "该标记段落既未补入事实也未处理") for r in flagged - handled}
    result = ErrorMap(unsettled, error_details=errors.error_details)
    for ref in flagged - handled:
        if ref not in result.error_details:
            result.error_details[ref] = {
                "path": "$",
                "actual": reply,
                "expected": "a valid item that handles this flagged paragraph",
                "source": texts.get(ref),
                "message": "No valid gap-fill item handled this flagged paragraph.",
                "instruction": "Handle every flagged paragraph exactly once, or explain why it has no saveable fact.",
            }
    return facts, result
