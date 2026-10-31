"""Write complete facts and relation notes, then search facts directly."""
from ..agent_loop import AgentLoop
from ..errors import ModelOutputError, SearchLoopError
from ..model import encode
from .retrieval import CollectionSearcher
from ..schema import digest, normalize_add, require, source_records, text, unit_text
from ..store import Scope
from . import prompts
from . import gap_fill
from . import contracts


def _source_text(source_texts, refs):
    if not source_texts or not isinstance(refs, (list, tuple, set)):
        return None
    return "\n".join(f"{ref}: {source_texts[ref]}" for ref in refs
                       if isinstance(ref, str) and ref in source_texts) or None


def string_list(value, name, *, nonempty=False, path=None, source=None):
    location = path or name
    require(isinstance(value, list), name + " 必须是编号数组", path=location, actual=value,
            expected="an array of unique non-empty string paragraph references", source=source,
            model_message=f"{location} must be an array of paragraph reference strings.")
    for index, item in enumerate(value):
        require(isinstance(item, str) and bool(item), name + " 必须是编号数组",
                path=f"{location}[{index}]", actual=item,
                expected="a non-empty paragraph reference string", source=source,
                model_message=f"{location}[{index}] must be a non-empty paragraph reference string.")
    require(len(set(value)) == len(value), name + " 不能重复", path=location, actual=value,
            expected="an array with unique paragraph references", source=source,
            model_message=f"{location} must not contain duplicate references.")
    require(not nonempty or bool(value), name + " 不能为空", path=location, actual=value,
            expected="a non-empty array of paragraph references", source=source,
            model_message=f"{location} must contain at least one paragraph reference.")
    return value


class CollectionEngine:
    write_prompt = prompts.WRITE
    # Longer sessions are extracted part by part; a request carries one part and the earlier facts.
    PART_CHARS = 24000
    # Earlier facts carried by one part, newest first; older facts beyond this are left out of the request.
    EARLIER_FACT_CHARS = 120000

    def __init__(self, store, model, *, embedder=None, floor=None):
        self.store, self.model = store, model
        self.searcher = CollectionSearcher(store, embedder, floor=floor)
        self.trace = []

    def _complete(self, messages, purpose, *, validate=None):
        return AgentLoop(self.model, messages, purpose=purpose, trace=self.trace).complete(validate=validate)


    @staticmethod
    def source_view(payload):
        aliases, sources = {}, []
        for i, source in enumerate(source_records(payload)):
            for n in range(1, len(source["units"]) + 1):
                alias = f"m{i}.{n}"
                aliases[alias] = f"{source['id']}.{n}"
                sources.append({"ref": alias, "session_id": source["session_id"],
                                **{k: v for k, v in source["message"].items() if k != "content"},
                                "text": unit_text(source, n)})
        return aliases, sources

    def extract(self, payload):
        payload = normalize_add(payload)
        parts = self.extraction_parts(payload)
        if len(parts) == 1:
            return self._complete(self.extraction_messages(payload), "collection_extract",
                                  validate=lambda extracted: self.extraction_points(payload, extracted))
        # A long session is read part by part: each request holds one part in full and the facts
        # extracted so far, so its size does not grow with the session.
        messages, facts = self.extraction_messages(payload)[:1], []
        for index, sources in enumerate(parts):
            seen = frozenset(s["ref"] for s in sources)
            message = self.part_message(index, len(parts), sources, earlier=self.earlier_facts(facts))
            reply = self._complete(
                [messages[0], message], "collection_extract",
                validate=lambda r, seen=seen, sources=sources:
                self.part_points(r, seen, {s["ref"]: s["text"] for s in sources}))
            messages += [message, {"role": "assistant", "content": encode(reply)}]
            facts += reply["facts"]
        merged = {"facts": facts, "conversation": messages}
        self.extraction_points(payload, merged)
        return merged

    @classmethod
    def earlier_facts(cls, facts):
        kept, size = [], 0
        for fact in reversed(facts):
            size += len(fact["text"])
            if size > cls.EARLIER_FACT_CHARS:
                break
            kept.append(fact["text"])
        return kept[::-1]

    @staticmethod
    def part_points(reply, seen, source_texts=None):
        """One part's reply: cites only that part's paragraphs."""
        require(isinstance(reply, dict) and isinstance(reply.get("facts"), list), "facts 必须是数组",
                path="facts", actual=(reply.get("facts") if isinstance(reply, dict) else reply),
                expected="an array of fact objects",
                model_message="facts must be an array of fact objects.")
        for index, fact in enumerate(reply["facts"]):
            path = f"facts[{index}]"
            refs = fact.get("evidence", []) if isinstance(fact, dict) else []
            source = _source_text(source_texts, refs)
            require(isinstance(fact, dict) and set(fact) == {"text", "evidence"}, "事实需要 text/evidence",
                    path=path, actual=fact, expected="an object with exactly text and evidence keys",
                    source=source,
                    model_message=f"{path} must be an object with exactly text and evidence keys.")
            text(fact["text"], "事实", path=f"{path}.text", source=source,
                 model_message=f"{path}.text must be a non-empty fact string.")
            cited = string_list(fact["evidence"], "事实依据", nonempty=True,
                                path=f"{path}.evidence", source=source)
            invalid = [ref for ref in cited if ref not in seen]
            require(not invalid, "事实引用了本段以外或不存在的原文",
                    path=(f"{path}.evidence[{cited.index(invalid[0])}]" if len(invalid) == 1
                          else f"{path}.evidence"),
                    actual=invalid[0] if len(invalid) == 1 else invalid,
                    expected="an existing paragraph reference from this part", source=source,
                    model_message=(f"{path}.evidence contains a reference outside this part. "
                                   "Use only an existing paragraph reference from the supplied source."))

    @classmethod
    def conversation(cls, payload, extracted):
        """The write conversation so far, ending with the latest extraction or gap-fill reply."""
        if "conversation" in extracted:
            return [dict(m) for m in extracted["conversation"]]
        return [*cls.extraction_messages(payload), {"role": "assistant", "content": encode(extracted)}]

    def fill_gaps(self, payload, extracted):
        """Ask about program-flagged paragraphs in small batches; only rejected paragraphs are asked again."""
        payload = normalize_add(payload)
        aliases, sources = self.source_view(payload)
        gaps = gap_fill.find_gaps(sources, extracted)
        texts = {s["ref"]: s["text"] for s in sources}
        self.trace.append({"tool": "find_gaps", "result": {"flagged": gaps}})
        if not gaps:
            return extracted
        facts, messages, reply = [dict(f) for f in extracted["facts"]], [], {}
        for start in range(0, len(gaps), gap_fill.BATCH):
            pending, errors, rejected = gaps[start:start + gap_fill.BATCH], {}, set()
            failures = {}
            while pending:
                messages = self.gap_request(
                    payload, extracted, facts, pending, errors,
                    error_details=getattr(errors, "error_details", None))
                reply = self._complete(messages, gap_fill.PURPOSE, validate=gap_fill.check_shape)
                facts, errors = gap_fill.apply_fill(facts, reply, pending, aliases, texts)
                pending = [g for g in pending if g["ref"] in errors]
                signature = digest([reply, errors])
                if pending:
                    for ref in errors:
                        failures[ref] = failures.get(ref, 0) + 1
                    exhausted = any(count >= AgentLoop.MAX_ATTEMPTS for count in failures.values())
                    repeated = signature in rejected
                    detail = {"type": "ValidationError", "message": encode(errors),
                              "model_message": "Some flagged paragraphs are still unresolved; correct the listed items.",
                              "error_details": getattr(errors, "error_details", {}),
                              "attempt": max(failures.values()), "retry_scheduled": not (repeated or exhausted)}
                    self.trace[-1]["output_validation"] = detail
                    calls = getattr(self.model, "calls", [])
                    last = calls[-1] if calls and isinstance(calls[-1], dict) else {}
                    if last:
                        last["output_validation"] = detail
                    channel = (f"；最后渠道 {last['channel_index']}（{last.get('requested_model', '')}）"
                               if "channel_index" in last else "")
                    reason = (f"{gap_fill.PURPOSE} 阶段在 {max(failures.values())} 次补漏后仍未通过校验"
                              f"{channel}：{encode(errors)}")
                    if repeated:
                        raise SearchLoopError(reason + "；模型重复返回相同无效内容；未写入替代结果")
                    if exhausted:
                        raise ModelOutputError(reason)
                rejected.add(signature)
        filled = {"facts": facts}
        self.extraction_points(payload, filled)
        # Preserve the last accepted request for checkpoints.
        return {**filled, "conversation": [*messages, {"role": "assistant", "content": encode(reply)}]}

    @classmethod
    def gap_request(cls, payload, extracted, facts, gaps, errors=None, *, error_details=None):
        """One batch: a single-part session continues its conversation; a long one sends only the surroundings."""
        if len(cls.extraction_parts(payload)) == 1:
            aliases, _ = cls.source_view(payload)
            indexes = range(len(facts))
            return gap_fill.gap_messages(cls.conversation(payload, extracted), facts, gaps, indexes, aliases,
                                         errors=errors, error_details=error_details)
        _, sources = cls.source_view(payload)
        shown, indexes = gap_fill.surroundings(sources, facts, gaps)
        refs = {s["ref"] for s in shown} | {r for i in indexes for r in facts[i]["evidence"]}
        return gap_fill.gap_messages(cls.extraction_messages(payload)[:1], facts, gaps, indexes, refs,
                                     paragraphs=cls.session_blocks(shown), errors=errors,
                                     error_details=error_details)

    @classmethod
    def extraction_messages(cls, payload):
        """Rebuild the same conversation prefix for live and cached extraction.

        No history is kept on the engine: another request cannot inherit these
        source messages. Cached extraction is validated against this same input.
        """
        parts = cls.extraction_parts(payload)
        return [{"role": "system", "content": cls.write_prompt}, cls.part_message(0, len(parts), parts[0])]

    @classmethod
    def extraction_parts(cls, payload):
        """Paragraphs split into parts of at most PART_CHARS characters.

        A session within the limit stays one part (the unchanged single request). Parts break at message
        boundaries, and only a message longer than the limit is split, between its paragraphs.
        """
        _, sources = cls.source_view(payload)
        parts, size, message, whole = [[]], 0, None, 0
        for source in sources:
            index = source["ref"].split(".")[0]
            if index != message:
                message, whole = index, cls.message_chars(sources, index)
                if parts[-1] and size + whole > cls.PART_CHARS:
                    parts.append([])
                    size = 0
            elif whole > cls.PART_CHARS and parts[-1] and size + len(source["text"]) > cls.PART_CHARS:
                parts.append([])
                size = 0
            parts[-1].append(source)
            size += len(source["text"])
        return parts

    @staticmethod
    def message_chars(sources, index):
        return sum(len(s["text"]) for s in sources if s["ref"].split(".")[0] == index)

    @staticmethod
    def session_blocks(sources):
        """Session id and date once; each message's role once, its paragraphs as [ref, text]."""
        blocks, message = [], None
        for source in sources:
            head = {k: source[k] for k in ("session_id", "session_date") if k in source}
            if not blocks or {k: v for k, v in blocks[-1].items() if k != "turns"} != head:
                blocks.append({**head, "turns": []})
                message = None
            if source["ref"].split(".")[0] != message:
                message = source["ref"].split(".")[0]
                blocks[-1]["turns"].append({**{k: v for k, v in source.items() if k not in (*head, "ref", "text")},
                                            "paragraphs": []})
            blocks[-1]["turns"][-1]["paragraphs"].append([source["ref"], source["text"]])
        return blocks

    @classmethod
    def part_message(cls, index, total, sources, *, earlier=()):
        sessions = cls.session_blocks(sources)
        instruction = contracts.extraction({s["ref"] for s in sources})
        if total == 1:
            body = {"instructions": prompts.EXTRACT + "\n" + instruction, "sessions": sessions}
        else:
            body = {"instructions": prompts.EXTRACT + "\n" + prompts.EXTRACT_PARTS + "\n" + instruction,
                    "part": f"{index + 1} of {total}"}
            if earlier:
                body["earlier_facts"] = list(earlier)
            body["sessions"] = sessions
        return {"role": "user", "content": encode(body)}

    @classmethod
    def extraction_points(cls, payload, extracted):
        """Validate references before caching or placing any facts."""
        payload = normalize_add(payload)
        aliases, sources = cls.source_view(payload)
        source_texts = {s["ref"]: s["text"] for s in sources}
        require(isinstance(extracted, dict), "抽取结果必须是对象", path="$", actual=extracted,
                expected="a JSON object with a facts array",
                model_message="The extraction response must be a JSON object with a facts array.")
        facts = extracted.get("facts")
        require(isinstance(facts, list), "facts 必须是数组", path="facts", actual=facts,
                expected="an array of fact objects",
                model_message="facts must be an array of fact objects.")
        points = []
        for index, fact in enumerate(facts):
            path = f"facts[{index}]"
            refs = fact.get("evidence", []) if isinstance(fact, dict) else []
            source = _source_text(source_texts, refs)
            require(isinstance(fact, dict) and set(fact) == {"text", "evidence"}, "事实需要 text/evidence",
                    path=path, actual=fact, expected="an object with exactly text and evidence keys",
                    source=source,
                    model_message=f"{path} must be an object with exactly text and evidence keys.")
            text(fact["text"], "事实", path=f"{path}.text", source=source,
                 model_message=f"{path}.text must be a non-empty fact string.")
            refs = string_list(fact["evidence"], "事实依据", nonempty=True,
                               path=f"{path}.evidence", source=source)
            invalid = [ref for ref in refs if ref not in aliases]
            require(not invalid, "事实引用了不存在的原文",
                    path=(f"{path}.evidence[{refs.index(invalid[0])}]" if len(invalid) == 1
                          else f"{path}.evidence"),
                    actual=invalid[0] if len(invalid) == 1 else invalid,
                    expected="an existing paragraph reference from the source conversation", source=source,
                    model_message=(f"{path}.evidence contains an unknown paragraph reference. "
                                   "Use an existing reference from the supplied source conversation."))
            points.append({"id": "cf_" + digest([payload["request_id"], index]),
                           "text": fact["text"], "evidence": [aliases[r] for r in refs]})
        return points

    def add(self, payload, *, extraction=None):
        """Save complete facts and sources, then annotate user-side fact relations."""
        self.trace = []
        payload = normalize_add(payload)
        prior = self.store.receipt(payload)
        if prior:
            self.ensure_relations(payload)
            return prior
        scope = self.store.snapshot(Scope(payload["user_id"], payload["scope_id"]))
        extracted = self.fill_gaps(payload, self.extract(payload)) if extraction is None else extraction
        points = self.extraction_points(payload, extracted)
        receipt = self.store.commit(payload, {"points": points}, expected_revision=scope.as_of)
        self.ensure_relations(payload)
        return receipt

    def relations_pending(self, payload):
        from .relations import PROMPT_HASH
        payload = normalize_add(payload)
        return self.searcher.embedder is not None and not self.store.relations_done(
            Scope(payload["user_id"], payload["scope_id"]), payload["request_id"], prompt_hash=PROMPT_HASH)

    def ensure_relations(self, payload):
        """Relation stage of one committed request, once; it needs fact vectors to choose candidates."""
        from .relations import RelationAnnotator
        payload = normalize_add(payload)
        if self.searcher.embedder is None:
            self.trace.append({"tool": "fact_relations", "result": {"skipped": "no embedder"}})
            return
        if self.relations_pending(payload):
            RelationAnnotator(self).run(Scope(payload["user_id"], payload["scope_id"]), self.store.receipt(payload))

    def query(self, question, scope, *, question_date=None, fact_limit=None):
        """Formal Search: direct fact ranking by the full question (text, BM25, named time), then session context."""
        from .query import DirectFactsQuery
        self.trace = []
        return DirectFactsQuery(self).run(question, scope, question_date=question_date,
                                          fact_limit=fact_limit)

