"""Write complete facts and relation notes, then search facts directly."""
from ..agent_loop import AgentLoop
from ..errors import SearchLoopError
from ..model import encode
from .retrieval import CollectionSearcher
from ..schema import digest, normalize_add, require, source_records, text, unit_text
from ..store import Scope
from . import prompts
from . import gap_fill
from . import contracts


def string_list(value, name, *, nonempty=False):
    require(isinstance(value, list) and all(isinstance(v, str) and v for v in value), name + " 必须是编号数组")
    require(len(set(value)) == len(value), name + " 不能重复")
    require(not nonempty or bool(value), name + " 不能为空")
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
            reply = self._complete([messages[0], message], "collection_extract",
                                   validate=lambda r, seen=seen: self.part_points(r, seen))
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
    def part_points(reply, seen):
        """One part's reply: cites only that part's paragraphs."""
        require(isinstance(reply, dict) and isinstance(reply.get("facts"), list), "facts 必须是数组")
        for fact in reply["facts"]:
            require(isinstance(fact, dict) and set(fact) == {"text", "evidence"}, "事实需要 text/evidence")
            text(fact["text"], "事实")
            cited = string_list(fact["evidence"], "事实依据", nonempty=True)
            require(set(cited) <= seen, "事实引用了本段以外或不存在的原文")

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
            while pending:
                messages = self.gap_request(payload, extracted, facts, pending, errors)
                reply = self._complete(messages, gap_fill.PURPOSE, validate=gap_fill.check_shape)
                facts, errors = gap_fill.apply_fill(facts, reply, pending, aliases, texts)
                pending = [g for g in pending if g["ref"] in errors]
                signature = digest([reply, errors])
                if pending and signature in rejected:
                    raise SearchLoopError("模型重复返回相同无效内容且未取得进展；未写入替代结果")
                rejected.add(signature)
        filled = {"facts": facts}
        self.extraction_points(payload, filled)
        # Preserve the last accepted request for checkpoints.
        return {**filled, "conversation": [*messages, {"role": "assistant", "content": encode(reply)}]}

    @classmethod
    def gap_request(cls, payload, extracted, facts, gaps, errors=None):
        """One batch: a single-part session continues its conversation; a long one sends only the surroundings."""
        if len(cls.extraction_parts(payload)) == 1:
            aliases, _ = cls.source_view(payload)
            indexes = range(len(facts))
            return gap_fill.gap_messages(cls.conversation(payload, extracted), facts, gaps, indexes, aliases,
                                         errors=errors)
        _, sources = cls.source_view(payload)
        shown, indexes = gap_fill.surroundings(sources, facts, gaps)
        refs = {s["ref"] for s in shown} | {r for i in indexes for r in facts[i]["evidence"]}
        return gap_fill.gap_messages(cls.extraction_messages(payload)[:1], facts, gaps, indexes, refs,
                                     paragraphs=cls.session_blocks(shown), errors=errors)

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
        aliases, _ = cls.source_view(payload)
        require(isinstance(extracted, dict), "抽取结果必须是对象")
        facts = extracted.get("facts")
        require(isinstance(facts, list), "facts 必须是数组")
        points = []
        for index, fact in enumerate(facts):
            require(isinstance(fact, dict) and set(fact) == {"text", "evidence"}, "事实需要 text/evidence")
            text(fact["text"], "事实")
            refs = string_list(fact["evidence"], "事实依据", nonempty=True)
            require(set(refs) <= set(aliases), "事实引用了不存在的原文")
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

