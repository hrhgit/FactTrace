"""Direct fact search: the full question ranks all facts (text, BM25, named time), then session context is attached."""
from copy import deepcopy
import time

from ..schema import text
from ..errors import ModelError
from .question_flags import BROAD, CHRONOLOGICAL, check_question, material_notes

DIRECT_POLICY = "direct-facts-question-flags-v4"
DEFAULT_POLICY = DIRECT_POLICY
# Facts kept after ranking; there is no score threshold. Try 16 first by
# project rule (AGENTS.md); 32 and 64 are explicit comparisons.
DIRECT_LIMIT = 16


class DirectFactsQuery:
    """The full question ranks all active facts; the top facts get session context.

    One model call marks question attributes (when the engine has a model); ranking itself makes no model call.
    No collection read, no score threshold, and no source text in the packet.
    """

    def __init__(self, engine):
        self.engine, self.store = engine, engine.store

    def step(self, name, arguments, execute):
        entry = {"tool": name, "arguments": deepcopy(arguments)}
        self.engine.trace.append(entry)
        start = time.monotonic()
        try:
            result = execute()
            entry["result"] = deepcopy(result)
            return result
        except Exception as error:
            entry["error"] = {"type": type(error).__name__, "message": str(error)}
            raise
        finally:
            entry["elapsed_seconds"] = time.monotonic() - start

    def run(self, question, scope, *, question_date=None, fact_limit=None):
        limit = DIRECT_LIMIT if fact_limit is None else fact_limit
        text(question, "question")
        self.scope = self.store.snapshot(scope)
        if self.engine.model is None:
            raise ModelError("正式查找需要模型判断问题所需的事实材料")
        flags, flags_error = self.step("mark_question", {"question": question},
                                       lambda: check_question(self.engine.model, question))
        if any(flags.get(k) for k in BROAD):
            limit *= 2
        found = self.step("rank_question_facts", {"question": question, "limit": limit},
                          lambda: self.engine.searcher.rank_question_facts(self.scope, question, limit=limit))
        hits = [fact["id"] for fact in found["facts"]]
        packet = (self.step("complete_fact_context", {"fact_ids": hits},
                            lambda: self.engine.searcher.complete_fact_context(
                                self.scope, hits))
                  if hits else {"evidence": [], "sources": [], "matched_fact_ids": []})
        return {"search_policy": DIRECT_POLICY, "retriever": dict(self.engine.searcher.identity),
                "question_date": question_date, "as_of": self.scope.as_of,
                "question_time_ranges": found["question_time_ranges"],
                "fact_limit": limit, "question_flags": flags, "question_flags_error": flags_error,
                "chronological": any(flags.get(k) for k in CHRONOLOGICAL), "notes": material_notes(flags),
                "coverage": "按完整问题排序的前列事实及其会话事实上下文；不附原文，未证明已查全。",
                **packet}
