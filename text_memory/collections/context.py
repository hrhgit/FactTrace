"""Attach the confirmed session-fact context after retrieval is finished.

This adapter supplies stored fact order, vectors to
the existing algorithm. It never changes search scores or reads raw paragraphs
into the answer packet. Derived profiles are scoped by all their exact inputs.
"""
import json
import math

from ..context_completion import complete_context
from ..context_segmentation import prepare_segments
from ..context_windows import same_segment_support
from ..errors import ConfigurationError, ModelError
from ..schema import digest, require

POLICY = "collections-session-fact-context-v1"


class SessionFactContext:
    def __init__(self, searcher):
        self.searcher, self.store = searcher, searcher.store
        self.views, self.profiles = {}, {}

    def _sessions(self, scope, facts):
        sources = {s["id"]: s for s in self.store.records(scope, "source")}
        receipts = [json.loads(row[0]) for row in self.store.db.execute(
            "SELECT result FROM requests WHERE user_id=? AND scope_id=?",
            (scope.user_id, scope.scope_id))]
        receipts = sorted((r for r in receipts if r["revision"] <= scope.as_of),
                          key=lambda r: r["revision"])
        # Receipts preserve extraction order, unlike hashed fact IDs or a sort
        # by source position (several extracted facts can cite one message).
        order = {}
        for receipt in receipts:
            for ident in receipt["point_ids"]:
                order.setdefault(ident, len(order))
        sessions = {}
        for source in sorted(sources.values(), key=lambda s: (s["revision"], s["position"])):
            session = sessions.setdefault(source["session_id"], {
                "session_id": source["session_id"], "request_ids": [],
                "participants": [], "facts": [], "speakers": []})
            if source["request_id"] not in session["request_ids"]:
                session["request_ids"].append(source["request_id"])
            speaker = source["message"].get("speaker")
            if speaker and speaker not in session["participants"]:
                session["participants"].append(speaker)
        for fact in sorted(facts.values(), key=lambda f: order[f["id"]]):
            local = {}
            for ref in fact["evidence"]:
                source = sources[ref.rsplit(".", 1)[0]]
                local.setdefault(source["session_id"], source["message"].get("speaker"))
            for session_id, speaker in local.items():
                sessions[session_id]["facts"].append(fact)
                sessions[session_id]["speakers"].append(speaker)
        return [s for s in sessions.values() if s["facts"]]

    def _profile(self, scope, session):
        facts = session["facts"]
        ids = [f["id"] for f in facts]
        request_ids = session["request_ids"]
        seed_id = request_ids[0] if len(request_ids) == 1 else digest(request_ids)
        fingerprint = digest({"policy": POLICY, "user": scope.user_id, "scope": scope.scope_id,
            "session": session, "links": [], "embedding": self.searcher.embedder.identity})
        if fingerprint in self.profiles:
            return self.profiles[fingerprint]
        self.store.db.execute("CREATE TABLE IF NOT EXISTS collection_context_cache "
                              "(fingerprint TEXT PRIMARY KEY, profile TEXT NOT NULL)")
        row = self.store.db.execute("SELECT profile FROM collection_context_cache WHERE fingerprint=?",
                                    (fingerprint,)).fetchone()
        if row:
            self.profiles[fingerprint] = json.loads(row[0])
            return self.profiles[fingerprint]
        vectors = self.searcher._vectors([f["text"] for f in facts])
        normalized, width = [], len(vectors[0])
        for vector in vectors:
            if not width or len(vector) != width or not all(math.isfinite(x) for x in vector):
                raise ModelError("会话上下文向量不完整或无效")
            norm = math.sqrt(sum(x*x for x in vector))
            if norm == 0:
                raise ModelError("会话上下文向量不能为零")
            normalized.append([x/norm for x in vector])
        kernel = [[sum(a*b for a, b in zip(x, y)) for y in normalized] for x in normalized]
        fit = prepare_segments(kernel, seed_id)
        support = same_segment_support(kernel, fit["k"], fit["temperature"])["same_segment"]
        windows = complete_context(fit["blocks"], support, kernel, [f["text"] for f in facts],
                                   participants=session["participants"], speakers=session["speakers"])
        result = {"session_id": session["session_id"], "ordered_fact_ids": ids,
                  **fit, "windows": windows}
        self.store.db.execute("INSERT OR IGNORE INTO collection_context_cache VALUES (?,?)",
                              (fingerprint, json.dumps(result, ensure_ascii=False)))
        self.profiles[fingerprint] = result
        return result

    def assemble(self, scope, hit_ids):
        scope = self.store.snapshot(scope)
        hits = list(dict.fromkeys(hit_ids))
        key = (scope.user_id, scope.scope_id, scope.as_of)
        if key not in self.views:
            facts = {f["id"]: f for f in self.store.records(scope, "point")
                     if f.get("state", "active") != "superseded"}
            self.views[key] = (facts, self._sessions(scope, facts))
        facts, all_sessions = self.views[key]
        require(set(hits) <= set(facts), "上下文起点必须是当前范围内的有效事实")
        sessions = all_sessions if hits else []
        selected, contexts = dict.fromkeys(hits), []
        for session in sessions:
            ids = [f["id"] for f in session["facts"]]
            session_hits = [i for i in hits if i in set(ids)]
            if not session_hits:
                continue
            if self.searcher.embedder is None:
                # If all facts are already selected, no completion can change
                # the returned material. Otherwise the confirmed method needs
                # vectors; do not invent a lexical substitute or silently skip.
                if set(ids) <= set(hits):
                    contexts.append({"session_id": session["session_id"],
                                     "status": "already_complete", "hit_ids": session_hits,
                                     "fact_ids": ids})
                    continue
                raise ConfigurationError("会话事实上下文需要配置向量模型，不能退回纯关键词补全")
            profile = self._profile(scope, session)
            for hit in session_hits:
                window = profile["windows"][ids.index(hit)]
                returned = [ids[n-1] for n in window["fact_numbers"]]
                selected.update(dict.fromkeys(returned))
                contexts.append({"session_id": session["session_id"], "hit_id": hit,
                                 "fact_ids": returned, "core_fact_ids": ids[window["core"][0]-1:window["core"][1]],
                                 "supplement_fact_ids": list(dict.fromkeys(
                                     ids[n-1] for p in window["supplements"] for n in p["fact_numbers"]))})
        evidence = [{"id": ident, "text": facts[ident]["text"], "source_ids": facts[ident]["evidence"]}
                    for ident in selected]
        added = [i for i in selected if i not in set(hits)]
        return {"context_policy": POLICY, "matched_fact_ids": hits, "context_fact_ids": added,
                "evidence": evidence, "sources": [], "read_fact_count": len(evidence), "read_source_count": 0,
                "context_completion": {"policy": POLICY, "contexts": contexts,
                                       "added_fact_count": len(added), "recursive": False}}
