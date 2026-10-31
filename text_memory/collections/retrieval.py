"""Full-question fact ranking and the confirmed session-fact context completion."""
import math

from ..retrieval import Searcher, STOPWORDS
from ..schema import require, text
from ..store import terms

QUESTION_RANKING = "question-text-bm25-time-v1"
# Added to a fact's score when its time overlaps a period the question names (Mem0's boost ceiling).
TIME_BONUS = 0.5
BM25_K1, BM25_B = 1.2, 0.75


def unit(vector):
    norm = math.sqrt(sum(x * x for x in vector)) or 1.0
    return [x / norm for x in vector]


def bm25_sigmoid(query, documents):
    """BM25 of the query against each document, squashed to [0, 1) with Mem0's query-length sigmoid.

    Terms are the project's bilingual terms() minus STOPWORDS (distinct terms, so term frequency is 1); offline
    this matched Mem0's spaCy lemmas (outputs/mem0-ranking-absorb-20261029/lemma-check.json). 0 when no term is shared.
    """
    asked = [t for t in terms(query) if t not in STOPWORDS]
    docs = [{t for t in terms(d) if t not in STOPWORDS} for d in documents]
    if not asked or not docs:
        return [0.0] * len(docs)
    count, mean = len(docs), (sum(map(len, docs)) / len(docs)) or 1.0
    frequency = {t: sum(t in d for d in docs) for t in asked}
    n = len(asked)
    midpoint, steepness = ((5.0, .7) if n <= 3 else (7.0, .6) if n <= 6 else (9.0, .5) if n <= 9
                           else (10.0, .5) if n <= 15 else (12.0, .5))
    out = []
    for d in docs:
        raw = sum(math.log(1 + (count - frequency[t] + .5) / (frequency[t] + .5))
                  * (BM25_K1 + 1) / (1 + BM25_K1 * (1 - BM25_B + BM25_B * len(d) / mean))
                  for t in asked if t in d)
        out.append(1 / (1 + math.exp(-steepness * (raw - midpoint))) if raw > 0 else 0.0)
    return out


REPORTING_TERMS = {"w" + word for word in """said says told asked replied shared photo photos picture pictures
image caption search query conversation chat message dated shown showing included""".split()}


class CollectionSearcher(Searcher):
    def __init__(self, store, embedder=None, *, floor=None):
        super().__init__(store, embedder, floor=floor)
        self.identity.update(policy=QUESTION_RANKING, question_ranking=QUESTION_RANKING, time_bonus=TIME_BONUS)

    def rank_question_facts(self, scope, question, *, limit):
        """Formal search ranking: the full question against every active fact's text; keep the top ``limit``.

        score = cosine(question, fact text) + sigmoid(BM25(question, fact text)) + TIME_BONUS when the fact's
        time (source conversation dates, resolved relative dates, written dates) overlaps a period the question
        names. Scores are added, not rank-fused; only full fact text is compared (offline
        and re-answer checks: outputs/mem0-ranking-absorb-20261029, outputs/time-lookup-luna-answers-20261029).
        There is no threshold: only the order and the limit decide what is kept.
        """
        from .time_ranges import fact_ranges, overlaps, question_ranges
        from .answer_context import source_date
        require(type(limit) is int and limit > 0, "排序保留条数必须是正整数")
        text(question, "question")
        require(self.embedder is not None, "直接查事实需要配置向量模型")
        facts = sorted((fact for fact in self.store.records(scope, "point")
                        if fact.get("state", "active") != "superseded"), key=lambda fact: fact["id"])
        periods = question_ranges(question)
        if not facts:
            return {"facts": [], "question_time_ranges": [], "limit": limit,
                    "retrieval_policy": QUESTION_RANKING}
        vectors = self._vectors([question, *(fact["text"] for fact in facts)])
        asked = unit(vectors[0])
        semantic = [sum(a * b for a, b in zip(asked, unit(v))) for v in vectors[1:]]
        lexical = bm25_sigmoid(question, [fact["text"] for fact in facts])
        dated = [False] * len(facts)
        if periods:
            sources = {s["id"]: s for s in self.store.records(scope, "source")}
            for n, fact in enumerate(facts):
                days = [source_date(sources[ref.rsplit(".", 1)[0]]["message"]) for ref in fact["evidence"]
                        if ref.rsplit(".", 1)[0] in sources]
                spans = fact_ranges(fact["text"], [d for d in dict.fromkeys(days) if d])
                dated[n] = any(overlaps(span, period) for span in spans for period in periods)
        hits = [{"kind": "point", "id": fact["id"], "semantic": semantic[n], "bm25": lexical[n], "time_match": dated[n],
                 "score": semantic[n] + lexical[n] + (TIME_BONUS if dated[n] else 0.0)} for n, fact in enumerate(facts)]
        hits.sort(key=lambda hit: (-hit["score"], hit["id"]))
        facts_by_id = {fact["id"]: fact for fact in facts}
        found = {"facts": [hit | {"rank": rank, "text": facts_by_id[hit["id"]]["text"]}
                           for rank, hit in enumerate(hits[:limit], 1)]}
        found.update(question_time_ranges=[[str(a), str(b)] for a, b in periods], limit=limit,
                     retrieval_policy=QUESTION_RANKING)
        return found

    def complete_fact_context(self, scope, fact_ids):
        from .context import SessionFactContext
        if not hasattr(self, "_context_assembler"):
            self._context_assembler = SessionFactContext(self)
        return self._context_assembler.assemble(scope, fact_ids)
