"""Collection Add/Search adapter, using the existing authenticated HTTP transport."""
from ..schema import digest, require, text
from ..errors import NotFoundError
from ..store import Scope
from .engine import CollectionEngine
from .store import CollectionStore

from .answer_context import render_fact_context


class CollectionApplication:
    def __init__(self, database, model_factory, *, embedder=None, floor=None, fact_limit=None):
        self.fact_limit = fact_limit
        self.database, self.model_factory = database, model_factory
        self.embedder, self.floor = embedder, floor
        with CollectionStore(database):
            pass

    def handle(self, path, body):
        require(isinstance(body, dict), "请求必须是 JSON 对象")
        if path not in {"/add", "/search"}:
            raise NotFoundError("没有这个接口")
        if path == "/search":
            require(type(body.get("top_k")) is int and body["top_k"] > 0,
                    "AML Search 请求需要正整数 top_k")
        with CollectionStore(self.database) as store:
            # Add extracts facts with the memory model; Search ranks by the full question, and the model only marks question attributes.
            model = self.model_factory()
            engine = CollectionEngine(store, model, embedder=self.embedder, floor=self.floor)
            if path == "/add":
                result = engine.add(body)
                return {k: result[k] for k in ("success", "request_id", "user_id", "session_id")}
            result, data = self._search(store, engine, body)
            return {"data": data}

    def search(self, body):
        """The /search reply's data together with the full search result (evaluation runners read both)."""
        with CollectionStore(self.database) as store:
            engine = CollectionEngine(store, self.model_factory(), embedder=self.embedder, floor=self.floor)
            return self._search(store, engine, body)

    def _search(self, store, engine, body):
        result = engine.query(text(body.get("query"), "query"),
                              Scope(body.get("user_id"), body.get("scope_id", "default"), body.get("as_of")),
                              question_date=body.get("question_date"),
                              fact_limit=self.fact_limit)
        if not result["evidence"]:
            return result, []
        content = render_fact_context(store, Scope(body["user_id"], body.get("scope_id", "default"),
                                                  result.get("as_of")), result)
        # A single complete packet avoids losing necessary members to candidate top_k truncation.
        return result, [{"id": "collection_" + digest([body["user_id"], content]), "content": content}]
