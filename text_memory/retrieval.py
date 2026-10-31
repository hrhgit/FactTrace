"""Exact-text vector cache and configured local or HTTP embedding provider."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import threading

from .errors import ConfigurationError, ModelError
from .model import encode
from .schema import digest, require
from .store import MemoryStore

STOPWORDS = {"w" + word for word in """a an and are as at be been but by can could did do does for from had has have
    how i if in is it its me my no not of on or our so that the their them then there these they this to too us
    was we were what when where which who why will with would you your about also just like some any""".split()}


class Searcher:
    def __init__(self, store: MemoryStore, embedder=None, *, floor: float | None = None):
        if embedder is not None:
            require(isinstance(floor, (int, float)) and not isinstance(floor, bool) and -1 <= floor <= 1,
                    "向量检索下限 floor 必须是 -1 到 1 之间的数")
        self.store = store
        self.embedder = embedder
        self.floor = float(floor) if embedder is not None else None
        self.identity = ({"retriever": "hybrid", **embedder.identity, "floor": self.floor} if embedder is not None
                         else {"retriever": "unconfigured"})
        if embedder is not None:
            self.model_key = digest(embedder.identity)
            store.db.execute("""CREATE TABLE IF NOT EXISTS vectors (
                model TEXT NOT NULL, text_hash TEXT NOT NULL, vector TEXT NOT NULL,
                PRIMARY KEY(model, text_hash))""")

    def _vectors(self, texts: list[str]) -> list[list[float]]:
        """Embeddings cached by exact text, so unchanged records are encoded once."""
        hashes = [digest(value) for value in texts]
        cached = {}
        for ident in set(hashes):
            row = self.store.db.execute("SELECT vector FROM vectors WHERE model=? AND text_hash=?",
                                        (self.model_key, ident)).fetchone()
            if row:
                cached[ident] = json.loads(row["vector"])
        missing = [(ident, value) for ident, value in dict(zip(hashes, texts)).items() if ident not in cached]
        if missing:
            vectors = self.embedder.encode([value for _, value in missing])
            require(len(vectors) == len(missing), "向量数量与文本数量不一致")
            for (ident, _), vector in zip(missing, vectors):
                cached[ident] = vector
                self.store.db.execute("INSERT OR IGNORE INTO vectors VALUES(?,?,?)",
                                      (self.model_key, ident, encode(vector)))
        return [cached[ident] for ident in hashes]


class SubprocessEmbedder:
    """A long-lived worker in another Python environment that already has the model offline."""

    def __init__(self, python: str, model: str, revision: str | None = None):
        self.python, self.model, self.revision = python, model, revision
        self.identity = {"model": model, "revision": revision}
        self.lock = threading.Lock()
        self.process: subprocess.Popen | None = None

    def _start(self):
        worker = Path(__file__).with_name("embed_worker.py")
        command = [self.python, "-I", str(worker), self.model] + ([self.revision] if self.revision else [])
        try:
            self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                            stderr=subprocess.PIPE, text=True,
                                            env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
        except OSError as error:
            raise ConfigurationError("无法启动向量模型进程：" + str(error)) from error
        ready = self.process.stdout.readline()
        if json.loads(ready or "{}").get("ready") is not True:
            raise ConfigurationError("向量模型不可用：" + self.process.stderr.read()[-500:])

    def encode(self, texts: list[str]) -> list[list[float]]:
        with self.lock:
            if self.process is None or self.process.poll() is not None:
                self._start()
            self.process.stdin.write(json.dumps({"texts": texts}, ensure_ascii=False) + "\n")
            self.process.stdin.flush()
            line = self.process.stdout.readline()
            if not line:
                raise ModelError("向量模型进程意外退出：" + self.process.stderr.read()[-500:])
            reply = json.loads(line)
            if "error" in reply:
                raise ModelError("向量模型出错：" + reply["error"])
            return reply["vectors"]

    def close(self):
        if self.process is not None and self.process.poll() is None:
            self.process.stdin.close()
            self.process.wait()


def embedder_from_env() -> SubprocessEmbedder | None:
    """MEMORY_EMBED_PYTHON, MEMORY_EMBED_MODEL and MEMORY_EMBED_FLOOR select semantic candidates.

    All unset returns None; formal lookup then reports missing vectors. The floor is
    calibrated per model; use embed_floor_from_env alongside this.
    """
    python, model = os.environ.get("MEMORY_EMBED_PYTHON"), os.environ.get("MEMORY_EMBED_MODEL")
    base, key = os.environ.get("MEMORY_EMBED_API_BASE"), os.environ.get("MEMORY_EMBED_API_KEY")
    if base or key:
        if python:
            raise ConfigurationError("远程向量与本地 MEMORY_EMBED_PYTHON 不能同时配置")
        embed_floor_from_env()
        from .http_embeddings import HttpEmbedder
        try:
            batch_size = int(os.environ.get("MEMORY_EMBED_BATCH_SIZE", "10"))
        except ValueError:
            raise ConfigurationError("MEMORY_EMBED_BATCH_SIZE 必须为正整数") from None
        from .usage import usage_path_from_env
        return HttpEmbedder(base, key, model, batch_size=batch_size,
                            usage_path=usage_path_from_env(), source=os.environ.get("MEMORY_EMBED_KEY_SOURCE") or None)
    if not python and not model:
        return None
    if not (python and model):
        raise ConfigurationError("向量检索需要同时配置 MEMORY_EMBED_PYTHON 与 MEMORY_EMBED_MODEL")
    embed_floor_from_env()
    return SubprocessEmbedder(python, model, os.environ.get("MEMORY_EMBED_REVISION") or None)


def embed_floor_from_env() -> float:
    value = os.environ.get("MEMORY_EMBED_FLOOR")
    if not value:
        raise ConfigurationError("向量检索需要配置 MEMORY_EMBED_FLOOR（按模型实测的相似度下限）")
    try:
        return float(value)
    except ValueError as error:
        raise ConfigurationError("MEMORY_EMBED_FLOOR 必须是数字") from error


def candidate_identity(embedder, floor=None) -> dict:
    if embedder is None:
        return {"retriever": "unconfigured"}
    return {"retriever": "question-text-bm25-time-v1", **embedder.identity, "floor": float(floor)}
