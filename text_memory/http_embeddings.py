"""Remote text embeddings; explicit configuration, no truncation or substitute vectors."""
import math

import httpx

from .errors import ConfigurationError, ModelError
from .usage import UsageLedger, default_source, key_identity


class HttpEmbedder:
    def __init__(self, base_url, api_key, model, *, batch_size=10, transport=None,
                 usage_path=None, source=None):
        if not all(isinstance(v, str) and v.strip() for v in (base_url, api_key, model)):
            raise ConfigurationError("远程向量需要 API_BASE、API_KEY 和 MODEL")
        if type(batch_size) is not int or batch_size <= 0:
            raise ConfigurationError("MEMORY_EMBED_BATCH_SIZE 必须为正整数")
        self.base_url, self.api_key, self.model = base_url, api_key, model
        self.batch_size, self.transport = batch_size, transport
        self.identity = {"model": model, "backend": "http-embeddings"}
        self.usage = UsageLedger(usage_path)
        self.key_id = key_identity(base_url, api_key)
        self.usage.register(self.key_id, source or default_source(base_url), explicit=source is not None)

    def encode(self, texts):
        if not isinstance(texts, list) or any(not isinstance(s, str) or not s.strip() for s in texts):
            raise ModelError("向量输入必须为非空文字组成的数组")
        vectors, dimension = [], None
        with httpx.Client(timeout=None, transport=self.transport) as client:
            for offset in range(0, len(texts), self.batch_size):
                batch = texts[offset:offset + self.batch_size]
                attempt_id = self.usage.begin(self.key_id, "MEMORY_EMBED", self.model, "embedding")
                response_usage, status = None, "transport_error"
                try:
                    try:
                        response = client.post(self.base_url.rstrip("/") + "/embeddings",
                            headers={"Authorization": "Bearer " + self.api_key},
                            json={"model": self.model, "input": batch, "encoding_format": "float"})
                    except httpx.HTTPError:
                        raise ModelError("远程向量请求连接失败；未返回替代向量") from None
                    status = "http_error" if not response.is_success else "invalid_response"
                    try:
                        payload = response.json()
                    except ValueError:
                        if response.is_success:
                            raise
                        payload = None
                    response_usage = payload.get("usage") if isinstance(payload, dict) else None
                    if not response.is_success:
                        raise ModelError(f"远程向量请求失败（HTTP {response.status_code}）；未返回替代向量")
                    data = payload["data"]
                    if not isinstance(data, list) or len(data) != len(batch):
                        raise ValueError()
                    indexed = {}
                    for item in data:
                        index, vector = item["index"], item["embedding"]
                        if type(index) is not int or not 0 <= index < len(batch) or index in indexed:
                            raise ValueError()
                        if (not isinstance(vector, list) or not vector or
                            any(type(v) not in (int, float) or not math.isfinite(v) for v in vector)):
                            raise ValueError()
                        dimension = len(vector) if dimension is None else dimension
                        if len(vector) != dimension:
                            raise ValueError()
                        norm = math.hypot(*vector)
                        if not math.isfinite(norm) or norm == 0:
                            raise ValueError()
                        indexed[index] = [float(v) / norm for v in vector]
                    vectors.extend(indexed[i] for i in range(len(batch)))
                    status = "completed"
                except (ValueError, TypeError, KeyError, OverflowError):
                    raise ModelError("远程向量响应的数量、顺序或数值无效；未返回替代向量") from None
                finally:
                    self.usage.finish(attempt_id, response_usage, status)
        return vectors

    def close(self):
        self.usage.close()
