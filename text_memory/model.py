"""OpenAI-compatible HTTP boundary with bounded retries and channel failover."""
from __future__ import annotations

from dataclasses import dataclass, replace
import json
import os
import random
import re
import time
from typing import Callable, Protocol, Sequence

import httpx

from .errors import ConfigurationError, InvalidOutputError, ModelError
from .usage import (PRICE_DATE, PRICE_SOURCE, REFERENCE_PRICES, ChannelConcurrencyError, UsageLedger, UsageLimitError,
                    default_source, key_identity, reference_price, usage_path_from_env)


def _strip_json_fence(content: str) -> str:
    candidate = content.strip()
    lines = candidate.splitlines()
    if lines and lines[0].strip() in {"```json", "```"}:
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        candidate = "\n".join(lines)
    return candidate


def _one_missing_json_closer(candidate: str) -> str | None:
    """Repair only one missing final JSON closer, outside quoted strings."""
    stack = []
    in_string = False
    escaped = False
    pairs = {"{": "}", "[": "]"}
    closing = {"}": "{", "]": "["}
    for char in candidate:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in pairs:
            stack.append(char)
        elif char in closing:
            if not stack or stack[-1] != closing[char]:
                return None
            stack.pop()
    if in_string or len(stack) != 1:
        return None
    return candidate + pairs[stack[0]]


# Chat-template/protocol tokens some channels leak after an otherwise complete
# reply.  These are exact, fixed suffixes: stripping one after a complete JSON
# object cannot change the model-produced JSON value.  Keep this list narrow;
# arbitrary trailing prose is still an output failure.
END_OF_TURN_MARKERS = (
    "</|assistant|>",
    "</|im_end|>",
    "</assistant>",
    "</final>",
    "</｜｜DSML｜｜ parameter>\n</｜｜DSML｜｜ invoke>\n</｜｜DSML｜｜ calls>",
)
# Backward-compatible name for callers that imported the original constant.
END_OF_TURN_MARKER = END_OF_TURN_MARKERS[0]


def _without_extra_json_closers(candidate: str) -> str | None:
    """Remove a short suffix of extra closers or one exact terminal marker.

    The JSON decoder must first find a complete object.  Only the fixed
    protocol markers above and a short suffix made of JSON closing characters
    can then be removed; arbitrary trailing text remains invalid.
    """
    try:
        result, end = json.JSONDecoder().raw_decode(candidate)
    except ValueError:
        return None
    if not isinstance(result, dict):
        return None
    suffix = candidate[end:].strip()
    if not suffix:
        return None
    for marker in END_OF_TURN_MARKERS:
        if suffix == marker:
            suffix = ""
            break
        if suffix.endswith(marker):
            suffix = suffix[:-len(marker)].strip()
            break
    closers = suffix
    if len(closers) > 8 or any(char not in ']}"' for char in closers):
        return None
    return candidate[:end]


def decode_json_object_with_meta(content: str) -> tuple[dict, bool]:
    """Decode JSON and apply only the approved syntax repairs.

    Removing a surrounding code fence, adding one missing final closer, and
    removing a short suffix made only of extra JSON closing characters, and
    removing one exact terminal protocol marker, do not invent model content.
    The suffix repair requires a complete object, a non-empty suffix of at
    most eight characters, and only ``]``, ``}``, and ``"`` in that suffix.
    Any other malformed or structurally invalid output remains an error and is
    retried as model output.
    """
    if not content.strip():
        raise InvalidOutputError(content, "模型没有返回非空 JSON 内容")
    candidate = _strip_json_fence(content)
    try:
        result = json.loads(candidate)
    except ValueError as error:
        repaired = _one_missing_json_closer(candidate)
        if repaired is None:
            repaired = _without_extra_json_closers(candidate)
        if repaired is None:
            raise InvalidOutputError(content, "模型回复无法解析为 JSON") from error
        try:
            result = json.loads(repaired)
        except ValueError:
            raise InvalidOutputError(content, "模型回复无法解析为 JSON") from error
        repaired_output = True
    else:
        repaired_output = candidate != content.strip()
    if not isinstance(result, dict):
        raise InvalidOutputError(content, "模型输出必须是 JSON 对象")
    return result, repaired_output


def decode_json_object(content: str) -> dict:
    """Decode a JSON object with the approved syntax repairs."""
    result, _ = decode_json_object_with_meta(content)
    return result


class JsonModel(Protocol):
    def complete(self, messages: list[dict], *, purpose: str) -> dict: ...

    def complete_text(self, messages: list[dict], *, purpose: str, temperature: float | None = None) -> str: ...


@dataclass(frozen=True)
class ModelConfig:
    base_url: str
    api_key: str
    model: str
    reasoning_effort: str | None = None
    fallbacks: tuple["ModelConfig", ...] = ()
    strategy: str = "failover"
    source: str | None = None
    channel_group: str = "MEMORY"
    usage_path: str | None = None
    exhausted_fallback: int | None = None

    def __post_init__(self):
        if self.strategy not in {"failover", "round_robin"}:
            raise ConfigurationError("模型渠道策略必须是 failover 或 round_robin")
        if self.source is not None and (not isinstance(self.source, str) or not self.source.strip()):
            raise ConfigurationError("模型渠道来源名称不能为空")
        if self.exhausted_fallback is not None and (
                type(self.exhausted_fallback) is not int
                or not 1 <= self.exhausted_fallback <= len(self.channels)):
            raise ConfigurationError("额度耗尽后的兜底渠道编号无效")

    @property
    def channels(self) -> tuple["ModelConfig", ...]:
        return (self, *self.fallbacks)

    @classmethod
    def from_env(cls, prefix: str = "MEMORY") -> "ModelConfig":
        def channel(suffix: str = "") -> "ModelConfig":
            keys = [f"{prefix}_{key}{suffix}" for key in ("API_BASE", "API_KEY", "MODEL")]
            missing = [key for key in keys if not os.environ.get(key)]
            if missing:
                raise ConfigurationError("请配置模型：" + ", ".join(missing))
            return cls(*(os.environ[key] for key in keys),
                       reasoning_effort=os.environ.get(f"{prefix}_REASONING_EFFORT{suffix}") or None,
                       source=(os.environ.get(f"{prefix}_KEY_SOURCE{suffix}") or "").strip() or None)

        primary = channel()
        marker = re.compile(rf"^{re.escape(prefix)}_(?:API_BASE|API_KEY|MODEL|REASONING_EFFORT|KEY_SOURCE)_(\d+)$")
        numbers = sorted({int(match.group(1)) for key in os.environ
                          if (match := marker.fullmatch(key)) is not None})
        if any(number < 2 for number in numbers):
            raise ConfigurationError("备用模型渠道编号从 2 开始")
        strategy = os.environ.get(f"{prefix}_CHANNEL_STRATEGY", "failover").strip().lower()
        fallback_value = os.environ.get(f"{prefix}_EXHAUSTED_FALLBACK", "").strip()
        if fallback_value:
            try:
                exhausted_fallback = int(fallback_value)
            except ValueError as error:
                raise ConfigurationError("额度耗尽后的兜底渠道编号无效") from error
        else:
            exhausted_fallback = None
        channel_count = 1 + len(numbers)
        if exhausted_fallback is not None and not 1 <= exhausted_fallback <= channel_count:
            raise ConfigurationError("额度耗尽后的兜底渠道编号无效")
        return replace(primary, fallbacks=tuple(channel(f"_{number}") for number in numbers),
                       strategy=strategy, channel_group=prefix, usage_path=usage_path_from_env(),
                       exhausted_fallback=exhausted_fallback)


class ModelChannelPool:
    def __init__(self, config: ModelConfig, *, usage_ledger: UsageLedger | None = None,
                 usage_path: str | os.PathLike | None = None):
        if any(channel.fallbacks for channel in config.fallbacks):
            raise ValueError("fallback channels cannot have nested fallbacks")
        if usage_ledger is not None and usage_path is not None:
            raise ValueError("usage_ledger and usage_path cannot both be set")
        self.config = config
        self.channels = config.channels
        self._key_ids = tuple(key_identity(channel.base_url, channel.api_key) for channel in self.channels)
        self.usage = usage_ledger or UsageLedger(usage_path if usage_path is not None else config.usage_path)
        self.usage_ledger = self.usage
        self._sources = tuple(self.usage.register(key_id, channel.source or default_source(channel.base_url),
                                                 explicit=channel.source is not None)
                              for key_id, channel in zip(self._key_ids, self.channels))
        self.usage.settings(config.channel_group, config.strategy, len(self.channels))

    def channel_source(self, channel_index: int) -> str:
        if type(channel_index) is not int or not 1 <= channel_index <= len(self.channels):
            raise ConfigurationError("模型渠道编号无效")
        return self._sources[channel_index - 1]

    def begin(self, channel_index: int, purpose: str, *, exhausted_fallback=False) -> str:
        self.channel_source(channel_index)  # Validate before indexing.
        if exhausted_fallback and channel_index != self.config.exhausted_fallback:
            raise ConfigurationError("该渠道未配置为额度耗尽后的兜底")
        return self.usage.begin(self._key_ids[channel_index - 1], self.config.channel_group,
                                self.channels[channel_index - 1].model, purpose,
                                exhausted_fallback_keys=self._key_ids if exhausted_fallback else None)

    def exhausted_fallback_ready(self) -> bool:
        return self.config.exhausted_fallback is not None and self.usage.all_exhausted(self._key_ids)

    def selection(self) -> dict:
        settings = self.usage.settings(self.config.channel_group, self.config.strategy,
                                       len(self.channels), advance=True)
        channels = [(i, self.channels[i - 1]) for i in settings["priority"]]
        fallback_index = self.config.exhausted_fallback
        terminal_fallback = False
        if self.exhausted_fallback_ready():
            channels = [(fallback_index, self.channels[fallback_index - 1])]
            terminal_fallback = True
        return settings | {"channels": channels, "exhausted_fallback": fallback_index,
                           "terminal_fallback": terminal_fallback}

    def order(self) -> list[tuple[int, ModelConfig]]:
        # This snapshot remains valid for the whole request, including failover.
        return self.selection()["channels"]

    def priority(self) -> list[int]:
        return self.usage.settings(self.config.channel_group, self.config.strategy, len(self.channels))["priority"]

    def update_priority(self, priority: Sequence[int | str], *, strategy: str | None = None) -> dict:
        """Change future request order without interrupting in-flight calls."""
        if isinstance(priority, (str, bytes)) or not isinstance(priority, Sequence) or not priority:
            raise ConfigurationError("渠道优先级必须是非空数组")
        selected = []
        for item in priority:
            if type(item) is int and 1 <= item <= len(self.channels):
                index = item
            elif isinstance(item, str) and self._sources.count(item) == 1:
                index = self._sources.index(item) + 1
            else:
                raise ConfigurationError("渠道优先级包含未知或不明确的渠道")
            if index in selected:
                raise ConfigurationError("渠道优先级不能重复")
            selected.append(index)
        selected.extend(index for index in self.priority() if index not in selected)
        self.usage.update_settings(self.config.channel_group, priority=selected, strategy=strategy)
        return self.snapshot()

    set_priority = update_priority

    def update_limits(self, channel_index: int, **changes) -> dict:
        self.channel_source(channel_index)
        key_id = self._key_ids[channel_index - 1]
        current = self.usage.report(key_id)
        if changes.get("max_cost_usd", current["limits"]["max_cost_usd"]) is not None:
            model = changes.get("reference_model", current["reference_model"]) or self.channels[channel_index - 1].model
            if reference_price(model) is None:
                raise ConfigurationError("该渠道没有参考单价；请同时指定 reference_model")
        self.usage.update_limits(key_id, **changes)
        return self.snapshot()

    def update_concurrency(self, channel_index: int, max_concurrency: int | None) -> dict:
        """Change a channel's live request ceiling; zero disables new requests."""
        self.channel_source(channel_index)
        self.usage.update_limits(self._key_ids[channel_index - 1], max_concurrency=max_concurrency)
        return self.snapshot()

    def update_total_concurrency(self, max_concurrency: int | None) -> dict:
        self.usage.update_run_concurrency(self.config.channel_group, max_concurrency)
        return self.snapshot()

    def snapshot(self) -> dict:
        settings = self.usage.settings(self.config.channel_group, self.config.strategy, len(self.channels))
        channels = []
        for index, channel in enumerate(self.channels, start=1):
            report = self.usage.report(self._key_ids[index - 1])
            channels.append({"index": index, "model": channel.model, **report,
                             "source": self.channel_source(index)})
        return {"group": self.config.channel_group, "usage_database": str(self.usage.path) if self.usage.path else None,
                **settings, "channels": channels,
                "max_concurrency": self.usage.run_concurrency(self.config.channel_group, None),
                "exhausted_fallback": self.config.exhausted_fallback,
                "exhausted_fallback_active": self.config.exhausted_fallback is not None
                    and all(channel["exhausted"] for channel in channels),
                "pricing_reference": {"checked_on": PRICE_DATE, "source": PRICE_SOURCE,
                    "usd_per_million_tokens": {name: {"input": rates[0], "cached_input": rates[1], "output": rates[2]}
                                               for name, rates in REFERENCE_PRICES.items()}}}


class _ChannelRequestError(ModelError):
    pass


class HttpJsonModel:
    _RETRYABLE_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504})
    _NON_RETRYABLE_MARKERS = (
        "quota", "exceed quota", "insufficient", "balance", "billing", "plan", "1113",
    )

    def __init__(self, config: ModelConfig, *, transport: httpx.BaseTransport | None = None,
                 channel_pool: ModelChannelPool | None = None,
                 max_retries: int = 2, retry_base_delay_s: float = 0.5,
                 retry_jitter_s: float = 0.1, sleep: Callable[[float], None] | None = None,
                 usage_ledger: UsageLedger | None = None,
                 usage_path: str | os.PathLike | None = None):
        if channel_pool is not None and (usage_ledger is not None or usage_path is not None):
            raise ValueError("configure accounting on the shared channel pool")
        self.channel_pool = channel_pool or ModelChannelPool(config, usage_ledger=usage_ledger,
                                                             usage_path=usage_path)
        if self.channel_pool.config != config:
            raise ValueError("channel pool must use the same model config")
        self.config = config
        self.channels = self.channel_pool.channels
        self.usage = self.channel_pool.usage
        self.transport = transport
        self.calls: list[dict] = []
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if retry_base_delay_s < 0 or retry_jitter_s < 0:
            raise ValueError("retry delays must be non-negative")
        self.max_retries = max_retries
        self.retry_base_delay_s = retry_base_delay_s
        self.retry_jitter_s = retry_jitter_s
        self._sleep = sleep or time.sleep

    @classmethod
    def _is_retryable_http_error(cls, status: int, detail: str) -> bool:
        if status not in cls._RETRYABLE_STATUSES:
            return False
        lowered = detail.lower()
        return not any(marker in lowered for marker in cls._NON_RETRYABLE_MARKERS)

    def _retry_delay(self, retry_index: int) -> float:
        # retry_index is the index of the failed attempt (0 for the first one).
        return self.retry_base_delay_s * (2 ** retry_index) + random.uniform(0, self.retry_jitter_s)

    def _post(self, body: dict, messages: list[dict], purpose: str, finished: set[str], parse):
        """Try channels in configured order, with bounded retries within each channel."""
        selection = self.channel_pool.selection()
        channels = selection["channels"]
        last_error = None
        for failover_index, (channel_index, channel) in enumerate(channels):
            try:
                return self._post_channel(channel, channel_index, failover_index,
                                          body, messages, purpose, finished, parse,
                                          selection=selection)
            except _ChannelRequestError as error:
                last_error = error
                self.calls[-1]["failover_scheduled"] = failover_index + 1 < len(channels)
        # A failed HTTP attempt can consume the last remaining allowance. Check
        # again so that this same request can continue on the authorized key.
        if not selection["terminal_fallback"] and self.channel_pool.exhausted_fallback_ready():
            fallback_index = selection["exhausted_fallback"]
            self.calls[-1]["failover_scheduled"] = True
            try:
                return self._post_channel(self.channels[fallback_index - 1], fallback_index, len(channels),
                                          body, messages, purpose, finished, parse,
                                          selection=selection | {"terminal_fallback": True})
            except _ChannelRequestError as error:
                last_error = error
                self.calls[-1]["failover_scheduled"] = False
        if len(channels) == 1:
            raise last_error
        raise ModelError(f"{len(channels)} 个模型渠道均请求失败；最后错误：{last_error}") from last_error

    def _post_channel(self, channel: ModelConfig, channel_index: int, failover_index: int,
                      body: dict, messages: list[dict], purpose: str, finished: set[str], parse,
                      *, selection: dict):
        """Post to one channel; successful HTTP responses are never replayed elsewhere."""
        url = channel.base_url.rstrip("/") + "/chat/completions"
        channel_body = dict(body)
        channel_body["model"] = channel.model
        if channel.reasoning_effort is None:
            channel_body.pop("reasoning_effort", None)
        else:
            channel_body["reasoning_effort"] = channel.reasoning_effort
        for retry_index in range(self.max_retries + 1):
            record = {"purpose": purpose, "requested_model": channel.model,
                      "channel_index": channel_index, "channel_strategy": selection["strategy"],
                      "channel_generation": selection["generation"],
                      "channel_source": self.channel_pool.channel_source(channel_index),
                      "failover_index": failover_index, "retry_index": retry_index,
                      "retryable": False, "retry_delay_s": 0,
                      "messages": json.loads(json.dumps(messages, ensure_ascii=False))}
            request_parameters = {}
            if "temperature" in channel_body:
                request_parameters["temperature"] = channel_body["temperature"]
            if "reasoning_effort" in channel_body:
                request_parameters["reasoning_effort"] = channel_body["reasoning_effort"]
            if request_parameters:
                record["request_parameters"] = request_parameters
            self.calls.append(record)
            terminal_fallback = (selection.get("terminal_fallback")
                                 and channel_index == selection.get("exhausted_fallback"))
            if terminal_fallback:
                record["exhausted_fallback"] = True
            while True:
                try:
                    attempt_id = self.channel_pool.begin(channel_index, purpose,
                                                         exhausted_fallback=terminal_fallback)
                    break
                except ChannelConcurrencyError:
                    # A full channel is throttled, rather than treated as a
                    # provider failure. The controller can raise the limit or
                    # set it to zero while this request waits.
                    record["waiting_for_channel_slot"] = True
                    self._sleep(0.1)
                except UsageLimitError as error:
                    record.update(status="usage_limit", retry_scheduled=False)
                    raise _ChannelRequestError(str(error)) from error
            record["usage_attempt_id"] = attempt_id
            response_usage = None
            accounted = False

            def account():
                nonlocal accounted
                if not accounted:
                    accounted = True
                    try:
                        self.usage.finish(attempt_id, response_usage, record.get("status", "interrupted"))
                    except ConfigurationError:
                        record["accounting_error"] = True
                        raise

            try:
                with httpx.Client(timeout=None, transport=self.transport) as client:
                    response = client.post(url, headers={"Authorization": "Bearer " + channel.api_key},
                                           json=channel_body)
                if response.status_code >= 400:
                    record.update(status="http_error", http_status=response.status_code)
                    try:
                        error_payload = response.json()
                        response_usage = error_payload.get("usage") if isinstance(error_payload, dict) else None
                    except ValueError:
                        pass
                    if response_usage is not None:
                        record["usage"] = response_usage
                    # Preserve provider diagnostics locally without retaining credentials.
                    detail = response.text.replace(channel.api_key, "[REDACTED]")
                    record["http_error_body"] = detail
                    retryable = self._is_retryable_http_error(response.status_code, detail)
                    record["retryable"] = retryable
                    if retryable and retry_index < self.max_retries:
                        delay = self._retry_delay(retry_index)
                        record.update(retry_scheduled=True, retry_delay_s=delay)
                        account()
                        self._sleep(delay)
                        continue
                    record["retry_scheduled"] = False
                    suffix = "，已停止自动重试" if retryable else "，不会自动重试"
                    raise _ChannelRequestError(f"模型请求失败（HTTP {response.status_code}）{suffix}")
                payload = response.json()
                response_usage = payload.get("usage") if isinstance(payload, dict) else None
                record["usage"] = response_usage
                if not isinstance(payload, dict) or not isinstance(payload.get("choices"), list):
                    raise ModelError("模型返回的响应结构无效")
                choice = payload["choices"][0]
                if not isinstance(choice, dict):
                    raise ModelError("模型返回的响应结构无效")
                record.update(actual_model=payload.get("model"), usage=payload.get("usage"),
                              finish_reason=choice.get("finish_reason"), response=payload)
                if choice.get("finish_reason") not in finished:
                    raise ModelError("模型未正常完成输出：" + str(choice.get("finish_reason")))
                message = choice["message"]
                if not isinstance(message, dict) or "content" not in message:
                    raise ModelError("模型返回的响应结构无效")
                content = message["content"]
                if content is None:
                    content = ""
                if not isinstance(content, str):
                    raise ModelError("模型返回的响应内容结构无效")
                result = parse(message, content)
                record["status"] = "completed"
                return result
            except httpx.HTTPError as error:
                record.update(status="transport_error", retryable=True,
                              transport_error_type=type(error).__name__)
                if retry_index < self.max_retries:
                    delay = self._retry_delay(retry_index)
                    record.update(retry_scheduled=True, retry_delay_s=delay)
                    account()
                    self._sleep(delay)
                    continue
                record["retry_scheduled"] = False
                # Do not include provider URLs, headers or secrets in public errors.
                raise _ChannelRequestError("无法连接模型服务，自动重试已用尽") from error
            except ModelError:
                record.setdefault("status", "invalid_output")
                raise
            except (KeyError, IndexError, TypeError, ValueError) as error:
                record["status"] = "invalid_response"
                raise ModelError("模型返回的结构或 JSON 无效") from error
            finally:
                # One accounting write per actual HTTP attempt. Even malformed
                # model output can be billed, so record usage before callers retry it.
                account()

    def complete(self, messages: list[dict], *, purpose: str) -> dict:
        repaired_output = False

        def parse(message: dict, content: str) -> dict:
            nonlocal repaired_output
            result, repaired_output = decode_json_object_with_meta(content)
            return result
        body = {"model": self.config.model, "messages": messages, "response_format": {"type": "json_object"}}
        if self.config.reasoning_effort is not None:
            body["reasoning_effort"] = self.config.reasoning_effort
        result = self._post(body, messages, purpose, {"stop"}, parse)
        if repaired_output:
            self.calls[-1]["json_repaired"] = True
        return result

    def complete_text(self, messages: list[dict], *, purpose: str, temperature: float | None = None) -> str:
        """A plain-text reply, for answers that need no structure."""
        def parse(message: dict, content: str) -> str:
            if not content.strip():
                raise InvalidOutputError(content, "模型没有返回回答文字")
            return content.strip()
        body = {"model": self.config.model, "messages": messages}
        if temperature is not None:
            body["temperature"] = temperature
        if self.config.reasoning_effort is not None:
            body["reasoning_effort"] = self.config.reasoning_effort
        return self._post(body, messages, purpose, {"stop"}, parse)

    def usage_snapshot(self) -> dict:
        return self.channel_pool.snapshot()

    def update_priority(self, priority: Sequence[int | str]) -> dict:
        return self.channel_pool.update_priority(priority)


def encode(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
