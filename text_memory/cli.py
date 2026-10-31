"""正式事实记忆入口：写入、查找与 HTTP 服务。"""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys

from dotenv import load_dotenv

from .collections import CollectionEngine, CollectionStore
from .collections.query import DIRECT_LIMIT
from .collections.service import CollectionApplication
from .errors import MemoryErrorBase, ConfigurationError
from .evaluation import TimedModel
from .model import ModelChannelPool, ModelConfig, encode
from .retrieval import embedder_from_env, embed_floor_from_env
from .service import make_server
from .store import Scope
from .usage import PRICE_DATE

ROOT = Path(__file__).resolve().parents[1]


def _optional_limit(value, convert):
    return None if value.lower() in {"none", "unlimited"} else convert(value)


def manage_channels(args):
    """A separate CLI process changes settings read by running model pools."""
    limit_fields = ("max_requests", "max_total_tokens", "max_cost_usd", "reference_model")
    changes = {name: getattr(args, name) for name in limit_fields if hasattr(args, name)}
    has_concurrency = hasattr(args, "max_concurrency")
    has_total_concurrency = hasattr(args, "total_concurrency")
    concurrency = getattr(args, "max_concurrency", None)
    total_concurrency = getattr(args, "total_concurrency", None)
    if args.action not in {"limit", "concurrency"} and (changes or args.channel is not None):
        raise ConfigurationError("设置上限或参考模型请使用 channels limit")
    if args.action != "priority" and (args.order or args.strategy):
        raise ConfigurationError("调整顺序或选择方式请使用 channels priority")
    if args.action != "concurrency" and (has_concurrency or has_total_concurrency):
        raise ConfigurationError("调整并发请使用 channels concurrency")
    if args.prefix == "MEMORY_EMBED" and args.action == "priority":
        raise ConfigurationError("远程向量来源支持查看和限额，保持原有向量模型与渠道")
    config = ModelConfig.from_env(args.prefix)
    if args.usage_db:
        config = replace(config, usage_path=str(args.usage_db))
    pool = ModelChannelPool(config)
    try:
        if args.action == "priority":
            if not args.order and not args.strategy:
                raise ConfigurationError("请提供渠道顺序或选择策略")
            pool.update_priority(args.order or pool.priority(), strategy=args.strategy)
        elif args.action == "limit":
            if args.channel is None:
                raise ConfigurationError("设置上限需要 --channel 渠道编号")
            pool.update_limits(args.channel, **changes)
        elif args.action == "concurrency":
            if has_total_concurrency:
                if args.channel is not None:
                    raise ConfigurationError("总并发不能同时指定渠道")
                pool.update_total_concurrency(total_concurrency)
            elif has_concurrency and args.channel is not None:
                pool.update_concurrency(args.channel, concurrency)
            else:
                raise ConfigurationError("请指定 --channel 或 --total-concurrency")
        result = pool.snapshot()
        if args.json:
            print(encode(result))
        else:
            print(f"{args.prefix} 渠道顺序：" + " → ".join(str(i) for i in result["priority"]))
            print(f"选择方式：{result['strategy']}；调整会在下一次请求生效。")
            if result["exhausted_fallback"] is not None:
                state = "已启用" if result["exhausted_fallback_active"] else "等待所有渠道达到上限"
                print(f"超额兜底：渠道 {result['exhausted_fallback']}（{state}）；继续使用并记账。")
            for channel in result["channels"]:
                usage, limits = channel["usage"], channel["limits"]
                print(f"{channel['index']}. {channel['source']} ({channel['key_id'][-8:]}) · {channel['model']}")
                amount = f"${usage['estimated_cost_usd']:.6f}" if usage["cost_complete"] or usage["estimated_cost_usd"] else "未知"
                if not usage["cost_complete"] and usage["estimated_cost_usd"]:
                    amount += "（已算部分）"
                print(f"   请求 {usage['requests']}；已测得 {usage['total_tokens']:,} token；参考金额 {amount}")
                if limits.get("max_concurrency") is not None:
                    print(f"   当前并发 {usage['active_requests']}/{limits['max_concurrency']}")
                if usage["exhausted_fallback_requests"]:
                    print(f"   其中超额兜底 {usage['exhausted_fallback_requests']} 次")
                print(f"   未测得用量 {usage['unmeasured_requests']} 次；未算金额 {usage['unpriced_requests']} 次；进行中或未结算 {usage['unfinished_requests']} 次")
                labels = {"max_requests": "请求数", "max_total_tokens": "总 token", "max_cost_usd": "美元",
                          "max_concurrency": "并发"}
                caps = [f"{labels[name]}={value}" for name, value in limits.items() if value is not None]
                print("   上限：" + ("，".join(caps) if caps else "未设置") + ("（已达到）" if channel["exhausted"] else ""))
            if result.get("max_concurrency") is not None:
                print(f"总并发上限：{result['max_concurrency']}")
            print(f"金额是 OpenAI 文本单价参考值（{PRICE_DATE}），不代表实际账单。")
        return 0
    finally:
        pool.usage.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for action in ("add", "query", "serve"):
        p = commands.add_parser(action)
        p.add_argument("--database", required=True)
        p.add_argument("--env-file", type=Path, default=ROOT / ".env")
        if action != "serve":
            p.add_argument("--input", required=True, help="JSON 写入请求或 question/user_id/scope_id 查询")
            p.add_argument("--trace", help="保存模型调用和程序步骤")
        if action in {"query", "serve"}:
            p.add_argument("--fact-limit", type=int, default=DIRECT_LIMIT, help="排序后保留的事实条数")
        if action == "serve":
            p.add_argument("--host", default="127.0.0.1")
            p.add_argument("--port", type=int, default=8766)
    p = commands.add_parser("channels", help="查看各 key 来源用量、设置上限或调整运行中的渠道顺序")
    p.add_argument("action", choices=("status", "priority", "limit", "concurrency"))
    p.add_argument("order", nargs="*", type=int, help="优先使用的渠道编号，例如 2 1")
    p.add_argument("--env-file", type=Path, default=ROOT / ".env")
    p.add_argument("--prefix", choices=("MEMORY", "ANSWER", "JUDGE", "MEMORY_EMBED"), default="MEMORY")
    p.add_argument("--usage-db", type=Path, help="用量数据库；须与运行中的任务相同")
    p.add_argument("--strategy", choices=("failover", "round_robin"))
    p.add_argument("--channel", type=int)
    p.add_argument("--max-requests", type=lambda v: _optional_limit(v, int), default=argparse.SUPPRESS)
    p.add_argument("--max-total-tokens", type=lambda v: _optional_limit(v, int), default=argparse.SUPPRESS)
    p.add_argument("--max-cost-usd", type=lambda v: _optional_limit(v, float), default=argparse.SUPPRESS)
    p.add_argument("--reference-model", type=lambda v: _optional_limit(v, str), default=argparse.SUPPRESS,
                   help="使用该 OpenAI 模型的单价作为金额参考")
    p.add_argument("--max-concurrency", type=lambda v: _optional_limit(v, int), default=argparse.SUPPRESS,
                   help="单个模型渠道的并发上限；设为 0 暂停该渠道")
    p.add_argument("--total-concurrency", type=lambda v: _optional_limit(v, int), default=argparse.SUPPRESS,
                   help="整个运行阶段的并发上限")
    p.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    load_dotenv(args.env_file, override=False)
    embedder = None
    try:
        if args.command == "channels":
            return manage_channels(args)
        config = ModelConfig.from_env()
        embedder = embedder_from_env()
        if embedder is None:
            raise ConfigurationError("正式写入、关系标注和查找需要配置向量模型")
        floor = embed_floor_from_env()
        channel_pool = ModelChannelPool(config)
        if args.command == "serve":
            app = CollectionApplication(args.database, lambda: TimedModel(config, channel_pool=channel_pool),
                                        embedder=embedder, floor=floor, fact_limit=args.fact_limit)
            with make_server(app, args.host, args.port, os.environ.get("MEMORY_SERVICE_TOKEN", "")) as server:
                server.serve_forever()
            return 0
        body = json.loads(sys.stdin.read() if args.input == "-" else Path(args.input).read_text())
        with CollectionStore(args.database) as store:
            engine = CollectionEngine(store, TimedModel(config, channel_pool=channel_pool), embedder=embedder, floor=floor)
            try:
                if args.command == "add":
                    result = engine.add(body)
                else:
                    result = engine.query(body["question"], Scope(body["user_id"], body.get("scope_id", "default"), body.get("as_of")),
                                          question_date=body.get("question_date"), fact_limit=args.fact_limit)
            finally:
                if args.trace:
                    destination = Path(args.trace)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_text(encode({"calls": engine.model.calls, "steps": engine.trace}) + "\n")
        print(encode(result))
        return 0
    except (MemoryErrorBase, ValueError, OSError) as error:
        print(str(error), file=sys.stderr)
        return 2
    finally:
        if embedder:
            embedder.close()
