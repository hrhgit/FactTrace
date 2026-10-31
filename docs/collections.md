# FactTrace 正式事实记忆流程

FactTrace 的正式流程以带来源的事实为记忆单位：写入时从完整会话提取并核对事实，查找时按问题组织事实和相关会话上下文。项目展示名是 FactTrace，运行包名仍是 `text_memory`。

## 写入

统一实现是 `text_memory/collections/engine.py` 的 `CollectionEngine.add`。

1. 保存完整输入会话。长会话按消息边界分段读取，沿用同一模型对话；不截断原文。
2. 提取完整、具体的一句话事实，并逐段检查来源。相关背景、原因、进度和结果尽量保存在同一事项里。
3. 对程序发现的疑似遗漏，请模型在原写入对话里核对和补充；没有遗漏标记时不调用。
4. 将事实、来源、原事实顺序和写入回执一次保存。重复的同一请求返回原回执，内容冲突会报错。
5. 对新用户侧事实标注与更早用户侧事实的关系。每条事实取 5 条最相近候选，逐对判断同一件事、另一次、取代、矛盾或计划结果；否定说法另与 30 条近邻核对矛盾。只来自助手消息的事实不参与标注和候选。事件日期不重叠时，将“同一件事”改为“另一次”。

关键词、归组和旧点线结构不在此流程中，也没有启用它们的参数。新数据库不再建立关键词表、归组表、旧关联线表或旧全文查找表；已有事实库可以继续读取，历史多余表保留原样而不使用。

事实保存成功而关系标注中断时，仍报告失败；重试复用回执，只补关系标注。事实提取和补漏的有效检查点可以复用。模型输出无效或服务失败要保持可见，不用默认值或程序生成的内容替代模型工作。

模型要求返回 JSON 时，解析器只做不改变内容的修复：去掉 JSON 代码围栏、补一个唯一缺失的末尾括号、去掉短的多余结束符，以及去掉白名单中的固定协议结束标记。其他格式或字段错误仍会进入模型重试并保留失败记录。历史输出审计、合同对照和失败重放脚本保存在 `outputs/archive/mainline-cleanup-followup-20261031/removed/scripts/`，不属于正式流程或源码包。

正式请求会由程序把本批允许的证据编号、配对编号和冲突编号写进输出合同；模型只能从这些编号中选择，程序随后仍会再次核对来源。

## 查找与答题材料

正式策略为 `direct-facts-question-flags-v4`。

- 模型只判断问题是否需要汇总多条记录、是否要以最新说法为准、是否问“是否曾经”；不回答或推理问题。判断失败会重试，仍失败则停止查找，不返回默认材料。
- 程序用完整问题比较所有有效事实：事实正文向量余弦 + BM25 压缩分值 + 命中问题日期范围时的 0.5 加分。没有分数门槛，默认选前 16 条；需汇总时选 32 条。
- 对命中事实执行已确认的会话事实补全，保留命中与补充的区别，不递归查找，不展开集合。沿用原核心段、0.4 同段支持度、0.55 边界、0.66 远处关联、0.7 伴随支持度和 0.68 伴随相似度。
- 每条事实最多展示 2 条写入时关系标注，每对单独说明，不做传递归组或自动计数。关系只用于材料说明，不给排序传分。
- 同一会话内按来源顺序展示事实，普通问题按最佳命中名次排列会话；需要汇总或以新值为准的问题，按会话时间先后排列，并说明排列方式。保留会话日期，程序解析相对日期，事实正文不在数据库中改写。
- 材料只显示事实、日期和关系说明；内部事实编号、来源编号、原对话和运行日志不展示给答题模型。

## 接口

```bash
./.venv/bin/python -m text_memory add --database outputs/memory.sqlite --input examples/conversation.json
./.venv/bin/python -m text_memory query --database outputs/memory.sqlite --input examples/question.json --trace outputs/query-trace.json
./.venv/bin/python -m text_memory serve --database outputs/memory.sqlite --env-file .env --port 8766
```

写入 JSON 使用 `user_id`、`scope_id`（默认 `default`）、`session_id`、`request_id`、`messages`。每条消息保留原角色、正文和原有日期/说话人字段。

本地查询 JSON 使用 `question`、`user_id`、`scope_id`，可带 `question_date` 和 `as_of` 历史修订号。`query` 返回已整理的材料包供核对；最终自然语言回答由正式评测答题步骤生成。

HTTP `/add` 使用同一写入形状；`/search` 使用 `query`、`user_id` 和正整数 `top_k`，可带 `scope_id`、`question_date`、`as_of`。返回 `data` 数组，每个元素有 `id` 和 `content`；无事实返回空数组。完整材料作为一个元素，避免按 `top_k` 截断事实。鉴权支持 `Authorization: Bearer ...`、`Authorization: Token ...` 和 `X-Api-Key`。

## 配置

`MEMORY_API_BASE`、`MEMORY_API_KEY`、`MEMORY_MODEL` 用于提取、补漏、问题需求判断和关系标注。同一模型配置贯穿这些步骤。

向量可使用已缓存的本地模型（`MEMORY_EMBED_PYTHON`、`MEMORY_EMBED_MODEL`、`MEMORY_EMBED_REVISION`），或 HTTP 接口（`MEMORY_EMBED_API_BASE`、`MEMORY_EMBED_API_KEY`、`MEMORY_EMBED_MODEL`）；两者不能同时配置。`MEMORY_EMBED_FLOOR` 沿用已验证的会话补全配置，正式整句排序本身不使用这一门槛。未配置向量时，统一命令行入口正常报错。

`MEMORY_*`、`ANSWER_*`、`JUDGE_*` 支持已配置的备用渠道，以及 `failover` / `round_robin` 渠道策略。向量空间不随生成模型渠道切换。具体字段见根目录 `.env.example`。
