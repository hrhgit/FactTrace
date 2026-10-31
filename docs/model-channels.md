# 渠道用量、上限和运行中调整

正式模型调用共用本地用量记录。写入、查找、答题、评分，以及 HTTP 远程向量请求都会记录；本地向量计算没有 key 消耗。同一服务地址和 key 在不同阶段、不同模型间合并计数，换 key 后另记。记录中没有 key 原文、请求文字或模型回复。

默认数据库是项目下的 `state/model-usage.sqlite`，重启后继续累计。可在 `.env` 设置 `MODEL_USAGE_DB` 为绝对路径；需要共用上限的进程必须指向同一个本地数据库。`.env` 中的 `MEMORY_KEY_SOURCE`、`ANSWER_KEY_SOURCE`、`JUDGE_KEY_SOURCE` 及 `_2` 等后缀可设置来源名称；名称只是说明，相同名称的不同 key 不会被混为一条。

## 查看

在项目目录另开终端，使用与运行任务相同的 Python 环境和配置：

```bash
./.venv/bin/python -m text_memory channels status
./.venv/bin/python -m text_memory channels status --prefix ANSWER
./.venv/bin/python -m text_memory channels status --prefix JUDGE --json
./.venv/bin/python -m text_memory channels status --prefix MEMORY_EMBED
```

最后一个命令只适用于已配置 HTTP 远程向量的情况。`--env-file` 可指定另一份配置，`--usage-db` 可指定用量数据库。渠道编号与该配置加载后的渠道列表对应，先用 `status` 核对。

报告包含请求数、输入/输出/总 token、缓存输入、推理 token、参考美元金额、上限，以及未测得用量、未算出金额和未结算请求的数量。缓存和推理 token 是总数中的明细，不再次加进总数。总数缺失或小于已知输入与输出之和时，完整的输入和输出可以相加得到总数；只返回一部分时，已测得的部分仍计入额度，同时保留“未测得完整用量”的标记。服务错误（包括远程向量）、重试、格式错误回复也留下记录；复制已有结果、复用本地记录不会增加消耗。程序退出时未完成的请求保留为“未结算”，不会按零消耗冒充已完成。

## 同时设置美元和 token 上限

下面是使用示例，命令执行后才会启用对应上限；项目默认不设置任何上限。

```bash
./.venv/bin/python -m text_memory channels limit --channel 1 --max-cost-usd 10 --max-total-tokens 10000000
./.venv/bin/python -m text_memory channels limit --prefix ANSWER --channel 2 --max-requests 1000
```

上限针对该来源从启用记账起的累计消耗，包含此前已记下的消耗。美元、token、请求数中任一达到上限，就不再向这个来源发起新请求；有其他可用渠道时继续按顺序尝试。需要在所有渠道达到上限后继续运行时，可设置 `MEMORY_EXHAUSTED_FALLBACK=1`（`ANSWER_`、`JUDGE_` 同理），把渠道 1 作为最终兜底；进入兜底后会越过该渠道的本地上限，但仍继续记账并在报告中显示超出上限。没有设置兜底时，所有来源都用尽后明确报错，批量任务保留失败项供以后重试。

例如两把密钥轮询、渠道 1 上限 40 美元、渠道 2 上限 43 美元：只要还有一把未达到上限，就使用未到上限的密钥；两把都达到后，后续请求持续使用渠道 1，不再设本地金额上限。不会因为某个服务暂时出错而提前启用超额兜底；OpenAI 自身的余额、限流和权限错误仍正常报错。每次发送前会重新核对所有渠道的最新上限，兜底请求继续计入原密钥的累计金额，并记录 `exhausted_fallback` 标记；`channels status` 会显示配置、是否已进入兜底和兜底请求数。只对设置了该项的模型用途生效，修改后需重启对应服务或任务。

可以在运行中提高上限，也可以用 `none` 移除某一项：

```bash
./.venv/bin/python -m text_memory channels limit --channel 1 --max-cost-usd 20
./.venv/bin/python -m text_memory channels limit --channel 1 --max-total-tokens none
```

请求数在发出前计数，因此多个进程不会重复占用最后一次请求额度。token 和美元根据回复后得到的用量累计，**已发出的请求不会中断，因此最后一个请求或同时进行的一批请求可能超过上限**。没有返回用量的请求会单独显示，金额/token 上限只能约束已测得的部分，不是服务商账户的绝对扣费上限。

## 美元参考值

token 按 [OpenAI Chat Completions 的用量字段](https://developers.openai.com/api/reference/resources/chat) 记录。金额按 [OpenAI 官方文本单价](https://developers.openai.com/api/docs/pricing)折算，目前收录 GPT-4o mini、GPT-6 Luna、Terra、Sol、Astra 的 Standard 参考单价，核对日期为 2026-10-30。报告的 JSON 附参考表及来源。`openai/`、`codex/` 前缀可识别；金额始终是参考估算，不代表订阅、第三方渠道的实际账单，也不包含工具调用等额外费用。

例如 [GPT-6 Luna 官方说明](https://developers.openai.com/api/docs/models/gpt-6-luna)的每百万 token 单价为：普通输入 $0.10、缓存输入 $0.01、输出 $0.50；超过 272,000 输入 token 时按官方长输入倍率计算。未提供缓存明细时，输入按普通输入参考价计算。

其他模型只记录能取得的用量，未收录价格时金额显示未知。设置美元上限后，每次实际发送前都会检查当前模型是否有参考单价；同一 key 在其他阶段调用未定价模型时也会被拦下，并按渠道顺序尝试备用来源，避免无金额记录的请求绕过美元上限。若希望把某来源按同一个 OpenAI 模型的单价比较，可以明确指定：

```bash
./.venv/bin/python -m text_memory channels limit --channel 2 --reference-model gpt-6-luna --max-cost-usd 5
```

这只选择金额参考，不改变实际调用的模型。参考设置只影响之后开始的请求；此前未算出的金额仍保留为未知。远程向量同样可以设置 token 上限；选择文字模型作金额参考时，仅以输入 token 折算。单价表是本地核对版本，不会在运行中自行从网络更新。

## 运行中调整优先级

```bash
./.venv/bin/python -m text_memory channels priority 2 1 --strategy failover
./.venv/bin/python -m text_memory channels priority 2 1 --prefix ANSWER --strategy failover
./.venv/bin/python -m text_memory channels priority 1 2 --strategy round_robin
```

`failover` 优先使用排在前面的来源，失败或达到上限后才试后面的来源；`round_robin` 按这个顺序轮流选起始来源。可以只列一个编号，把它移到最前，其他来源保持当前先后。省略 `--strategy` 保留现有选择方式。

设置保存后，运行中的服务或本地批量任务在**下一次模型请求**开始时读取新顺序；当前请求和其重试、备用顺序继续使用开始时的顺序，不会被打断。每次实际发送前仍检查最新上限。远程向量保持原来的唯一渠道，不支持切换顺序。

手动设置的顺序和选择方式会跨重启保留；未手动设置选择方式时仍读取 `.env` 的 `*_CHANNEL_STRATEGY`。要改变正在运行的任务，请使用上述命令；单独编辑 `.env` 不会热更新已经加载的 key、服务地址或模型。

## 模型出错时的处理

服务连接失败、可重试的 HTTP 错误，以及 HTTP 成功但回复缺字段、内容类型错误或输出未完成时，默认在当前渠道最多请求 3 次，仍失败再按本次请求开始时的顺序切换渠道。认证或余额错误直接尝试下一渠道。每次实际请求都记录用量，包括失败的请求；请求间等待会先结算用量并释放渠道名额。

模型回复能读取但内容不符合要求时，会把原回复和具体错误原因接回对话，请模型返回完整修正结果。事实提取、关系判断、评分的每轮内容校验最多尝试 3 次，完全相同的无效回复会提前停止；补漏逐项保留已通过的内容，同一未解决段落最多校验 3 轮。空回答默认最多尝试 3 次。问题需求判断保持最多 3 次，并在重试前等待；批量查找仍可重复整个失败请求，但只重试模型错误，程序和数据库错误直接记录。

最终失败会报告阶段、尝试次数、最后渠道和具体原因；服务错误保留安全处理后的简短提示。原始请求和完整回复留在本地调用记录中，评分等内容校验失败另记为 `output_validation`。接口把模型失败返回为 HTTP 502，不会误报成调用方 JSON 错误。失败项不保存默认事实、答案或分数；批量查找失败会清除本轮目录内旧的材料导出，恢复成功后清除旧失败清单。

这些限制按服务请求、内容校验、批量请求分别计算，总 HTTP 请求数可能超过 3 次。未新增默认等待超时或内容截断；服务若一直不返回，请求仍会等待。本次修改需要相应 Python 进程重新启动后生效。

## 运行中调整并发

任务启动时的 `--workers` 是初始总并发。运行中的任务会从同一个用量数据库读取新的总并发上限；可以降低或提高它，已经开始的任务不会被中断，等待中的任务会按新上限继续：

```bash
./.venv/bin/python -m text_memory channels concurrency --total-concurrency 8 \
  --prefix MEMORY --usage-db path/to/model-usage.sqlite
```

也可以单独限制某个模型渠道。渠道并发设为 `0` 就是暂停该渠道的新请求；正在处理的请求仍会完成，恢复时设回正整数或用 `none` 移除限制：

```bash
./.venv/bin/python -m text_memory channels concurrency --channel 2 --max-concurrency 0 \
  --prefix MEMORY --usage-db path/to/model-usage.sqlite
./.venv/bin/python -m text_memory channels concurrency --channel 2 --max-concurrency 4 \
  --prefix MEMORY --usage-db path/to/model-usage.sqlite
```

总并发限制批量任务数量，渠道并发限制同一渠道同时发出的模型请求；两者同时生效，取较小的实际效果。渠道并发和总并发设置都保存在用量数据库中，运行中的新请求会读取最新值。

此功能只对已加载新代码的进程生效。升级前已启动的 Python 进程继续使用旧逻辑；本次没有停止它们，也没有从历史调用日志回填用量。
