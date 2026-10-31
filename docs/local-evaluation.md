# FactTrace 正式主线本地评测

所有记忆写入和查找都经过正式 `CollectionEngine`，答题和评分采用 `vendor/aml` 固定版本的各题库原模板。公开题集是本地评测材料，不代表正式比赛的完整题包。

## 输入与隔离

题集目录包含 `inputs/histories.jsonl`、`inputs/queries.jsonl`、`inputs/tasks.jsonl`；参考答案只供独立评分步骤读取，不进入写入、查找或答题。现有基础集和扩展集位于 `data/local-eval/`。

## 执行

```bash
./.venv/bin/python scripts/run_mixed_memory.py write --set data/local-eval/mixed-v2 --output outputs/final-check
./.venv/bin/python scripts/run_mixed_memory.py search --set data/local-eval/mixed-v2 --output outputs/final-check
./.venv/bin/python scripts/run_mixed_answers.py answer --set data/local-eval/mixed-v2 --contexts outputs/final-check/search-top16/contexts.jsonl --output outputs/final-check/answers
./.venv/bin/python scripts/run_mixed_answers.py grade --output outputs/final-check/answers
./.venv/bin/python scripts/run_mixed_answers.py summary --output outputs/final-check/answers
```

运行命令的其他参数以 `--help` 为准。写入和查找配置 `MEMORY_*` 及向量；作答配置 `ANSWER_*`，开放题评分配置 `JUDGE_*`。

用量会按 key 来源合并保存，可同时设置参考美元和 token 上限。运行期间可以另开终端调整渠道顺序，下一次请求生效，见 [渠道用量与上限](model-channels.md)。这些调整有独立记录，不改动记忆提取、查找、答题或评分的方法。

同一用户会话按原顺序提交；会话内事实提取、补漏可以先并行完成并记录。中断后复用成功结果，已经保存事实的请求只补未完成的关系标注。`--reuse-writes` 可复制原始历史一致的已完成用户数据库；复制和复用都会保留事实来源。

查找使用同一正式 `/search` 实现。`contexts.jsonl` 来自返回的完整材料；问题需求判断失败会重试，失败题记录在 `search-failures.json`，该轮不导出新的答题材料。执行失败与答错分开。

每个运行目录固定题集、模型、输入、代码指纹和查找参数。代码或配置改变后使用新目录，不接着写旧目录冒充同一版本。旧数据库可被新版本读取，不表示旧运行分数自动代表新版本。

## 数据工具

`build_mixed_eval.py`、`prepare_locomo.py`、`prepare_longmemeval.py`、`prepare_personamem.py` 和 `sample_eval_sets.py` 用于准备输入；`check_data.py`、`restore_data.py`、`record_sources.py` 用于数据来源和完整性维护。它们不提供其他记忆算法。原始题库保留在 `data/public/`，来源清单保留在 `data/`。
