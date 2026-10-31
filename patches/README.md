# AML 文件上下文兼容性补丁

`aml-async-file-context.patch` 和 `aml-locomo-answer-parser.patch` 适用于官方固定提交
`1b8142bfe0f20f1c5218d6b554aa0012de34e504`。前者修复官方流程打开结果文件的方式，后者保留 FactTrace 当前使用的 LoCoMo 答案格式提示和宽容 JSON 标签解析。

五个 pipeline（locomo-refined、longmemeval-s、clbench、beam、scriptmem）的
九处代码把普通文件放入 `async with`，在作答或评分循环开始前触发
`TypeError`。补丁保留 `httpx.AsyncClient` 的 `async with`，文件改用嵌套的
普通 `with`，仅调整相应缩进，不改变作答、评分或请求逻辑。

2026-10-21 浅克隆核对上游默认分支，最新提交仍为上述固定提交，九处均未修复。
检查副本位于 `outputs/aml-upstream-check/`；没有升级或切换本地 vendor 版本。
上游代码：[固定提交](https://github.com/AML-memory/agent-memory-leaderboard/tree/1b8142bfe0f20f1c5218d6b554aa0012de34e504)。

重新下载 vendor 并恢复到固定提交后，从项目根目录应用：

```bash
git -C vendor/aml apply --check ../../patches/aml-async-file-context.patch
git -C vendor/aml apply ../../patches/aml-async-file-context.patch
git -C vendor/aml apply --check ../../patches/aml-locomo-answer-parser.patch
git -C vendor/aml apply ../../patches/aml-locomo-answer-parser.patch
.venv/bin/python scripts/check_data.py --patch-only
```

`check_data.py` 用反向应用检查报告补丁是否已应用；完整数据检查也包含此项。
检查不修改 vendor 的索引、提交或工作文件。
