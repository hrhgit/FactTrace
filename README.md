# FactTrace

**Evidence-grounded temporal fact memory for long conversations.**

FactTrace turns complete conversations into source-linked facts, records how user facts change or conflict over time, and assembles question-specific evidence from the selected sessions. Its core innovation is treating traceable facts—not isolated text chunks—as the unit of long-term memory.

**中文简介：** FactTrace 把完整会话整理成带来源的具体事实，记录事实随时间的变化和关系，再根据问题组织可核对的证据材料。

[Project introduction](docs/project.md) · [Competition submission blurb](docs/submission.md) · [Formal pipeline](docs/collections.md) · [部署说明](deploy/aws/README.md)

需要运行完整评测测试时，请连同固定版本的官方评测模板一起克隆：

```bash
git clone --recurse-submodules https://github.com/hrhgit/FactTrace.git
cd FactTrace
git -C vendor/aml apply ../../patches/aml-async-file-context.patch
git -C vendor/aml apply ../../patches/aml-locomo-answer-parser.patch
```

已有副本可以执行 `git submodule update --init --recursive` 补齐该子模块，再执行上面的两个补丁命令；题库、运行数据库和评测输出仍按需在本地准备，不提交到仓库。

FactTrace 现在只保留完整事实写入和直接查事实这一条正式流程。统一运行入口为 `./.venv/bin/python -m text_memory`，HTTP 和本地评测调用同一个 `CollectionEngine`。

写入：**完整会话 → 提取事实 → 补漏 → 保存事实与来源 → 用户侧关系标注**。

查找：**判断问题所需材料 → 完整问题给事实排序 → 取前 16 条（汇总问题 32 条）→ 补充相关会话事实 → 展示事实与关系标注**。答题模型使用这些材料，原始对话仅用于写入和核对。

关键词、归组、二次查找、旧点线、关系传分和其他研究代码已从正式目录移走，主线没有相关可选开关。此次清理保留现行提取提示、问题判断提示、排序公式、关系标注规则和会话补全参数。

## 运行

Python 3.12，依赖见 `requirements.lock.txt`。复制 `.env.example` 为自己的 `.env`，填写 `MEMORY_*` 与向量配置；本地答题和评分再配置 `ANSWER_*`、`JUDGE_*`。密钥和本地数据库保留在本机。

```bash
./.venv/bin/python -m pip install -r requirements.lock.txt
./.venv/bin/python -m text_memory add --database outputs/memory.sqlite --input examples/conversation.json
./.venv/bin/python -m text_memory query --database outputs/memory.sqlite --input examples/question.json
./.venv/bin/python -m text_memory serve --database outputs/memory.sqlite --host 127.0.0.1 --port 8766
```

服务提供 `/health`、`/add` 和 `/search`。启动服务需配置 `MEMORY_SERVICE_TOKEN`；查询请求示例和配置说明见 [主线说明](docs/collections.md)。部署模板见 [部署说明](deploy/aws/README.md)。

各 key 来源的请求数、token 和 OpenAI 单价参考金额会累计到本地。可同时设置美元、token 上限，并在任务运行中调整渠道优先级，当前请求继续完成，后续请求采用新顺序。查看用量：`./.venv/bin/python -m text_memory channels status`；设置方法见 [渠道用量与上限](docs/model-channels.md)。

本地写入、查找、答题和评分的统一运行方法见 [评测说明](docs/local-evaluation.md)。这些步骤执行相同的正式方法，不包含旧方案对照。现有题集和结果仍保留在原位置，历史分数不视作此次清理后的新成绩。

## 测试与打包

```bash
./.venv/bin/python -m unittest discover -v
./.venv/bin/python scripts/package_submission.py --output outputs/final-mainline-20261030/source.zip
```

打包只读取正式源码、测试、文档、示例、固定官方评分代码与来源清单；包内含逐文件校验清单。`.env`、数据库、题库、实验输出、工作记录和归档不会进入提交源码包。打包与实际向平台提交是不同步骤，见 [提交源码说明](docs/submission.md)。

## 归档

本次清理前的完整源码快照、被拆出的文件、文件校验值和行为核对结果位于 工作区的 `outputs/archive/final-mainline-cleanup-20261030/`。此前的 `outputs/`、题库、数据库和工作记录继续保留。需要复现旧实验时，在独立目录恢复对应快照，不把旧实现放回正式源码。

清理后新增的历史格式审计和对照脚本另存于 `outputs/archive/mainline-cleanup-followup-20261031/`，不属于正式流程或源码包。
