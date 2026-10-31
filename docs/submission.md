# FactTrace 最终源码包

FactTrace 的正式入口是 `./.venv/bin/python -m text_memory`，仅提供 `add`、`query`、`serve`。项目展示名是 FactTrace，运行包名仍是 `text_memory`；运行包与本地使用相同源码、提示词、排序方法和关系标注，不另设比赛专用分支。

FactTrace 的核心创新是以带来源的事实作为长期记忆的基本单位，并在写入后记录事实变化，在查找时按问题需要组织事实和相关会话上下文。完整方法说明见[项目介绍](project.md)和[正式事实记忆流程](collections.md)。

**Competition submission blurb:** FactTrace is a long-conversation memory system built around evidence-grounded temporal facts. It extracts source-linked facts from complete sessions, records how user facts change or conflict over time, and assembles answer material with question-aware semantic, lexical, and time-based retrieval plus same-session fact completion. Its core innovation is treating traceable facts as the unit of long-term memory.

仓库以 Git 子模块固定官方评测模板版本；完整克隆请使用 `git clone --recurse-submodules`，已有副本执行 `git submodule update --init --recursive`，再按[补丁说明](../patches/README.md)应用两个项目补丁。评测题库和运行产物仍在本地准备，不进入提交包。

## 生成

```bash
./.venv/bin/python scripts/package_submission.py --output outputs/final-mainline-20261030/source.zip
```

包内保留正式源码、运行与数据准备脚本、主线测试、当前文档与示例、部署模板、依赖版本、固定官方答题评分代码和来源清单。`MANIFEST.json` 对每个文件记录 SHA-256，打包完成后逐个核对。

本地密钥、模型文件、数据库、原始题库、工作记录、旧源码和实验结果不打入此包。旧实验的恢复包在工作区的 `outputs/archive/final-mainline-cleanup-20261030/`，与提交源码分开保存。

## 解压核对

在新的空目录解压，安装 `requirements.lock.txt`，即可执行 `./.venv/bin/python -m unittest discover -v` 和 `./.venv/bin/python -m text_memory --help`。需要真实写入或服务时，再提供自己的模型、向量和服务鉴权配置，具体字段见 `.env.example`。

单元测试使用明确注入的测试模型，不需要真实模型密钥。测试通过说明本地程序行为得到核对，不代表真实模型效果、远端部署或官方平台评测已经完成。

此次操作只准备并核对源码包，没有发布、部署或向比赛平台实际提交。
