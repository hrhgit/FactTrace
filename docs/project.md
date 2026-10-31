# FactTrace

**面向长会话的可追溯时序事实记忆**

FactTrace 是一个面向长会话的个人事实记忆系统。它把完整会话整理成一条条可以独立理解、可以回到原文核对的事实，再根据问题需要组织事实和相关会话脉络，为回答提供一份清楚的证据材料。

项目展示名是 **FactTrace**，运行包名仍是 `text_memory`。这样可以保持已有命令和服务入口不变，同时让项目的用途和方法更容易被理解。

## 核心创新

FactTrace 最重要的设计是：**把长期记忆的基本对象从相似的文本片段改成带来源、能处理时间变化、可以恢复上下文的事实。**

这一点贯穿整个正式流程：

1. **事实本身完整且可核对。** 每条记忆用一句自然语言记录一个具体事项，保留必要的背景、原因、进度和结果，并附上它来自哪些消息。
2. **事实变化被明确记录。** 新事实写入后，只在用户自己的事实之间标注同一件事、另一次、取代、冲突或计划结果；事实正文和原始来源继续保留。
3. **查找按问题组织证据。** 模型只判断问题是否需要汇总、最新说法或历史冲突；程序再用完整问题、文字匹配和时间范围给事实排序，并补回命中事实所在会话中的相关事实。
4. **回答材料保持清楚的边界。** 原始对话用于写入、核对和审计；回答模型看到的是已经整理好的事实、关系说明和必要的来源信息，避免把整段聊天直接塞进答案上下文。

FactTrace 的重点是建立一条可追溯的事实记忆闭环：

**完整会话 → 事实提取 → 补漏 → 保存事实与来源 → 用户侧关系标注 → 问题需求判断 → 完整问题排序 → 会话事实补全 → 展示关系标注**

关键词、模型归组、程序分组、二次查找、旧点线和关系传分等研究变体不属于正式主线。

## English project introduction

**FactTrace is an evidence-grounded temporal fact memory system for long conversations.** It turns complete conversations into concise, self-contained facts that retain links to their source messages. When new user facts arrive, the system records whether they describe the same matter, a separate occurrence, a replacement, a conflict, or the outcome of a plan. For a question, a small model judgment identifies whether the answer needs aggregation, the latest value, or historical conflict; programmatic retrieval then ranks facts using the full question, text matching, and time, and restores related facts from the selected sessions. The answer model receives organized facts and relationship notes instead of raw chat history. The central idea is to make traceable, time-aware facts the unit of long-term memory.

## Competition submission blurb

> FactTrace is a long-conversation memory system built around evidence-grounded temporal facts. It extracts source-linked facts from complete sessions, records how user facts change or conflict over time, and assembles answer material with question-aware semantic, lexical, and time-based retrieval plus same-session fact completion. Its core innovation is treating traceable facts—not isolated text chunks—as the unit of long-term memory.

## 正式入口

```bash
./.venv/bin/python -m text_memory add --database outputs/memory.sqlite --input examples/conversation.json
./.venv/bin/python -m text_memory query --database outputs/memory.sqlite --input examples/question.json
./.venv/bin/python -m text_memory serve --database outputs/memory.sqlite --host 127.0.0.1 --port 8766
```

写入、查找和服务使用同一套 `CollectionEngine`。完整流程、接口和配置见[正式事实记忆流程](collections.md)，源码包准备步骤见[提交说明](submission.md)。
