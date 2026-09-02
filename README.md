# ainy

基于大语言模型的智能陪伴 Agent（AI 智能陪伴 / AI 女友），支持多轮上下文对话、长期记忆、向量召回与 MCP 工具调用。

## 架构一览

```
                    ┌───────────────────────────── Agent (chat_completion.py) ─────────────────────────────┐
   用户输入 ────────▶ │  1. 短期窗口：超过阈值 → LLM 摘要 → SQLite 长期记忆（含向量）                        │
                    │  2. 向量召回：按当前问题召回 top-K 相关记忆 → 注入 system 上下文                      │
                    │  3. Agent Loop：流式调用 → finish_reason==tool_calls → 聚合参数 → 执行 → 结果回注     │
                    │  4. 工具：MCP Client 动态发现 (tools/list) 并执行 (tools/call)                        │
                    └───────────────┬───────────────────────────────────────────────┬──────────────────────┘
                                    │ stdio (JSON-RPC / MCP 协议)                     │
                         ┌──────────▼──────────┐                          ┌──────────▼──────────┐
                         │  SQLite: memory.db  │                          │  MCP Server         │
                         │  memories 表 + 向量 │                          │  (mcp_server.py)    │
                         │  conversation.json  │                          │  get_weather / get_date
                         └─────────────────────┘                          └─────────────────────┘
```

- **Agent Loop**：基于 Tool Calling 的执行循环，按模型返回的 `finish_reason` 判断是否执行工具，结果回注上下文，形成「模型决策 → 工具执行 → 结果反馈 → 继续决策」闭环，并设轮次上限防止死循环。
- **长期记忆分层**：短期窗口（`conversation.json`，原子写入）+ 长期记忆（SQLite `memories` 表）。上下文超过阈值时由 LLM 把最早的对话摘要固化入库；**按用户当前问题向量召回 top-K 相关记忆**（`embedding.py` + 余弦相似度），只注入相关内容，降低 Token 消耗、提升长期一致性。Embedding 不可用时自动退化为关键词重叠召回。
- **Streaming Tool Call**：流式响应中函数参数会被拆散，按 `tool_call index` 增量拼接，聚合完成后才做 JSON 解析与工具执行。
- **MCP 工具扩展**：天气 / 日期查询用官方 mcp SDK 封装为 MCP Server（`mcp_server.py`，`@server.tool` 装饰器注册，stdio 传输）；Agent 侧 `mcp_client.py` 基于 `ClientSession` + 后台事件循环线程桥接同步代码，动态 `tools/list` 发现工具并统一 `tools/call`。工具与 Agent 解耦：新增能力 = Server 加一个 `@server.tool` 函数，Agent 零改动；接入更多 Server（含第三方）只需在 `DEFAULT_MCP_SERVERS` 追加配置。

## 快速开始

```bash
pip install -r requirements.txt
cp .env_example .env        # 按需填模型 / Embedding 配置
python call.py              # 终端对话（exit / quit / 退出 结束）
```

## 配置 (.env)

| 变量 | 说明 |
|---|---|
| `OPENAI_BASE_URL` / `OPENAI_API_KEY` / `AI_MYSISTER` | OpenAI 兼容端点（Ollama、vLLM 等） |
| `EMBEDDING_MODEL` | Embedding 模型名，留空则记忆召回退化为关键词重叠 |
| `EMBEDDING_BASE_URL` / `EMBEDDING_API_KEY` | 可选，缺省复用 OPENAI_* |

## 说明

- 天气接口优先调 wttr.in（免密钥），网络不可用时自动降级为演示数据。
- 知识库 RAG（文档解析 → Chunk → Embedding → 检索增强）尚未接入，`embedding.py` / `memory.py` 的向量能力可复用。
- 长期记忆召回为 SQLite 内存暴力余弦（记忆量 ≤ 60 条量级足够）；如需更大规模可平滑替换为专用向量库。

## 测试

```bash
python test_agent.py        # 离线冒烟测试：fake LLM + 真实 MCP 子进程
```
