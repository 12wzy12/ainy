import atexit
import json
import logging
import os
import httpx

from openai import OpenAI
from dotenv import load_dotenv

import sys_prompt_zh as sys_prompt
from memory import MemoryStore
from mcp_client import McpToolRegistry

load_dotenv(override=True)

for key in [
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
]:
    os.environ.pop(key, None)

os.environ.update({"http_proxy": "http://10.64.3.228:10800","https_proxy": "http://10.64.3.228:10800"})


logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MEMORY_PATH = os.path.join(_MODULE_DIR, "conversation.json")
DEFAULT_DB_PATH = os.path.join(_MODULE_DIR, "memory.db")

# 长期记忆在上下文中出现的角色标记；加载旧对话时按此前缀识别并迁入 SQLite
MEMORY_PREFIX = "[长期记忆] "


class Sister:

    def __init__(
        self,
        max_messages: int = 40,
        max_tool_rounds: int = 6,
        memory_path: str = DEFAULT_MEMORY_PATH,
        db_path: str = DEFAULT_DB_PATH,
        max_memories: int = 60,
        recall_k: int = 3,
    ):

        self.model = os.getenv("AI_MYSISTER")
        self.max_messages = max_messages
        self.max_tool_rounds = max_tool_rounds
        self.memory_path = memory_path
        self.db_path = db_path
        self.recall_k = recall_k

        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
            http_client=httpx.Client(proxy=None)
        )

        # 工具全部来自 MCP Server：动态发现 schema，通过统一协议执行
        self.registry = McpToolRegistry()
        self.tools = self.registry.tools
        if not self.tools:
            logging.warning("没有任何可用 MCP 工具，将以纯对话模式运行")

        self.store = MemoryStore(db_path, max_rows=max_memories)
        atexit.register(self.store.close)

        # self.messages 只放「系统提示词 + 对话历史」；
        # 长期记忆按用户问题逐轮动态召回，不常驻在窗口里（见 memory_msgs）
        base = [{"role": "system", "content": sys_prompt.SISTER_PROMPT}]
        short_term = self._load_history() or []
        self.messages = base + short_term
        self.memory_msgs = []  # 当前这轮召回出来的长期记忆（system 消息）

    # ---------- 记忆 ----------

    def _load_history(self) -> list:
        """从 conversation.json 恢复短期窗口；顺带把旧的 [长期记忆] 迁入 SQLite。"""
        try:
            with open(self.memory_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            return None
        except Exception as e:
            logging.warning(f"load history failed: {e}")
            return None

        if not isinstance(data, list) or not data or data[0].get("role") != "system":
            return None

        short = []
        for m in data:
            content = str(m.get("content", ""))
            if m.get("role") == "system" and content.startswith(MEMORY_PREFIX):
                # 旧版本把长期记忆内联在对话里 → 迁入 SQLite 并移出短期窗口
                self.store.add_if_missing(content[len(MEMORY_PREFIX):])
            elif m.get("role") == "system" and content == sys_prompt.SISTER_PROMPT:
                # 旧存档开头带一份系统提示词副本 → 丢弃，避免重复注入
                continue
            else:
                short.append(m)

        logging.info(f"已恢复上次对话记忆：{self.memory_path}")
        return short or None

    def _save(self):
        """把当前短期窗口写入磁盘（临时文件 + 原子替换）。"""
        try:
            tmp = self.memory_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.messages, f, ensure_ascii=False, indent=2)
            os.replace(tmp, self.memory_path)
        except Exception as e:
            logging.warning(f"save history failed: {e}")

    def _summarize(self, old_messages: list) -> str:
        """把一段对话压缩成一段要点的长期记忆文本（非流式）。"""
        prompt = (
            "把下面这段对话压缩成一段简短的、要点式的长期记忆（第三人称），"
            "用于以后继续闲聊时记住用户的偏好和聊过的事，不要编造：\n"
            + json.dumps(old_messages, ensure_ascii=False)
        )
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                stream=False,
            )
            return resp.choices[0].message.content
        except Exception as e:
            logging.warning(f"summarize failed: {e}")
            return None

    @staticmethod
    def _safe_boundary(history: list, min_keep: int = 6) -> int:
        """找一个不会拆散「assistant 的 tool_call ↔ tool 结果」的折叠边界。

        返回 history 前 i 条将被折叠成长期记忆；剩余 recent >= min_keep 条。
        折叠边界若落在 assistant(tool_calls) 与其 tool 结果之间，会向模型
        上下文注入「孤儿 tool 消息」，导致部分兼容端点报错。
        """
        if len(history) <= min_keep:
            return max(1, len(history))
        for i in range(len(history) - min_keep, 0, -1):  # 尽量多折叠
            if history[i].get("role") != "tool":        # recent 首条不是 tool 即可
                return i
        return len(history) - min_keep

    def _condense_memory(self):
        """窗口超限时：把最早一段对话压成长期记忆写入 SQLite，只保留最近几轮。

        长期记忆固化后不再留在对话窗口里（避免被重复摘要成新记忆），
        之后按用户当前问题动态召回进上下文。
        """
        history = self.messages[1:]  # 不含系统提示词
        while len(history) > self.max_messages:
            boundary = self._safe_boundary(history)
            old, recent = history[:boundary], history[boundary:]

            summary = self._summarize(old)
            if summary:
                self.store.add(summary.strip())  # 摘要 + 向量持久化到 SQLite
                logging.info(f"已把 {boundary} 条旧消息固化为长期记忆")
            else:
                logging.warning("摘要失败，直接裁剪超限的历史消息")

            self.messages = self.messages[:1] + recent
            history = recent

    def _recall_memory(self, query: str):
        """按用户当前问题向量召回相关长期记忆，组装成 context 注入消息。"""
        memories = self.store.recall(query, k=self.recall_k)
        self.memory_msgs = [
            {"role": "system", "content": f"{MEMORY_PREFIX}{m}"} for m in memories
        ]
        if memories:
            logging.info(f"召回 {len(memories)} 条相关长期记忆")

    # ---------- 对话 ----------

    def _context_messages(self) -> list:
        """组装本次模型调用的完整上下文：系统提示词 + 召回记忆 + 对话历史。"""
        return self.messages[:1] + self.memory_msgs + self.messages[1:]

    def _call_sister(self, with_tools: bool = True):
        """用 OpenAI SDK 流式调用 OpenAI 兼容的 /v1 端点。"""
        kwargs = {
            "model": self.model,
            "messages": self._context_messages(),
            "stream": True,
            "extra_body": {
                "thinking": "enable",
            }
        }
        if with_tools and self.tools:
            kwargs["tools"] = self.tools
        return self.client.chat.completions.create(**kwargs)

    def _parse_response(self, response):

        has_print_reasoning = False
        has_print_content = False
        has_print_tool = False

        tool_calls = {}
        finish_reason = ""

        assistant_message = {
            "role": "assistant",
            "reasoning_content": "",
            "content": "",
            "tool_calls": []
        }

        for chunk in response:
            delta = chunk.choices[0].delta
            chunk_finish = getattr(chunk.choices[0], "finish_reason", "")
            if chunk_finish:
                finish_reason = chunk_finish

            assistant_reasoning = (
                getattr(delta, "reasoning_content", "") or getattr(delta, "reasoning", "") or ""
            )
            assistant_content = getattr(delta, "content", "") or ""
            assistant_tool_calls = getattr(delta, "tool_calls", None)

            if assistant_reasoning:
                if not has_print_reasoning:
                    print()
                    print("思考： ", end="", flush=True)
                    has_print_reasoning = True
                assistant_message["reasoning_content"] += assistant_reasoning
                print(assistant_reasoning, end="", flush=True)

            if assistant_content:
                if not has_print_content:
                    print()
                    print("回答： ", end="", flush=True)
                    has_print_content = True
                assistant_message["content"] += assistant_content
                print(assistant_content, end="", flush=True)

            # Streaming Tool Call：函数参数可能被拆在多个 chunk 里，
            # 按 tool_call index 增量聚合，最后拼成完整 JSON 再解析
            if assistant_tool_calls:
                if not has_print_tool:
                    print()
                    print("工具调用： ", end="", flush=True)
                    has_print_tool = True
                for tc in assistant_tool_calls:
                    index = tc.index
                    if index not in tool_calls:
                        tool_calls[index] = {"id": "", "type": "function", "function": {"arguments": "", "name": ""}}
                    # 流式首片可能只带 id 不带 function，逐字段判空增量拼接
                    if tc.id:
                        tool_calls[index]["id"] = tc.id
                    if tc.function is not None:
                        if tc.function.name:
                            tool_calls[index]["function"]["name"] = tc.function.name
                        if tc.function.arguments:
                            tool_calls[index]["function"]["arguments"] += tc.function.arguments

        if tool_calls:
            assistant_message["tool_calls"] = list(tool_calls.values())
        else:
            assistant_message.pop("tool_calls", None)  # 无工具调用时不带该字段

        self.messages.append(assistant_message)

        return assistant_message, finish_reason

    def _parse_tool_calls(self, assistant_message: dict, finish_reason: str):
        if finish_reason != "tool_calls":
            return

        tool_call_result_message = []

        for tool in assistant_message["tool_calls"]:
            name = tool["function"]["name"]
            try:
                # 参数在流式传输中被拆散，这里才对聚合完成的 JSON 做解析
                arg = json.loads(tool["function"]["arguments"] or "{}")
                tool_result = self.registry.call_tool(name, arg)  # 经 MCP 统一协议执行
            except Exception as e:
                logging.warning(f"tool {name} failed: {e}")
                tool_result = f"工具执行出错: {e}"
            tool_call_result_message.append({"role": "tool", "tool_call_id": tool["id"], "content": tool_result})

        self.messages.extend(tool_call_result_message)

    def chat(self, user_input: str):
        self.messages.append({
            "role": "user",
            "content": user_input
        })

        # 上下文分层：超阈值则把最早的对话摘要固化为长期记忆
        self._condense_memory()
        # 长期记忆按当前问题召回，只把相关的注入本轮的上下文
        self._recall_memory(user_input)

        for _ in range(self.max_tool_rounds):

            try:
                response = self._call_sister()
            except Exception as e:
                logging.error(f"call_sister failed : {e}")
                raise

            # Agent Loop：按 finish_reason 判断模型是否要调用工具
            assistant_message, finish_reason = self._parse_response(response)

            if finish_reason != "tool_calls":
                self._save()
                return assistant_message["content"]

            # 执行工具并把结果重新注入上下文，让模型继续决策
            self._parse_tool_calls(assistant_message, finish_reason)

        # 达到工具调用轮次上限：强制一次不带工具的回答，避免无限循环且让会话自洽
        self.messages.append({
            "role": "user",
            "content": "（请直接给出最终回答，不要再调用工具）"
        })
        try:
            response = self._call_sister(with_tools=False)
        except Exception as e:
            logging.error(f"call_sister failed : {e}")
            raise
        assistant_message, _ = self._parse_response(response)
        self._save()
        return assistant_message["content"]
