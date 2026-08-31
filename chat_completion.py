import atexit
import json
import logging
import os
import sqlite3
import httpx

from openai import OpenAI
from dotenv import load_dotenv

import sys_prompt_zh as sys_prompt
import function_calls as function

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


class MemoryStore:
    """长期记忆的 SQLite 持久化：一条记录 = 一段长期记忆摘要。"""

    def __init__(self, db_path: str = DEFAULT_DB_PATH, max_rows: int = 60):
        self.db_path = db_path
        self.max_rows = max_rows
        self.conn = sqlite3.connect(db_path)
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
                content    TEXT NOT NULL
            )
            """
        )
        self.conn.commit()

    def add(self, content: str) -> int:
        """新增一条长期记忆，随后裁剪到上限。"""
        cur = self.conn.execute(
            "INSERT INTO memories(content) VALUES (?)", (content,)
        )
        self.conn.commit()
        self._prune()
        return cur.lastrowid

    def add_if_missing(self, content: str) -> int:
        """内容已存在则忽略（去重），否则新增。"""
        row = self.conn.execute(
            "SELECT id FROM memories WHERE content = ?", (content,)
        ).fetchone()
        if row:
            return row[0]
        return self.add(content)

    def all(self) -> list:
        """按时间顺序返回全部长期记忆内容。"""
        rows = self.conn.execute(
            "SELECT content FROM memories ORDER BY id"
        ).fetchall()
        return [r[0] for r in rows]

    def count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]

    def _prune(self):
        self.conn.execute(
            "DELETE FROM memories WHERE id NOT IN "
            "(SELECT id FROM memories ORDER BY id DESC LIMIT ?)",
            (self.max_rows,),
        )
        self.conn.commit()

    def close(self):
        self.conn.close()


class Sister:

    def __init__(
        self,
        max_messages: int = 40,
        max_tool_rounds: int = 6,
        memory_path: str = DEFAULT_MEMORY_PATH,
        db_path: str = DEFAULT_DB_PATH,
        max_memories: int = 60,
    ):

        self.model = os.getenv("AI_MYSISTER")
        self.max_messages = max_messages
        self.max_tool_rounds = max_tool_rounds
        self.memory_path = memory_path
        self.db_path = db_path

        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
            http_client=httpx.Client(proxy=None)
        )

        self.tools = function.tools

        self.tool_call_map = {
            "get_weather": function._get_weather,
            "get_date": function._get_date,
        }

        self.store = MemoryStore(db_path, max_rows=max_memories)
        atexit.register(self.store.close)

        base = [{"role": "system", "content": sys_prompt.SISTER_PROMPT}]
        long_term = self._memory_context()      # 长期记忆：来自 SQLite
        short_term = self._load_history() or []  # 短期窗口：来自 conversation.json
        self.messages = base + long_term + short_term

    # ---------- 记忆 ----------

    def _memory_context(self) -> list:
        """把 SQLite 里的长期记忆组装成 system 消息（供模型当上下文）。"""
        memories = self.store.all()
        if not memories:
            return []
        return [{"role": "system", "content": f"[长期记忆] {m}"} for m in memories]

    def _load_history(self) -> list:
        """从 conversation.json 恢复短期窗口；顺带把其中旧的 [长期记忆] 迁入 SQLite。"""
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
            if m.get("role") == "system" and str(m.get("content", "")).startswith("[长期记忆]"):
                # 旧版的内联长期记忆 → 迁入 SQLite 并移出短期窗口
                self.store.add_if_missing(m["content"])
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

    def _condense_memory(self):
        """窗口超限时：把最早一段对话压成长期记忆写入 SQLite，并保留最近几轮。"""
        if len(self.messages) <= self.max_messages:
            return

        system = self.messages[:1]           # 始终保留系统提示词
        keep_recent = 8
        split = max(1, len(self.messages) - keep_recent)
        old = self.messages[1:split]
        recent = self.messages[split:]

        summary = self._summarize(old)
        if summary:
            self.store.add(summary)          # 长期记忆持久化到 SQLite
            memory_msg = {"role": "system", "content": f"[长期记忆] {summary.strip()}"}
            self.messages = system + [memory_msg] + recent
        else:
            self.messages = system + recent

    # ---------- 对话 ----------

    def _call_sister(self, with_tools: bool = True):
        """用 OpenAI SDK 流式调用 OpenAI 兼容的 /v1 端点。"""
        kwargs = {
            "model": self.model,
            "messages": self.messages,
            "stream": True,
            "extra_body": {
                "thinking": "enable",
            }
        }
        if with_tools:
            kwargs["tools"] = self.tools
        return self.client.chat.completions.create(**kwargs)

    def _parse_response(self, response):

        has_print_reasoning = False
        has_print_content = False
        has_print_tool = False

        tool_calls = {}

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

            if assistant_tool_calls:
                if not has_print_tool:
                    print()
                    print("工具调用： ", end="", flush=True)
                    has_print_tool = True
                for tc in assistant_tool_calls:
                    index = tc.index
                    if index not in tool_calls:
                        tool_calls[index] = {"id": "", "type": "function", "function": {"arguments": "", "name": ""}}
                    if tc.id:
                        tool_calls[index]["id"] = tc.id
                    if tc.function.name:
                        tool_calls[index]["function"]["name"] = tc.function.name
                    if tc.function.arguments:
                        tool_calls[index]["function"]["arguments"] += tc.function.arguments

        assistant_message["tool_calls"] = list(tool_calls.values())

        self.messages.append(assistant_message)

        return assistant_message, finish_reason

    def _parse_tool_calls(self, assistant_message: dict, finish_reason: str):
        if finish_reason != "tool_calls":
            return

        tool_call_result_message = []

        for tool in assistant_message["tool_calls"]:
            name = tool["function"]["name"]
            try:
                tool_function = self.tool_call_map[name]
                arg = json.loads(tool["function"]["arguments"] or "{}")
                tool_result = str(tool_function(**arg))
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

        self._condense_memory()

        for _ in range(self.max_tool_rounds):

            try:
                response = self._call_sister()
            except Exception as e:
                logging.error(f"call_sister failed : {e}")
                raise

            assistant_message, finish_reason = self._parse_response(response)

            if finish_reason != "tool_calls":
                self._save()
                return assistant_message["content"]

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
