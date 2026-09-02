"""MCP 客户端 / 工具注册中心（Agent 进程内使用，基于官方 mcp SDK）。

Agent 侧是同步代码，而官方 SDK 的 stdio 客户端是异步接口，因此这里用
「后台事件循环线程」做桥接：对每个 MCP Server 拉起的子进程连接
（stdio transport + ClientSession），initialize / tools/list / tools/call
都在同一个 loop 中执行，对上层保持同步调用语义。

工作方式：
    1. 按配置拉起 MCP Server 子进程，完成 initialize 握手；
    2. tools/list 动态拉取工具清单，翻译成 OpenAI Function Calling 格式；
    3. Agent 决策后把 tool call 转发为 tools/call，结果回填对话上下文。

解耦收益：Agent 不感知任何工具实现；新增 Server 只需在 DEFAULT_MCP_SERVERS
追加一条配置，第三方 MCP Server 也能直接接入。
"""
import asyncio
import atexit
import logging
import os
import sys
import threading

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logger = logging.getLogger(__name__)

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

# 默认 MCP Server 配置；接入更多 Server（含第三方）时在列表里追加即可
DEFAULT_MCP_SERVERS = [
    {
        "name": "companion-tools",
        "command": sys.executable,
        "args": [os.path.join(_MODULE_DIR, "mcp_server.py")],
    },
]

# 仅保留对 OpenAI Function Calling 有意义的 schema 键
_DROP_SCHEMA_KEYS = {"title", "$schema", "additionalProperties", "definitions"}


def _sanitize_schema(schema: dict) -> dict:
    """把 MCP 工具返回的 JSON Schema 清洗成 OpenAI function.parameters。"""
    if not isinstance(schema, dict):
        return {"type": "object", "properties": {}}
    return {k: v for k, v in schema.items() if k not in _DROP_SCHEMA_KEYS}


class _AsyncLoopThread:
    """持有一个永不停止的 asyncio loop，供所有 MCP 连接共用。"""

    def __init__(self):
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run, daemon=True, name="mcp-loop")
        self._thread.start()

    def _run(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def run(self, coro, timeout: float):
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return fut.result(timeout=timeout)

    def stop(self):
        try:
            self._loop.call_soon_threadsafe(self._loop.stop)
        except Exception:
            pass
        self._thread.join(timeout=3)


class McpServerSync:
    """一个 MCP Server 子进程的同步封装（会话常驻，跨多次调用复用）。"""

    def __init__(self, loop: _AsyncLoopThread, name: str, command: str, args: list,
                 timeout: float = 15.0):
        self.name = name
        self.timeout = timeout
        self._loop = loop
        env = dict(os.environ)
        env["PYTHONIOENCODING"] = "utf-8"  # Windows 下子进程编码统一为 UTF-8
        self._params = StdioServerParameters(
            command=command, args=args, env=env,
        )
        self._client_ctx = None
        self._session_ctx = None
        self._session = None

    def open(self):
        """启动子进程并完成 initialize 握手（失败抛异常，由注册中心兜底）。"""
        async def _open():
            self._client_ctx = stdio_client(self._params)
            read, write = await self._client_ctx.__aenter__()
            self._session_ctx = ClientSession(read, write)
            self._session = await self._session_ctx.__aenter__()
            await self._session.initialize()

        self._loop.run(_open(), self.timeout)

    def list_tools(self) -> list:
        async def _list():
            result = await self._session.list_tools()
            return list(result.tools)

        return self._loop.run(_list(), self.timeout)

    def call_tool(self, name: str, args: dict) -> str:
        async def _call():
            result = await self._session.call_tool(name, arguments=args)
            texts = "".join(
                c.text for c in result.content if getattr(c, "type", "") == "text"
            )
            if getattr(result, "isError", False):
                raise RuntimeError(texts or "工具执行失败")
            return texts

        return self._loop.run(_call(), self.timeout)

    def close(self):
        async def _close():
            try:
                if self._session_ctx is not None:
                    await self._session_ctx.__aexit__(None, None, None)
            except Exception:
                pass
            try:
                if self._client_ctx is not None:
                    await self._client_ctx.__aexit__(None, None, None)
            except Exception:
                pass

        try:
            self._loop.run(_close(), self.timeout)
        except Exception:
            pass


class McpToolRegistry:
    """管理一组 MCP Server：动态发现工具（→ OpenAI schema）并执行工具调用。"""

    def __init__(self, servers: list = None, timeout: float = 15.0):
        self._loop = _AsyncLoopThread()
        self.connections = []
        self.tools = []          # OpenAI function-calling 格式的工具清单
        self._tool_owner = {}    # 工具名 -> McpServerSync

        for cfg in servers if servers is not None else DEFAULT_MCP_SERVERS:
            self._connect_server(cfg, timeout)

        atexit.register(self.close)

    def _connect_server(self, cfg: dict, timeout: float):
        """连接一个 MCP Server 并拉取/翻译它的工具清单；失败不影响其它 Server。"""
        conn = None
        try:
            conn = McpServerSync(
                self._loop, cfg["name"], cfg["command"],
                cfg.get("args", []), timeout=timeout,
            )
            conn.open()
            raw_tools = conn.list_tools()
        except Exception as e:
            logger.error(f"MCP server {cfg['name']} 连接失败，跳过: {e}")
            if conn is not None:
                conn.close()
            return

        for t in raw_tools:
            self.tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description or "",
                        "parameters": _sanitize_schema(t.input_schema),
                    },
                }
            )
            self._tool_owner[t.name] = conn
        self.connections.append(conn)
        logger.info(f"MCP server {cfg['name']} 就绪，工具: {[t.name for t in raw_tools]}")

    def call_tool(self, name: str, args: dict) -> str:
        """执行一次工具调用，返回文本结果。"""
        conn = self._tool_owner.get(name)
        if conn is None:
            raise KeyError(f"未注册的工具: {name}")
        return conn.call_tool(name, args or {})

    def close(self):
        for conn in self.connections:
            try:
                conn.close()
            except Exception:
                pass
        self.connections = []
        self._loop.stop()
