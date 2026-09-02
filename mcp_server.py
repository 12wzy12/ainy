"""MCP Server —— 把「天气查询 / 日期查询」等能力封装为标准 MCP Server。

基于官方 mcp Python SDK（mcp v2 的 MCPServer，即原 FastMCP）：
    1. 用 @server.tool 装饰器声明工具，参数类型注解自动生成 JSON Schema；
    2. Agent 通过协议动态发现工具（tools/list）、统一执行（tools/call），
       不感知任何工具实现细节；
    3. 新增工具 = 加一个 @server.tool 函数并重启，Agent 端零改动。

单独启动（调试）：python mcp_server.py   （stdio 传输）
"""
import logging
import sys
from datetime import datetime
from urllib.parse import quote

from mcp.server.mcpserver import MCPServer

try:
    import httpx
except ImportError:  # 无 httpx 时天气接口直接走降级
    httpx = None

server = MCPServer("companion-tools")


@server.tool(description="根据用户说的地点获取该地的当前天气")
def get_weather(location: str) -> str:
    """查询某地当前天气。

    优先调公开天气接口 wttr.in（无需密钥），接口不可达/超时则降级为
    演示数据，保证工具始终可用。
    """
    if httpx is not None:
        try:
            resp = httpx.get(
                f"https://wttr.in/{quote(location)}",
                params={"format": "%l: %t, %C, 风速%w", "lang": "zh"},
                timeout=4,
                follow_redirects=True,
            )
            if resp.status_code == 200 and resp.text.strip():
                return resp.text.strip()
        except Exception as e:
            logging.getLogger(__name__).warning(f"weather api failed: {e}")
    return f"{location}天气良好，23℃，微风（演示数据：外部天气接口不可用）"


@server.tool(description="获取今天的日期、时间与星期")
def get_date() -> str:
    """返回当前日期时间与星期。"""
    now = datetime.now()
    week = "一二三四五六日"[now.weekday()]
    return f"{now:%Y-%m-%d %H:%M:%S} 星期{week}"


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
    server.run()  # stdio 传输
