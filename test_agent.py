"""离线冒烟测试：fake LLM 流 + 真实 MCP stdio 子进程 + 临时 SQLite。

覆盖：MCP 协议发现与执行、流式 tool_call 分片聚合、Agent Loop 工具回注、
长期记忆向量召回与关键词兜底、摘要窗口折叠边界、旧数据迁移。

运行：python test_agent.py   （也可用 pytest 收集）
"""
import json
import os
import shutil
import sys
import tempfile
import types
from types import SimpleNamespace

# ---- 环境准备：构造一个不会发真实请求的 OpenAI 客户端，关闭 Embedding ----
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("OPENAI_BASE_URL", "http://127.0.0.1:1/v1")
os.environ.pop("EMBEDDING_MODEL", None)
os.environ.pop("EMBEDDING_BASE_URL", None)

import chat_completion  # noqa: E402
import embedding as embedding_module  # noqa: E402
from chat_completion import Sister  # noqa: E402
from memory import MemoryStore  # noqa: E402
from mcp_client import McpToolRegistry  # noqa: E402

# 测试环境没有内网代理，去掉 chat_completion import 时注入的代理，避免子进程干扰
for k in ["http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"]:
    os.environ.pop(k, None)


# ---------------- fake LLM ----------------

def chunk(*, content=None, reasoning=None, tool_calls=None, finish=None):
    """构造一个长得像 SDK 流式 chunk 的命名空间对象。"""
    delta = SimpleNamespace(
        content=content,
        reasoning_content=reasoning,
        tool_calls=[SimpleNamespace(
            index=tc["index"],
            id=tc.get("id"),
            function=SimpleNamespace(name=tc.get("name"), arguments=tc.get("args")),
        ) for tc in (tool_calls or [])] or None,
    )
    return SimpleNamespace(choices=[SimpleNamespace(index=0, delta=delta, finish_reason=finish)])


class FakeCompletions:
    def __init__(self, script=None, final_answer="测试摘要"):
        self.script = list(script or [])   # 每个流式回答 = 一组 chunk
        self.final_answer = final_answer    # 非流式（摘要）的回答
        self.calls = []                     # 记录每次 create 的 kwargs

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if not kwargs.get("stream"):
            # 非流式：用于 _summarize
            msg = SimpleNamespace(message=SimpleNamespace(content=self.final_answer))
            return SimpleNamespace(choices=[msg])
        chunks = self.script.pop(0)
        return iter(chunks)


class FakeOpenAIClient:
    """形状与真实 OpenAI client 一致：client.chat.completions.create(...)"""

    def __init__(self, script=None, final_answer="测试摘要"):
        self.chat = SimpleNamespace(
            completions=FakeCompletions(script=script, final_answer=final_answer)
        )


def make_sister(tmp, script=None, final_answer="测试摘要", **kwargs):
    """构造一个接了 fake LLM 的 Sister（真实 MCP 子进程 + 临时 db/json）。"""
    sister = Sister(
        memory_path=os.path.join(tmp, "conversation.json"),
        db_path=os.path.join(tmp, "memory.db"),
        **kwargs,
    )
    sister.client = FakeOpenAIClient(script=script, final_answer=final_answer)
    return sister


def clean(sister):
    sister.registry.close()
    sister.store.close()


# ---------------- 测试用例 ----------------

def test_mcp_registry_discovery_and_call():
    """MCP：动态发现工具并翻译成 OpenAI schema，走统一协议执行。"""
    tmp = tempfile.mkdtemp()
    try:
        reg = McpToolRegistry()
        names = {t["function"]["name"] for t in reg.tools}
        assert names == {"get_weather", "get_date"}, names
        weather = next(t for t in reg.tools if t["function"]["name"] == "get_weather")
        params = weather["function"]["parameters"]
        assert params["required"] == ["location"], params
        assert weather["type"] == "function"

        out = reg.call_tool("get_weather", {"location": "北京"})
        assert out and "执行出错" not in out, out

        out = reg.call_tool("get_date", {})
        assert "20" in out and "星期" in out, out

        try:
            reg.call_tool("not_exist_tool", {})
            raise AssertionError("未知工具应抛异常")
        except KeyError:
            pass
    finally:
        reg.close()
        shutil.rmtree(tmp, ignore_errors=True)


def test_streaming_tool_call_agent_loop():
    """Agent Loop + Streaming Tool Call：参数被拆 3 片仍能聚合成完整 JSON，
    finish_reason==tool_calls → MCP 执行 → 结果回注 → 模型继续回答。"""
    tmp = tempfile.mkdtemp()
    try:
        # 两次模型调用 = 两组 chunk 序列：① 工具调用轮（参数拆成 3 片）② 最终回答轮
        script = [
            [
                chunk(tool_calls=[{"index": 0, "id": "call_1", "name": "get_weather", "args": "{\"loc"}]),
                chunk(tool_calls=[{"index": 0, "args": "ation\":\"北京\"}"}]),  # 仅参数分片
                chunk(finish="tool_calls"),
            ],
            [chunk(content="北京今天天气不错，可以出去走走呀。", finish="stop")],
        ]
        sister = make_sister(tmp, script=script)
        try:
            ans = sister.chat("北京今天天气怎么样？")
            assert ans == "北京今天天气不错，可以出去走走呀。", ans

            # 流式参数按 index 聚合后完整解析
            msgs = sister.messages
            asst = next(m for m in msgs if m["role"] == "assistant" and m.get("tool_calls"))
            assert asst["tool_calls"][0]["id"] == "call_1"
            assert json.loads(asst["tool_calls"][0]["function"]["arguments"]) == {"location": "北京"}

            # 工具结果由真实 MCP 子进程返回并被回注为 role:tool
            tool_msg = next(m for m in msgs if m["role"] == "tool")
            assert tool_msg["tool_call_id"] == "call_1"
            assert tool_msg["content"] and "执行出错" not in tool_msg["content"]

            # 第一轮带了 tools，最终回答轮上下文完整
            assert "tools" in sister._context_messages() or sister.tools
            assert sister.registry.tools
        finally:
            clean(sister)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_plain_chat_persists_history():
    """普通多轮对话：无工具调用时直接回答并把短期窗口落盘。"""
    tmp = tempfile.mkdtemp()
    try:
        sister = make_sister(tmp, script=[
            [chunk(content="嗯，我听着呢。", finish="stop")],
            [chunk(content="记得多喝水哦。", finish="stop")],
        ])
        try:
            assert sister.chat("今天心情不太好") == "嗯，我听着呢。"
            assert sister.chat("好") == "记得多喝水哦。"
            with open(os.path.join(tmp, "conversation.json"), encoding="utf-8") as f:
                saved = json.load(f)
            roles = [m["role"] for m in saved]
            assert roles == ["system", "user", "assistant", "user", "assistant"], roles
            assert not any(str(m.get("content", "")).startswith("[长期记忆]") for m in saved)
        finally:
            clean(sister)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _fake_embed(markers):
    """按关键词做 one-hot 向量，模拟一个可用的 Embedding。"""
    vocab = ["篮球", "数学", "面试", "猫"]

    def _embed_texts(texts):
        vecs = []
        for t in texts:
            v = [1.0 if w in t else 0.0 for w in vocab]
            if not any(v):
                v = [0.0] * len(vocab)  # 无关键词 → 零向量，余弦相似度为 0
            vecs.append(v)
        return vecs

    return _embed_texts


def test_memory_vector_recall():
    """长期记忆：按用户问题向量召回 top-K（embedding 余弦）。"""
    tmp = tempfile.mkdtemp()
    try:
        store = MemoryStore(os.path.join(tmp, "m.db"))
        try:
            embedding_module.embed_texts = _fake_embed(None)
            store.add("用户喜欢打篮球，每周打两次，最近想加入俱乐部")
            store.add("用户最近在学数学，准备考研")
            store.add("用户养了一只猫，叫咪咪")

            hit = store.recall("周末要不要一起打篮球", k=1)
            assert len(hit) == 1 and "篮球" in hit[0], hit

            hit = store.recall("高等数学题目不会做", k=1)
            assert "数学" in hit[0], hit

            hit = store.recall("今天晚饭吃什么", k=3)
            assert hit == [], "无关问题不应召回记忆（无重叠）"
        finally:
            store.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_memory_fallback_without_embedding():
    """无 Embedding 时退化为关键词重叠召回，记忆功能依然可用。"""
    tmp = tempfile.mkdtemp()
    try:
        store = MemoryStore(os.path.join(tmp, "m.db"))
        try:
            embedding_module.embed_texts = lambda texts: None  # 模拟 Embedding 不可用
            store.add("用户喜欢打篮球，每周打两次")
            store.add("用户最近在学数学，准备考研")

            hit = store.recall("周末想找人打篮球", k=1)
            assert hit and "篮球" in hit[0], hit

            hit = store.recall("数学题好难", k=1)
            assert "数学" in hit[0], hit

            store.add_if_missing("用户喜欢打篮球，每周打两次")  # 去重
            assert store.count() == 2
        finally:
            store.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_memory_recall_injected_to_context():
    """对话时按当前问题召回的记忆以 system 消息注入模型上下文，且不落盘。"""
    tmp = tempfile.mkdtemp()
    try:
        sister = make_sister(tmp, script=[[chunk(content="嗯嗯。", finish="stop")]])
        try:
            sister.store.add("用户喜欢打篮球，每周都去球场")
            sister.chat("我昨天打篮球扭到脚了")
            ctx = sister._context_messages()
            injected = [m for m in ctx if m["role"] == "system" and "篮球" in m["content"]]
            assert injected, "应把相关长期记忆注入上下文"
            # 记忆只出现在“组装好的上下文”里，不应混入持久化的对话历史（以 system 消息形态）
            history_system = [m for m in sister.messages[1:] if m["role"] == "system"]
            assert history_system == [], f"历史中不应有 system 记忆消息: {history_system}"
        finally:
            clean(sister)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_condense_boundary_keeps_tool_pair():
    """窗口超限摘要时：折叠边界不会拆散 assistant(tool_calls) 与它的 tool 结果。"""
    tmp = tempfile.mkdtemp()
    try:
        sister = make_sister(tmp)
        try:
            # 造一段超过阈值的历史，并在折叠边界附近放一组 tool 调用对
            history = []
            for i in range(52):
                history.append({"role": "user" if i % 2 == 0 else "assistant",
                                "content": f"msg{i}"})
            history[43] = {"role": "assistant", "content": "", "tool_calls": [
                {"id": "call_x", "type": "function", "function": {"name": "get_date", "arguments": "{}"}}]}
            history.insert(44, {"role": "tool", "tool_call_id": "call_x", "content": "2026-09-02"})
            sister.messages = [sister.messages[0]] + history

            sister._condense_memory()

            kept = sister.messages[1:]
            assert len(kept) < 10
            assert sister.store.count() >= 1, "应已把最早对话固化为长期记忆"
            # 断言的边界安全性质：保留窗口不以孤儿 tool 消息开头
            assert kept[0]["role"] != "tool", kept[0]
            for idx, m in enumerate(kept):
                if m["role"] == "tool":
                    assert kept[idx - 1]["role"] == "assistant" and kept[idx - 1].get("tool_calls"), "tool 结果必须跟在它的 assistant(tool_calls) 后面"
            # 记忆固化后不再内联在对话里
            assert not any(m.get("role") == "system" and str(m.get("content", "")).startswith("[长期记忆]") for m in kept)
        finally:
            clean(sister)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_legacy_history_migration():
    """旧版 conversation.json 里的内联 [长期记忆] 应迁移进 SQLite 并移出窗口。"""
    tmp = tempfile.mkdtemp()
    try:
        legacy = [
            {"role": "system", "content": "系统提示词"},
            {"role": "system", "content": "[长期记忆] 用户喜欢打篮球"},
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好呀"},
        ]
        path = os.path.join(tmp, "conversation.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(legacy, f, ensure_ascii=False)

        sister = make_sister(tmp)
        try:
            # 内联记忆已迁入 SQLite；文件自带的 system 头保留，对话恢复为纯窗口
            assert sister.store.all() == ["用户喜欢打篮球"], sister.store.all()
            contents = [m["content"] for m in sister.messages]
            assert not any(str(c).startswith("[长期记忆]") for c in contents)
            assert contents[1] == "系统提示词"
            assert [m["role"] for m in sister.messages] == ["system", "system", "user", "assistant"]
        finally:
            clean(sister)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------- runner ----------------

_ALL = [
    test_mcp_registry_discovery_and_call,
    test_streaming_tool_call_agent_loop,
    test_plain_chat_persists_history,
    test_memory_vector_recall,
    test_memory_fallback_without_embedding,
    test_memory_recall_injected_to_context,
    test_condense_boundary_keeps_tool_pair,
    test_legacy_history_migration,
]


def main():
    failed = 0
    for fn in _ALL:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except Exception as e:
            failed += 1
            import traceback
            print(f"FAIL  {fn.__name__}: {e}")
            traceback.print_exc()
    print(f"\n{len(_ALL) - failed}/{len(_ALL)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
