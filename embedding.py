"""轻量 Embedding 客户端（OpenAI 兼容 /v1/embeddings）。

供「长期记忆向量召回」使用，后续接入 RAG 知识库时可直接复用：
  1. 文本入库时算出向量与正文一起持久化；
  2. 检索时用用户问题算出 query 向量，再做相似度排序。

配置（.env）：
    EMBEDDING_MODEL     必须：Embedding 模型名（如 ollama 的 bge-m3）
    EMBEDDING_BASE_URL  可选：缺省复用 OPENAI_BASE_URL
    EMBEDDING_API_KEY   可选：缺省复用 OPENAI_API_KEY
未配置 EMBEDDING_MODEL 或调用失败时，embed_texts 返回 None，
调用方（memory.py）会自动退化为关键词重叠召回，保证离线可用。
"""
import logging
import os
import threading

logger = logging.getLogger(__name__)

_client = None
_client_lock = threading.Lock()


def _get_client():
    """懒加载单例；与 chat_completion 保持一致，走直连不走代理。"""
    global _client
    if _client is None:
        from openai import OpenAI
        import httpx

        with _client_lock:
            if _client is None:
                api_key = os.getenv("EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY")
                base_url = os.getenv("EMBEDDING_BASE_URL") or os.getenv("OPENAI_BASE_URL")
                _client = OpenAI(
                    api_key=api_key,
                    base_url=base_url,
                    http_client=httpx.Client(proxy=None),
                )
    return _client


def embed_texts(texts):
    """把一组文本转成向量列表。

    返回 list[list[float]]，与 texts 一一对应；
    Embedding 未配置 / 调用失败返回 None（由调用方兜底）。
    """
    model = os.getenv("EMBEDDING_MODEL")
    if not model:
        return None
    try:
        resp = _get_client().embeddings.create(model=model, input=list(texts))
        return [d.embedding for d in resp.data]
    except Exception as e:
        logger.warning(f"embed failed (fallback to keyword recall): {e}")
        return None


def embed_available() -> bool:
    return bool(os.getenv("EMBEDDING_MODEL"))
