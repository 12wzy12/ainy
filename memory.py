"""长期记忆：SQLite 持久化 + 按用户问题向量召回。

数据模型：一条记忆 = 一段由 LLM 从旧对话压缩出的要点摘要。
    memories(id, created_at, content, embedding)
    - content    记忆正文
    - embedding  写入时的文本向量（JSON），Embedding 不可用时为 NULL

召回：以用户当前问题为 query 做余弦相似度取 top-K，只把相关记忆注入
上下文，避免全量加载导致的 Token 浪费；Embedding 不可用（未配置模型或
调用失败）时自动退化为「关键词重叠 + 时间近邻」，保证无 Embedding 也能跑。
"""
import json
import logging
import os
import re
import sqlite3

import embedding

logger = logging.getLogger(__name__)

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB_PATH = os.path.join(_MODULE_DIR, "memory.db")

_CJK_RE = re.compile(r"[一-鿿]")
_WORD_RE = re.compile(r"[a-zA-Z0-9_]+")


def _tokenize(text):
    """把文本切成 token 集合：英文/数字按词，中文按字 + 相邻字二元组。"""
    tokens = set()
    lowered = (text or "").lower()
    for w in _WORD_RE.findall(lowered):
        tokens.add(w)
    for m in _CJK_RE.finditer(lowered):
        tokens.add(m.group(0))  # 单字（兼顾短文本）
    # 相邻中文字符的二元组，能更好表达词义
    cjk_chars = "".join(_CJK_RE.findall(lowered))
    for i in range(len(cjk_chars) - 1):
        tokens.add(cjk_chars[i : i + 2])
    return tokens


def _overlap_score(query, doc):
    """Jaccard 式关键词重叠分 ∈ [0, 1]，用于无向量时的召回兜底。"""
    q, d = _tokenize(query), _tokenize(doc)
    if not q or not d:
        return 0.0
    return len(q & d) / len(q | d)


def _cosine(a, b):
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class MemoryStore:
    """长期记忆的 SQLite 持久化与向量召回。"""

    def __init__(self, db_path: str = DEFAULT_DB_PATH, max_rows: int = 60):
        self.db_path = db_path
        self.max_rows = max_rows
        self.conn = sqlite3.connect(db_path)
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
                content    TEXT NOT NULL,
                embedding  TEXT
            )
            """
        )
        # 兼容旧库：早期版本建表时没有 embedding 列
        cols = {r[1] for r in self.conn.execute("PRAGMA table_info(memories)")}
        if "embedding" not in cols:
            self.conn.execute("ALTER TABLE memories ADD COLUMN embedding TEXT")
        self.conn.commit()

    # ---------- 写入 ----------

    def _to_vector(self, content: str):
        """算向量；返回 list[float] 或 None（不可用）。"""
        try:
            vecs = embedding.embed_texts([content])
            return vecs[0] if vecs else None
        except Exception as e:
            logger.warning(f"embed failed: {e}")
            return None

    def add(self, content: str) -> int:
        """新增一条长期记忆（写入时顺带落向量），随后裁剪到上限。"""
        cur = self.conn.execute(
            "INSERT INTO memories(content, embedding) VALUES (?, ?)",
            (content, json.dumps(self._to_vector(content), ensure_ascii=False)),
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

    # ---------- 召回 ----------

    def recall(self, query: str, k: int = 3) -> list:
        """按用户当前问题召回最相关的 k 条记忆（按相似度降序）。

        有向量：余弦相似度为主；历史遗留的无向量记录用关键词重叠×0.5 兜底；
        无向量可用：统一用关键词重叠分（分数并列时新的优先）。
        """
        rows = self.conn.execute(
            "SELECT id, content, embedding FROM memories"
        ).fetchall()
        if not rows:
            return []

        qvec = self._to_vector(query)  # None 表示 Embedding 不可用
        scored = []
        for rid, content, raw_vec in rows:
            if qvec and raw_vec:
                vec = json.loads(raw_vec)
                sim = _cosine(qvec, vec)
            elif qvec:
                # 旧数据没有向量：弱化处理，仍有机会被召回
                sim = _overlap_score(query, content) * 0.5
            else:
                sim = _overlap_score(query, content)
            if sim > 0:
                scored.append((rid, content, sim))

        scored.sort(key=lambda x: (-x[2], -x[0]))  # 相似度降序，同等取较新
        return [content for _, content, _ in scored[:k]]

    def all(self) -> list:
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
        try:
            self.conn.close()
        except Exception:
            pass
