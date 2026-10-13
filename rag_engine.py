"""
RAG 核心引擎 — PDF解析 / 文本分块 / 向量化 / 检索 / 问答
"""

import json
import os
import re
from pathlib import Path

import dashscope
import faiss
import numpy as np
import pdfplumber

# ── 配置 ──────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
INDEX_DIR = BASE_DIR / "index"
INDEX_DIR.mkdir(exist_ok=True)

FAISS_INDEX_PATH = INDEX_DIR / "faiss.index"
CHUNKS_PATH = INDEX_DIR / "chunks.json"

EMBEDDING_MODEL = "text-embedding-v3"
LLM_MODEL = "qwen-plus"
EMBEDDING_DIM = 1024


def set_api_key(key: str):
    dashscope.api_key = key


# ── PDF 解析 ──────────────────────────────────────────

def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    """提取 PDF 每页文本，返回 [{page: int, text: str}]"""
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            text = text.strip()
            if text:
                pages.append({"page": i + 1, "text": text})
    return pages


# ── 文本分块 ──────────────────────────────────────────

def chunk_pages(pages: list[dict], chunk_size: int = 500, overlap: int = 100) -> list[dict]:
    """
    将页面文本分块。
    返回 [{text, source_file, page, chunk_id}]
    """
    chunks = []
    chunk_id = 0

    for page_info in pages:
        text = page_info["text"]
        page_num = page_info["page"]
        source = page_info.get("source_file", "")

        # 按段落先切
        paragraphs = re.split(r"\n{2,}", text)
        buffer = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if len(buffer) + len(para) <= chunk_size:
                buffer = buffer + "\n" + para if buffer else para
            else:
                # buffer 已满，保存
                if buffer:
                    chunks.append({
                        "text": buffer,
                        "source_file": source,
                        "page": page_num,
                        "chunk_id": chunk_id,
                    })
                    chunk_id += 1
                    # 滑动窗口：保留尾部 overlap 字符
                    tail = buffer[-overlap:] if len(buffer) > overlap else buffer
                    buffer = tail + "\n" + para
                else:
                    buffer = para

        # 页面剩余
        if buffer:
            chunks.append({
                "text": buffer,
                "source_file": source,
                "page": page_num,
                "chunk_id": chunk_id,
            })
            chunk_id += 1

    return chunks


def process_pdf(pdf_path: str, chunk_size: int = 500, overlap: int = 100) -> list[dict]:
    """解析单个 PDF 并分块"""
    pages = extract_text_from_pdf(pdf_path)
    filename = os.path.basename(pdf_path)
    for p in pages:
        p["source_file"] = filename
    return chunk_pages(pages, chunk_size, overlap)


# ── Embedding ─────────────────────────────────────────

def get_embeddings(texts: list[str]) -> np.ndarray:
    """调用阿里 text-embedding-v3 批量获取向量"""
    all_embeddings = []
    batch_size = 25  # DashScope 单次限制

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = dashscope.TextEmbedding.call(
            model=EMBEDDING_MODEL,
            input=batch,
            dimension=EMBEDDING_DIM,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Embedding 调用失败: {resp.code} - {resp.message}")
        for item in resp.output["embeddings"]:
            all_embeddings.append(item["embedding"])

    return np.array(all_embeddings, dtype="float32")


# ── FAISS 索引 ────────────────────────────────────────

class KnowledgeIndex:
    """管理 FAISS 索引和文本块元数据"""

    def __init__(self):
        self.index: faiss.IndexFlatIP | None = None
        self.chunks: list[dict] = []

    def build(self, all_chunks: list[dict]):
        """从文本块列表构建索引"""
        if not all_chunks:
            raise ValueError("没有可索引的文本块")

        self.chunks = all_chunks
        texts = [c["text"] for c in all_chunks]
        embeddings = get_embeddings(texts)

        # 归一化后用内积 = 余弦相似度
        faiss.normalize_L2(embeddings)
        self.index = faiss.IndexFlatIP(EMBEDDING_DIM)
        self.index.add(embeddings)

        self._save()

    def _save(self):
        # 使用字节序列化避免 faiss C++ 层不支持中文路径的问题
        index_bytes = faiss.serialize_index(self.index)
        with open(FAISS_INDEX_PATH, "wb") as f:
            f.write(index_bytes.tobytes())
        with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)

    def load(self) -> bool:
        """加载已有索引，成功返回 True"""
        if FAISS_INDEX_PATH.exists() and CHUNKS_PATH.exists():
            try:
                with open(FAISS_INDEX_PATH, "rb") as f:
                    index_bytes = np.frombuffer(f.read(), dtype="uint8")
                self.index = faiss.deserialize_index(index_bytes)
                with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
                    self.chunks = json.load(f)
                return True
            except Exception:
                self.index = None
                self.chunks = []
                return False
        return False

    @property
    def ready(self) -> bool:
        return self.index is not None and len(self.chunks) > 0

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """检索最相关的 top_k 个文本块"""
        if not self.ready:
            return []

        q_emb = get_embeddings([query])
        faiss.normalize_L2(q_emb)
        scores, indices = self.index.search(q_emb, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            chunk = self.chunks[idx].copy()
            chunk["score"] = float(score)
            results.append(chunk)
        return results

    def stats(self) -> dict:
        """返回索引统计信息"""
        if not self.ready:
            return {"doc_count": 0, "chunk_count": 0, "files": []}
        files = list({c["source_file"] for c in self.chunks})
        return {
            "doc_count": len(files),
            "chunk_count": len(self.chunks),
            "files": files,
        }


# ── LLM 问答 ─────────────────────────────────────────

SYSTEM_PROMPT = """你是一个制度知识库助手。请根据以下参考内容回答用户的问题。
要求：
1. 回答要准确、简洁，基于参考内容作答
2. 如果参考内容中没有相关信息，请如实说明"知识库中未找到相关内容"
3. 在回答末尾标注信息来源（文件名和页码）"""


def _build_messages(query: str, context_chunks: list[dict]) -> list[dict]:
    """构建 LLM 消息列表"""
    context_parts = []
    for i, c in enumerate(context_chunks, 1):
        source = f"[来源: {c['source_file']} 第{c['page']}页]"
        context_parts.append(f"片段{i} {source}:\n{c['text']}")
    context = "\n\n".join(context_parts)

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"【参考内容】\n{context}\n\n【用户问题】\n{query}"},
    ]


def ask(query: str, context_chunks: list[dict]) -> str:
    """基于检索到的上下文调用 LLM 生成回答（非流式）"""
    if not context_chunks:
        return "知识库中未找到相关内容，请确认知识库已建立索引。"

    messages = _build_messages(query, context_chunks)
    resp = dashscope.Generation.call(
        model=LLM_MODEL,
        messages=messages,
        result_format="message",
    )

    if resp.status_code != 200:
        return f"LLM 调用失败: {resp.code} - {resp.message}"

    return resp.output.choices[0].message.content


def ask_stream(query: str, context_chunks: list[dict]):
    """
    流式问答生成器。
    yield ("thinking", text) 或 ("content", text)
    """
    if not context_chunks:
        yield ("content", "知识库中未找到相关内容，请确认知识库已建立索引。")
        return

    messages = _build_messages(query, context_chunks)

    responses = dashscope.Generation.call(
        model=LLM_MODEL,
        messages=messages,
        result_format="message",
        stream=True,
        incremental_output=True,
        enable_thinking=True,
    )

    for chunk in responses:
        if chunk.status_code != 200:
            yield ("content", f"\n\nLLM 调用失败: {chunk.code} - {chunk.message}")
            return

        choice = chunk.output.choices[0]
        msg = choice.message

        # thinking 内容
        thinking = getattr(msg, "reasoning_content", None) or ""
        if thinking:
            yield ("thinking", thinking)

        # 正文内容
        content = getattr(msg, "content", None) or ""
        if content:
            yield ("content", content)
