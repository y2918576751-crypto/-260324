"""
制度知识库问答系统 — Streamlit 主界面
"""

import os
import tempfile

import streamlit as st

from rag_engine import KnowledgeIndex, ask_stream, process_pdf, set_api_key

# ── 页面配置 ──────────────────────────────────────────
st.set_page_config(
    page_title="制度知识库问答系统",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 自定义样式 ────────────────────────────────────────
st.markdown("""
<style>
    /* 全局 */
    .stApp {
        background: linear-gradient(160deg, #f8f9fc 0%, #eef1f8 100%);
    }
    .block-container { max-width: 900px; padding-top: 2rem; }

    /* 侧边栏 */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1c2e 0%, #2d2f45 100%);
    }
    section[data-testid="stSidebar"] * {
        color: #e0e0e0 !important;
    }
    section[data-testid="stSidebar"] .stSlider label,
    section[data-testid="stSidebar"] .stTextInput label {
        color: #a0a4b8 !important; font-size: 13px;
    }
    section[data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.08);
    }
    section[data-testid="stSidebar"] .stMetric label {
        color: #8b8fa3 !important;
    }
    section[data-testid="stSidebar"] .stMetric [data-testid="stMetricValue"] {
        color: #7c8cf8 !important; font-size: 28px !important;
    }

    /* 标题区 */
    .title-area {
        text-align: center; padding: 1.5rem 0 1rem;
    }
    .title-area h1 {
        font-size: 1.8rem; font-weight: 700;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .title-area p {
        color: #888; font-size: 0.9rem;
    }

    /* 来源标签 */
    .source-chip {
        display: inline-flex; align-items: center; gap: 4px;
        background: #f0f4ff; color: #5b6abf;
        padding: 3px 10px; border-radius: 20px;
        font-size: 12px; margin: 2px 4px 2px 0;
        border: 1px solid #dde3ff;
    }
    .score-chip {
        display: inline-flex; align-items: center;
        background: #fff8ee; color: #c47d10;
        padding: 3px 10px; border-radius: 20px;
        font-size: 12px; margin: 2px 4px 2px 0;
        border: 1px solid #ffe8c2;
    }

    /* thinking 块 */
    .thinking-block {
        background: #f9f9fb; border-left: 3px solid #b8bfff;
        padding: 10px 14px; margin: 8px 0 12px;
        border-radius: 0 8px 8px 0; font-size: 13px;
        color: #666; line-height: 1.6;
        max-height: 300px; overflow-y: auto;
    }
    .thinking-label {
        font-size: 12px; color: #9a9ec4; font-weight: 600;
        margin-bottom: 4px; display: flex; align-items: center; gap: 4px;
    }

    /* 空状态 */
    .empty-state {
        text-align: center; padding: 4rem 2rem; color: #aaa;
    }
    .empty-state .icon { font-size: 3rem; margin-bottom: 1rem; }
    .empty-state p { font-size: 1rem; }

    /* chat input 美化 */
    .stChatInput textarea { border-radius: 24px !important; }
</style>
""", unsafe_allow_html=True)

# ── 初始化 Session State ─────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "knowledge_index" not in st.session_state:
    st.session_state.knowledge_index = KnowledgeIndex()
if "api_key_set" not in st.session_state:
    st.session_state.api_key_set = False

ki = st.session_state.knowledge_index

# ── 侧边栏 ───────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🗂️ 知识库管理")

    # API Key
    api_key = st.text_input(
        "DashScope API Key",
        value="sk-fe35d14176d74c33892f2dfe011b58e0",
        type="password",
    )
    if api_key:
        set_api_key(api_key)
        st.session_state.api_key_set = True

    st.divider()

    # 文件上传
    st.markdown("##### 📄 上传文档")
    uploaded_files = st.file_uploader(
        "拖拽或点击上传 PDF",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    # 参数
    st.divider()
    st.markdown("##### 🎛️ 检索参数")
    chunk_size = st.slider("分块大小 (字)", 200, 1000, 500, step=50)
    chunk_overlap = st.slider("重叠大小 (字)", 0, 200, 100, step=20)
    top_k = st.slider("Top-K 召回数量", 1, 10, 5)

    # 构建索引
    st.divider()
    if st.button("🔄 构建 / 重建索引", use_container_width=True, type="primary"):
        if not uploaded_files:
            st.warning("请先上传 PDF 文件")
        elif not st.session_state.api_key_set:
            st.warning("请先设置 API Key")
        else:
            all_chunks = []
            progress = st.progress(0, text="正在处理文档...")
            for i, f in enumerate(uploaded_files):
                progress.progress(i / len(uploaded_files), text=f"正在解析: {f.name}")
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                    tmp.write(f.read())
                    tmp_path = tmp.name
                try:
                    chunks = process_pdf(tmp_path, chunk_size, chunk_overlap)
                    all_chunks.extend(chunks)
                finally:
                    os.unlink(tmp_path)

            progress.progress(0.8, text="正在向量化并构建索引...")
            try:
                ki.build(all_chunks)
                progress.progress(1.0, text="完成!")
                st.success(f"索引构建成功 — {len(all_chunks)} 个文本块")
            except Exception as e:
                st.error(f"索引构建失败: {e}")

    # 加载已有索引
    if not ki.ready:
        ki.load()

    # 索引状态
    st.divider()
    st.markdown("##### 📊 索引状态")
    stats = ki.stats()
    if stats["chunk_count"] > 0:
        col1, col2 = st.columns(2)
        col1.metric("文档", stats["doc_count"])
        col2.metric("文本块", stats["chunk_count"])
        for fname in stats["files"]:
            st.caption(f"✅ {fname}")
    else:
        st.caption("暂无索引，请上传文档并构建")

    # 清空对话
    st.divider()
    if st.button("🗑️ 清空对话", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ── 辅助函数 ──────────────────────────────────────────

def render_sources(sources: list[dict]):
    """渲染检索来源标签"""
    if not sources:
        return
    with st.expander("🔍 检索来源", expanded=False):
        for src in sources:
            score_pct = f"{src['score']:.0%}"
            st.markdown(
                f"<span class='source-chip'>📖 {src['source_file']} 第{src['page']}页</span>"
                f"<span class='score-chip'>相似度 {score_pct}</span>",
                unsafe_allow_html=True,
            )
            preview = src["text"][:200] + "..." if len(src["text"]) > 200 else src["text"]
            st.caption(preview)


def render_thinking(thinking_text: str):
    """渲染 thinking 块"""
    if not thinking_text:
        return
    with st.expander("💭 思考过程", expanded=False):
        st.markdown(
            f"<div class='thinking-block'>{thinking_text}</div>",
            unsafe_allow_html=True,
        )


# ── 主界面 ────────────────────────────────────────────

st.markdown("""
<div class="title-area">
    <h1>📚 制度知识库问答系统</h1>
    <p>上传制度文档，构建索引，即可智能问答</p>
</div>
""", unsafe_allow_html=True)

# 空状态提示
if not ki.ready and not st.session_state.messages:
    st.markdown("""
    <div class="empty-state">
        <div class="icon">📂</div>
        <p>请在左侧上传 PDF 文档并构建索引<br/>然后就可以开始提问了</p>
    </div>
    """, unsafe_allow_html=True)

# 历史消息
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    else:
        with st.chat_message("assistant", avatar="✨"):
            # thinking
            render_thinking(msg.get("thinking", ""))
            st.markdown(msg["content"])
            render_sources(msg.get("sources", []))

# ── 用户输入 & 流式回复 ──────────────────────────────
if query := st.chat_input("输入你的问题..."):
    if not ki.ready:
        st.warning("请先构建知识库索引")
        st.stop()

    # 用户消息
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # AI 回复（流式）
    with st.chat_message("assistant", avatar="✨"):
        # 1) 检索
        with st.spinner("正在检索知识库..."):
            results = ki.search(query, top_k=top_k)

        # 2) 流式生成
        thinking_placeholder = st.empty()
        content_placeholder = st.empty()

        thinking_buf = ""
        content_buf = ""
        is_thinking_phase = True

        for chunk_type, chunk_text in ask_stream(query, results):
            if chunk_type == "thinking":
                thinking_buf += chunk_text
                thinking_placeholder.markdown(
                    f"<div class='thinking-label'>💭 思考中...</div>"
                    f"<div class='thinking-block'>{thinking_buf}</div>",
                    unsafe_allow_html=True,
                )
            elif chunk_type == "content":
                if is_thinking_phase and thinking_buf:
                    # thinking 阶段结束，折叠起来
                    is_thinking_phase = False
                    thinking_placeholder.empty()
                content_buf += chunk_text
                content_placeholder.markdown(content_buf)

        # 3) 最终渲染 thinking（折叠）
        if thinking_buf:
            thinking_placeholder.empty()
            content_placeholder.empty()
            render_thinking(thinking_buf)
            st.markdown(content_buf)

        # 4) 检索来源
        sources = []
        for r in results:
            sources.append({
                "source_file": r["source_file"],
                "page": r["page"],
                "score": r["score"],
                "text": r["text"],
            })
        render_sources(sources)

        # 5) 保存到历史
        st.session_state.messages.append({
            "role": "assistant",
            "content": content_buf,
            "thinking": thinking_buf,
            "sources": sources,
        })
