"""
Folio — Polished RAG Book Assistant v4
Run:  streamlit run folio_app.py
Requires in .env:
    MISTRAL_API_KEY   (optional)
    GROQ_API_KEY      (optional)
    GOOGLE_API_KEY    (optional)
At least one LLM key must be present.

Supported file formats:
    PDF, DOCX, TXT, MD, CSV, PPTX
Install extras:
    pip install python-docx docx2txt unstructured python-pptx
"""

import os, time, json, textwrap, re
from datetime import datetime
import tempfile

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    CSVLoader,
    UnstructuredPowerPointLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Folio",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Session defaults
# ─────────────────────────────────────────────────────────────────────────────
DEFAULTS = {
    "chat_history": [],
    "pinned": [],
    "doc_stats": None,
    "db_ready": False,
    "summary": None,
    "theme": "dark",
    "llm_choice": "Auto",
    "conv_messages": [],
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

IS_DARK = st.session_state.theme == "dark"

# ─────────────────────────────────────────────────────────────────────────────
# Themes
# ─────────────────────────────────────────────────────────────────────────────
DARK = {
    "bg":        "#0b0c0f",
    "surface":   "#13141a",
    "surface2":  "#1c1d26",
    "border":    "#252630",
    "accent":    "#e2b96f",
    "accent2":   "#6fb5e2",
    "danger":    "#e26f6f",
    "success":   "#6fe2a0",
    "text":      "#eae6dd",
    "muted":     "#666070",
    "pin":       "#e2b96f22",
}
LIGHT = {
    "bg":        "#f7f4ee",
    "surface":   "#ffffff",
    "surface2":  "#f0ece3",
    "border":    "#ddd8cc",
    "accent":    "#9a6f28",
    "accent2":   "#2876a0",
    "danger":    "#a03028",
    "success":   "#287040",
    "text":      "#1a1810",
    "muted":     "#8a8070",
    "pin":       "#9a6f2818",
}
T = DARK if IS_DARK else LIGHT

# ─────────────────────────────────────────────────────────────────────────────
# Supported formats config
# ─────────────────────────────────────────────────────────────────────────────
FORMAT_INFO = {
    "pdf":  {"icon": "📕", "label": "PDF"},
    "docx": {"icon": "📘", "label": "Word"},
    "doc":  {"icon": "📘", "label": "Word"},
    "txt":  {"icon": "📄", "label": "Text"},
    "md":   {"icon": "📝", "label": "Markdown"},
    "csv":  {"icon": "📊", "label": "CSV"},
    "pptx": {"icon": "📙", "label": "PowerPoint"},
}
ACCEPTED_TYPES = list(FORMAT_INFO.keys())

# ─────────────────────────────────────────────────────────────────────────────
# CSS injection
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,400;0,600;1,400&family=Fira+Code:wght@300;400&family=Jost:wght@300;400;500&display=swap');

:root {{
  --bg:       {T['bg']};
  --surf:     {T['surface']};
  --surf2:    {T['surface2']};
  --bdr:      {T['border']};
  --acc:      {T['accent']};
  --acc2:     {T['accent2']};
  --err:      {T['danger']};
  --ok:       {T['success']};
  --txt:      {T['text']};
  --muted:    {T['muted']};
  --pin:      {T['pin']};
  --r:        7px;
  --trans:    .18s ease;
}}

html, body, [class*="css"] {{
  background-color: var(--bg) !important;
  color: var(--txt) !important;
  font-family: 'Jost', sans-serif !important;
}}
#MainMenu, footer, header {{ visibility: hidden; }}
.block-container {{ padding: 1.2rem 2rem 4rem !important; max-width: 1200px !important; }}

.mast {{
  display:flex; align-items:baseline; gap:1.1rem;
  border-bottom: 1px solid var(--bdr);
  padding-bottom: 1rem; margin-bottom: 1.6rem;
}}
.mast-title {{
  font-family:'Cormorant Garamond',serif;
  font-size:2.8rem; font-weight:600; color:var(--acc);
  letter-spacing:-1px; line-height:1; margin:0;
}}
.mast-sub {{
  font-family:'Fira Code',monospace; font-size:.65rem;
  color:var(--muted); letter-spacing:3px; text-transform:uppercase;
}}
.mast-badge {{
  margin-left:auto;
  font-family:'Fira Code',monospace; font-size:.62rem;
  background:var(--surf2); border:1px solid var(--bdr);
  color:var(--muted); border-radius:20px; padding:3px 10px;
}}

.mlabel {{
  font-family:'Fira Code',monospace; font-size:.6rem;
  letter-spacing:3px; text-transform:uppercase; color:var(--muted);
  margin-bottom:.45rem; display:block;
}}

.panel {{
  background:var(--surf); border:1px solid var(--bdr);
  border-radius:var(--r); padding:1.1rem 1.3rem; margin-bottom:.9rem;
}}
.panel-accent {{ border-left:3px solid var(--acc); }}
.panel-err    {{ border-left:3px solid var(--err); }}

/* ── format badge strip ── */
.fmt-strip {{
  display:flex; gap:.4rem; flex-wrap:wrap; margin:.5rem 0 .9rem;
}}
.fmt-badge {{
  display:inline-flex; align-items:center; gap:.3rem;
  font-family:'Fira Code',monospace; font-size:.58rem; letter-spacing:.8px;
  background:var(--surf2); border:1px solid var(--bdr);
  color:var(--muted); border-radius:4px; padding:3px 8px;
}}
.fmt-badge.active {{
  background:{T['accent']}18; border-color:{T['accent']}44; color:var(--acc);
}}

/* ── file type chip on upload ── */
.file-info {{
  display:inline-flex; align-items:center; gap:.5rem;
  background:var(--surf2); border:1px solid var(--bdr);
  border-radius:var(--r); padding:.5rem .85rem;
  font-size:.83rem; color:var(--txt);
  margin-bottom:.5rem;
}}
.file-icon {{ font-size:1.1rem; }}
.file-type {{
  font-family:'Fira Code',monospace; font-size:.58rem; letter-spacing:1px;
  background:{T['accent']}18; border:1px solid {T['accent']}33;
  color:var(--acc); border-radius:3px; padding:1px 6px; margin-left:.3rem;
}}

.answer-text {{ font-size:.95rem; line-height:1.8; color:var(--txt); }}

.conf-row {{ display:flex; align-items:center; gap:.7rem; margin:.5rem 0 .85rem; }}
.conf-track {{ flex:1; height:3px; background:var(--bdr); border-radius:2px; overflow:hidden; }}
.conf-fill {{ height:3px; border-radius:2px; transition: width .6s cubic-bezier(.4,0,.2,1); }}
.conf-pct {{ font-family:'Fira Code',monospace; font-size:.62rem; color:var(--muted); white-space:nowrap; }}

.chip {{
  display:inline-block; font-family:'Fira Code',monospace;
  font-size:.62rem; letter-spacing:.8px;
  border-radius:3px; padding:2px 7px; margin:2px 2px 2px 0;
}}
.chip-src  {{ background:rgba(226,185,111,.1); border:1px solid rgba(226,185,111,.3); color:var(--acc); }}
.chip-llm  {{ background:rgba(111,181,226,.1); border:1px solid rgba(111,181,226,.3); color:var(--acc2); }}
.chip-time {{ background:var(--surf2); border:1px solid var(--bdr); color:var(--muted); }}
.chip-fmt  {{ background:rgba(111,226,160,.1); border:1px solid rgba(111,226,160,.3); color:var(--ok); }}

.stat-grid {{ display:grid; grid-template-columns:repeat(4,1fr); gap:.5rem; margin:.3rem 0 .6rem; }}
.stat-card {{
  background:var(--surf2); border:1px solid var(--bdr);
  border-radius:var(--r); padding:.7rem .8rem; text-align:center;
}}
.stat-num {{ font-family:'Cormorant Garamond',serif; font-size:1.9rem; color:var(--acc); line-height:1; }}
.stat-lbl {{ font-family:'Fira Code',monospace; font-size:.55rem; letter-spacing:2px; text-transform:uppercase; color:var(--muted); margin-top:2px; }}

.summary-box {{
  background:var(--surf2); border:1px solid var(--bdr);
  border-left:3px solid var(--acc2);
  border-radius:var(--r); padding:1rem 1.2rem;
  font-size:.88rem; line-height:1.7; color:var(--txt); margin:.6rem 0;
}}

.passage {{
  background:var(--surf2); border:1px solid var(--bdr);
  border-radius:var(--r); padding:.8rem 1rem; margin:.4rem 0;
  font-size:.82rem; line-height:1.65; color:var(--muted);
}}
.passage-hdr {{
  font-family:'Fira Code',monospace; font-size:.6rem;
  letter-spacing:2px; text-transform:uppercase; color:var(--acc); margin-bottom:.4rem;
}}

.h-item {{ border-bottom:1px solid var(--bdr); padding:.6rem 0; cursor:default; }}
.h-q {{ font-style:italic; color:var(--acc); font-size:.83rem; }}
.h-a {{ color:var(--muted); font-size:.75rem; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}

.pin-item {{ background:var(--pin); border:1px solid var(--bdr); border-radius:var(--r); padding:.7rem .9rem; margin:.4rem 0; }}
.pin-q {{ font-style:italic; color:var(--acc); font-size:.8rem; }}
.pin-a {{ font-size:.82rem; line-height:1.6; color:var(--txt); margin-top:.3rem; }}

.diff-badge {{
  display:inline-block; font-family:'Fira Code',monospace;
  font-size:.6rem; letter-spacing:1.5px; text-transform:uppercase;
  border-radius:3px; padding:2px 8px; margin-left:.5rem;
}}

.stButton > button {{
  background:var(--acc) !important; color:#0b0c0f !important;
  border:none !important; border-radius:var(--r) !important;
  font-family:'Fira Code',monospace !important; font-size:.68rem !important;
  letter-spacing:1.5px; text-transform:uppercase;
  padding:.5rem 1.2rem !important; transition:opacity var(--trans);
}}
.stButton > button:hover {{ opacity:.82 !important; }}

.stTextInput > div > div > input {{
  background:var(--surf) !important; border:1px solid var(--bdr) !important;
  color:var(--txt) !important; border-radius:var(--r) !important;
  font-family:'Jost',sans-serif !important; font-size:.93rem !important;
  padding:.6rem .9rem !important;
}}
.stTextInput > div > div > input:focus {{
  border-color:var(--acc) !important;
  box-shadow:0 0 0 2px rgba(226,185,111,.12) !important;
}}

[data-testid="stFileUploader"] {{
  background:var(--surf) !important; border:1px dashed var(--bdr) !important;
  border-radius:var(--r) !important;
}}
[data-testid="stFileUploader"]:hover {{ border-color:var(--acc) !important; }}

[data-testid="stSidebar"] {{
  background:var(--surf) !important; border-right:1px solid var(--bdr) !important;
}}

.stSelectbox > div > div {{
  background:var(--surf) !important; border-color:var(--bdr) !important;
  color:var(--txt) !important; border-radius:var(--r) !important;
}}
.stAlert {{ background:var(--surf2) !important; border-radius:var(--r) !important; font-size:.84rem !important; }}
.streamlit-expanderHeader {{
  background:var(--surf2) !important; border:1px solid var(--bdr) !important;
  color:var(--muted) !important; border-radius:var(--r) !important;
  font-family:'Fira Code',monospace !important; font-size:.62rem !important;
  letter-spacing:1.5px; text-transform:uppercase;
}}
hr {{ border-color:var(--bdr) !important; }}
.stSpinner > div {{ border-top-color:var(--acc) !important; }}
[data-testid="stRadio"] label {{ font-size:.82rem !important; }}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def get_llm(choice: str):
    order = {"Mistral": ["mistral","groq","gemini"],
             "Groq":    ["groq","gemini","mistral"],
             "Gemini":  ["gemini","groq","mistral"]}.get(choice, ["groq","gemini","mistral"])
    for name in order:
        if name == "mistral" and os.getenv("MISTRAL_API_KEY"):
            try:
                from langchain_mistralai import ChatMistralAI
                return ChatMistralAI(model="open-mistral-7b"), "Mistral · open-mistral-7b"
            except Exception: pass
        if name == "groq" and os.getenv("GROQ_API_KEY"):
            try:
                from langchain_groq import ChatGroq
                return ChatGroq(model="llama-3.1-8b-instant"), "Groq · llama-3.1-8b-instant"
            except Exception: pass
        if name == "gemini" and os.getenv("GOOGLE_API_KEY"):
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                return ChatGoogleGenerativeAI(model="gemini-2.5-flash"), "Gemini · 2.5-flash"
            except Exception: pass
    return None, None


def load_document(fpath: str, ext: str) -> list:
    """Load any supported file type → list of LangChain Documents."""
    ext = ext.lower().lstrip(".")
    try:
        if ext == "pdf":
            return PyPDFLoader(fpath).load()
        elif ext in ("docx", "doc"):
            return Docx2txtLoader(fpath).load()
        elif ext in ("txt", "md"):
            return TextLoader(fpath, encoding="utf-8").load()
        elif ext == "csv":
            return CSVLoader(fpath).load()
        elif ext == "pptx":
            return UnstructuredPowerPointLoader(fpath).load()
        else:
            # Generic fallback: read raw text
            with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            return [Document(page_content=text, metadata={"source": fpath})]
    except Exception as e:
        st.error(f"Could not load file: {e}")
        return []


def flesch_score(text: str) -> float:
    sentences = max(len(re.findall(r'[.!?]+', text)), 1)
    words     = max(len(text.split()), 1)
    syllables = sum(max(1, len(re.findall(r'[aeiouAEIOU]', w))) for w in text.split())
    return 206.835 - 1.015 * (words / sentences) - 84.6 * (syllables / words)


def difficulty_label(score: float):
    if score >= 70: return ("Easy",      T['success'])
    if score >= 50: return ("Medium",    T['accent'])
    if score >= 30: return ("Hard",      T['danger'])
    return               ("Very Hard",  T['danger'])


def confidence_html(score: float) -> str:
    pct   = int(score * 100)
    color = T['success'] if pct>=70 else (T['accent'] if pct>=40 else T['danger'])
    return (f'<div class="conf-row"><div class="conf-track">'
            f'<div class="conf-fill" style="width:{pct}%;background:{color};"></div>'
            f'</div><span class="conf-pct">{pct}% confidence</span></div>')


def compute_confidence(docs) -> float:
    if not docs: return 0.0
    return min(sum(len(d.page_content) for d in docs) / (len(docs) * 900), 1.0)


def source_chips(docs) -> str:
    seen, chips = set(), []
    for d in docs:
        pg  = d.metadata.get("page")
        lbl = f"p.{int(pg)+1}" if pg is not None else "§"
        if lbl not in seen:
            seen.add(lbl)
            chips.append(f'<span class="chip chip-src">{lbl}</span>')
    return "".join(chips)


def export_txt(history, pinned) -> str:
    ts  = datetime.now().strftime("%Y-%m-%d %H:%M")
    out = [f"FOLIO — Export  {ts}", "="*56, ""]
    out.append("── CONVERSATION ──────────────────────────────────────")
    for i, m in enumerate(history, 1):
        if m["role"] == "user":
            out.append(f"\nQ{i//2+1}: {m['content']}")
        else:
            out.append(f"A:  {m['content'][:500]}{'…' if len(m['content'])>500 else ''}")
    if pinned:
        out += ["", "── PINNED ANSWERS ────────────────────────────────────"]
        for p in pinned:
            out.append(f"\n★  {p['q']}\n   {p['a'][:400]}")
    return "\n".join(out)


def invoke_with_retry(llm, messages, retries=3, wait=8):
    for attempt in range(retries):
        try:
            return llm.invoke(messages)
        except Exception as e:
            if "429" in str(e) and attempt < retries - 1:
                st.toast(f"Rate limited — retrying in {wait}s…", icon="⏳")
                time.sleep(wait); wait *= 2
            else:
                raise


# ─────────────────────────────────────────────────────────────────────────────
# Prompt templates
# ─────────────────────────────────────────────────────────────────────────────
QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are Folio, an expert literary research assistant.
Answer questions ONLY from the provided document context.
Be precise, insightful, and cite page numbers when visible in the context.
If the answer is absent, say exactly: "I could not find the answer in the document."
Format your answer in clean, readable prose. Avoid bullet noise unless listing is genuinely clearer."""),
    ("human", "Document context:\n{context}\n\n---\nConversation so far:\n{history}\n\nQuestion: {question}"),
])

SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a concise academic summariser. Produce a 3-sentence executive summary of the document excerpt provided. Be factual, no padding."),
    ("human", "Document excerpt:\n{text}"),
])


# ─────────────────────────────────────────────────────────────────────────────
# Masthead
# ─────────────────────────────────────────────────────────────────────────────
theme_icon = "☀️" if IS_DARK else "🌙"
db_badge   = "● Index loaded" if os.path.exists("chroma_db") else "○ No index"

st.markdown(f"""
<div class="mast">
  <h1 class="mast-title">Folio</h1>
  <span class="mast-sub">Document Intelligence</span>
  <span class="mast-badge">{db_badge}</span>
</div>
""", unsafe_allow_html=True)

components.html(f"""
<script>
(function() {{
  var SELECTORS = [
    '[data-testid="collapsedControl"]',
    '[data-testid="stSidebarCollapsedControl"]',
    'button[aria-label="open sidebar"]',
    'button[aria-label="close sidebar"]',
  ];
  function findToggleBtn() {{
    for (var i = 0; i < SELECTORS.length; i++) {{
      var el = window.parent.document.querySelector(SELECTORS[i]);
      if (el) return el;
    }}
    return null;
  }}
  var existing = window.parent.document.getElementById('folio-toggle-btn');
  if (existing) existing.remove();
  var btn = window.parent.document.createElement('button');
  btn.id = 'folio-toggle-btn';
  btn.title = 'Toggle sidebar';
  btn.textContent = '☰';
  btn.style.cssText = [
    'position:fixed','top:14px','left:14px','z-index:999999',
    'width:36px','height:36px','background:{T["surface"]}',
    'border:1px solid {T["border"]}','border-radius:7px',
    'color:{T["accent"]}','font-size:1.1rem','line-height:36px',
    'text-align:center','cursor:pointer','padding:0','font-family:sans-serif',
  ].join(';');
  btn.addEventListener('click', function() {{
    var target = findToggleBtn();
    if (target) target.click();
  }});
  window.parent.document.body.appendChild(btn);
}})();
</script>
""", height=0)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    t_col1, t_col2 = st.columns([3, 1])
    t_col1.markdown('<span class="mlabel">Settings</span>', unsafe_allow_html=True)
    if t_col2.button(theme_icon, key="theme_btn"):
        st.session_state.theme = "light" if IS_DARK else "dark"
        st.rerun()

    st.markdown('<span class="mlabel" style="margin-top:.5rem;">Language Model</span>',
                unsafe_allow_html=True)
    llm_choice = st.radio(
        "LLM", ["Auto", "Groq", "Gemini", "Mistral"],
        index=["Auto","Groq","Gemini","Mistral"].index(st.session_state.llm_choice),
        horizontal=True, label_visibility="collapsed"
    )
    st.session_state.llm_choice = llm_choice

    st.markdown("---")
    st.markdown('<span class="mlabel">📌 Pinned Answers</span>', unsafe_allow_html=True)
    if st.session_state.pinned:
        for p in reversed(st.session_state.pinned[-5:]):
            st.markdown(f"""<div class="pin-item">
                <div class="pin-q">❝ {p['q'][:60]}…</div>
                <div class="pin-a">{p['a'][:120]}…</div>
            </div>""", unsafe_allow_html=True)
    else:
        st.markdown('<p style="color:var(--muted);font-size:.78rem;">Pin answers with ★ below.</p>',
                    unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<span class="mlabel">💬 History</span>', unsafe_allow_html=True)
    msgs = st.session_state.chat_history
    history_pairs = [(msgs[i]["content"], msgs[i+1]["content"])
                     for i in range(0, len(msgs)-1, 2) if msgs[i]["role"]=="user"]
    if history_pairs:
        for q, a in reversed(history_pairs[-6:]):
            st.markdown(f"""<div class="h-item">
                <div class="h-q">❝ {q[:55]}{'…' if len(q)>55 else ''}</div>
                <div class="h-a">{a[:70]}…</div>
            </div>""", unsafe_allow_html=True)
    else:
        st.markdown('<p style="color:var(--muted);font-size:.78rem;">No questions yet.</p>',
                    unsafe_allow_html=True)

    st.markdown("---")
    if st.session_state.chat_history:
        st.download_button(
            "⬇ Export session",
            data=export_txt(st.session_state.chat_history, st.session_state.pinned),
            file_name=f"folio_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            mime="text/plain",
            use_container_width=True,
        )
    if st.button("🗑 Clear session", use_container_width=True):
        for k in ["chat_history", "pinned", "conv_messages", "summary"]:
            st.session_state[k] = [] if k != "summary" else None
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN AREA
# ─────────────────────────────────────────────────────────────────────────────

# ── Supported format badges ───────────────────────────────────────────────────
st.markdown('<span class="mlabel">Document</span>', unsafe_allow_html=True)

badges_html = '<div class="fmt-strip">'
for ext, info in FORMAT_INFO.items():
    badges_html += f'<span class="fmt-badge">{info["icon"]} {info["label"]}</span>'
badges_html += "</div>"
st.markdown(badges_html, unsafe_allow_html=True)

# ── File uploader (multi-format) ──────────────────────────────────────────────
up_col, btn_col = st.columns([4, 1])
with up_col:
    uploaded = st.file_uploader(
        "Upload a document",
        type=ACCEPTED_TYPES,
        label_visibility="collapsed",
        help="Supported: PDF, DOCX, TXT, MD, CSV, PPTX"
    )

with btn_col:
    build_clicked = st.button("⚙ Build Index", use_container_width=True)

# ── Show file info card when a file is selected ───────────────────────────────
if uploaded:
    ext      = uploaded.name.rsplit(".", 1)[-1].lower()
    fmt      = FORMAT_INFO.get(ext, {"icon": "📄", "label": ext.upper()})
    size_kb  = round(uploaded.size / 1024, 1)
    st.markdown(
        f'<div class="file-info">'
        f'<span class="file-icon">{fmt["icon"]}</span>'
        f'<span>{uploaded.name}</span>'
        f'<span class="file-type">{fmt["label"]}</span>'
        f'<span style="color:var(--muted);font-size:.78rem;margin-left:auto;">{size_kb} KB</span>'
        f'</div>',
        unsafe_allow_html=True
    )

# ── Index build ───────────────────────────────────────────────────────────────
if uploaded and build_clicked:
    ext = uploaded.name.rsplit(".", 1)[-1].lower()
    suffix = f".{ext}"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.read())
        fpath = tmp.name

    with st.spinner(f"Parsing {FORMAT_INFO.get(ext, {}).get('label','file')} · Splitting · Embedding…"):
        raw_docs = load_document(fpath, ext)

        if not raw_docs:
            st.error("No content could be extracted from this file.")
            st.stop()

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks   = splitter.split_documents(raw_docs)
        emb      = get_embeddings()
        Chroma.from_documents(documents=chunks, embedding=emb, persist_directory="chroma_db")

        full_text = " ".join(d.page_content for d in raw_docs[:20])
        fscore    = flesch_score(full_text[:5000])
        dlabel, dcolor = difficulty_label(fscore)

        st.session_state.doc_stats = {
            "pages":  len(raw_docs),
            "chunks": len(chunks),
            "avg_ch": int(sum(len(c.page_content) for c in chunks) / max(len(chunks), 1)),
            "flesch": round(fscore, 1),
            "dlabel": dlabel,
            "dcolor": dcolor,
            "fmt":    FORMAT_INFO.get(ext, {"icon":"📄","label":ext.upper()}),
        }
        st.session_state.db_ready      = True
        st.session_state.summary       = None
        st.session_state.conv_messages = []

    st.success(f"Index built from **{uploaded.name}**!")
    st.rerun()

elif uploaded and not build_clicked:
    st.info(f"Ready — click **⚙ Build Index** to index this file.")

# ── Doc stats ─────────────────────────────────────────────────────────────────
if st.session_state.doc_stats:
    s = st.session_state.doc_stats
    fmt_chip = (f'<span class="chip chip-fmt">'
                f'{s["fmt"]["icon"]} {s["fmt"]["label"]}</span>')
    diff_badge = (f'<span class="diff-badge" '
                  f'style="background:{s["dcolor"]}22;border:1px solid {s["dcolor"]}44;'
                  f'color:{s["dcolor"]};">{s["dlabel"]}</span>')
    st.markdown(f'<span class="mlabel" style="margin-top:.4rem;">'
                f'Stats {fmt_chip} {diff_badge}</span>', unsafe_allow_html=True)
    st.markdown(f"""<div class="stat-grid">
      <div class="stat-card"><div class="stat-num">{s['pages']}</div><div class="stat-lbl">Pages</div></div>
      <div class="stat-card"><div class="stat-num">{s['chunks']}</div><div class="stat-lbl">Chunks</div></div>
      <div class="stat-card"><div class="stat-num">{s['avg_ch']}</div><div class="stat-lbl">Avg chars</div></div>
      <div class="stat-card"><div class="stat-num">{s['flesch']}</div><div class="stat-lbl">Flesch</div></div>
    </div>""", unsafe_allow_html=True)

elif os.path.exists("chroma_db") and not st.session_state.db_ready:
    st.info("Existing index loaded.")
    st.session_state.db_ready = True

# ─────────────────────────────────────────────────────────────────────────────
# Ask the Document
# ─────────────────────────────────────────────────────────────────────────────
if not (os.path.exists("chroma_db") or st.session_state.db_ready):
    st.markdown('<div class="panel"><p style="color:var(--muted);font-size:.88rem;">'
                'Build an index from a document to start asking questions.</p></div>',
                unsafe_allow_html=True)
else:
    emb       = get_embeddings()
    vs        = Chroma(persist_directory="chroma_db", embedding_function=emb)
    retriever = vs.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 12, "lambda_mult": 0.5},
    )

    st.markdown("---")

    sum_col, gen_col = st.columns([6, 1])
    with sum_col:
        st.markdown('<span class="mlabel">Summary</span>', unsafe_allow_html=True)
    with gen_col:
        gen_sum = st.button("✦ Generate", key="gen_sum_btn")

    if gen_sum:
        llm, llm_name = get_llm(st.session_state.llm_choice)
        if not llm:
            st.error("No LLM available — check your API keys in .env")
        else:
            with st.spinner("Summarising…"):
                sample_docs = vs.similarity_search("introduction overview summary", k=6)
                sample_text = "\n\n".join(d.page_content for d in sample_docs)
                p    = SUMMARY_PROMPT.invoke({"text": sample_text[:4000]})
                resp = invoke_with_retry(llm, p)
                st.session_state.summary = resp.content
            st.rerun()

    if st.session_state.summary:
        st.markdown(f'<div class="summary-box">{st.session_state.summary}</div>',
                    unsafe_allow_html=True)

    st.markdown("---")

    st.markdown('<span class="mlabel">Ask the Document</span>', unsafe_allow_html=True)
    query = st.text_input("question", placeholder="What is the central argument?",
                          label_visibility="collapsed")

    SUGGESTIONS = [
        "What is the main theme?",
        "Summarise the introduction",
        "List key conclusions",
        "Who are the main characters?",
        "What evidence is presented?",
        "What is the author's argument?",
    ]
    sg_cols = st.columns(6)
    for idx, sg in enumerate(SUGGESTIONS):
        if sg_cols[idx].button(sg, key=f"sg_{idx}", use_container_width=True):
            query = sg

    if query:
        llm, llm_name = get_llm(st.session_state.llm_choice)
        if not llm:
            st.error("No LLM API key found. Add GROQ_API_KEY / GOOGLE_API_KEY / MISTRAL_API_KEY to .env")
        else:
            with st.spinner("Retrieving & reasoning…"):
                t0        = time.time()
                retrieved = retriever.invoke(query)
                context   = "\n\n".join(d.page_content for d in retrieved)
                hist_msgs = st.session_state.conv_messages[-8:]
                hist_str  = "\n".join(
                    f"{'User' if m.type=='human' else 'Assistant'}: {m.content}"
                    for m in hist_msgs) or "None"
                final_p   = QA_PROMPT.invoke({
                    "context": context, "history": hist_str, "question": query,
                })
                response = invoke_with_retry(llm, final_p)
                elapsed  = round(time.time() - t0, 2)

            answer = response.content
            no_ans = "could not find" in answer.lower()
            conf   = compute_confidence(retrieved)
            chips  = source_chips(retrieved)

            st.session_state.conv_messages.append(HumanMessage(content=query))
            st.session_state.conv_messages.append(AIMessage(content=answer))
            st.session_state.chat_history.append({"role": "user",      "content": query})
            st.session_state.chat_history.append({"role": "assistant", "content": answer})

            panel_cls = "panel panel-err" if no_ans else "panel panel-accent"
            st.markdown(f'<div class="{panel_cls}"><div class="answer-text">{answer}</div></div>',
                        unsafe_allow_html=True)

            st.markdown(confidence_html(conf), unsafe_allow_html=True)
            meta = (f'{chips}'
                    f'<span class="chip chip-llm">{llm_name}</span>'
                    f'<span class="chip chip-time">{elapsed}s</span>')
            st.markdown(meta, unsafe_allow_html=True)

            p_col, _ = st.columns([1, 5])
            if p_col.button("★ Pin this answer", key=f"pin_{len(st.session_state.pinned)}"):
                st.session_state.pinned.append({"q": query, "a": answer,
                                                 "ts": datetime.now().isoformat()})
                st.toast("Answer pinned!", icon="📌")

            with st.expander("View retrieved passages"):
                for i, doc in enumerate(retrieved, 1):
                    pg  = doc.metadata.get("page")
                    lbl = f"Passage {i}" + (f" · page {int(pg)+1}" if pg is not None else "")
                    st.markdown(f'<div class="passage">'
                                f'<div class="passage-hdr">{lbl}</div>'
                                f'{doc.page_content}</div>', unsafe_allow_html=True)