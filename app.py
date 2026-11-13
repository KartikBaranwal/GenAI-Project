"""AI Video Assistant — turn a recording into notes you can search and question."""

import html
import logging
import os
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

import streamlit as st
from dotenv import load_dotenv

from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import ask_question, build_rag_chain
from core.summarize import generate_title, summarize
from core.transcriber import transcribe_all
from utils.audio_processor import process_input

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("ai_video_assistant")

st.set_page_config(
    page_title="AI Video Assistant",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ═══════════════════════════════════════════════════════════════════════════════
# Design system
# ═══════════════════════════════════════════════════════════════════════════════
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,600;12..96,700;12..96,800&family=Hanken+Grotesk:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
  --bg: #F4F5F7;  --surface: #FFFFFF;  --line: #E2E5EA;  --line-strong: #CDD2DA;
  --ink: #12161C; --muted: #5B6675;    --faint: #8793A3;
  --brand: #0F766E; --brand-ink: #0B5A54; --brand-tint: #E4F3F1;
  --amber: #9A5B0B; --amber-tint: #FDF1DC;
  --blue: #1E4FBF;  --blue-tint: #E6EDFC;
  --violet: #5B34C4; --violet-tint: #EFEAFC;
  --radius: 14px;
  --shadow: 0 1px 2px rgba(18,22,28,.04), 0 4px 16px rgba(18,22,28,.04);
  --font-body: 'Hanken Grotesk', system-ui, -apple-system, 'Segoe UI', sans-serif;
  --font-display: 'Bricolage Grotesque', 'Hanken Grotesk', system-ui, sans-serif;
  --font-mono: 'IBM Plex Mono', ui-monospace, Menlo, monospace;
}

/* ── Base ── */
.stApp { background: var(--bg); color: var(--ink); }
html, body, .stApp, .stApp button, .stApp input, .stApp textarea, [data-baseweb] { font-family: var(--font-body) !important; }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"], [data-testid="collapsedControl"] { display: none; }
#MainMenu, footer { visibility: hidden; }
.block-container { max-width: 1120px; padding: 0 1.5rem 3rem; }

/* ── Top bar ── */
.topbar { display: flex; align-items: center; gap: .65rem; padding: 1.1rem 0 1.4rem; }
.logo { width: 32px; height: 32px; border-radius: 9px; background: var(--brand); display: grid; place-items: center; }
.logo svg { width: 16px; height: 16px; fill: #fff; margin-left: 2px; }
.wordmark { font-family: var(--font-display); font-weight: 700; font-size: 1.05rem; letter-spacing: -.01em; }

/* ── Landing ── */
.hero { text-align: center; max-width: 720px; margin: 2.2rem auto 1.8rem; }
.hero h1 { font-family: var(--font-display) !important; font-weight: 800; font-size: clamp(2rem, 4.6vw, 3.1rem); line-height: 1.08; letter-spacing: -.035em; margin: 0 0 1rem; padding: 0; }
.hero p { color: var(--muted); font-size: 1.08rem; line-height: 1.6; margin: 0 auto; max-width: 56ch; }

[data-testid="stForm"] { background: var(--surface); border: 1px solid var(--line); border-radius: 18px; padding: 1.4rem 1.4rem 1.2rem; box-shadow: var(--shadow); }
.field-note { color: var(--faint); font-size: .82rem; margin: .35rem 0 0; }

.features { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 1.4rem; margin: 2.4rem auto 0; max-width: 820px; }
.feature h3 { font-family: var(--font-display) !important; font-size: 1rem; font-weight: 700; margin: 0 0 .3rem; padding: 0; }
.feature p { color: var(--muted); font-size: .92rem; line-height: 1.55; margin: 0; }
.feature .chip { margin-bottom: .7rem; }

/* ── Inputs ── */
.stTextInput label, .stSelectbox label, .stFileUploader label { font-weight: 600; font-size: .85rem; color: var(--ink); }
.stTextInput input { background: #fff; border-radius: 10px; }
[data-baseweb="input"], [data-baseweb="select"] > div { border-radius: 10px !important; background: #fff !important; border-color: var(--line-strong) !important; }
[data-baseweb="input"]:focus-within, [data-baseweb="select"] > div:focus-within { border-color: var(--brand) !important; box-shadow: 0 0 0 3px var(--brand-tint) !important; }
[data-testid="stFileUploaderDropzone"] { background: #FAFBFC; border: 1.5px dashed var(--line-strong); border-radius: 12px; }

/* ── Buttons ── */
.stApp .stButton > button, .stApp [data-testid="stDownloadButton"] > button, .stApp [data-testid="stFormSubmitButton"] > button {
  border-radius: 10px; min-height: 2.75rem; font-weight: 600; font-size: .92rem; border: 1px solid var(--line-strong); background: #fff; color: var(--ink); box-shadow: none; transition: background .15s, border-color .15s;
}
.stApp .stButton > button:hover, .stApp [data-testid="stDownloadButton"] > button:hover { border-color: var(--ink); background: #fff; color: var(--ink); }
.stApp [data-testid="stBaseButton-primary"], .stApp [data-testid="stBaseButton-primaryFormSubmit"] { background: var(--brand) !important; border-color: var(--brand) !important; color: #fff !important; }
.stApp [data-testid="stBaseButton-primary"]:hover, .stApp [data-testid="stBaseButton-primaryFormSubmit"]:hover { background: var(--brand-ink) !important; border-color: var(--brand-ink) !important; }
button:focus-visible, input:focus-visible, textarea:focus-visible, [role="tab"]:focus-visible, .transcript:focus-visible { outline: 2px solid var(--brand) !important; outline-offset: 2px; }

/* ── Tabs ── */
[data-baseweb="tab-list"] { gap: 1.6rem; border-bottom: 1px solid var(--line); }
button[data-baseweb="tab"] { padding: .7rem 0; background: transparent; }
button[data-baseweb="tab"] p { font-weight: 600; font-size: .95rem; color: var(--muted); }
button[data-baseweb="tab"][aria-selected="true"] p { color: var(--ink); }
[data-baseweb="tab-highlight"] { background: var(--brand) !important; height: 3px !important; }
[data-baseweb="tab-border"] { display: none; }
[data-baseweb="tab-panel"] { padding-top: 1.4rem; }

/* ── Chips, cards ── */
.chip { width: 36px; height: 36px; border-radius: 10px; display: inline-grid; place-items: center; flex-shrink: 0; }
.chip svg { width: 18px; height: 18px; fill: none; stroke: currentColor; stroke-width: 2; stroke-linecap: round; stroke-linejoin: round; }
.chip-brand { background: var(--brand-tint); color: var(--brand); }
.chip-amber { background: var(--amber-tint); color: var(--amber); }
.chip-blue { background: var(--blue-tint); color: var(--blue); }
.chip-violet { background: var(--violet-tint); color: var(--violet); }

.card { background: var(--surface); border: 1px solid var(--line); border-radius: var(--radius); padding: 1.3rem 1.4rem; box-shadow: var(--shadow); }
.card-head { display: flex; align-items: center; gap: .75rem; margin-bottom: 1rem; }
.card-head h3 { font-family: var(--font-display) !important; font-size: 1.05rem; font-weight: 700; margin: 0; padding: 0; flex: 1; letter-spacing: -.01em; }
.count { font-size: .78rem; font-weight: 700; color: var(--muted); background: var(--bg); border: 1px solid var(--line); border-radius: 999px; padding: .1rem .55rem; }
.grid-3 { display: grid; grid-template-columns: repeat(auto-fit, minmax(290px, 1fr)); gap: 1rem; margin-top: 1rem; align-items: start; }

.prose { font-size: 1rem; line-height: 1.72; max-width: 74ch; color: #222833; }
.prose p { margin: 0 0 .85rem; }
.prose p:last-child, .prose ul:last-child { margin-bottom: 0; }
.prose h4 { font-family: var(--font-display) !important; font-size: .98rem; margin: 1.1rem 0 .4rem; padding: 0; }
.prose ul { margin: 0 0 .85rem; padding-left: 1.25rem; }
.prose li { margin-bottom: .3rem; }
.prose code, .item code { font-family: var(--font-mono); font-size: .85em; background: var(--bg); padding: .1em .35em; border-radius: 5px; }

.items { list-style: none; margin: 0; padding: 0; display: grid; gap: .7rem; }
.item { display: flex; gap: .7rem; font-size: .95rem; line-height: 1.55; align-items: flex-start; }
.item .mark { width: 18px; height: 18px; flex-shrink: 0; margin-top: 2px; }
.mark-box { border: 1.8px solid var(--amber); border-radius: 5px; }
.mark-dot { display: grid; place-items: center; }
.mark-dot::after { content: ''; width: 7px; height: 7px; border-radius: 50%; background: var(--blue); }
.mark-q { display: grid; place-items: center; font-weight: 700; font-size: .72rem; color: var(--violet); background: var(--violet-tint); border-radius: 50%; }
.empty-note { color: var(--faint); font-size: .92rem; margin: 0; }

/* ── Result header ── */
.result-title { font-family: var(--font-display) !important; font-weight: 800; font-size: clamp(1.6rem, 3vw, 2.2rem); line-height: 1.14; letter-spacing: -.03em; margin: 0; padding: 0; }
.meta { display: flex; flex-wrap: wrap; gap: .5rem; margin-top: .7rem; }
.pill { font-size: .8rem; font-weight: 600; color: var(--muted); background: #fff; border: 1px solid var(--line); border-radius: 999px; padding: .2rem .7rem; max-width: 100%; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: .8rem; margin: 1.4rem 0 1.6rem; }
.stat { background: var(--surface); border: 1px solid var(--line); border-radius: 12px; padding: .85rem 1rem; }
.stat b { display: block; font-family: var(--font-display); font-size: 1.55rem; font-weight: 700; letter-spacing: -.02em; line-height: 1.15; }
.stat span { font-size: .82rem; color: var(--muted); font-weight: 500; }

/* ── Transcript ── */
.transcript { background: var(--surface); border: 1px solid var(--line); border-radius: var(--radius); padding: .6rem 1.3rem; max-height: 540px; overflow-y: auto; box-shadow: var(--shadow); }
.tline { margin: 0; padding: .75rem 0; border-bottom: 1px solid var(--line); font-size: .96rem; line-height: 1.7; display: grid; grid-template-columns: 1fr; gap: .1rem .9rem; max-width: 80ch; }
.tline:last-child { border-bottom: 0; }
.tline.has-ts { grid-template-columns: 62px 1fr; }
.ts { font-family: var(--font-mono); font-size: .78rem; color: var(--brand); padding-top: .2rem; }
mark { background: #FFE9A8; color: inherit; border-radius: 3px; padding: 0 2px; }
.result-count { color: var(--muted); font-size: .85rem; margin: .5rem 0 .8rem; }

/* ── Chat ── */
[data-testid="stChatMessage"] { background: var(--surface); border: 1px solid var(--line); border-radius: var(--radius); padding: 1rem 1.2rem; margin-bottom: .75rem; }
[data-testid="stChatInput"] { border-radius: 14px; border-color: var(--line-strong); background: #fff; }
.chat-intro { color: var(--muted); font-size: .98rem; margin: 0 0 1rem; }

/* ── Progress stepper ── */
.steps { display: grid; gap: .7rem; margin-top: 1rem; }
.step { display: flex; align-items: center; gap: .9rem; padding: .95rem 1.1rem; border: 1px solid var(--line); border-radius: 12px; background: var(--surface); }
.step .mark { width: 28px; height: 28px; border-radius: 50%; border: 2px solid var(--line-strong); display: grid; place-items: center; font-size: .78rem; font-weight: 700; color: var(--faint); flex-shrink: 0; }
.step b { display: block; font-size: .96rem; }
.step span.d { font-size: .84rem; color: var(--muted); }
.step.pending { opacity: .62; }
.step.active { border-color: var(--brand); box-shadow: 0 0 0 3px var(--brand-tint); }
.step.active .mark { border-color: var(--brand); border-top-color: transparent; animation: spin .9s linear infinite; }
.step.done .mark { background: var(--brand); border-color: var(--brand); color: #fff; }
@keyframes spin { to { transform: rotate(360deg); } }
@media (prefers-reduced-motion: reduce) { .step.active .mark { animation: none; border-top-color: var(--brand); } }

.footnote { text-align: center; color: var(--faint); font-size: .8rem; margin-top: 2.5rem; }
[data-testid="stAlert"] { border-radius: 12px; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════
LANGUAGES = {"English": "english", "Hinglish": "hinglish"}
ACCEPTED_TYPES = ["mp4", "mov", "mkv", "webm", "mp3", "wav", "m4a", "ogg"]
MAX_UPLOAD_MB = 200
URL_RE = re.compile(r"^https?://\S+$", re.IGNORECASE)
BULLET_RE = re.compile(r"^\s*(?:[-*•–]|\d+[.)]|\[[ xX]\])\s+")
NONE_RE = re.compile(r"^\s*(?:no|none|n/a|nothing)\b", re.IGNORECASE)
TS_RE = re.compile(r"^\[?(\d{1,2}:\d{2}(?::\d{2})?)\]?\s*[-–:]?\s*")
STARTERS = ["What were the main decisions?", "Who owns which action item?", "What was left unresolved?"]

STEPS = [
    ("Preparing audio", "Downloading and splitting the recording"),
    ("Transcribing speech", "Converting audio to text"),
    ("Analysing the transcript", "Summary, action items, decisions, open questions and chat index"),
]

ICONS = {
    "logo": '<svg viewBox="0 0 24 24"><path d="M6 4l14 8-14 8z"/></svg>',
    "summary": '<svg viewBox="0 0 24 24"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg>',
    "actions": '<svg viewBox="0 0 24 24"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>',
    "decisions": '<svg viewBox="0 0 24 24"><path d="M4 15s1-1 4-1 5 2 8 2 4-1 4-1V3s-1 1-4 1-5-2-8-2-4 1-4 1z"/><line x1="4" y1="22" x2="4" y2="15"/></svg>',
    "questions": '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>',
    "chat": '<svg viewBox="0 0 24 24"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>',
    "search": '<svg viewBox="0 0 24 24"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>',
}

# ═══════════════════════════════════════════════════════════════════════════════
# Text helpers (all user/LLM content is escaped before it touches HTML)
# ═══════════════════════════════════════════════════════════════════════════════
def inline(text: str) -> str:
    t = html.escape(str(text).strip())
    t = re.sub(r"`([^`]+)`", r"<code>\1</code>", t)
    t = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", t)
    t = re.sub(r"(?<![*\w])\*(?!\s)(.+?)(?<!\s)\*(?![*\w])", r"<em>\1</em>", t)
    return t


def strip_marker(line: str) -> str:
    """Removes a leading bullet, number or [ ] checkbox from a line."""
    line = BULLET_RE.sub("", line, count=1)
    return re.sub(r"^\[[ xX]\]\s+", "", line).strip()


def to_items(value) -> list[str]:
    raw = [str(v) for v in value] if isinstance(value, (list, tuple)) else str(value or "").splitlines()
    items = [strip_marker(ln) for ln in raw if ln.strip() and not re.match(r"^\s*#{1,6}\s", ln)]
    if len(items) == 1 and NONE_RE.match(items[0]):
        return []
    return items


def rich_html(value) -> str:
    text = "\n".join(map(str, value)) if isinstance(value, (list, tuple)) else str(value or "")
    out: list[str] = []
    para: list[str] = []
    bullets: list[str] = []

    def flush():
        if para:
            out.append(f"<p>{inline(' '.join(para))}</p>")
            para.clear()
        if bullets:
            out.append("<ul>" + "".join(f"<li>{inline(b)}</li>" for b in bullets) + "</ul>")
            bullets.clear()

    for ln in text.splitlines():
        s = ln.strip()
        if not s:
            flush()
        elif re.match(r"^#{1,6}\s", s):
            flush()
            out.append(f"<h4>{inline(s.lstrip('#').strip())}</h4>")
        elif BULLET_RE.match(ln):
            if para:
                flush()
            bullets.append(strip_marker(ln))
        else:
            if bullets:
                flush()
            para.append(s)
    flush()
    return "".join(out) or '<p class="empty-note">Nothing to show for this recording.</p>'


def split_paragraphs(text: str) -> list[str]:
    lines = [ln.strip() for ln in str(text).splitlines() if ln.strip()]
    if len(lines) >= 5:
        return lines
    sentences = re.split(r"(?<=[.!?])\s+", " ".join(lines))
    return [" ".join(sentences[i : i + 4]) for i in range(0, len(sentences), 4)] or lines


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:60] or "analysis"


def source_label(link: str, upload) -> str:
    if upload is not None:
        return upload.name
    link = link.strip()
    if URL_RE.match(link):
        parsed = urlparse(link)
        return parsed.netloc.removeprefix("www.") or "Link"
    return Path(link).name or "File"


def validate_source(link: str, upload) -> str | None:
    """Returns an error message, or None when the input is usable."""
    if upload is not None:
        if upload.size > MAX_UPLOAD_MB * 1024 * 1024:
            return f"That file is over {MAX_UPLOAD_MB} MB. Upload a smaller file or paste a link instead."
        return None
    link = link.strip()
    if not link:
        return "Paste a YouTube link or upload a file to get started."
    if URL_RE.match(link) or Path(link).expanduser().exists():
        return None
    return "That isn't a valid link or an existing file path. Check it and try again."


def build_report(r: dict) -> str:
    def bullets(v):
        items = to_items(v)
        return "\n".join(f"- {i}" for i in items) if items else "_None found._"

    return (
        f"# {r['title']}\n\nSource: {r['source']}\n\n"
        f"## Summary\n\n{str(r['summary']).strip()}\n\n"
        f"## Action items\n\n{bullets(r['action_items'])}\n\n"
        f"## Key decisions\n\n{bullets(r['key_decisions'])}\n\n"
        f"## Open questions\n\n{bullets(r['open_questions'])}\n\n"
        f"## Transcript\n\n{r['transcript']}\n\n"
        "_Generated by AI. Verify important details against the recording._\n"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# UI components
# ═══════════════════════════════════════════════════════════════════════════════
def topbar():
    st.markdown(
        f'<div class="topbar"><div class="logo">{ICONS["logo"]}</div><div class="wordmark">AI Video Assistant</div></div>',
        unsafe_allow_html=True,
    )


def card_head(icon: str, tone: str, title: str, count: int | None = None) -> str:
    badge = f'<span class="count">{count}</span>' if count is not None else ""
    return f'<div class="card-head"><span class="chip chip-{tone}">{ICONS[icon]}</span><h3>{title}</h3>{badge}</div>'


def list_card(icon: str, tone: str, title: str, value, marker: str) -> str:
    items = to_items(value)
    mark = {"box": '<span class="mark mark-box"></span>', "dot": '<span class="mark mark-dot"></span>', "q": '<span class="mark mark-q">?</span>'}[marker]
    body = (
        '<ul class="items">' + "".join(f'<li class="item">{mark}<span>{inline(i)}</span></li>' for i in items) + "</ul>"
        if items
        else '<p class="empty-note">None found in this recording.</p>'
    )
    return f'<section class="card">{card_head(icon, tone, title, len(items))}{body}</section>'


def render_stepper(slot, current: int, done: bool = False):
    rows = []
    for i, (title, desc) in enumerate(STEPS):
        state = "done" if done or i < current else "active" if i == current else "pending"
        mark = "✓" if state == "done" else ("" if state == "active" else str(i + 1))
        rows.append(
            f'<div class="step {state}"><div class="mark">{mark}</div>'
            f'<div><b>{title}</b><span class="d">{desc}</span></div></div>'
        )
    slot.markdown(f'<div class="steps" role="status" aria-live="polite">{"".join(rows)}</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline
# ═══════════════════════════════════════════════════════════════════════════════
def run_pipeline(source: str, language: str, slot) -> dict:
    render_stepper(slot, 0)
    chunks = process_input(source)

    render_stepper(slot, 1)
    transcript = transcribe_all(chunks, language)
    if not transcript or not str(transcript).strip():
        raise ValueError("No speech was detected in this recording.")

    render_stepper(slot, 2)
    # Independent LLM calls: run them in parallel. Lower max_workers if you hit rate limits.
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs = {
            "title": pool.submit(generate_title, transcript),
            "summary": pool.submit(summarize, transcript),
            "action_items": pool.submit(extract_action_items, transcript),
            "key_decisions": pool.submit(extract_key_decisions, transcript),
            "open_questions": pool.submit(extract_questions, transcript),
            "rag_chain": pool.submit(build_rag_chain, transcript),
        }
        outputs = {name: job.result() for name, job in jobs.items()}

    render_stepper(slot, 2, done=True)
    return {"transcript": transcript, **outputs}


def answer_question(question: str) -> None:
    st.session_state.chat_history.append({"role": "user", "content": question})
    try:
        answer = ask_question(st.session_state.result["rag_chain"], question)
    except Exception:  # noqa: BLE001
        log.exception("Chat failed")
        answer = "I couldn't answer that just now. Try again or rephrase the question."
    st.session_state.chat_history.append({"role": "assistant", "content": answer})


# ═══════════════════════════════════════════════════════════════════════════════
# State
# ═══════════════════════════════════════════════════════════════════════════════
st.session_state.setdefault("result", None)
st.session_state.setdefault("chat_history", [])
st.session_state.setdefault("error", None)
st.session_state.setdefault("last_link", "")

topbar()

# ═══════════════════════════════════════════════════════════════════════════════
# View 1: landing + intake
# ═══════════════════════════════════════════════════════════════════════════════
if st.session_state.result is None:
    st.markdown(
        """
<div class="hero">
  <h1>Turn any recording into notes you can question</h1>
  <p>Paste a YouTube link or upload a file. You get a transcript, a summary, action items and a chat that answers from what was actually said.</p>
</div>""",
        unsafe_allow_html=True,
    )

    _, centre, _ = st.columns([1, 3.4, 1])
    with centre:
        if st.session_state.error:
            st.error(st.session_state.error)

        progress_slot = st.empty()
        form_slot = st.empty()

        with form_slot.container():
            with st.form("intake", border=True):
                tab_link, tab_file = st.tabs(["YouTube link or path", "Upload a file"])
                with tab_link:
                    link = st.text_input(
                        "Link or file path",
                        value=st.session_state.last_link,
                        placeholder="https://www.youtube.com/watch?v=…",
                    )
                with tab_file:
                    upload = st.file_uploader(
                        "Audio or video file",
                        type=ACCEPTED_TYPES,
                        help=f"MP4, MOV, MKV, WebM, MP3, WAV, M4A or OGG, up to {MAX_UPLOAD_MB} MB.",
                    )
                language_label = st.selectbox("Spoken language", list(LANGUAGES))
                submitted = st.form_submit_button("Analyse recording", type="primary", use_container_width=True)
                st.markdown('<p class="field-note">Uploaded files are deleted as soon as processing finishes.</p>', unsafe_allow_html=True)

        if submitted:
            st.session_state.error = None
            st.session_state.last_link = link
            problem = validate_source(link, upload)
            if problem:
                st.session_state.error = problem
                st.rerun()

            form_slot.empty()
            tmp_path = None
            try:
                if upload is not None:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(upload.name).suffix) as tmp:
                        tmp.write(upload.getbuffer())
                        tmp_path = tmp.name
                    source = tmp_path
                    fallback_title = Path(upload.name).stem
                else:
                    source = link.strip() if URL_RE.match(link.strip()) else str(Path(link.strip()).expanduser())
                    fallback_title = "Untitled recording"

                result = run_pipeline(source, LANGUAGES[language_label], progress_slot)
                result["title"] = str(result.get("title") or "").strip() or fallback_title
                result["source"] = source_label(link, upload)
                result["language"] = language_label
                st.session_state.result = result
                st.session_state.chat_history = []
            except ValueError as exc:
                st.session_state.error = str(exc)
            except Exception:  # noqa: BLE001
                log.exception("Pipeline failed")
                st.session_state.error = (
                    "We couldn't analyse that recording. Check that the link is public or the file plays, then try again."
                )
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.remove(tmp_path)
            st.rerun()

        st.markdown(
            f"""
<div class="features">
  <div class="feature"><span class="chip chip-brand">{ICONS['summary']}</span><h3>Searchable transcript</h3><p>Find the exact moment something was said.</p></div>
  <div class="feature"><span class="chip chip-amber">{ICONS['actions']}</span><h3>Actions and decisions</h3><p>Pulled out and listed, so nothing gets lost.</p></div>
  <div class="feature"><span class="chip chip-violet">{ICONS['chat']}</span><h3>Ask follow-ups</h3><p>Chat with the recording and get answers from the transcript.</p></div>
</div>""",
            unsafe_allow_html=True,
        )

# ═══════════════════════════════════════════════════════════════════════════════
# View 2: results
# ═══════════════════════════════════════════════════════════════════════════════
else:
    r = st.session_state.result
    transcript_text = str(r["transcript"])
    words = len(transcript_text.split())
    n_actions, n_decisions, n_questions = (len(to_items(r[k])) for k in ("action_items", "key_decisions", "open_questions"))

    head, dl, new = st.columns([5, 1.6, 1.4], vertical_alignment="top", gap="small")
    with head:
        st.markdown(
            f'<h1 class="result-title">{html.escape(r["title"])}</h1>'
            f'<div class="meta"><span class="pill">{html.escape(r["source"])}</span><span class="pill">{html.escape(r["language"])}</span></div>',
            unsafe_allow_html=True,
        )
    with dl:
        st.download_button(
            "Download report",
            data=build_report(r),
            file_name=f"{slugify(r['title'])}.md",
            mime="text/markdown",
            use_container_width=True,
        )
    with new:
        if st.button("New analysis", use_container_width=True, type="primary"):
            st.session_state.result = None
            st.session_state.chat_history = []
            st.session_state.error = None
            st.session_state.last_link = ""
            st.rerun()

    st.markdown(
        f"""
<div class="stats">
  <div class="stat"><b>{words:,}</b><span>Words transcribed</span></div>
  <div class="stat"><b>{max(1, round(words / 200))} min</b><span>Reading time</span></div>
  <div class="stat"><b>{n_actions}</b><span>Action items</span></div>
  <div class="stat"><b>{n_decisions}</b><span>Key decisions</span></div>
  <div class="stat"><b>{n_questions}</b><span>Open questions</span></div>
</div>""",
        unsafe_allow_html=True,
    )

    tab_overview, tab_transcript, tab_chat = st.tabs(["Overview", "Transcript", "Ask questions"])

    # ── Overview
    with tab_overview:
        st.markdown(
            f'<section class="card">{card_head("summary", "brand", "Summary")}<div class="prose">{rich_html(r["summary"])}</div></section>'
            '<div class="grid-3">'
            + list_card("actions", "amber", "Action items", r["action_items"], "box")
            + list_card("decisions", "blue", "Key decisions", r["key_decisions"], "dot")
            + list_card("questions", "violet", "Open questions", r["open_questions"], "q")
            + "</div>",
            unsafe_allow_html=True,
        )

    # ── Transcript
    with tab_transcript:
        s_col, d_col = st.columns([4, 1.4], gap="small", vertical_alignment="bottom")
        with s_col:
            query = st.text_input("Search transcript", placeholder="Search for a word or phrase").strip()
        with d_col:
            st.download_button(
                "Download transcript",
                data=transcript_text,
                file_name=f"{slugify(r['title'])}-transcript.txt",
                mime="text/plain",
                use_container_width=True,
            )

        paragraphs = split_paragraphs(transcript_text)
        pattern = re.compile(re.escape(html.escape(query)), re.IGNORECASE) if query else None
        if query:
            paragraphs = [p for p in paragraphs if query.lower() in p.lower()]
            total = sum(p.lower().count(query.lower()) for p in paragraphs)
            st.markdown(
                f'<div class="result-count">{total} match{"es" if total != 1 else ""} for “{html.escape(query)}”</div>',
                unsafe_allow_html=True,
            )

        rows = []
        for p in paragraphs:
            m = TS_RE.match(p)
            ts, body = (m.group(1), p[m.end():]) if m else (None, p)
            safe = html.escape(body)
            if pattern:
                safe = pattern.sub(lambda mm: f"<mark>{mm.group(0)}</mark>", safe)
            rows.append(
                f'<p class="tline{" has-ts" if ts else ""}">'
                + (f'<span class="ts">{html.escape(ts)}</span>' if ts else "")
                + f"<span>{safe}</span></p>"
            )
        body_html = "".join(rows) or '<p class="empty-note" style="padding:1rem 0">No lines match that search.</p>'
        st.markdown(
            f'<div class="transcript" tabindex="0" role="region" aria-label="Transcript">{body_html}</div>',
            unsafe_allow_html=True,
        )

    # ── Chat
    with tab_chat:
        pending = None
        if not st.session_state.chat_history:
            st.markdown('<p class="chat-intro">Ask anything about this recording. Answers come from the transcript.</p>', unsafe_allow_html=True)
            cols = st.columns(len(STARTERS))
            for col, q in zip(cols, STARTERS):
                if col.button(q, key=f"starter-{q}", use_container_width=True):
                    pending = q

        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        typed = st.chat_input("Ask about this recording")
        question = (typed or pending or "").strip()
        if question:
            with st.chat_message("user"):
                st.markdown(question)
            with st.chat_message("assistant"):
                with st.spinner("Searching the transcript…"):
                    answer_question(question)
            st.rerun()

        if st.session_state.chat_history and st.button("Clear conversation"):
            st.session_state.chat_history = []
            st.rerun()

st.markdown(
    '<p class="footnote">Summaries and answers are AI-generated. Check important details against the recording.</p>',
    unsafe_allow_html=True,
)