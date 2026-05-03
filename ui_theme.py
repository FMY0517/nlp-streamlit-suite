from __future__ import annotations

import html
from typing import Iterable

import streamlit as st


def inject_iekg_theme(extra_css: str = "") -> None:
    st.markdown(
        f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');
    :root {{
        --bg-top: #f7fbff;
        --bg-bottom: #edf4f7;
        --ink-strong: #0f172a;
        --ink-soft: #475569;
        --border-soft: rgba(15, 23, 42, 0.08);
        --panel-bg: rgba(255, 255, 255, 0.9);
        --shadow-soft: 0 18px 40px rgba(15, 23, 42, 0.08);
        --shadow-card: 0 14px 30px rgba(15, 23, 42, 0.06);
        --accent-a: #0f766e;
        --accent-b: #0ea5a4;
        --accent-c: #2563eb;
    }}
    .stApp {{
        font-family: 'IBM Plex Sans', 'Microsoft YaHei', sans-serif;
        background:
            radial-gradient(circle at 0% 0%, rgba(37, 99, 235, 0.16), transparent 26%),
            radial-gradient(circle at 100% 0%, rgba(14, 165, 164, 0.14), transparent 24%),
            radial-gradient(circle at 50% 100%, rgba(249, 115, 22, 0.09), transparent 26%),
            linear-gradient(180deg, var(--bg-top) 0%, var(--bg-bottom) 100%);
    }}
    .block-container {{
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        max-width: 1380px;
    }}
    h1, h2, h3, .stMarkdown, .stCaption, label {{
        font-family: 'IBM Plex Sans', 'Microsoft YaHei', sans-serif !important;
    }}
    .stTextArea textarea,
    .stTextInput input {{
        border-radius: 18px !important;
        border: 1px solid rgba(148, 163, 184, 0.28) !important;
    }}
    .stTabs [data-baseweb="tab-list"] {{
        gap: 0.6rem;
        background: rgba(255, 255, 255, 0.56);
        padding: 0.4rem;
        border-radius: 20px;
        border: 1px solid var(--border-soft);
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.7);
    }}
    .stTabs [data-baseweb="tab"] {{
        height: 52px;
        padding: 0 1rem;
        border-radius: 16px;
        background: transparent;
        color: var(--ink-soft);
        font-weight: 600;
    }}
    .stTabs [aria-selected="true"] {{
        background: linear-gradient(135deg, rgba(15, 118, 110, 0.14), rgba(37, 99, 235, 0.16)) !important;
        color: var(--ink-strong) !important;
        box-shadow: 0 8px 18px rgba(15, 23, 42, 0.08);
    }}
    .stButton > button {{
        border-radius: 16px;
        border: 0;
        min-height: 48px;
        font-weight: 700;
        background: linear-gradient(135deg, #0f766e, #2563eb);
        color: white;
        box-shadow: 0 14px 26px rgba(37, 99, 235, 0.24);
    }}
    .stButton > button:hover {{
        filter: brightness(1.05);
    }}
    .stDataFrame, div[data-testid="stTable"] {{
        background: rgba(255, 255, 255, 0.8);
        border-radius: 18px;
        border: 1px solid var(--border-soft);
        overflow: hidden;
    }}
    .stAlert {{
        border-radius: 18px;
    }}
    .hero-shell {{
        position: relative;
        overflow: hidden;
        border-radius: 28px;
        padding: 1px;
        background: linear-gradient(135deg, rgba(15, 118, 110, 0.36), rgba(37, 99, 235, 0.28), rgba(249, 115, 22, 0.16));
        box-shadow: var(--shadow-soft);
        margin-bottom: 1rem;
    }}
    .hero-card {{
        position: relative;
        overflow: hidden;
        background:
            radial-gradient(circle at top right, rgba(191, 219, 254, 0.9), transparent 28%),
            linear-gradient(135deg, rgba(255, 255, 255, 0.96), rgba(246, 250, 255, 0.92));
        border: 1px solid rgba(255, 255, 255, 0.55);
        border-radius: 27px;
        padding: 1.5rem 1.5rem 1.3rem;
    }}
    .hero-grid {{
        display: grid;
        grid-template-columns: 1.6fr 1fr;
        gap: 1rem;
        align-items: center;
    }}
    .hero-title {{
        font-size: 2.35rem;
        font-weight: 700;
        color: var(--ink-strong);
        margin-bottom: 0.4rem;
        letter-spacing: -0.03em;
    }}
    .hero-desc {{
        color: var(--ink-soft);
        line-height: 1.65;
        max-width: 760px;
    }}
    .hero-kicker {{
        display: inline-block;
        padding: 0.38rem 0.78rem;
        border-radius: 999px;
        background: rgba(15, 118, 110, 0.1);
        color: #0f766e;
        font-size: 0.9rem;
        font-weight: 700;
        margin-bottom: 0.8rem;
    }}
    .hero-mini-panel {{
        background: linear-gradient(160deg, #0f172a, #113a67);
        color: #f8fafc;
        border-radius: 22px;
        padding: 1.1rem 1.1rem 1rem;
        box-shadow: 0 18px 38px rgba(15, 23, 42, 0.18);
    }}
    .hero-mini-title {{
        font-size: 0.92rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        opacity: 0.76;
        margin-bottom: 0.65rem;
    }}
    .hero-mini-list {{
        display: grid;
        gap: 0.7rem;
    }}
    .hero-mini-item {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        padding-bottom: 0.55rem;
    }}
    .hero-mini-item:last-child {{
        border-bottom: 0;
        padding-bottom: 0;
    }}
    .hero-mini-item span {{
        color: rgba(255, 255, 255, 0.78);
    }}
    .metric-chip-wrap {{
        display: flex;
        gap: 0.9rem;
        flex-wrap: wrap;
        margin: 1rem 0 1.15rem;
    }}
    .metric-chip {{
        flex: 1 1 180px;
        background: linear-gradient(145deg, rgba(15, 118, 110, 0.96), rgba(37, 99, 235, 0.92));
        color: white;
        padding: 1rem 1.05rem;
        border-radius: 20px;
        min-width: 150px;
        box-shadow: 0 16px 30px rgba(37, 99, 235, 0.18);
        position: relative;
        overflow: hidden;
    }}
    .metric-chip::after {{
        content: '';
        position: absolute;
        right: -18px;
        top: -18px;
        width: 82px;
        height: 82px;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.12);
    }}
    .metric-chip b {{
        display: block;
        font-size: 1.5rem;
        margin-top: 0.16rem;
    }}
    .metric-chip span {{
        font-size: 0.92rem;
        opacity: 0.92;
    }}
    .section-card {{
        background: var(--panel-bg);
        border: 1px solid var(--border-soft);
        border-radius: 24px;
        padding: 1.15rem 1.15rem 1.05rem;
        box-shadow: var(--shadow-card);
        backdrop-filter: blur(8px);
        margin-bottom: 1rem;
    }}
    .section-heading {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 0.75rem;
    }}
    .section-heading h3 {{
        margin: 0;
        font-size: 1.2rem;
        color: var(--ink-strong);
    }}
    .section-kicker {{
        display: inline-block;
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: var(--accent-c);
        background: rgba(37, 99, 235, 0.08);
        padding: 0.28rem 0.64rem;
        border-radius: 999px;
    }}
    .subgrid-card {{
        background: linear-gradient(180deg, rgba(255,255,255,0.95), rgba(248,251,255,0.9));
        border: 1px solid rgba(15, 23, 42, 0.07);
        border-radius: 20px;
        padding: 1rem;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.8);
    }}
    .guide-card {{
        background: linear-gradient(135deg, rgba(14, 165, 164, 0.08), rgba(37, 99, 235, 0.08));
        border: 1px solid rgba(37, 99, 235, 0.12);
        border-radius: 18px;
        padding: 0.95rem 1rem;
        margin: 0.55rem 0 0.95rem;
    }}
    .guide-card strong {{
        color: var(--ink-strong);
    }}
    .guide-card p {{
        margin: 0.18rem 0;
        color: var(--ink-soft);
        line-height: 1.65;
    }}
    @media (max-width: 980px) {{
        .hero-grid {{
            grid-template-columns: 1fr;
        }}
    }}
    {extra_css}
</style>
        """,
        unsafe_allow_html=True,
    )


def render_hero(title: str, kicker: str, description: str, steps: Iterable[tuple[str, str]]) -> None:
    items = []
    for step, label in steps:
        items.append(
            f"<div class='hero-mini-item'><span>{html.escape(step)}</span><b>{html.escape(label)}</b></div>"
        )

    st.markdown(
        f"""
<div class="hero-shell">
    <div class="hero-card">
        <div class="hero-grid">
            <div>
                <div class="hero-kicker">{html.escape(kicker)}</div>
                <div class="hero-title">{html.escape(title)}</div>
                <div class="hero-desc">{html.escape(description)}</div>
            </div>
            <div class="hero-mini-panel">
                <div class="hero-mini-title">Pipeline Snapshot</div>
                <div class="hero-mini-list">
                    {''.join(items)}
                </div>
            </div>
        </div>
    </div>
</div>
        """,
        unsafe_allow_html=True,
    )


def render_guide_card(what_it_does: str, how_to_read: str, notes: str) -> None:
    st.markdown(
        f"""
        <div class="guide-card">
            <p><strong>功能说明：</strong>{html.escape(what_it_does)}</p>
            <p><strong>结果解读：</strong>{html.escape(how_to_read)}</p>
            <p><strong>展示提示：</strong>{html.escape(notes)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
