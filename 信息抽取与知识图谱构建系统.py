import html
import json
import re
from collections import Counter
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

try:
    import spacy
except ImportError:
    spacy = None


st.set_page_config(
    page_title="信息抽取与知识图谱构建系统",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');
    :root {
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
    }
    .stApp {
        font-family: 'IBM Plex Sans', 'Microsoft YaHei', sans-serif;
        background:
            radial-gradient(circle at 0% 0%, rgba(37, 99, 235, 0.16), transparent 26%),
            radial-gradient(circle at 100% 0%, rgba(14, 165, 164, 0.14), transparent 24%),
            radial-gradient(circle at 50% 100%, rgba(249, 115, 22, 0.09), transparent 26%),
            linear-gradient(180deg, var(--bg-top) 0%, var(--bg-bottom) 100%);
    }
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        max-width: 1380px;
    }
    h1, h2, h3, .stMarkdown, .stCaption, label {
        font-family: 'IBM Plex Sans', 'Microsoft YaHei', sans-serif !important;
    }
    [data-testid="stSidebar"] {
        background:
            linear-gradient(180deg, rgba(10, 37, 64, 0.98), rgba(15, 118, 110, 0.92));
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }
    [data-testid="stSidebar"] * {
        color: #eef6ff;
    }
    [data-testid="stSidebar"] .stTextArea textarea {
        background: rgba(255, 255, 255, 0.96) !important;
        color: #0f172a !important;
        caret-color: #0f172a !important;
        border: 1px solid rgba(255, 255, 255, 0.22) !important;
        border-radius: 18px !important;
    }
    [data-testid="stSidebar"] .stTextArea textarea::placeholder {
        color: #64748b !important;
    }
    [data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] > div {
        background: rgba(255, 255, 255, 0.96) !important;
        color: #0f172a !important;
        border: 1px solid rgba(255, 255, 255, 0.22) !important;
        border-radius: 14px !important;
    }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSelectbox label p {
        color: #0f172a !important;
    }
    [data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] * {
        color: #0f172a !important;
        -webkit-text-fill-color: #0f172a !important;
        opacity: 1 !important;
    }
    [data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] span,
    [data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] svg,
    [data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] input {
        color: #0f172a !important;
        fill: #0f172a !important;
        caret-color: #0f172a !important;
        opacity: 1 !important;
    }
    [data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] input {
        font-weight: 600 !important;
    }
    [data-testid="stSidebar"] .stCheckbox label p,
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #eef6ff !important;
    }
    .stTextArea textarea {
        border-radius: 18px !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.6rem;
        background: rgba(255, 255, 255, 0.56);
        padding: 0.4rem;
        border-radius: 20px;
        border: 1px solid var(--border-soft);
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.7);
    }
    .stTabs [data-baseweb="tab"] {
        height: 52px;
        padding: 0 1rem;
        border-radius: 16px;
        background: transparent;
        color: var(--ink-soft);
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(15, 118, 110, 0.14), rgba(37, 99, 235, 0.16)) !important;
        color: var(--ink-strong) !important;
        box-shadow: 0 8px 18px rgba(15, 23, 42, 0.08);
    }
    .stButton > button {
        border-radius: 16px;
        border: 0;
        min-height: 48px;
        font-weight: 700;
        background: linear-gradient(135deg, #0f766e, #2563eb);
        color: white;
        box-shadow: 0 14px 26px rgba(37, 99, 235, 0.24);
    }
    .stButton > button:hover {
        filter: brightness(1.05);
    }
    .stDataFrame, div[data-testid="stTable"] {
        background: rgba(255, 255, 255, 0.8);
        border-radius: 18px;
        border: 1px solid var(--border-soft);
        overflow: hidden;
    }
    .hero-shell {
        position: relative;
        overflow: hidden;
        border-radius: 28px;
        padding: 1px;
        background: linear-gradient(135deg, rgba(15, 118, 110, 0.36), rgba(37, 99, 235, 0.28), rgba(249, 115, 22, 0.16));
        box-shadow: var(--shadow-soft);
    }
    .hero-card {
        position: relative;
        overflow: hidden;
        background:
            radial-gradient(circle at top right, rgba(191, 219, 254, 0.9), transparent 28%),
            linear-gradient(135deg, rgba(255, 255, 255, 0.96), rgba(246, 250, 255, 0.92));
        border: 1px solid rgba(255, 255, 255, 0.55);
        border-radius: 27px;
        padding: 1.5rem 1.5rem 1.3rem;
    }
    .hero-grid {
        display: grid;
        grid-template-columns: 1.6fr 1fr;
        gap: 1rem;
        align-items: center;
    }
    .hero-title {
        font-size: 2.35rem;
        font-weight: 700;
        color: var(--ink-strong);
        margin-bottom: 0.4rem;
        letter-spacing: -0.03em;
    }
    .hero-desc {
        color: var(--ink-soft);
        line-height: 1.65;
        max-width: 760px;
    }
    .hero-kicker {
        display: inline-block;
        padding: 0.38rem 0.78rem;
        border-radius: 999px;
        background: rgba(15, 118, 110, 0.1);
        color: #0f766e;
        font-size: 0.9rem;
        font-weight: 700;
        margin-bottom: 0.8rem;
    }
    .hero-mini-panel {
        background: linear-gradient(160deg, #0f172a, #113a67);
        color: #f8fafc;
        border-radius: 22px;
        padding: 1.1rem 1.1rem 1rem;
        box-shadow: 0 18px 38px rgba(15, 23, 42, 0.18);
    }
    .hero-mini-title {
        font-size: 0.92rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        opacity: 0.76;
        margin-bottom: 0.65rem;
    }
    .hero-mini-list {
        display: grid;
        gap: 0.7rem;
    }
    .hero-mini-item {
        display: flex;
        align-items: center;
        justify-content: space-between;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        padding-bottom: 0.55rem;
    }
    .hero-mini-item:last-child {
        border-bottom: 0;
        padding-bottom: 0;
    }
    .hero-mini-item span {
        color: rgba(255, 255, 255, 0.78);
    }
    .metric-chip-wrap {
        display: flex;
        gap: 0.9rem;
        flex-wrap: wrap;
        margin: 1rem 0 1.15rem;
    }
    .metric-chip {
        flex: 1 1 180px;
        background: linear-gradient(145deg, rgba(15, 118, 110, 0.96), rgba(37, 99, 235, 0.92));
        color: white;
        padding: 1rem 1.05rem;
        border-radius: 20px;
        min-width: 150px;
        box-shadow: 0 16px 30px rgba(37, 99, 235, 0.18);
        position: relative;
        overflow: hidden;
    }
    .metric-chip::after {
        content: '';
        position: absolute;
        right: -18px;
        top: -18px;
        width: 82px;
        height: 82px;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.12);
    }
    .metric-chip b {
        display: block;
        font-size: 1.5rem;
        margin-top: 0.16rem;
    }
    .metric-chip span {
        font-size: 0.92rem;
        opacity: 0.92;
    }
    .tag-legend {
        display: flex;
        gap: 0.55rem;
        flex-wrap: wrap;
        margin: 0.6rem 0 0.8rem;
    }
    .tag-pill {
        display: inline-block;
        padding: 0.34rem 0.78rem;
        border-radius: 999px;
        font-size: 0.9rem;
        font-weight: 600;
        color: #0f172a;
        box-shadow: inset 0 0 0 1px rgba(15, 23, 42, 0.06);
    }
    .section-card {
        background: var(--panel-bg);
        border: 1px solid var(--border-soft);
        border-radius: 24px;
        padding: 1.15rem 1.15rem 1.05rem;
        box-shadow: var(--shadow-card);
        backdrop-filter: blur(8px);
    }
    .section-heading {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 0.75rem;
    }
    .section-heading h3 {
        margin: 0;
        font-size: 1.2rem;
        color: var(--ink-strong);
    }
    .section-kicker {
        display: inline-block;
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: var(--accent-c);
        background: rgba(37, 99, 235, 0.08);
        padding: 0.28rem 0.64rem;
        border-radius: 999px;
    }
    .subgrid-card {
        background: linear-gradient(180deg, rgba(255,255,255,0.95), rgba(248,251,255,0.9));
        border: 1px solid rgba(15, 23, 42, 0.07);
        border-radius: 20px;
        padding: 1rem;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.8);
    }
    .entity-span {
        display: inline-block;
        padding: 0.22rem 0.48rem;
        border-radius: 0.7rem;
        margin: 0.12rem 0.12rem;
        line-height: 1.7;
        border: 1px solid rgba(15, 23, 42, 0.08);
        box-shadow: 0 6px 12px rgba(15, 23, 42, 0.05);
    }
    .entity-label {
        font-size: 0.72rem;
        font-weight: 700;
        margin-left: 0.28rem;
        opacity: 0.75;
    }
    .text-box {
        background: linear-gradient(180deg, #ffffff, #f9fbff);
        border: 1px solid #dbe4ee;
        border-radius: 22px;
        padding: 1.15rem;
        min-height: 220px;
        line-height: 2.02;
        font-size: 1.02rem;
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.9);
    }
    .bio-box {
        background: linear-gradient(180deg, #08111f, #0f172a);
        color: #dbeafe;
        border-radius: 22px;
        padding: 1.15rem;
        min-height: 220px;
        font-family: 'IBM Plex Mono', Consolas, Monaco, monospace;
        white-space: pre-wrap;
        line-height: 1.6;
        box-shadow: 0 14px 32px rgba(15, 23, 42, 0.18);
    }
    .stats-note {
        color: var(--ink-soft);
        font-size: 0.95rem;
        margin-top: -0.1rem;
    }
    .guide-card {
        background: linear-gradient(135deg, rgba(14, 165, 164, 0.08), rgba(37, 99, 235, 0.08));
        border: 1px solid rgba(37, 99, 235, 0.12);
        border-radius: 18px;
        padding: 0.95rem 1rem;
        margin: 0.55rem 0 0.95rem;
    }
    .guide-card strong {
        color: var(--ink-strong);
    }
    .guide-card p {
        margin: 0.18rem 0;
        color: var(--ink-soft);
        line-height: 1.65;
    }
    @media (max-width: 980px) {
        .hero-grid {
            grid-template-columns: 1fr;
        }
    }
</style>
""",
    unsafe_allow_html=True,
)


ENTITY_CONFIG: Dict[str, Dict[str, str]] = {
    "PERSON": {"label": "Person", "color": "#ffd6a5", "size": 28},
    "ORG": {"label": "Organization", "color": "#bde0fe", "size": 30},
    "LOC": {"label": "Location", "color": "#caffbf", "size": 26},
    "DATE": {"label": "Date", "color": "#e9d5ff", "size": 24},
    "EVENT": {"label": "Event", "color": "#fecdd3", "size": 24},
}

RELATION_COLORS = {
    "FOUNDER_OF": "#fb7185",
    "WORKS_FOR": "#2563eb",
    "LOCATED_IN": "#16a34a",
    "HELD_IN": "#7c3aed",
    "HAPPENED_ON": "#ea580c",
    "PARTNERS_WITH": "#0f766e",
    "RELATED_TO": "#475569",
}

DEFAULT_TEXT = """Steve Jobs founded Apple in California in 1976.
Tim Cook now leads Apple.
北京大学位于北京。
马云创立了阿里巴巴。
2024世界人工智能大会在上海举行。"""

DEFAULT_TEXT_2 = """Researchers at the University of California, Los Angeles collaborated with Apple on a medical AI project in California."""

DEFAULT_TEXT_3 = """After Tim Cook visited Shanghai, he said that Apple, which Steve Jobs founded in California, would expand a lab that helps developers."""

DEFAULT_TEXT_4 = """On Tuesday, Tim Cook said Apple was discussing a new enterprise AI partnership with Alibaba after Steve Jobs's former design team met researchers from Peking University in Shanghai. Later, Satya Nadella told Microsoft investors that OpenAI, Nvidia and Goldman Sachs were also shaping the market, while analysts said the rivalry could intensify in 2025."""

DEMO_TEXTS = {
    "默认文本1：课程示例": DEFAULT_TEXT,
    "默认文本2：嵌套实体": DEFAULT_TEXT_2,
    "默认文本3：复杂从句与指代": DEFAULT_TEXT_3,
    "默认文本4：商业新闻段落": DEFAULT_TEXT_4,
}

ENTITY_PRIORITY = {
    "ORG": 5,
    "EVENT": 4,
    "LOC": 3,
    "PERSON": 2,
    "DATE": 1,
}

CHINESE_TOKEN_HINTS = [
    "世界人工智能大会",
    "北京大学",
    "阿里巴巴",
    "位于",
    "创立了",
    "创立",
    "举行",
    "召开",
    "发布",
    "成立",
    "领导",
    "担任",
    "在",
    "于",
    "了",
]


@st.cache_resource
def load_spacy_model():
    if spacy is None:
        return None
    try:
        return spacy.load("en_core_web_sm")
    except Exception:
        return None


def is_cjk_char(char: str) -> bool:
    return "\u4e00" <= char <= "\u9fff"


def is_ascii_word_char(char: str) -> bool:
    return char.isascii() and char.isalnum()


def entity_sort_key(item: Dict[str, object]) -> Tuple[int, int, int]:
    return (
        item["start"],
        -(item["end"] - item["start"]),
        -ENTITY_PRIORITY.get(str(item["label"]), 0),
    )


def segment_plain_cjk(text: str, start_offset: int) -> List[Dict[str, int]]:
    tokens: List[Dict[str, int]] = []
    i = 0
    while i < len(text):
        matched_hint = None
        for hint in CHINESE_TOKEN_HINTS:
            if text.startswith(hint, i) and (matched_hint is None or len(hint) > len(matched_hint)):
                matched_hint = hint
        if matched_hint:
            end = i + len(matched_hint)
            tokens.append({"text": matched_hint, "start": start_offset + i, "end": start_offset + end})
            i = end
            continue
        tokens.append({"text": text[i], "start": start_offset + i, "end": start_offset + i + 1})
        i += 1
    return tokens


def segment_cjk_span(
    text: str,
    start: int,
    end: int,
    entities: List[Dict[str, object]],
) -> List[Dict[str, int]]:
    tokens: List[Dict[str, int]] = []
    span_entities = sorted(
        [
            entity
            for entity in entities
            if entity["start"] >= start and entity["end"] <= end and any(is_cjk_char(ch) for ch in entity["text"])
        ],
        key=entity_sort_key,
    )
    cursor = start

    while cursor < end:
        matching_entities = [entity for entity in span_entities if entity["start"] == cursor]
        if matching_entities:
            entity = max(
                matching_entities,
                key=lambda item: (
                    item["end"] - item["start"],
                    ENTITY_PRIORITY.get(str(item["label"]), 0),
                ),
            )
            tokens.append({"text": text[cursor:entity["end"]], "start": cursor, "end": entity["end"]})
            cursor = entity["end"]
            continue

        next_entity_start = min(
            [entity["start"] for entity in span_entities if entity["start"] > cursor],
            default=end,
        )
        tokens.extend(segment_plain_cjk(text[cursor:next_entity_start], cursor))
        cursor = next_entity_start

    return tokens


def tokenize_with_spans(text: str, entities: List[Dict[str, object]] | None = None) -> List[Dict[str, int]]:
    tokens: List[Dict[str, int]] = []
    entity_items = entities or []
    i = 0
    while i < len(text):
        char = text[i]
        if char.isspace():
            i += 1
            continue
        start = i
        if is_cjk_char(char):
            while i < len(text) and is_cjk_char(text[i]):
                i += 1
            tokens.extend(segment_cjk_span(text, start, i, entity_items))
            continue
        if is_ascii_word_char(char):
            while i < len(text) and (is_ascii_word_char(text[i]) or text[i] in {"-", "_", "'"}):
                i += 1
            tokens.append({"text": text[start:i], "start": start, "end": i})
            continue
        tokens.append({"text": char, "start": start, "end": start + 1})
        i += 1
    return tokens


def add_entity(
    entities: List[Dict[str, object]],
    seen: set,
    text: str,
    start: int,
    end: int,
    label: str,
):
    if start < 0 or end <= start:
        return
    mention = text[start:end].strip()
    if label in {"PERSON", "ORG"} and mention.endswith("'s"):
        mention = mention[:-2].rstrip()
        end -= 2
    if label in {"PERSON", "ORG"} and mention.endswith("’s"):
        mention = mention[:-2].rstrip()
        end -= 2
    key = (start, end, label)
    if key in seen:
        return
    if not mention:
        return
    seen.add(key)
    entities.append(
        {
            "text": mention,
            "start": start,
            "end": end,
            "label": label,
        }
    )


def extract_entities_rule_based(text: str) -> List[Dict[str, object]]:
    entities: List[Dict[str, object]] = []
    seen = set()

    known_entities = {
        "Steve Jobs": "PERSON",
        "Tim Cook": "PERSON",
        "Satya Nadella": "PERSON",
        "Apple": "ORG",
        "Alibaba": "ORG",
        "Microsoft": "ORG",
        "OpenAI": "ORG",
        "Nvidia": "ORG",
        "Goldman Sachs": "ORG",
        "Peking University": "ORG",
        "University of California, Los Angeles": "ORG",
        "California": "LOC",
        "Beijing": "LOC",
        "Shanghai": "LOC",
        "Los Angeles": "LOC",
        "北京大学": "ORG",
        "北京": "LOC",
        "马云": "PERSON",
        "阿里巴巴": "ORG",
        "上海": "LOC",
        "世界人工智能大会": "EVENT",
    }

    for mention, label in known_entities.items():
        for match in re.finditer(re.escape(mention), text, flags=re.IGNORECASE):
            add_entity(entities, seen, text, match.start(), match.end(), label)

    date_patterns = [
        r"\b(?:On|on)\s+(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b",
        r"(?<!\d)(?:19|20)\d{2}(?!\d)",
        r"\b(?:19|20)\d{2}-\d{1,2}-\d{1,2}\b",
        r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+(?:19|20)\d{2}\b",
        r"(?:19|20)\d{2}年(?:\d{1,2}月(?:\d{1,2}日)?)?",
    ]
    for pattern in date_patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            add_entity(entities, seen, text, match.start(), match.end(), "DATE")

    chinese_org_suffix = r"[\u4e00-\u9fff]{2,20}(?:大学|公司|集团|学院|研究院|委员会)"
    chinese_loc_suffix = r"[\u4e00-\u9fff]{2,12}(?:市|省|国|区|县|州)"
    chinese_event_suffix = r"[\u4e00-\u9fff]{2,20}(?:大会|峰会|论坛|会议|赛事|展览)"
    english_org_pattern = r"\b[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*\s+(?:University|Inc|Corp|Corporation|Company|Group|Committee)\b"
    english_loc_pattern = r"\b(?:New York|San Francisco|Los Angeles|London|Paris|Tokyo|China|USA)\b"

    for pattern, label in [
        (chinese_org_suffix, "ORG"),
        (chinese_loc_suffix, "LOC"),
        (chinese_event_suffix, "EVENT"),
        (english_org_pattern, "ORG"),
        (english_loc_pattern, "LOC"),
    ]:
        for match in re.finditer(pattern, text):
            add_entity(entities, seen, text, match.start(), match.end(), label)

    contextual_patterns = [
        (r"位于([\u4e00-\u9fff]{2,12})", "LOC"),
        (r"在([\u4e00-\u9fff]{2,12})举行", "LOC"),
        (r"([\u4e00-\u9fff]{2,20}(?:大学|公司|集团|学院|研究院|委员会))位于", "ORG"),
        (r"([\u4e00-\u9fff]{2,20}(?:大会|峰会|论坛|会议|赛事|展览))在", "EVENT"),
    ]
    for pattern, label in contextual_patterns:
        for match in re.finditer(pattern, text):
            add_entity(entities, seen, text, match.start(1), match.end(1), label)

    # 简单的人名启发式，优先补充英文双词人名。
    for match in re.finditer(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", text):
        mention = text[match.start():match.end()]
        if any(token in mention for token in ["University", "Company", "Group", "Inc", "Corp"]):
            continue
        if mention in {"On Tuesday", "On Monday", "On Wednesday", "On Thursday", "On Friday", "On Saturday", "On Sunday"}:
            continue
        add_entity(entities, seen, text, match.start(), match.end(), "PERSON")

    entities.sort(key=entity_sort_key)
    return deduplicate_overlaps(entities)


def merge_spacy_entities(text: str, entities: List[Dict[str, object]]) -> List[Dict[str, object]]:
    nlp = load_spacy_model()
    if nlp is None:
        return entities

    label_map = {
        "PERSON": "PERSON",
        "ORG": "ORG",
        "GPE": "LOC",
        "LOC": "LOC",
        "DATE": "DATE",
        "EVENT": "EVENT",
    }
    seen = {(item["start"], item["end"], item["label"]) for item in entities}
    doc = nlp(text)
    for ent in doc.ents:
        if any(is_cjk_char(char) for char in ent.text):
            continue
        if ent.label_ == "PERSON" and re.fullmatch(
            r"(?:On|on)\s+(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)",
            ent.text,
        ):
            continue
        mapped = label_map.get(ent.label_)
        if mapped:
            add_entity(entities, seen, text, ent.start_char, ent.end_char, mapped)
    entities.sort(key=entity_sort_key)
    return deduplicate_overlaps(entities)


def deduplicate_overlaps(entities: List[Dict[str, object]]) -> List[Dict[str, object]]:
    filtered: List[Dict[str, object]] = []
    for entity in entities:
        overlap = False
        for kept in filtered:
            if entity["start"] < kept["end"] and entity["end"] > kept["start"]:
                current_len = entity["end"] - entity["start"]
                kept_len = kept["end"] - kept["start"]
                if current_len <= kept_len:
                    overlap = True
                    break
        if not overlap:
            filtered.append(entity)
    return filtered


def extract_entities(text: str) -> List[Dict[str, object]]:
    entities = extract_entities_rule_based(text)
    return merge_spacy_entities(text, entities)


def entity_lookup(entities: List[Dict[str, object]]) -> Dict[str, Dict[str, object]]:
    return {entity["text"]: entity for entity in entities}


def split_sentences_with_spans(text: str) -> List[Tuple[str, int, int]]:
    sentences: List[Tuple[str, int, int]] = []
    start = 0
    for match in re.finditer(r"[.!?。！？]", text):
        end = match.end()
        chunk = text[start:end].strip()
        if chunk:
            sentences.append((text[start:end], start, end))
        start = end
    if start < len(text):
        tail = text[start:].strip()
        if tail:
            sentences.append((text[start:], start, len(text)))
    return sentences


def has_relation_between(
    relations: List[Dict[str, str]],
    left: str,
    right: str,
    relation_name: str | None = None,
) -> bool:
    for relation in relations:
        same_pair = relation["source"] == left and relation["target"] == right
        reverse_pair = relation["source"] == right and relation["target"] == left
        if not (same_pair or reverse_pair):
            continue
        if relation_name is None or relation["relation"] == relation_name:
            return True
    return False


def sentence_entities(
    entities: List[Dict[str, object]],
    start: int,
    end: int,
) -> List[Dict[str, object]]:
    return [
        entity
        for entity in entities
        if entity["start"] >= start and entity["end"] <= end
    ]


def enrich_sentence_level_relations(
    text: str,
    entities: List[Dict[str, object]],
    relations: List[Dict[str, str]],
    seen: set,
):
    for sentence_text, start, end in split_sentences_with_spans(text):
        lower_sentence = sentence_text.lower()
        sent_entities = sentence_entities(entities, start, end)
        persons = [entity for entity in sent_entities if entity["label"] == "PERSON"]
        orgs = [entity for entity in sent_entities if entity["label"] == "ORG"]
        locs = [entity for entity in sent_entities if entity["label"] == "LOC"]

        if "founded" in lower_sentence and persons and orgs:
            add_relation(relations, seen, persons[0]["text"], orgs[0]["text"], "FOUNDER_OF")

        if persons and orgs and (" said " in lower_sentence or " told " in lower_sentence or " leads " in lower_sentence):
            add_relation(relations, seen, persons[0]["text"], orgs[0]["text"], "WORKS_FOR")

        if "partnership with" in lower_sentence and len(orgs) >= 2:
            add_relation(relations, seen, orgs[0]["text"], orgs[1]["text"], "PARTNERS_WITH")

        if "investors" in lower_sentence and persons and orgs:
            add_relation(relations, seen, persons[0]["text"], orgs[0]["text"], "WORKS_FOR")

        if "steve jobs" in lower_sentence and any(org["text"] == "Apple" for org in orgs):
            add_relation(relations, seen, "Steve Jobs", "Apple", "FOUNDER_OF")

        if len(orgs) >= 2:
            anchor_org = orgs[0]["text"]
            for org in orgs[1:]:
                if not has_relation_between(relations, anchor_org, org["text"]):
                    add_relation(relations, seen, anchor_org, org["text"], "RELATED_TO")

        if locs and orgs:
            for loc in locs:
                nearest_org = min(
                    orgs,
                    key=lambda org: min(
                        abs(int(loc["start"]) - int(org["end"])),
                        abs(int(org["start"]) - int(loc["end"])),
                    ),
                )
                if not has_relation_between(relations, nearest_org["text"], loc["text"]):
                    add_relation(relations, seen, nearest_org["text"], loc["text"], "LOCATED_IN")


def extract_relations(text: str, entities: List[Dict[str, object]]) -> List[Dict[str, str]]:
    relations: List[Dict[str, str]] = []
    seen = set()
    entity_map = entity_lookup(entities)

    patterns = [
        (r"([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*) founded ([A-Z][A-Za-z]+)", "FOUNDER_OF"),
        (r"([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*) now leads ([A-Z][A-Za-z]+)", "WORKS_FOR"),
        (r"([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*) said ([A-Z][A-Za-z]+) was discussing .*? partnership with ([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*)", "SPOKES_COMPLEX"),
        (r"([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*) told ([A-Z][A-Za-z]+) investors", "SPOKES_FOR"),
        (r"([\u4e00-\u9fff]{2,8})创立了([\u4e00-\u9fff]{2,20})", "FOUNDER_OF"),
        (r"([\u4e00-\u9fff]{2,20})位于([\u4e00-\u9fff]{2,12})", "LOCATED_IN"),
        (r"([\u4e00-\u9fff]{2,20})在([\u4e00-\u9fff]{2,12})举行", "HELD_IN"),
    ]

    for pattern, relation in patterns:
        for match in re.finditer(pattern, text):
            if relation == "SPOKES_COMPLEX":
                person = match.group(1).strip()
                org_a = match.group(2).strip()
                org_b = match.group(3).strip()
                add_relation(relations, seen, person, org_a, "WORKS_FOR")
                add_relation(relations, seen, org_a, org_b, "PARTNERS_WITH")
                continue
            if relation == "SPOKES_FOR":
                source = match.group(1).strip()
                target = match.group(2).strip()
                add_relation(relations, seen, source, target, "WORKS_FOR")
                continue
            source = match.group(1).strip()
            target = match.group(2).strip()
            add_relation(relations, seen, source, target, relation)

    dates = [entity["text"] for entity in entities if entity["label"] == "DATE"]
    events = [entity["text"] for entity in entities if entity["label"] == "EVENT"]
    orgs = [entity["text"] for entity in entities if entity["label"] == "ORG"]
    locs = [entity["text"] for entity in entities if entity["label"] == "LOC"]

    for event in events:
        for date in dates:
            if event in text and date in text:
                add_relation(relations, seen, event, date, "HAPPENED_ON")
        for loc in locs:
            if event in text and loc in text:
                add_relation(relations, seen, event, loc, "HELD_IN")

    for org in orgs:
        if org in text:
            for loc in locs:
                if loc in text and org != loc and abs(text.find(org) - text.find(loc)) < 24:
                    add_relation(relations, seen, org, loc, "LOCATED_IN")

    enrich_sentence_level_relations(text, entities, relations, seen)

    for relation in relations:
        relation["source_type"] = entity_map.get(relation["source"], {}).get("label", "UNKNOWN")
        relation["target_type"] = entity_map.get(relation["target"], {}).get("label", "UNKNOWN")

    return relations


def add_relation(
    relations: List[Dict[str, str]],
    seen: set,
    source: str,
    target: str,
    relation: str,
):
    if not source or not target or source == target:
        return
    key = (source, target, relation)
    if key in seen:
        return
    seen.add(key)
    relations.append({"source": source, "target": target, "relation": relation})


def bio_tagging(text: str, entities: List[Dict[str, object]]) -> List[Tuple[str, str]]:
    tokens = tokenize_with_spans(text, entities)
    tags: List[Tuple[str, str]] = []

    for token in tokens:
        tag = "O"
        for entity in entities:
            if token["start"] >= entity["start"] and token["end"] <= entity["end"]:
                prefix = "B" if token["start"] == entity["start"] else "I"
                tag = f"{prefix}-{entity['label']}"
                break
        tags.append((token["text"], tag))

    return tags


def render_highlighted_text(text: str, entities: List[Dict[str, object]]) -> str:
    if not entities:
        return f"<div class='text-box'>{html.escape(text)}</div>"

    parts = ["<div class='text-box'>"]
    cursor = 0
    for entity in entities:
        start = entity["start"]
        end = entity["end"]
        label = entity["label"]
        color = ENTITY_CONFIG.get(label, {}).get("color", "#e2e8f0")
        if start > cursor:
            parts.append(html.escape(text[cursor:start]))
        mention = html.escape(text[start:end])
        short_label = ENTITY_CONFIG.get(label, {}).get("label", label)
        parts.append(
            f"<span class='entity-span' style='background:{color};'>"
            f"{mention}<span class='entity-label'>{short_label}</span></span>"
        )
        cursor = end
    if cursor < len(text):
        parts.append(html.escape(text[cursor:]))
    parts.append("</div>")
    return "".join(parts)


def render_bio_text(tags: List[Tuple[str, str]]) -> str:
    lines = [f"{token:<20} {tag}" for token, tag in tags]
    return "<div class='bio-box'>" + html.escape("\n".join(lines)) + "</div>"


def render_tab_guide(what_it_does: str, how_to_read: str, notes: str):
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


def build_relation_table(relations: List[Dict[str, str]]) -> pd.DataFrame:
    if not relations:
        return pd.DataFrame(columns=["Subject", "Predicate", "Object", "Subject Type", "Object Type"])
    rows = [
        {
            "Subject": item["source"],
            "Predicate": item["relation"],
            "Object": item["target"],
            "Subject Type": item.get("source_type", "UNKNOWN"),
            "Object Type": item.get("target_type", "UNKNOWN"),
        }
        for item in relations
    ]
    return pd.DataFrame(rows)


def build_graph_data(entities: List[Dict[str, object]], relations: List[Dict[str, str]]) -> Tuple[List[Dict], List[Dict]]:
    nodes = []
    node_seen = set()
    for entity in entities:
        if entity["text"] in node_seen:
            continue
        config = ENTITY_CONFIG.get(entity["label"], {"color": "#cbd5e1", "size": 24, "label": entity["label"]})
        nodes.append(
            {
                "id": entity["text"],
                "label": entity["text"],
                "title": f"{entity['text']} ({entity['label']})",
                "group": entity["label"],
                "color": config["color"],
                "value": config["size"],
                "font": {"size": 18},
            }
        )
        node_seen.add(entity["text"])

    edges = []
    for idx, relation in enumerate(relations):
        edges.append(
            {
                "id": f"edge-{idx}",
                "from": relation["source"],
                "to": relation["target"],
                "label": relation["relation"],
                "color": {"color": RELATION_COLORS.get(relation["relation"], "#64748b")},
                "arrows": "to",
                "font": {"align": "middle", "size": 14},
                "smooth": {"type": "dynamic"},
            }
        )
    return nodes, edges


def render_kg(nodes: List[Dict], edges: List[Dict], height: int = 560):
    graph_html = f"""
    <div id="kg-network" style="width:100%; height:{height}px; border-radius:18px; background:linear-gradient(180deg,#ffffff,#f8fbff);"></div>
    <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <script>
        const nodes = new vis.DataSet({json.dumps(nodes, ensure_ascii=False)});
        const edges = new vis.DataSet({json.dumps(edges, ensure_ascii=False)});
        const container = document.getElementById("kg-network");
        const data = {{ nodes: nodes, edges: edges }};
        const options = {{
            autoResize: true,
            interaction: {{
                hover: true,
                zoomView: true,
                dragView: true,
                dragNodes: true
            }},
            physics: {{
                enabled: true,
                solver: "forceAtlas2Based",
                forceAtlas2Based: {{
                    gravitationalConstant: -45,
                    springLength: 150,
                    springConstant: 0.05
                }},
                stabilization: {{ iterations: 180 }}
            }},
            nodes: {{
                borderWidth: 1.5,
                shadow: true,
                shape: "dot",
                scaling: {{ min: 18, max: 36 }}
            }},
            edges: {{
                width: 2.2,
                shadow: false
            }}
        }};
        new vis.Network(container, data, options);
    </script>
    """
    components.html(graph_html, height=height + 20)


def render_echarts(entity_counts: Dict[str, int], relation_counts: Dict[str, int], height: int = 450):
    entity_series = [{"name": key, "value": value} for key, value in entity_counts.items()]
    relation_x = list(relation_counts.keys())
    relation_y = list(relation_counts.values())

    chart_html = f"""
    <div id="ie-chart" style="width:100%; height:{height}px;"></div>
    <script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
    <script>
        const chart = echarts.init(document.getElementById('ie-chart'));
        const option = {{
            backgroundColor: 'rgba(255,255,255,0)',
            tooltip: {{ trigger: 'item' }},
            color: ['#0ea5a4', '#3b82f6', '#f97316', '#8b5cf6', '#ef4444', '#64748b'],
            grid: {{ left: '52%', right: '4%', top: 48, bottom: 72 }},
            title: [
                {{
                    text: '实体类别分布',
                    left: '3%',
                    top: 10,
                    textStyle: {{ fontSize: 16, fontWeight: 'bold' }}
                }},
                {{
                    text: '关系类型统计',
                    left: '55%',
                    top: 10,
                    textStyle: {{ fontSize: 16, fontWeight: 'bold' }}
                }}
            ],
            xAxis: {{
                type: 'category',
                data: {json.dumps(relation_x, ensure_ascii=False)},
                axisLabel: {{ rotate: 20, margin: 16 }}
            }},
            yAxis: {{
                type: 'value',
                splitLine: {{ lineStyle: {{ type: 'dashed' }} }}
            }},
            series: [
                {{
                    type: 'pie',
                    radius: ['36%', '62%'],
                    center: ['24%', '56%'],
                    label: {{ formatter: '{{b}}: {{c}}' }},
                    data: {json.dumps(entity_series, ensure_ascii=False)}
                }},
                {{
                    type: 'bar',
                    barWidth: 26,
                    data: {json.dumps(relation_y, ensure_ascii=False)},
                    itemStyle: {{
                        borderRadius: [8, 8, 0, 0]
                    }},
                    label: {{
                        show: true,
                        position: 'top'
                    }}
                }}
            ]
        }};
        chart.setOption(option);
        window.addEventListener('resize', () => chart.resize());
    </script>
    """
    components.html(chart_html, height=height)


st.markdown(
    """
<div class="hero-shell">
    <div class="hero-card">
        <div class="hero-grid">
            <div>
                <div class="hero-kicker">Information Extraction Studio</div>
                <div class="hero-title">信息抽取与知识图谱构建系统</div>
                <div class="hero-desc">
                    基于 Streamlit 的课堂演示应用，覆盖命名实体识别、BIO 标注、关系抽取、知识图谱交互可视化与统计分析。
                    当前版本采用“规则 + Mock + 可选 spaCy 增强”的轻量实现，适合直接展示完整抽取链路。
                </div>
            </div>
            <div class="hero-mini-panel">
                <div class="hero-mini-title">Pipeline Snapshot</div>
                <div class="hero-mini-list">
                    <div class="hero-mini-item"><span>Step 01</span><b>NER + BIO</b></div>
                    <div class="hero-mini-item"><span>Step 02</span><b>Relation Extraction</b></div>
                    <div class="hero-mini-item"><span>Step 03</span><b>Knowledge Graph</b></div>
                    <div class="hero-mini-item"><span>Step 04</span><b>Analytics Dashboard</b></div>
                </div>
            </div>
        </div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("参数设置")
    if "demo_choice" not in st.session_state:
        st.session_state["demo_choice"] = "默认文本1：课程示例"
    if "input_text_value" not in st.session_state:
        st.session_state["input_text_value"] = DEFAULT_TEXT
    if "last_demo_choice" not in st.session_state:
        st.session_state["last_demo_choice"] = None

    demo_options = ["自定义输入"] + list(DEMO_TEXTS.keys())
    selected_demo = st.selectbox("选择示例文本", demo_options, key="demo_choice")
    if selected_demo != st.session_state["last_demo_choice"]:
        if selected_demo != "自定义输入":
            st.session_state["input_text_value"] = DEMO_TEXTS[selected_demo]
        st.session_state["last_demo_choice"] = selected_demo

    input_text = st.text_area("输入待分析文本", key="input_text_value", height=220)
    show_bio = st.checkbox("查看底层 BIO 标注", value=False)
    st.caption("可在上方切换 4 组默认文本，也可以选择“自定义输入”后手动输入。")
    run_button = st.button("开始抽取", type="primary", use_container_width=True)

    st.markdown("### 实体颜色图例")
    legend_html = ["<div class='tag-legend'>"]
    for key, value in ENTITY_CONFIG.items():
        legend_html.append(
            f"<span class='tag-pill' style='background:{value['color']};'>{value['label']}</span>"
        )
    legend_html.append("</div>")
    st.markdown("".join(legend_html), unsafe_allow_html=True)

if run_button or "last_result" not in st.session_state:
    entities = extract_entities(input_text)
    relations = extract_relations(input_text, entities)
    bio_tags = bio_tagging(input_text, entities)
    st.session_state["last_result"] = {
        "text": input_text,
        "entities": entities,
        "relations": relations,
        "bio_tags": bio_tags,
    }

result = st.session_state["last_result"]
entities = result["entities"]
relations = result["relations"]
bio_tags = result["bio_tags"]
nodes, edges = build_graph_data(entities, relations)

entity_counts = Counter(item["label"] for item in entities)
relation_counts = Counter(item["relation"] for item in relations)

st.markdown(
    f"""
<div class="metric-chip-wrap">
    <div class="metric-chip"><span>实体数量</span><b>{len(entities)}</b></div>
    <div class="metric-chip"><span>关系数量</span><b>{len(relations)}</b></div>
    <div class="metric-chip"><span>图谱节点</span><b>{len(nodes)}</b></div>
    <div class="metric-chip"><span>BIO Token</span><b>{len(bio_tags)}</b></div>
</div>
""",
    unsafe_allow_html=True,
)

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "模块 1：NER 与 BIO 标注",
        "模块 2：关系抽取",
        "模块 3：知识图谱",
        "模块 4：统计分析",
    ]
)

with tab1:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="section-heading">
            <div>
                <div class="section-kicker">Module 01</div>
                <h3>命名实体识别与 BIO 标注</h3>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("支持中文与英文混合文本的课堂演示。开启复选框后，展示将切换为底层 BIO 序列。")
    render_tab_guide(
        "这一部分负责从原文中识别人名、机构、地点、时间和事件等实体，并将实体在原文中高亮显示。",
        "左侧可以看高亮文本或 BIO 序列，右侧表格给出实体文本、类别和在原句中的位置。",
        "讲解时可以先看高亮结果，再切换到 BIO 标注，说明系统如何从字符级或词级标签还原出实体。",
    )

    col1, col2 = st.columns([1.7, 1.0], gap="large")
    with col1:
        st.markdown("<div class='subgrid-card'>", unsafe_allow_html=True)
        if show_bio:
            st.markdown(render_bio_text(bio_tags), unsafe_allow_html=True)
        else:
            st.markdown(render_highlighted_text(result["text"], entities), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='subgrid-card'>", unsafe_allow_html=True)
        entity_df = pd.DataFrame(entities) if entities else pd.DataFrame(columns=["text", "label", "start", "end"])
        st.dataframe(
            entity_df.rename(
                columns={
                    "text": "Entity",
                    "label": "Type",
                    "start": "Start",
                    "end": "End",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with tab2:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="section-heading">
            <div>
                <div class="section-kicker">Module 02</div>
                <h3>实体关系抽取</h3>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("当前关系抽取结合模式匹配与实体共现规则，适合课程展示 Subject-Predicate-Object 结构。")
    render_tab_guide(
        "这一部分在实体识别的基础上抽取三元组关系，例如谁创立了哪个公司、哪个机构位于哪里、哪个事件在何处举行。",
        "表格中的 Subject、Predicate、Object 对应知识图谱里的头实体、关系和尾实体，后两列展示实体类型。",
        "展示时可以强调关系抽取把离散实体连接起来，是从 NER 走向知识图谱构建的关键一步。",
    )
    relation_df = build_relation_table(relations)
    st.dataframe(relation_df, use_container_width=True, hide_index=True)

    if relation_df.empty:
        st.info("当前文本未抽取到关系，可尝试包含“创立了 / founded / 位于 / 在…举行 / now leads”等模式。")
    st.markdown("</div>", unsafe_allow_html=True)

with tab3:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="section-heading">
            <div>
                <div class="section-kicker">Module 03</div>
                <h3>知识图谱交互可视化</h3>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("节点支持拖拽，滚轮缩放，边带箭头并显示关系标签。")
    render_tab_guide(
        "这一部分把前面抽取出的实体和关系转换成图结构，让知识之间的连接方式可以被直观看到。",
        "图中的节点代表实体，颜色区分实体类型，带箭头的连线代表关系方向，标签显示关系名称。",
        "演示时可以拖动节点或缩放图谱，突出某个核心实体如何与多类对象形成关联网络。",
    )
    if nodes:
        render_kg(nodes, edges)
    else:
        st.warning("暂无可渲染的图谱节点。请先输入包含实体的信息。")
    st.markdown("</div>", unsafe_allow_html=True)

with tab4:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="section-heading">
            <div>
                <div class="section-kicker">Module 04</div>
                <h3>抽取结果统计分析</h3>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("左侧为实体类别分布，右侧为关系类型统计。")
    render_tab_guide(
        "这一部分对抽取结果做汇总统计，帮助观察当前文本中哪些实体类别更多、哪些关系最常出现。",
        "左侧饼图反映实体类别占比，右侧柱状图展示关系频次，下方表格给出更清晰的计数结果。",
        "展示时可以把它作为整体总结，用来说明模型输出不仅能看样例，还能支持更宏观的分析。",
    )
    if entity_counts or relation_counts:
        render_echarts(entity_counts, relation_counts)
    else:
        st.info("暂无统计数据。")

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 实体类别计数")
        entity_count_df = pd.DataFrame(
            [{"Entity Type": key, "Count": value} for key, value in entity_counts.items()]
        )
        st.dataframe(entity_count_df, use_container_width=True, hide_index=True)
    with col2:
        st.markdown("#### 关系类型计数")
        relation_count_df = pd.DataFrame(
            [{"Relation Type": key, "Count": value} for key, value in relation_counts.items()]
        )
        st.dataframe(relation_count_df, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

with st.expander("实现说明", expanded=False):
    st.markdown(
        """
1. `模块 1`：先做规则与词典匹配，若本地已安装 `spaCy + en_core_web_sm`，会自动补充英文实体识别。
2. `模块 2`：输出标准三元组表格，包括主体、关系词、客体及实体类型。
3. `模块 3`：通过 `vis-network` 渲染知识图谱，支持交互探索。
4. `模块 4`：通过 `ECharts` 展示实体和关系的统计结果，形成完整抽取链路闭环。
        """
    )
