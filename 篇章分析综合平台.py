from __future__ import annotations

import html
import importlib.util
import re
import subprocess
import sys
from typing import Any

import streamlit as st
from ui_theme import inject_iekg_theme, render_guide_card, render_hero

try:
    import requests
except ImportError:
    requests = None


_original_find_spec = importlib.util.find_spec


def _safe_find_spec(name: str, *args: Any, **kwargs: Any):
    if name == "torchvision" or name.startswith("torchvision."):
        return None
    return _original_find_spec(name, *args, **kwargs)


st.set_page_config(
    page_title="篇章分析综合平台",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="auto",
)


inject_iekg_theme(
    """
    .segment-card {
        background: #ffffff;
        border: 1px solid rgba(148, 163, 184, 0.26);
        border-left: 6px solid #2563eb;
        border-radius: 14px;
        padding: 0.85rem 1rem;
        margin-bottom: 0.65rem;
    }
    .segment-card.gold {
        border-left-color: #0f766e;
    }
    .segment-meta {
        color: #475569;
        font-size: 0.92rem;
        margin-bottom: 0.8rem;
    }
    .boundary-shell {
        background: rgba(255,255,255,0.94);
        border: 1px solid rgba(148, 163, 184, 0.24);
        border-radius: 16px;
        padding: 0.9rem 1rem;
        margin-bottom: 0.8rem;
    }
    .boundary-flow {
        display: flex;
        flex-wrap: wrap;
        gap: 0.45rem 0.4rem;
        line-height: 1.8;
    }
    .boundary-token {
        display: inline-flex;
        align-items: center;
        gap: 0.25rem;
        padding: 0.15rem 0.42rem;
        border-radius: 0.55rem;
        background: #f8fafc;
        color: #0f172a;
        border: 1px solid rgba(148, 163, 184, 0.18);
        font-size: 0.96rem;
    }
    .boundary-token.predicted {
        background: #fff7ed;
        border-color: rgba(234, 88, 12, 0.24);
        color: #9a3412;
        font-weight: 700;
    }
    .boundary-token.gold {
        background: #f0fdf4;
        border-color: rgba(15, 118, 110, 0.22);
        color: #166534;
        font-weight: 700;
    }
    .boundary-badge {
        display: inline-block;
        border-radius: 999px;
        padding: 0.08rem 0.35rem;
        font-size: 0.7rem;
        font-weight: 700;
        background: rgba(255,255,255,0.72);
    }
    .arg-box {
        border-radius: 14px;
        padding: 0.85rem 1rem;
        margin-bottom: 0.7rem;
        border: 1px solid rgba(148, 163, 184, 0.24);
    }
    .arg-before {
        background: #eff6ff;
    }
    .arg-conn {
        background: #fff7ed;
        text-align: center;
        font-weight: 700;
    }
    .arg-after {
        background: #f0fdf4;
    }
    .coref-box {
        background: white;
        border: 1px solid rgba(148, 163, 184, 0.26);
        border-radius: 16px;
        padding: 1rem;
        line-height: 2;
        font-size: 1.02rem;
    }
    .cluster-chip {
        display: inline-block;
        border-radius: 999px;
        padding: 0.15rem 0.55rem;
        margin-right: 0.35rem;
        font-size: 0.85rem;
        font-weight: 700;
        color: #0f172a;
    }
    """
)


CONNECTIVE_FAMILIES = {
    "Comparison": {"although", "though", "but", "however", "whereas"},
    "Contingency": {"because", "since", "therefore", "so"},
    "Temporal": {"when", "while", "after", "before", "once", "since"},
    "Expansion": {"and", "also", "moreover", "furthermore", "besides"},
}

CONNECTIVES_FOR_DEMO = ("because", "although", "since", "when", "however", "but")

NEURAL_EDUSEG_CANDIDATES = [
    {
        "sample_id": "wsj_1103",
        "raw_url": "https://raw.githubusercontent.com/PKU-TANGENT/NeuralEDUSeg/master/data/rst/TRAINING/wsj_1103.out",
        "edu_url": "https://raw.githubusercontent.com/PKU-TANGENT/NeuralEDUSeg/master/data/rst/TRAINING/wsj_1103.out.edus",
    },
    {
        "sample_id": "wsj_1105",
        "raw_url": "https://raw.githubusercontent.com/PKU-TANGENT/NeuralEDUSeg/master/data/rst/TRAINING/wsj_1105.out",
        "edu_url": "https://raw.githubusercontent.com/PKU-TANGENT/NeuralEDUSeg/master/data/rst/TRAINING/wsj_1105.out.edus",
    },
    {
        "sample_id": "wsj_1101",
        "raw_url": "https://raw.githubusercontent.com/PKU-TANGENT/NeuralEDUSeg/master/data/rst/TRAINING/wsj_1101.out",
        "edu_url": "https://raw.githubusercontent.com/PKU-TANGENT/NeuralEDUSeg/master/data/rst/TRAINING/wsj_1101.out.edus",
    },
]

FALLBACK_SAMPLE = {
    "sample_id": "wsj_1103",
    "paragraph": (
        "Boston Co., the upper-crust financial services concern that was rocked by a "
        "management scandal late last year, has had a sharp drop in profitability -- "
        "mainly because a high-risk bet on interest rates backfired."
    ),
    "gold_segments": [
        "Boston Co., the upper-crust financial services concern",
        "that was rocked by a management scandal late last year,",
        "has had a sharp drop in profitability",
        "-- mainly because a high-risk bet on interest rates backfired.",
    ],
    "raw_url": "https://raw.githubusercontent.com/PKU-TANGENT/NeuralEDUSeg/master/data/rst/TRAINING/wsj_1103.out",
    "edu_url": "https://raw.githubusercontent.com/PKU-TANGENT/NeuralEDUSeg/master/data/rst/TRAINING/wsj_1103.out.edus",
    "source_note": "网络不可用时使用内置的 NeuralEDUSeg 样例缓存。",
}

COREF_COLORS = [
    "#fde68a",
    "#bfdbfe",
    "#c7f9cc",
    "#fbcfe8",
    "#ddd6fe",
    "#fecaca",
]
BOUNDARY_TOKEN_RE = re.compile(r"\w+(?:[-']\w+)*|[^\w\s]")


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def compact_text(text: str) -> str:
    return re.sub(r"\s+", "", text)


def run_pip_install(packages: list[str]) -> tuple[bool, str]:
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", *packages],
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception as exc:  # pragma: no cover - UI fallback
        return False, str(exc)

    output = (result.stdout or "") + "\n" + (result.stderr or "")
    return result.returncode == 0, output.strip()


def render_dependency_help() -> None:
    with st.expander("依赖说明", expanded=False):
        st.markdown(
            """
            - 本页依赖：`streamlit`、`requests`、`spacy`、`fastcoref`
            - `spaCy` 的 `en_core_web_sm` 模型会在第一次真正用到篇章关系模块时尝试自动下载
            - `fastcoref` 首次成功加载时，还会自动从 Hugging Face 下载其模型文件
            """
        )
        st.code(
            "\n".join(
                [
                    "pip install streamlit requests spacy fastcoref",
                    "python -m spacy download en_core_web_sm",
                ]
            ),
            language="bash",
        )


def ensure_requests_available() -> bool:
    if requests is not None:
        return True
    st.error("当前环境缺少 requests，第一页无法联网抓取 NeuralEDUSeg 示例数据。")
    return False


@st.cache_resource(show_spinner=False)
def load_spacy_model() -> tuple[Any | None, str | None]:
    try:
        import spacy
    except ImportError:
        return None, "未安装 spaCy，请先执行 `pip install spacy`。"

    try:
        return spacy.load("en_core_web_sm"), None
    except OSError:
        try:
            result = subprocess.run(
                [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
                capture_output=True,
                text=True,
                check=False,
            )
        except Exception as exc:  # pragma: no cover - UI fallback
            return None, f"自动下载 spaCy 模型失败：{exc}"

        if result.returncode != 0:
            stderr = (result.stderr or "").strip()
            return None, f"自动下载 spaCy 模型失败：{stderr or '请手动执行下载命令。'}"

        try:
            return spacy.load("en_core_web_sm"), None
        except Exception as exc:  # pragma: no cover - UI fallback
            return None, f"spaCy 模型下载后仍无法加载：{exc}"


def patch_fastcoref_model_class(model_class: Any) -> None:
    if not hasattr(model_class, "all_tied_weights_keys"):
        model_class.all_tied_weights_keys = {}
    if not hasattr(model_class, "_tied_weights_keys"):
        model_class._tied_weights_keys = {}


def ensure_fastcoref_transformers_compat() -> None:
    compatibility_targets = (
        ("fastcoref.coref_models.modeling_fcoref", "FCorefModel"),
        ("fastcoref.coref_models.modeling_lingmess", "LingMessModel"),
    )
    for module_name, class_name in compatibility_targets:
        try:
            module = __import__(module_name, fromlist=[class_name])
            model_class = getattr(module, class_name, None)
            if model_class is not None:
                patch_fastcoref_model_class(model_class)
        except Exception:
            continue


def install_fastcoref_ui(installed: bool = False) -> None:
    if installed:
        st.warning("当前环境里的 fastcoref 已安装，但暂时没有成功加载；可以尝试重新安装或稍后重试。")
    else:
        st.warning("当前环境还没有安装 fastcoref，第三个标签页需要它来做现代指代消解。")
    col1, col2 = st.columns([1, 1.4])
    with col1:
        if st.button("尝试自动安装 fastcoref", key="install_fastcoref"):
            with st.spinner("正在安装 fastcoref，这可能需要一点时间..."):
                ok, output = run_pip_install(["fastcoref"])
            if ok:
                st.session_state.pop("_fastcoref_model", None)
                st.success("fastcoref 安装完成，请再点击一次“开始分析”。")
                st.rerun()
            st.error("fastcoref 安装失败，请按下方命令手动安装。")
            st.code(output or "pip install fastcoref", language="bash")
    with col2:
        st.code("pip install fastcoref", language="bash")
        st.caption("如果还没有安装 PyTorch，pip 会一并处理；首次运行模型时还需要联网下载权重。")


def load_fastcoref_model() -> tuple[Any | None, str | None]:
    importlib.util.find_spec = _safe_find_spec
    try:
        from fastcoref import FCoref
    except ImportError:
        return None, "missing_package"

    try:
        ensure_fastcoref_transformers_compat()
        model = st.session_state.get("_fastcoref_model")
        if model is not None:
            return model, None

        model = FCoref(device="cpu")
        st.session_state["_fastcoref_model"] = model
        return model, None
    except Exception as exc:  # pragma: no cover - UI fallback
        return None, str(exc)


def split_into_paragraphs(raw_text: str) -> list[str]:
    blocks = re.split(r"\n\s*\n", raw_text)
    return [normalize_spaces(block) for block in blocks if normalize_spaces(block)]


def align_paragraphs_to_gold(raw_text: str, edu_text: str) -> list[dict[str, Any]]:
    paragraphs = split_into_paragraphs(raw_text)
    edu_lines = [normalize_spaces(line) for line in edu_text.splitlines() if normalize_spaces(line)]

    pairs: list[dict[str, Any]] = []
    edu_index = 0
    for paragraph in paragraphs:
        target = compact_text(paragraph)
        collected: list[str] = []
        start_index = edu_index

        while edu_index < len(edu_lines):
            collected.append(edu_lines[edu_index])
            edu_index += 1
            current = compact_text(" ".join(collected))
            if current == target:
                pairs.append({"paragraph": paragraph, "gold_segments": collected[:]})
                break
            if not target.startswith(current):
                edu_index = start_index
                break

    return pairs


def choose_demo_pair(pairs: list[dict[str, Any]]) -> dict[str, Any] | None:
    for pair in pairs:
        paragraph = pair["paragraph"].lower()
        if any(keyword in paragraph for keyword in CONNECTIVES_FOR_DEMO) and len(pair["gold_segments"]) >= 3:
            return pair
    for pair in pairs:
        if len(pair["gold_segments"]) >= 3:
            return pair
    return pairs[0] if pairs else None


@st.cache_data(show_spinner=False)
def fetch_neuraleduseg_demo() -> dict[str, Any]:
    if not ensure_requests_available():
        return FALLBACK_SAMPLE

    headers = {"User-Agent": "streamlit-discourse-demo/1.0"}
    last_error = "未知错误"

    for candidate in NEURAL_EDUSEG_CANDIDATES:
        try:
            raw_response = requests.get(candidate["raw_url"], timeout=20, headers=headers)
            edu_response = requests.get(candidate["edu_url"], timeout=20, headers=headers)
            raw_response.raise_for_status()
            edu_response.raise_for_status()
        except Exception as exc:
            last_error = str(exc)
            continue

        pairs = align_paragraphs_to_gold(raw_response.text, edu_response.text)
        chosen = choose_demo_pair(pairs)
        if chosen is None:
            last_error = f"{candidate['sample_id']} 对齐失败"
            continue

        return {
            "sample_id": candidate["sample_id"],
            "paragraph": chosen["paragraph"],
            "gold_segments": chosen["gold_segments"],
            "raw_url": candidate["raw_url"],
            "edu_url": candidate["edu_url"],
            "source_note": "已从 NeuralEDUSeg GitHub 示例数据实时抓取。",
        }

    fallback = FALLBACK_SAMPLE.copy()
    fallback["source_note"] = f"{FALLBACK_SAMPLE['source_note']} 最近一次抓取错误：{last_error}"
    return fallback


def rule_based_segment(text: str) -> list[str]:
    working = normalize_spaces(text)
    working = re.sub(r"([.!?])\s+", r"\1 <CUT> ", working)
    working = re.sub(r"\s+(because|although)\s+", r" <CUT> \1 ", working, flags=re.IGNORECASE)
    return [segment.strip() for segment in working.split("<CUT>") if segment.strip()]


def render_segment_list(title: str, segments: list[str], css_class: str, caption: str) -> None:
    st.markdown(f"### {title}")
    st.markdown(f"<div class='segment-meta'>{html.escape(caption)}</div>", unsafe_allow_html=True)
    for index, segment in enumerate(segments, start=1):
        st.markdown(
            f"<div class='segment-card {css_class}'><strong>{index}.</strong> {html.escape(segment)}</div>",
            unsafe_allow_html=True,
        )


def tokenize_boundary_view(text: str) -> list[str]:
    return BOUNDARY_TOKEN_RE.findall(text)


def get_boundary_token_indices(paragraph: str, segments: list[str]) -> tuple[list[str], set[int]]:
    paragraph_tokens = tokenize_boundary_view(paragraph)
    boundary_indices: set[int] = set()
    cursor = 0

    for segment in segments[:-1]:
        cursor += len(tokenize_boundary_view(segment))
        if 0 < cursor <= len(paragraph_tokens):
            boundary_indices.add(cursor - 1)

    return paragraph_tokens, boundary_indices


def render_boundary_token_view(title: str, tokens: list[str], boundary_indices: set[int], css_class: str, caption: str) -> None:
    token_html: list[str] = []
    for index, token in enumerate(tokens):
        classes = f"boundary-token {css_class}" if index in boundary_indices else "boundary-token"
        badge = "<span class='boundary-badge'>BT</span>" if index in boundary_indices else ""
        token_html.append(
            f"<span class='{classes}'>{html.escape(token)}{badge}</span>"
        )

    st.markdown(f"### {title}")
    st.markdown(
        f"""
        <div class='boundary-shell'>
            <div class='segment-meta'>{html.escape(caption)}</div>
            <div class='boundary-flow'>
                {''.join(token_html)}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def classify_since(token: Any, doc: Any) -> str:
    window = doc[token.i + 1 : min(len(doc), token.i + 8)]
    time_cues = {"ago", "today", "yesterday", "tomorrow", "earlier", "later", "last", "next", "then"}
    if any(getattr(item, "ent_type_", "") in {"DATE", "TIME"} for item in window):
        return "Temporal"
    if any(item.text.lower() in time_cues for item in window):
        return "Temporal"
    return "Contingency"


def split_arguments(doc: Any, token_index: int) -> tuple[str, str]:
    token = doc[token_index]
    connective = token.text.lower()

    if token_index <= 1 or (token_index > 0 and doc[token_index - 1].text in {"(", ";"}):
        comma_index = None
        for idx in range(token_index + 1, len(doc)):
            if doc[idx].text == ",":
                comma_index = idx
                break
        if comma_index is not None:
            before = doc[token_index + 1 : comma_index].text.strip()
            after = doc[comma_index + 1 :].text.strip()
            return before, after

    if connective in {"however"} and token_index + 1 < len(doc) and doc[token_index + 1].text == ",":
        before = doc[:token_index].text.strip()
        after = doc[token_index + 2 :].text.strip()
        return before, after

    before = doc[:token_index].text.strip()
    after = doc[token_index + 1 :].text.strip()
    return before, after


def extract_discourse_relations(sentence: str, nlp: Any) -> list[dict[str, str]]:
    doc = nlp(sentence)
    results: list[dict[str, str]] = []
    seen: set[int] = set()

    for token in doc:
        lower = token.text.lower()
        relation_type = None
        if lower == "since":
            relation_type = classify_since(token, doc)
        else:
            for label, lexicon in CONNECTIVE_FAMILIES.items():
                if lower in lexicon:
                    relation_type = label
                    break

        if relation_type is None or token.i in seen:
            continue

        before, after = split_arguments(doc, token.i)
        results.append(
            {
                "connective": token.text,
                "relation_type": relation_type,
                "before": before or "未抽取到前项",
                "after": after or "未抽取到后项",
            }
        )
        seen.add(token.i)

    return results


def deduplicate_mentions(mentions: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for mention in mentions:
        clean = mention.strip()
        key = clean.lower()
        if not clean or key in seen:
            continue
        seen.add(key)
        ordered.append(clean)
    return ordered


def filter_non_overlapping_spans(text: str, clusters: list[list[tuple[int, int]]]) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    for cluster_id, mentions in enumerate(clusters):
        if len(mentions) < 2:
            continue
        for start, end in mentions:
            spans.append(
                {
                    "cluster_id": cluster_id,
                    "start": int(start),
                    "end": int(end),
                    "text": text[int(start) : int(end)],
                }
            )

    spans.sort(key=lambda item: (item["start"], -(item["end"] - item["start"])))
    filtered: list[dict[str, Any]] = []
    current_end = -1
    for span in spans:
        if span["start"] < current_end:
            continue
        filtered.append(span)
        current_end = span["end"]
    return filtered


def render_highlighted_coref_text(text: str, clusters: list[list[tuple[int, int]]]) -> str:
    spans = filter_non_overlapping_spans(text, clusters)
    if not spans:
        return f"<div class='coref-box'>{html.escape(text)}</div>"

    parts = ["<div class='coref-box'>"]
    cursor = 0
    for span in spans:
        if span["start"] > cursor:
            parts.append(html.escape(text[cursor : span["start"]]))
        color = COREF_COLORS[span["cluster_id"] % len(COREF_COLORS)]
        parts.append(
            f"<span style='background:{color}; padding:0.18rem 0.3rem; border-radius:0.5rem;'>"
            f"{html.escape(span['text'])}</span>"
        )
        cursor = span["end"]
    if cursor < len(text):
        parts.append(html.escape(text[cursor:]))
    parts.append("</div>")
    return "".join(parts)


render_hero(
    "篇章分析综合平台",
    "Discourse Analysis Studio",
    "围绕课堂里的话语分割、篇章关系识别和现代指代消解，提供一个更统一也更适合展示的三模块演示界面。",
    [
        ("Step 01", "EDU Segmentation"),
        ("Step 02", "Relation Typing"),
        ("Step 03", "Coreference Resolution"),
    ],
)
render_dependency_help()

tab1, tab2, tab3 = st.tabs(["话语分割", "篇章关系", "指代消解"])


with tab1:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="section-heading">
            <div>
                <div class="section-kicker">Module 01</div>
                <h3>话语分割</h3>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("左边是你的规则切分器，右边是 NeuralEDUSeg 示例数据里的人工金标准 EDU。")
    render_guide_card(
        "这一部分演示句子或段落如何被切成更细的语义单元，也就是课堂里常说的 EDU。",
        "左列代表启发式规则，右列代表数据集里的标准切法；两边段数和边界差异越大，越能说明规则法的局限。",
        "展示时可以先读原文，再逐段对照，指出 because、从句和插入结构往往是规则切分容易出错的地方。",
    )
    sample = fetch_neuraleduseg_demo()
    rule_segments = rule_based_segment(sample["paragraph"])
    paragraph_tokens, predicted_boundary_indices = get_boundary_token_indices(sample["paragraph"], rule_segments)
    _, gold_boundary_indices = get_boundary_token_indices(sample["paragraph"], sample["gold_segments"])

    st.markdown("#### 示例原文")
    st.text_area(
        "NeuralEDUSeg 示例段落",
        value=sample["paragraph"],
        height=140,
        disabled=True,
        label_visibility="collapsed",
    )
    st.caption(
        f"{sample['source_note']} 样例：`{sample['sample_id']}`  Raw: {sample['raw_url']}  Gold: {sample['edu_url']}"
    )

    col1, col2 = st.columns(2)
    with col1:
        render_segment_list(
            "规则切分结果",
            rule_segments,
            "",
            "规则只有两条：句号后切分；遇到 because / although 前切一刀。",
        )
    with col2:
        render_segment_list(
            "NeuralEDUSeg 金标准",
            sample["gold_segments"],
            "gold",
            "右侧直接展示数据集中这段文本对应的人工 EDU 边界。",
        )

    st.markdown("#### Boundary Token 高亮")
    boundary_col1, boundary_col2 = st.columns(2)
    with boundary_col1:
        render_boundary_token_view(
            "规则预测边界词",
            paragraph_tokens,
            predicted_boundary_indices,
            "predicted",
            "橙色高亮表示规则切分器预测为 EDU 边界的 token。",
        )
    with boundary_col2:
        render_boundary_token_view(
            "金标准边界词",
            paragraph_tokens,
            gold_boundary_indices,
            "gold",
            "绿色高亮表示 NeuralEDUSeg 示例数据中的人工边界 token。",
        )

    st.info(
        f"规则切分得到 {len(rule_segments)} 段，NeuralEDUSeg 金标准为 {len(sample['gold_segments'])} 段。"
    )
    st.markdown("</div>", unsafe_allow_html=True)


with tab2:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="section-heading">
            <div>
                <div class="section-kicker">Module 02</div>
                <h3>篇章关系</h3>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("自动定位连接词，并把它们映射到对比、因果、时间、扩展四类关系。")
    render_guide_card(
        "这一部分演示句子内部的逻辑连接方式，尤其是连接词如何提示前后两个片段之间的关系。",
        "中间橙色卡片显示连接词和关系类型，上下两块分别表示它前后的论据内容。",
        "since 会先看后面是否跟时间表达，如果像 since 2020 这样的形式更偏时间，否则优先判作原因。",
    )
    relation_sentence = st.text_input(
        "输入英文句子",
        value="Since 2020, the company has grown, but investors remain cautious because costs are rising.",
    )

    if st.button("分析篇章关系", key="analyze_relations", type="primary"):
        nlp, error_message = load_spacy_model()
        if nlp is None:
            st.error(error_message or "spaCy 模型不可用。")
            st.code(
                "\n".join(
                    [
                        "pip install spacy",
                        "python -m spacy download en_core_web_sm",
                    ]
                ),
                language="bash",
            )
        else:
            relations = extract_discourse_relations(relation_sentence, nlp)
            if not relations:
                st.info("当前句子里没有识别到预置连接词，可以试试 because / although / since / when / but。")
            else:
                for index, relation in enumerate(relations, start=1):
                    st.markdown(f"### 关系 {index}")
                    st.markdown(
                        f"<div class='arg-box arg-before'><strong>前项 / Arg1</strong><br>{html.escape(relation['before'])}</div>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"<div class='arg-box arg-conn'><strong>{html.escape(relation['connective'])}</strong> → {html.escape(relation['relation_type'])}</div>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"<div class='arg-box arg-after'><strong>后项 / Arg2</strong><br>{html.escape(relation['after'])}</div>",
                        unsafe_allow_html=True,
                    )
    st.markdown("</div>", unsafe_allow_html=True)


with tab3:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="section-heading">
            <div>
                <div class="section-kicker">Module 03</div>
                <h3>指代消解</h3>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("利用 fastcoref 识别 John / he / his 这类同指对象，并给出可视化的共指链。")
    render_guide_card(
        "这一部分演示文本中哪些词实际上指向同一个人、事物或群体，也就是共指链。",
        "原文高亮里同色表示属于同一条指代链，下方清单则把每一组实体按顺序列出来。",
        "如果模型首次运行较慢，通常是因为正在下载权重；这属于正常现象。",
    )
    default_coref_text = (
        "John met Mary after she finished her class. He handed her a notebook because it "
        "had been left on his desk. Later, they said it was the first draft of their report."
    )
    coref_text = st.text_area("输入英文短文", value=default_coref_text, height=180)

    if st.button("开始指代消解", key="run_coref", type="primary"):
        model, error_message = load_fastcoref_model()
        if model is None:
            if error_message == "missing_package":
                install_fastcoref_ui()
            else:
                st.error(f"fastcoref 加载失败：{error_message}")
                if "all_tied_weights_keys" in error_message or "'list' object has no attribute 'keys'" in error_message:
                    st.caption("这属于 fastcoref 与当前 transformers 版本的兼容问题；页面已经尝试自动补丁，但本次加载仍未成功。")
                else:
                    st.caption("这通常意味着模型权重下载失败，或者本机缺少相关深度学习依赖。")
                install_fastcoref_ui(installed=True)
        else:
            with st.spinner("正在运行 fastcoref，首次加载模型会稍慢一些..."):
                try:
                    predictions = model.predict(texts=[coref_text])
                    prediction = predictions[0]
                    cluster_offsets = prediction.get_clusters(as_strings=False)
                    cluster_strings = prediction.get_clusters()
                except Exception as exc:  # pragma: no cover - UI fallback
                    st.error(f"指代消解运行失败：{exc}")
                    st.stop()

            valid_offsets = [cluster for cluster in cluster_offsets if len(cluster) >= 2]
            valid_strings = [deduplicate_mentions(cluster) for cluster in cluster_strings if len(cluster) >= 2]

            if not valid_offsets:
                st.info("这段文本里没有识别到明显的共指链，可以试试再多放一些 he / she / it / they。")
            else:
                st.markdown("#### 原文高亮")
                st.markdown(render_highlighted_coref_text(coref_text, valid_offsets), unsafe_allow_html=True)

                st.markdown("#### 指代链清单")
                for index, mentions in enumerate(valid_strings, start=1):
                    color = COREF_COLORS[(index - 1) % len(COREF_COLORS)]
                    st.markdown(
                        f"<span class='cluster-chip' style='background:{color};'>实体 {index}</span> "
                        + html.escape(", ".join(mentions)),
                        unsafe_allow_html=True,
                    )
    st.markdown("</div>", unsafe_allow_html=True)
