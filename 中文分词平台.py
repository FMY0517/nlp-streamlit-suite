from __future__ import annotations

from collections import Counter
from pathlib import Path
import html
import re
import unicodedata

import streamlit as st

from deploy_utils import find_available_chinese_font
from ui_theme import inject_iekg_theme, render_guide_card, render_hero


st.set_page_config(
    page_title="中文分词平台",
    page_icon="✂️",
    layout="wide",
    initial_sidebar_state="auto",
)


inject_iekg_theme(
    """
    .token-card {
        background: linear-gradient(180deg, rgba(255,255,255,0.97), rgba(248,251,255,0.92));
        border-radius: 22px;
        padding: 1rem;
        border: 1px solid rgba(15, 23, 42, 0.08);
        box-shadow: 0 14px 30px rgba(15, 23, 42, 0.06);
        min-height: 200px;
    }
    .normal-card {
        background: linear-gradient(180deg, rgba(255,255,255,0.98), rgba(246,250,255,0.94));
        border-radius: 20px;
        padding: 1rem;
        border: 1px solid rgba(15, 23, 42, 0.08);
        box-shadow: 0 12px 26px rgba(15, 23, 42, 0.06);
        min-height: 220px;
    }
    .normal-card h4, .token-card h4 {
        margin: 0 0 0.75rem;
        color: #0f172a;
        font-size: 1.05rem;
    }
    .result-text {
        color: #334155;
        line-height: 1.85;
        white-space: pre-wrap;
        word-break: break-word;
    }
    .subtle-label {
        display: inline-block;
        margin-bottom: 0.6rem;
        padding: 0.25rem 0.62rem;
        border-radius: 999px;
        background: rgba(37, 99, 235, 0.08);
        color: #2563eb;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.04em;
    }
    .pos-wrap {
        display: flex;
        flex-wrap: wrap;
        gap: 1rem 0.85rem;
        margin-top: 0.25rem;
    }
    .pos-token {
        position: relative;
        min-width: 66px;
        padding: 0.95rem 0.8rem 0.42rem;
        border-radius: 18px;
        background: rgba(255,255,255,0.94);
        border: 1px solid rgba(148, 163, 184, 0.2);
        box-shadow: 0 10px 20px rgba(15, 23, 42, 0.05);
        text-align: center;
    }
    .pos-badge {
        position: absolute;
        top: -0.46rem;
        left: 0.65rem;
        padding: 0.16rem 0.45rem;
        border-radius: 999px;
        color: white;
        font-size: 0.74rem;
        font-weight: 700;
        box-shadow: 0 8px 14px rgba(15, 23, 42, 0.12);
    }
    .pos-word {
        color: #0f172a;
        font-weight: 700;
        font-size: 1rem;
        line-height: 1.4;
    }
    .legend-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.55rem;
        margin-top: 0.85rem;
    }
    .legend-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.32rem 0.58rem;
        border-radius: 999px;
        background: rgba(255,255,255,0.92);
        border: 1px solid rgba(148, 163, 184, 0.16);
        color: #334155;
        font-size: 0.84rem;
        font-weight: 600;
    }
    .legend-dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        display: inline-block;
    }
    .dependency-code code {
        white-space: pre-wrap;
        word-break: break-word;
    }
    """
)


render_hero(
    "中文分词平台",
    "Chinese Tokenization Lab",
    "围绕歧义句和复杂短语，比较不同中文分词算法在切分边界、词频结构和词性标注上的差异，让课堂展示更直观。",
    [
        ("Step 01", "Text Normalize"),
        ("Step 02", "Segmentation Compare"),
        ("Step 03", "POS + Word Cloud"),
    ],
)


render_guide_card(
    "输入任意中文文本后，可以切换不同分词算法，观察歧义短语在不同模型下的切分差异。",
    "上方三个规范化结果分别展示特殊符号清洗、全角转半角和简繁转换效果；后续分词与词频统计默认基于“全角转半角 + 转简体”的文本。",
    "推荐先试“长春市长春节致辞”“南京市长江大桥”“研究生命起源”等典型歧义句，再切换算法做课堂演示。",
)


try:
    import jieba
    import jieba.posseg as pseg

    HAS_JIEBA = True
except ImportError:
    HAS_JIEBA = False
    jieba = None
    pseg = None

try:
    import pkuseg

    HAS_PKUSEG = True
except ImportError:
    HAS_PKUSEG = False
    pkuseg = None

try:
    import thulac

    HAS_THULAC = True
except ImportError:
    HAS_THULAC = False
    thulac = None

try:
    from opencc import OpenCC

    HAS_OPENCC = True
except ImportError:
    HAS_OPENCC = False
    OpenCC = None

try:
    from wordcloud import WordCloud

    HAS_WORDCLOUD = True
except ImportError:
    HAS_WORDCLOUD = False
    WordCloud = None

try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    pd = None


DEPENDENCY_TEXT = """必装依赖:
pip install streamlit jieba pandas wordcloud opencc-python-reimplemented

可选增强:
pip install pkuseg thulac
"""


SPECIAL_SYMBOL_RE = re.compile(r"[^\u4e00-\u9fffA-Za-z0-9，。！？；：、“”‘’（）《》【】,.!?;:()\[\]\-—\s]")
TOKEN_KEEP_RE = re.compile(r"[\u4e00-\u9fffA-Za-z0-9]")
FULLWIDTH_PUNCT_MAP = {
    "。": ".",
    "，": ",",
    "！": "!",
    "？": "?",
    "：": ":",
    "；": ";",
    "（": "(",
    "）": ")",
    "【": "[",
    "】": "]",
    "《": "<",
    "》": ">",
    "“": '"',
    "”": '"',
    "‘": "'",
    "’": "'",
}
POS_PALETTE = [
    "#0f766e",
    "#2563eb",
    "#ea580c",
    "#7c3aed",
    "#0891b2",
    "#b91c1c",
    "#4f46e5",
    "#15803d",
    "#c2410c",
    "#475569",
]
POS_LABELS = {
    "n": "名词",
    "nr": "人名",
    "ns": "地名",
    "t": "时间词",
    "nt": "机构名",
    "nrt": "音译人名",
    "nz": "其他专名",
    "v": "动词",
    "vd": "副动词",
    "vn": "名动词",
    "a": "形容词",
    "ad": "副形词",
    "an": "名形词",
    "d": "副词",
    "m": "数词",
    "q": "量词",
    "r": "代词",
    "p": "介词",
    "c": "连词",
    "u": "助词",
    "uj": "结构助词",
    "ul": "时态助词",
    "e": "叹词",
    "o": "拟声词",
    "x": "非语素字",
    "zg": "状态词语素",
    "w": "标点",
}


ALGORITHM_CONFIG = {
    "jieba_precise": {
        "label": "Jieba 精确模式",
        "available": HAS_JIEBA,
        "install": "pip install jieba",
    },
    "jieba_full": {
        "label": "Jieba 全模式",
        "available": HAS_JIEBA,
        "install": "pip install jieba",
    },
    "jieba_search": {
        "label": "Jieba 搜索引擎模式",
        "available": HAS_JIEBA,
        "install": "pip install jieba",
    },
    "pkuseg": {
        "label": "PKUSeg",
        "available": HAS_PKUSEG,
        "install": "pip install pkuseg",
    },
    "thulac": {
        "label": "THULAC",
        "available": HAS_THULAC,
        "install": "pip install thulac",
    },
}


@st.cache_resource(show_spinner=False)
def get_opencc_converters():
    if not HAS_OPENCC:
        return None, None
    return OpenCC("s2t"), OpenCC("t2s")


@st.cache_resource(show_spinner=False)
def get_pkuseg_segmenter():
    if not HAS_PKUSEG:
        return None
    return pkuseg.pkuseg()


@st.cache_resource(show_spinner=False)
def get_pkuseg_postagger():
    if not HAS_PKUSEG:
        return None
    return pkuseg.pkuseg(postag=True)


@st.cache_resource(show_spinner=False)
def get_thulac_segmenter(seg_only: bool = True):
    if not HAS_THULAC:
        return None
    return thulac.thulac(seg_only=seg_only)


def remove_special_symbols(text: str) -> str:
    cleaned = SPECIAL_SYMBOL_RE.sub("", text)
    return re.sub(r"\s+", " ", cleaned).strip()


def fullwidth_to_halfwidth(text: str) -> str:
    normalized_chars: list[str] = []
    for char in text:
        if char in FULLWIDTH_PUNCT_MAP:
            normalized_chars.append(FULLWIDTH_PUNCT_MAP[char])
            continue
        code = ord(char)
        if code == 12288:
            normalized_chars.append(" ")
        elif 65281 <= code <= 65374:
            normalized_chars.append(chr(code - 65248))
        else:
            normalized_chars.append(unicodedata.normalize("NFKC", char))
    return "".join(normalized_chars)


def convert_script_variants(text: str) -> tuple[str, str, bool]:
    if not HAS_OPENCC:
        return text, text, False
    s2t, t2s = get_opencc_converters()
    if s2t is None or t2s is None:
        return text, text, False
    return s2t.convert(text), t2s.convert(text), True


def prepare_segmentation_text(text: str) -> str:
    cleaned = remove_special_symbols(text)
    halfwidth = fullwidth_to_halfwidth(cleaned)
    _, simplified, has_converter = convert_script_variants(halfwidth)
    base_text = simplified if has_converter else halfwidth
    return re.sub(r"\s+", " ", base_text).strip()


def format_algorithm_label(key: str) -> str:
    label = ALGORITHM_CONFIG[key]["label"]
    return label if ALGORITHM_CONFIG[key]["available"] else f"{label}（需安装）"


def list_available_algorithms() -> list[str]:
    return [key for key, config in ALGORITHM_CONFIG.items() if config["available"]]


def segment_text(text: str, algorithm: str) -> list[str]:
    if not text:
        return []
    if algorithm == "jieba_precise" and HAS_JIEBA:
        return [token.strip() for token in jieba.lcut(text, cut_all=False) if token.strip()]
    if algorithm == "jieba_full" and HAS_JIEBA:
        return [token.strip() for token in jieba.lcut(text, cut_all=True) if token.strip()]
    if algorithm == "jieba_search" and HAS_JIEBA:
        return [token.strip() for token in jieba.lcut_for_search(text) if token.strip()]
    if algorithm == "pkuseg" and HAS_PKUSEG:
        segmenter = get_pkuseg_segmenter()
        return [token.strip() for token in segmenter.cut(text) if token.strip()]
    if algorithm == "thulac" and HAS_THULAC:
        segmenter = get_thulac_segmenter(seg_only=True)
        pairs = segmenter.cut(text)
        return [pair[0].strip() for pair in pairs if pair and pair[0].strip()]
    return []


def pos_tag_tokens(tokens: list[str], algorithm: str) -> list[tuple[str, str]]:
    text = " ".join(tokens)
    if not text:
        return []
    if algorithm == "pkuseg" and HAS_PKUSEG:
        segmenter = get_pkuseg_postagger()
        return [(word.strip(), tag.strip()) for word, tag in segmenter.cut(text) if word.strip()]
    if algorithm == "thulac" and HAS_THULAC:
        segmenter = get_thulac_segmenter(seg_only=False)
        pairs = segmenter.cut(text)
        return [(word.strip(), tag.strip()) for word, tag in pairs if word.strip()]
    if HAS_JIEBA:
        return [(item.word.strip(), item.flag.strip()) for item in pseg.cut(text, HMM=False) if item.word.strip()]
    return [(token, "unk") for token in tokens]


def meaningful_tokens(tokens: list[str]) -> list[str]:
    words: list[str] = []
    for token in tokens:
        normalized = token.strip()
        if len(normalized) <= 1:
            continue
        if TOKEN_KEEP_RE.search(normalized):
            words.append(normalized)
    return words


def build_wordcloud(freq_map: dict[str, int]):
    if not HAS_WORDCLOUD or not freq_map:
        return None
    font_path = find_chinese_font()
    if not font_path:
        return None
    cloud = WordCloud(
        width=900,
        height=520,
        background_color="white",
        font_path=font_path,
        colormap="viridis",
        prefer_horizontal=0.92,
    )
    return cloud.generate_from_frequencies(freq_map).to_array()


def find_chinese_font() -> str | None:
    return find_available_chinese_font()


def pos_color_map(tags: list[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for index, tag in enumerate(sorted(set(tags))):
        mapping[tag] = POS_PALETTE[index % len(POS_PALETTE)]
    return mapping


def render_text_card(title: str, subtitle: str, content: str) -> None:
    safe_title = html.escape(title)
    safe_subtitle = html.escape(subtitle)
    safe_content = html.escape(content or "（暂无内容）")
    st.markdown(
        f"""
        <div class="normal-card">
            <div class="subtle-label">{safe_subtitle}</div>
            <h4>{safe_title}</h4>
            <div class="result-text">{safe_content}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_pos_tags(tagged_tokens: list[tuple[str, str]]) -> None:
    if not tagged_tokens:
        st.info("当前没有可展示的词性标注结果。")
        return

    color_mapping = pos_color_map([tag for _, tag in tagged_tokens])
    token_html: list[str] = []
    legend_html: list[str] = []

    seen_tags: set[str] = set()
    for word, tag in tagged_tokens:
        color = color_mapping[tag]
        token_html.append(
            (
                f'<div class="pos-token">'
                f'<span class="pos-badge" style="background:{color};">{html.escape(tag)}</span>'
                f'<div class="pos-word">{html.escape(word)}</div>'
                f"</div>"
            )
        )
        if tag not in seen_tags:
            seen_tags.add(tag)
            legend_html.append(
                (
                    f'<span class="legend-chip">'
                    f'<span class="legend-dot" style="background:{color};"></span>'
                    f"{html.escape(tag)} / {html.escape(POS_LABELS.get(tag, '词性'))}"
                    f"</span>"
                )
            )

    st.markdown(
        (
            '<div class="token-card">'
            "<h4>词性标注结果</h4>"
            f'<div class="pos-wrap">{"".join(token_html)}</div>'
            f'<div class="legend-row">{"".join(legend_html)}</div>'
            "</div>"
        ),
        unsafe_allow_html=True,
    )


TEXT_INPUT_KEY = "segmentation_input_text"

example_text = (
    "长春市长春节致辞，南京市长江大桥风景宜人。"
    "研究生命起源的人常说：中文分词真的很考验算法。"
)
symbol_demo_text = (
    "今天，ＡＩ研究社說：“NLP真有趣！”同學们用Python、正則表達式和《分詞》工具，"
    "測試全角／半角、繁體與简体是否能同步清洗。"
)
frequency_demo_text = (
    "在自然语言处理课上，我们用文本、文本、文本构造小语料，再观察模型、模型、模型如何统计词频。"
    "老师提醒大家先清洗数据，再比较分词结果，因为数据质量会影响分析，数据越稳定，结果越清晰。"
    "同学们继续加入学习、学习、学习这样的高频词，也加入中文分词、词频统计、词频标注等表达，"
    "这样既能看到常见词反复出现，也能直观看到词云和高频词列表的变化。"
)


def load_demo_text(text: str) -> None:
    st.session_state[TEXT_INPUT_KEY] = text

available_algorithms = list_available_algorithms()
default_algorithm = available_algorithms[0] if available_algorithms else "jieba_precise"


with st.expander("依赖与运行说明", expanded=False):
    st.markdown("下面这组命令可以直接安装本页面需要的核心依赖。")
    st.code(DEPENDENCY_TEXT, language="bash")
    if not HAS_OPENCC:
        st.warning("未检测到 `opencc-python-reimplemented`，简繁转换展示将退化为原文。")
    if not HAS_WORDCLOUD:
        st.warning("未检测到 `wordcloud`，词云图区域将只显示提示信息。")
    elif not find_chinese_font():
        st.warning("未找到可用的中文字体文件，词云图可能无法正确渲染中文。")
    missing_optional = [config["label"] for config in ALGORITHM_CONFIG.values() if not config["available"]]
    if missing_optional:
        st.info(f"当前可选增强算法未全部安装：{'、'.join(missing_optional)}")


if TEXT_INPUT_KEY not in st.session_state:
    st.session_state[TEXT_INPUT_KEY] = example_text


text_input = st.text_area(
    "输入待分析的中文文本",
    key=TEXT_INPUT_KEY,
    height=120,
    help="适合输入带有歧义短语或多义结构的句子，也可以直接点击下方按钮载入课堂演示示例。",
)

demo_col1, demo_col2 = st.columns(2)
with demo_col1:
    st.button(
        "载入清晰展示示例",
        key="load_symbol_demo_text",
        on_click=load_demo_text,
        args=(symbol_demo_text,),
        use_container_width=True,
    )
with demo_col2:
    st.button(
        "载入词频标注示例",
        key="load_frequency_demo_text",
        on_click=load_demo_text,
        args=(frequency_demo_text,),
        use_container_width=True,
    )

algorithm = st.selectbox(
    "选择分词算法",
    options=list(ALGORITHM_CONFIG.keys()),
    format_func=format_algorithm_label,
    index=list(ALGORITHM_CONFIG.keys()).index(default_algorithm) if default_algorithm in ALGORITHM_CONFIG else 0,
)

if not ALGORITHM_CONFIG[algorithm]["available"]:
    st.error(
        f"当前未安装 {ALGORITHM_CONFIG[algorithm]['label']}。"
        f" 请先运行：`{ALGORITHM_CONFIG[algorithm]['install']}`"
    )
    st.stop()


removed_special = remove_special_symbols(text_input)
halfwidth_text = fullwidth_to_halfwidth(text_input)
traditional_text, simplified_text, has_converter = convert_script_variants(halfwidth_text)
segmentation_text = prepare_segmentation_text(text_input)
tokens = segment_text(segmentation_text, algorithm)
token_text = " ".join(tokens)
filtered_words = meaningful_tokens(tokens)
freq_counter = Counter(filtered_words)
top_items = freq_counter.most_common(5)
tagged_tokens = pos_tag_tokens(tokens, algorithm)


metric_col1, metric_col2, metric_col3 = st.columns(3)
metric_col1.metric("输入字符数", len(text_input.strip()))
metric_col2.metric("分词数量", len(tokens))
metric_col3.metric("有效双字词+", len(filtered_words))


st.markdown(
    """
    <div class="section-card">
        <div class="section-heading">
            <div>
                <div class="section-kicker">Block 01</div>
                <h3>文本规范化结果</h3>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

col1, col2, col3 = st.columns(3)
with col1:
    render_text_card("去除特殊符号效果", "Clean Symbols", removed_special or "（清洗后为空）")
with col2:
    render_text_card("全角转半角效果", "Fullwidth to Halfwidth", halfwidth_text or "（转换后为空）")
with col3:
    conversion_text = (
        f"简体 -> 繁体：{traditional_text}\n\n繁体 -> 简体：{simplified_text}"
        if has_converter
        else "未安装 OpenCC，当前展示原文。"
    )
    render_text_card("简体繁体转换效果", "Script Conversion", conversion_text)


st.markdown(
    """
    <div class="section-card">
        <div class="section-heading">
            <div>
                <div class="section-kicker">Block 02</div>
                <h3>分词结果</h3>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="token-card">
        <div class="subtle-label">实际分词输入</div>
        <div class="result-text">{html.escape(segmentation_text or '（暂无内容）')}</div>
        <hr style="border:none;border-top:1px solid rgba(148,163,184,0.18);margin:0.9rem 0;">
        <div class="subtle-label">{html.escape(ALGORITHM_CONFIG[algorithm]['label'])}</div>
        <div class="result-text">{html.escape(token_text or '（暂无分词结果）')}</div>
    </div>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    """
    <div class="section-card">
        <div class="section-heading">
            <div>
                <div class="section-kicker">Block 03</div>
                <h3>词频统计</h3>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

freq_col, cloud_col = st.columns(2)

with freq_col:
    st.markdown('<div class="token-card"><h4>高频词 Top 5（过滤单字词）</h4>', unsafe_allow_html=True)
    if top_items and HAS_PANDAS:
        chart_df = pd.DataFrame(top_items, columns=["词语", "频次"]).set_index("词语")
        st.bar_chart(chart_df)
        st.dataframe(chart_df, use_container_width=True)
    elif top_items:
        st.table({"词语": [item[0] for item in top_items], "频次": [item[1] for item in top_items]})
    else:
        st.info("过滤单字词后，没有足够的高频词可展示。")
    st.markdown("</div>", unsafe_allow_html=True)

with cloud_col:
    st.markdown('<div class="token-card"><h4>词云图</h4>', unsafe_allow_html=True)
    cloud_image = build_wordcloud(dict(freq_counter))
    if cloud_image is not None:
        st.image(cloud_image, use_container_width=True)
    elif not HAS_WORDCLOUD:
        st.warning("未安装 `wordcloud`，请先安装后查看词云图。")
    elif not find_chinese_font():
        st.warning("当前环境缺少中文字体，词云图无法正确生成。")
    else:
        st.info("当前有效词数量较少，暂时无法生成词云图。")
    st.markdown("</div>", unsafe_allow_html=True)


st.markdown(
    """
    <div class="section-card">
        <div class="section-heading">
            <div>
                <div class="section-kicker">Block 04</div>
                <h3>词性标注</h3>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

render_pos_tags(tagged_tokens)
