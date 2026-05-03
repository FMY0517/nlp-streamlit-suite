from __future__ import annotations

import html
import importlib.util
import os
import random
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
import streamlit as st

from deploy_utils import ensure_named_cache_dir
from ui_theme import inject_iekg_theme, render_guide_card, render_hero


st.set_page_config(
    page_title="情感分析与可视化仪表盘",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="auto",
)


inject_iekg_theme(
    """
    .sentiment-result-card {
        background: linear-gradient(180deg, rgba(255,255,255,0.97), rgba(248,251,255,0.92));
        border-radius: 22px;
        padding: 1rem 1.05rem;
        border: 1px solid rgba(15, 23, 42, 0.08);
        box-shadow: 0 14px 30px rgba(15, 23, 42, 0.06);
        min-height: 178px;
    }
    .sentiment-pill {
        display: inline-block;
        padding: 0.3rem 0.68rem;
        border-radius: 999px;
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 0.65rem;
    }
    .pill-positive {
        background: rgba(15, 118, 110, 0.12);
        color: #0f766e;
    }
    .pill-negative {
        background: rgba(220, 38, 38, 0.12);
        color: #b91c1c;
    }
    .pill-neutral {
        background: rgba(245, 158, 11, 0.14);
        color: #b45309;
    }
    .result-title {
        color: #0f172a;
        font-size: 1.5rem;
        font-weight: 700;
        margin: 0.1rem 0 0.45rem;
    }
    .result-meta {
        color: #475569;
        line-height: 1.7;
    }
    .dashboard-shell {
        background:
            radial-gradient(circle at top right, rgba(37, 99, 235, 0.18), transparent 26%),
            linear-gradient(145deg, rgba(8, 15, 33, 0.97), rgba(15, 35, 64, 0.96));
        border-radius: 28px;
        padding: 1.1rem;
        border: 1px solid rgba(148, 163, 184, 0.18);
        box-shadow: 0 24px 50px rgba(15, 23, 42, 0.2);
    }
    .dashboard-title {
        color: #e2e8f0;
        font-size: 1.25rem;
        font-weight: 700;
        margin-bottom: 0.9rem;
    }
    .dashboard-caption {
        color: rgba(226, 232, 240, 0.78);
        margin-bottom: 1rem;
        line-height: 1.7;
    }
    .stats-row {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.8rem;
        margin-bottom: 1rem;
    }
    .stats-box {
        border-radius: 20px;
        padding: 0.9rem 1rem;
        color: white;
        box-shadow: 0 18px 30px rgba(15, 23, 42, 0.18);
    }
    .stats-box b {
        display: block;
        font-size: 1.85rem;
        margin-top: 0.2rem;
    }
    .stats-positive {
        background: linear-gradient(145deg, #0f766e, #14b8a6);
    }
    .stats-neutral {
        background: linear-gradient(145deg, #b45309, #f59e0b);
    }
    .stats-negative {
        background: linear-gradient(145deg, #b91c1c, #ef4444);
    }
    .batch-card {
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(148, 163, 184, 0.16);
        border-radius: 20px;
        padding: 0.9rem;
        color: #e2e8f0;
    }
    .batch-item {
        padding: 0.65rem 0;
        border-bottom: 1px solid rgba(148, 163, 184, 0.12);
    }
    .batch-item:last-child {
        border-bottom: 0;
        padding-bottom: 0;
    }
    .batch-item strong {
        color: #f8fafc;
    }
    @media (max-width: 900px) {
        .stats-row {
            grid-template-columns: 1fr;
        }
    }
    """
)


render_hero(
    "情感分析与可视化仪表盘",
    "Sentiment Intelligence Dashboard",
    "面向电商评价与社交媒体舆情场景，把单句情感分类、显隐式情感识别和批量口碑可视化放到同一个交互式分析面板里。",
    [
        ("Step 01", "Sentiment Gauge"),
        ("Step 02", "Explicit vs Implicit"),
        ("Step 03", "Opinion Dashboard"),
    ],
)


_original_find_spec = importlib.util.find_spec


def _safe_find_spec(name: str, *args: Any, **kwargs: Any):
    if name == "torchvision" or name.startswith("torchvision."):
        return None
    return _original_find_spec(name, *args, **kwargs)


@contextmanager
def suppress_torchvision_for_transformers():
    previous_find_spec = importlib.util.find_spec
    importlib.util.find_spec = _safe_find_spec
    try:
        yield
    finally:
        importlib.util.find_spec = previous_find_spec


def get_pipeline_import():
    with suppress_torchvision_for_transformers():
        from transformers import pipeline as hf_pipeline

    return hf_pipeline


HF_CACHE_DIR = Path(os.environ.get("NLP_SENTIMENT_CACHE_DIR", ensure_named_cache_dir("hf_cache_sentiment")))
HF_CACHE_DIR.mkdir(exist_ok=True)
MODEL_NAME = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"


@st.cache_resource(show_spinner=False)
def load_sentiment_pipeline():
    hf_pipeline = get_pipeline_import()
    return hf_pipeline(
        task="text-classification",
        model=MODEL_NAME,
        tokenizer=MODEL_NAME,
        cache_dir=str(HF_CACHE_DIR),
    )


def map_model_label(label: str) -> str:
    lower = label.strip().lower()
    if lower.startswith("1") or lower.startswith("2"):
        return "Negative"
    if lower.startswith("3"):
        return "Neutral"
    if lower.startswith("4") or lower.startswith("5"):
        return "Positive"
    if "negative" in lower:
        return "Negative"
    if "neutral" in lower:
        return "Neutral"
    return "Positive"


def aggregate_sentiment_scores(raw_scores: list[dict[str, Any]]) -> dict[str, float]:
    sentiment_scores = {"Positive": 0.0, "Neutral": 0.0, "Negative": 0.0}
    for item in raw_scores:
        sentiment = map_model_label(str(item["label"]))
        sentiment_scores[sentiment] += float(item["score"])
    return sentiment_scores


def analyze_sentiment(text: str) -> dict[str, Any]:
    classifier = load_sentiment_pipeline()
    raw_scores = classifier(text, top_k=None)
    if raw_scores and isinstance(raw_scores[0], list):
        raw_scores = raw_scores[0]

    sentiment_scores = aggregate_sentiment_scores(raw_scores)
    best_label, best_score = max(sentiment_scores.items(), key=lambda pair: pair[1])

    return {
        "label": best_label,
        "confidence": float(best_score),
        "raw_scores": sentiment_scores,
    }


def label_to_chinese(label: str) -> str:
    mapping = {"Positive": "积极", "Neutral": "中性", "Negative": "消极"}
    return mapping[label]


def label_to_pill_class(label: str) -> str:
    mapping = {"Positive": "pill-positive", "Neutral": "pill-neutral", "Negative": "pill-negative"}
    return mapping[label]


def label_to_color(label: str) -> str:
    mapping = {"Positive": "#0f766e", "Neutral": "#d97706", "Negative": "#dc2626"}
    return mapping[label]


def build_gauge_chart(label: str, confidence: float) -> go.Figure:
    color = label_to_color(label)
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=round(confidence * 100, 2),
            number={"suffix": "%", "font": {"size": 34, "color": "#0f172a"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#94a3b8"},
                "bar": {"color": color, "thickness": 0.42},
                "bgcolor": "rgba(255,255,255,0)",
                "borderwidth": 0,
                "shape": "angular",
                "steps": [
                    {"range": [0, 35], "color": "rgba(148, 163, 184, 0.18)"},
                    {"range": [35, 70], "color": "rgba(148, 163, 184, 0.12)"},
                    {"range": [70, 100], "color": "rgba(148, 163, 184, 0.08)"},
                ],
            },
            title={"text": f"{label_to_chinese(label)} 置信度", "font": {"size": 18, "color": "#334155"}},
        )
    )
    fig.update_layout(
        margin=dict(l=20, r=20, t=60, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        height=320,
    )
    return fig


def render_sentiment_result(title: str, result: dict[str, Any] | None, placeholder: str) -> None:
    if result is None:
        body = f"<div class='result-meta'>{html.escape(placeholder)}</div>"
    else:
        label = result["label"]
        confidence = result["confidence"]
        body = (
            f"<div class='sentiment-pill {label_to_pill_class(label)}'>{html.escape(label)}</div>"
            f"<div class='result-title'>{html.escape(label_to_chinese(label))}</div>"
            f"<div class='result-meta'>"
            f"模型判定这段文本整体情绪偏<strong>{html.escape(label_to_chinese(label))}</strong>，"
            f"聚合置信度约为 <strong>{confidence:.1%}</strong>。"
            f"</div>"
        )

    st.markdown(
        f"""
        <div class='sentiment-result-card'>
            <div class='translation-label'>{html.escape(title)}</div>
            {body}
        </div>
        """,
        unsafe_allow_html=True,
    )


def generate_mock_reviews() -> list[str]:
    return [
        "物流特别快，昨天下单今天一早就送到了，包装也很完整。",
        "手机外观很好看，系统运行流畅，整体体验很满意。",
        "这个耳机音质一般，倒是佩戴起来还算舒服。",
        "用了不到一周充电口就松了，体验很差。",
        "客服回复速度挺快，不过问题最后也没有完全解决。",
        "屏幕显示细腻，晚上追剧观感很好。",
        "在太阳底下根本看不清屏幕上的字。",
        "价格还可以，但做工没有想象中精致。",
        "玩游戏半小时就烫得厉害，还掉电特别快。",
        "包装很有质感，送人也拿得出手。",
        "声音非常清楚，开会收音效果比预期好。",
        "说明书写得太简单，第一次安装花了很久。",
    ]


if "sentiment_single_text" not in st.session_state:
    st.session_state["sentiment_single_text"] = "这款手机外观很好看，运行也很流畅，我很喜欢。"
if "sentiment_single_result" not in st.session_state:
    st.session_state["sentiment_single_result"] = None
if "explicit_result" not in st.session_state:
    st.session_state["explicit_result"] = None
if "implicit_result" not in st.session_state:
    st.session_state["implicit_result"] = None
if "batch_reviews" not in st.session_state:
    st.session_state["batch_reviews"] = []
if "batch_results" not in st.session_state:
    st.session_state["batch_results"] = []


tab1, tab2, tab3 = st.tabs(
    [
        "模块 1：基础情感分类与置信度量化",
        "模块 2：显式情感 vs. 隐式情感识别",
        "模块 3：舆情挖掘与可视化仪表盘",
    ]
)


with tab1:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="section-heading">
            <div>
                <div class="section-kicker">Module 01</div>
                <h3>基础情感分类与置信度量化</h3>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("演示单条中文评论如何被映射到积极 / 中性 / 消极三种情感极性，并额外给出可信程度。")
    render_guide_card(
        "这一部分对应课件里“情感分类 + 概率输出”的概念。工程里光有类别还不够，置信度能帮助我们决定是否要人工复核。",  # noqa: E501
        "输入明显的好评、差评或模棱两可的评论，看看分类标签和仪表盘指针会如何变化。",
        "这里使用多语言轻量模型，并把细粒度标签聚合成 Positive / Neutral / Negative 三类，更适合课堂演示。",
    )

    single_text = st.text_area(
        "输入一段中文商品评论",
        value=st.session_state["sentiment_single_text"],
        height=140,
    )

    if st.button("分析单条评论情感", key="sentiment_single_run", type="primary"):
        if not single_text.strip():
            st.warning("请先输入评论文本。")
        else:
            with st.spinner("正在加载情感分析模型并计算结果..."):
                try:
                    st.session_state["sentiment_single_result"] = analyze_sentiment(single_text)
                    st.session_state["sentiment_single_text"] = single_text
                except Exception as exc:  # pragma: no cover - UI fallback
                    st.error(f"情感分析失败：{exc}")
                    st.stop()

    result = st.session_state["sentiment_single_result"]
    col1, col2 = st.columns([1, 1.15])
    with col1:
        render_sentiment_result(
            "Sentiment Result",
            result,
            "点击上方按钮后，这里会给出当前评论的情感标签与解释。",
        )
    with col2:
        if result is None:
            st.info("完成分析后，这里会显示置信度仪表盘。")
        else:
            st.plotly_chart(
                build_gauge_chart(result["label"], result["confidence"]),
                use_container_width=True,
                key="single_sentiment_gauge",
            )
    st.markdown("</div>", unsafe_allow_html=True)


with tab2:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="section-heading">
            <div>
                <div class="section-kicker">Module 02</div>
                <h3>显式情感 vs. 隐式情感识别</h3>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("比较带明显褒贬词的表达，和不带情绪词但暗含态度的客观描述，看看模型能不能“听懂”弦外之音。")
    render_guide_card(
        "显式情感通常会出现“太棒了”“太差了”“垃圾”这类直接的褒贬词；隐式情感则更像客观叙述，但事实本身已经透露满意或不满。",
        "先输入一句显式评价，再输入一句隐式描述，观察两边模型判断是否同样稳定。",
        "小模型对显式情感往往更敏感，对隐式负面通常更容易漏判，这正是课堂上值得讨论的误差来源。",
    )

    explicit_col, implicit_col = st.columns(2)
    with explicit_col:
        explicit_text = st.text_area(
            "显式情感评价",
            value="这屏幕画质太垃圾了。",
            height=120,
            key="explicit_sentiment_input",
        )
    with implicit_col:
        implicit_text = st.text_area(
            "隐式客观描述",
            value="在太阳底下根本看不清屏幕上的字。",
            height=120,
            key="implicit_sentiment_input",
        )

    if st.button("对比分析两种表达", key="compare_sentiment_styles", type="primary"):
        texts = {
            "explicit_result": explicit_text.strip(),
            "implicit_result": implicit_text.strip(),
        }
        if not all(texts.values()):
            st.warning("两个输入框都请填写内容。")
        else:
            with st.spinner("正在比较显式与隐式情感表达..."):
                try:
                    st.session_state["explicit_result"] = analyze_sentiment(texts["explicit_result"])
                    st.session_state["implicit_result"] = analyze_sentiment(texts["implicit_result"])
                except Exception as exc:  # pragma: no cover - UI fallback
                    st.error(f"情感分析失败：{exc}")
                    st.stop()

    col1, col2 = st.columns(2)
    with col1:
        render_sentiment_result(
            "显式情感分析结果",
            st.session_state["explicit_result"],
            "输入带明显褒贬词的句子后，这里显示模型判断。",
        )
    with col2:
        render_sentiment_result(
            "隐式情感分析结果",
            st.session_state["implicit_result"],
            "输入不带明显情绪词的客观描述后，这里显示模型判断。",
        )
    st.markdown("</div>", unsafe_allow_html=True)


with tab3:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="section-heading">
            <div>
                <div class="section-kicker">Module 03</div>
                <h3>舆情挖掘与可视化仪表盘</h3>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("面向一批模拟商品评价做批量情感分析，并用可视化图表展示整体口碑结构。")
    render_guide_card(
        "这一部分对应课件里的舆情监测面板思路：先批量跑情感模型，再把结果汇总成管理者一眼能看懂的可视化指标。",
        "点击按钮自动生成 10 到 15 条测试评论，系统会统计 Positive / Neutral / Negative 的数量并绘制口碑分布图。",
        "这里特意把视觉样式做成更偏“数据大屏”的科技感风格，模拟真实业务仪表盘的展示方式。",
    )

    if st.button("生成测试舆情数据", key="generate_batch_reviews", type="primary"):
        reviews = generate_mock_reviews()
        with st.spinner("正在批量分析测试评论..."):
            try:
                batch_results = [{"text": review, **analyze_sentiment(review)} for review in reviews]
            except Exception as exc:  # pragma: no cover - UI fallback
                st.error(f"批量情感分析失败：{exc}")
                st.stop()
        st.session_state["batch_reviews"] = reviews
        st.session_state["batch_results"] = batch_results

    batch_results = st.session_state["batch_results"]
    if not batch_results:
        st.info("点击上方按钮后，这里会生成一组模拟评论，并展示整体舆情分布。")
    else:
        counts = {"Positive": 0, "Neutral": 0, "Negative": 0}
        for item in batch_results:
            counts[item["label"]] += 1

        pie_fig = go.Figure(
            data=[
                go.Pie(
                    labels=["Positive", "Neutral", "Negative"],
                    values=[counts["Positive"], counts["Neutral"], counts["Negative"]],
                    hole=0.52,
                    marker=dict(colors=["#14b8a6", "#f59e0b", "#ef4444"]),
                    textinfo="label+percent",
                )
            ]
        )
        pie_fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e2e8f0"),
            margin=dict(l=10, r=10, t=20, b=10),
            height=360,
        )

        st.markdown("<div class='dashboard-shell'>", unsafe_allow_html=True)
        st.markdown("<div class='dashboard-title'>舆情挖掘与可视化仪表盘</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='dashboard-caption'>系统已经对模拟商品评论完成批量情感打标，下面展示的是当前样本集的整体口碑结构。</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div class='stats-row'>
                <div class='stats-box stats-positive'><span>Positive</span><b>{counts["Positive"]}</b></div>
                <div class='stats-box stats-neutral'><span>Neutral</span><b>{counts["Neutral"]}</b></div>
                <div class='stats-box stats-negative'><span>Negative</span><b>{counts["Negative"]}</b></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        chart_col, list_col = st.columns([1.15, 1])
        with chart_col:
            st.plotly_chart(pie_fig, use_container_width=True, key="sentiment_batch_pie")
        with list_col:
            st.markdown("<div class='batch-card'>", unsafe_allow_html=True)
            st.markdown("**本批次评论明细**")
            for item in batch_results:
                st.markdown(
                    f"""
                    <div class='batch-item'>
                        <strong>{html.escape(label_to_chinese(item["label"]))}</strong>
                        （{item["confidence"]:.1%}）<br/>
                        {html.escape(item["text"])}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
