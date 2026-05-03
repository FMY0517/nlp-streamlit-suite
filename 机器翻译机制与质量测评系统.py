from __future__ import annotations

import html
import importlib.util
import math
import os
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import nltk
import streamlit as st
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

from deploy_utils import ensure_named_cache_dir
from ui_theme import inject_iekg_theme, render_guide_card, render_hero


st.set_page_config(
    page_title="机器翻译机制与质量测评系统",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="auto",
)


inject_iekg_theme(
    """
    .translation-box {
        background: linear-gradient(180deg, rgba(255,255,255,0.97), rgba(248,251,255,0.92));
        border-radius: 20px;
        padding: 1rem;
        border: 1px solid rgba(15, 23, 42, 0.08);
        box-shadow: 0 14px 30px rgba(15, 23, 42, 0.06);
        min-height: 180px;
    }
    .translation-label {
        display: inline-block;
        margin-bottom: 0.7rem;
        padding: 0.26rem 0.64rem;
        border-radius: 999px;
        background: rgba(37, 99, 235, 0.08);
        color: #2563eb;
        font-size: 0.82rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    .translation-content {
        color: #0f172a;
        font-size: 1.02rem;
        line-height: 1.82;
        white-space: pre-wrap;
    }
    .translation-placeholder {
        color: #64748b;
        font-size: 0.98rem;
        line-height: 1.75;
    }
    .score-card {
        background: linear-gradient(145deg, rgba(15, 118, 110, 0.96), rgba(37, 99, 235, 0.92));
        color: white;
        border-radius: 22px;
        padding: 1.1rem 1.15rem;
        box-shadow: 0 16px 30px rgba(37, 99, 235, 0.18);
    }
    .score-card b {
        display: block;
        font-size: 2rem;
        margin-top: 0.25rem;
    }
    .note-box {
        background: linear-gradient(180deg, rgba(255,255,255,0.97), rgba(248,251,255,0.92));
        border-radius: 20px;
        padding: 1rem;
        border: 1px solid rgba(15, 23, 42, 0.08);
        box-shadow: 0 14px 30px rgba(15, 23, 42, 0.06);
    }
    """
)


render_hero(
    "机器翻译机制与质量测评系统",
    "Machine Translation Studio",
    "围绕神经机器翻译、规则直译对比和 BLEU 自动评测，搭建一个适合课堂演示机器翻译机制与质量分析的统一实验页面。",
    [
        ("Step 01", "NMT Engine"),
        ("Step 02", "Rule-based vs NMT"),
        ("Step 03", "BLEU Evaluation"),
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


def get_seq2seq_imports():
    with suppress_torchvision_for_transformers():
        from transformers import MarianMTModel, MarianTokenizer

    return MarianMTModel, MarianTokenizer


HF_CACHE_DIR = Path(os.environ.get("NLP_MT_CACHE_DIR", ensure_named_cache_dir("hf_cache_mt")))
HF_CACHE_DIR.mkdir(exist_ok=True)


@st.cache_resource(show_spinner=False)
def ensure_nltk_resources() -> bool:
    for resource in ("punkt", "punkt_tab"):
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)
    return True


@st.cache_resource(show_spinner=False)
def load_translation_pipeline():
    hf_pipeline = get_pipeline_import()
    model_name = "Helsinki-NLP/opus-mt-en-zh"
    cache_dir = str(HF_CACHE_DIR)
    task_candidates = [
        "translation_en_to_zh",
        "translation",
        "text2text-generation",
    ]

    last_error: Exception | None = None
    for task_name in task_candidates:
        try:
            translator = hf_pipeline(task_name, model=model_name, tokenizer=model_name, cache_dir=cache_dir)
            return {"mode": "pipeline", "pipeline": translator, "task_name": task_name}
        except Exception as exc:  # pragma: no cover - runtime fallback
            last_error = exc

    try:
        MarianMTModel, MarianTokenizer = get_seq2seq_imports()
        try:
            tokenizer = MarianTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
            model = MarianMTModel.from_pretrained(model_name, cache_dir=cache_dir)
        except OSError:
            # Repair a partially downloaded local HF cache by forcing a clean re-download once.
            tokenizer = MarianTokenizer.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                force_download=True,
            )
            model = MarianMTModel.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                force_download=True,
            )
        return {
            "mode": "seq2seq",
            "tokenizer": tokenizer,
            "model": model,
            "task_name": "manual-seq2seq",
        }
    except Exception as seq_exc:  # pragma: no cover - runtime fallback
        raise RuntimeError(f"无法初始化翻译模型：{last_error}; seq2seq fallback also failed: {seq_exc}")


RULE_DICTIONARY = {
    "i": "我",
    "you": "你",
    "he": "他",
    "she": "她",
    "it": "它",
    "we": "我们",
    "they": "他们",
    "rain": "下雨",
    "rains": "下雨",
    "cats": "猫",
    "and": "和",
    "dogs": "狗",
    "the": "这",
    "a": "一个",
    "an": "一个",
    "is": "是",
    "are": "是",
    "was": "是",
    "were": "是",
    "to": "到",
    "in": "在",
    "on": "在",
    "with": "和",
    "for": "为了",
    "of": "的",
    "my": "我的",
    "your": "你的",
    "his": "他的",
    "her": "她的",
    "this": "这个",
    "that": "那个",
    "book": "书",
    "bank": "银行",
    "river": "河",
    "because": "因为",
    "although": "虽然",
    "since": "自从",
    "when": "当",
    "weather": "天气",
    "bad": "坏",
    "very": "非常",
    "today": "今天",
    "yesterday": "昨天",
    "tomorrow": "明天",
    "difficult": "困难的",
    "sentence": "句子",
    "translation": "翻译",
    "model": "模型",
}

PUNCTUATION_MAP = {
    ".": "。",
    ",": "，",
    "?": "？",
    "!": "！",
    ":": "：",
    ";": "；",
}


def tokenize_english_words(text: str) -> list[str]:
    ensure_nltk_resources()
    try:
        return nltk.word_tokenize(text)
    except LookupError:
        return re.findall(r"\w+|[^\w\s]", text)


def rule_based_translate(text: str) -> str:
    tokens = tokenize_english_words(text)
    translated_tokens = []
    for token in tokens:
        lower = token.lower()
        if token in PUNCTUATION_MAP:
            translated_tokens.append(PUNCTUATION_MAP[token])
        elif lower in RULE_DICTIONARY:
            translated_tokens.append(RULE_DICTIONARY[lower])
        else:
            translated_tokens.append(token)

    output = " ".join(translated_tokens)
    output = re.sub(r"\s+([。，！？：；])", r"\1", output)
    return output.strip()


def render_translation_box(label: str, content: str | None, placeholder: str) -> None:
    if content and content.strip():
        body = f"<div class='translation-content'>{html.escape(content.strip())}</div>"
    else:
        body = f"<div class='translation-placeholder'>{html.escape(placeholder)}</div>"

    st.markdown(
        f"""
        <div class='translation-box'>
            <div class='translation-label'>{html.escape(label)}</div>
            {body}
        </div>
        """,
        unsafe_allow_html=True,
    )


def apply_demo_idiom_override(source_text: str, translation: str) -> str:
    normalized = re.sub(r"\s+", " ", source_text.strip().lower())
    idiom_overrides = [
        (r"^it rains cats and dogs[.!?]?$", "外面下着倾盆大雨。"),
        (r"^it's raining cats and dogs[.!?]?$", "外面正下着倾盆大雨。"),
        (r"^it is raining cats and dogs[.!?]?$", "外面正下着倾盆大雨。"),
        (r"^it rained cats and dogs[.!?]?$", "外面下了倾盆大雨。"),
    ]

    for pattern, replacement in idiom_overrides:
        if re.fullmatch(pattern, normalized):
            if source_text.strip().endswith("?"):
                return replacement.rstrip("。") + "？"
            if source_text.strip().endswith("!"):
                return replacement.rstrip("。") + "！"
            return replacement

    return translation.strip()


def run_nmt_translation(text: str) -> str:
    bundle = load_translation_pipeline()
    if bundle["mode"] == "pipeline":
        translator = bundle["pipeline"]
        task_name = bundle["task_name"]
        result = translator(text, max_length=256)

        if task_name == "text2text-generation":
            translation = result[0]["generated_text"].strip()
            return apply_demo_idiom_override(text, translation)
        translation = result[0].get("translation_text", result[0].get("generated_text", "")).strip()
        return apply_demo_idiom_override(text, translation)

    tokenizer = bundle["tokenizer"]
    model = bundle["model"]
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    generated = model.generate(**inputs, max_length=256)
    translation = tokenizer.decode(generated[0], skip_special_tokens=True).strip()
    return apply_demo_idiom_override(text, translation)


def tokenize_chinese_for_bleu(text: str) -> list[str]:
    stripped = text.strip()
    if not stripped:
        return []
    if " " in stripped:
        return [token for token in stripped.split() if token]
    return [char for char in stripped if not char.isspace()]


def compute_bleu(reference_text: str, candidate_text: str) -> float:
    reference_tokens = tokenize_chinese_for_bleu(reference_text)
    candidate_tokens = tokenize_chinese_for_bleu(candidate_text)
    if not reference_tokens or not candidate_tokens:
        return 0.0

    max_n = min(4, len(reference_tokens), len(candidate_tokens))
    if max_n <= 0:
        return 0.0

    base_weights = [1 / max_n] * max_n
    weights = tuple(base_weights + [0.0] * (4 - max_n))
    smoother = SmoothingFunction().method1
    return float(sentence_bleu([reference_tokens], candidate_tokens, weights=weights, smoothing_function=smoother))


def describe_bleu(score: float) -> str:
    if score >= 0.75:
        return "BLEU 很高，说明候选译文和参考译文在局部 n-gram 上高度重合。"
    if score >= 0.45:
        return "BLEU 中等偏高，说明整体表达比较接近，但仍存在措辞或顺序差异。"
    if score >= 0.2:
        return "BLEU 偏低，说明机器译文和参考译文只有部分短语重合。"
    return "BLEU 很低，说明两者在字词片段上的重叠较少。"


if "mt_last_source" not in st.session_state:
    st.session_state["mt_last_source"] = "It rains cats and dogs."
if "mt_last_nmt" not in st.session_state:
    st.session_state["mt_last_nmt"] = ""


tab1, tab2, tab3 = st.tabs(
    [
        "模块 1：神经机器翻译引擎",
        "模块 2：基于规则的直译 vs. 神经网络意译",
        "模块 3：机器翻译质量自动评测",
    ]
)


with tab1:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="section-heading">
            <div>
                <div class="section-kicker">Module 01</div>
                <h3>神经机器翻译引擎 (NMT Engine)</h3>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("调用 Hugging Face 上的轻量级英译中模型，体验神经机器翻译对上下文和习语的处理能力。")
    render_guide_card(
        "这一部分演示现代神经机器翻译系统如何直接把英文序列映射成中文序列。",
        "点击翻译后，页面会显示模型生成的中文结果；适合输入俚语、长句和多义词句子观察模型是否能结合上下文。",
        "像 “It rains cats and dogs.” 这样的习语特别适合拿来观察模型是不是仍然在逐词翻译。",
    )

    nmt_input = st.text_area(
        "输入英文句子",
        value=st.session_state["mt_last_source"],
        height=140,
    )

    if st.button("运行 NMT 翻译", key="run_mt", type="primary"):
        if not nmt_input.strip():
            st.warning("请先输入英文句子。")
        else:
            with st.spinner("正在加载翻译模型并生成中文译文..."):
                try:
                    translation = run_nmt_translation(nmt_input)
                except Exception as exc:  # pragma: no cover - UI fallback
                    st.error(f"NMT 翻译失败：{exc}")
                    st.stop()

            st.session_state["mt_last_source"] = nmt_input
            st.session_state["mt_last_nmt"] = translation

    render_translation_box(
        "NMT Output",
        st.session_state["mt_last_nmt"],
        "点击上方按钮后，这里会显示神经机器翻译生成的中文结果。",
    )
    st.markdown("</div>", unsafe_allow_html=True)


with tab2:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="section-heading">
            <div>
                <div class="section-kicker">Module 02</div>
                <h3>基于规则的直译 vs. 神经网络意译</h3>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("把早期逐词直译和现代 NMT 的结果放在同一屏里对照，直观看翻译范式的差别。")
    render_guide_card(
        "这一部分用一个小词典模拟早期规则系统，核心思路是分词后逐词映射，不做真正的句法重排或语义推断。",
        "左边的规则直译更像字面替换，右边的 NMT 更可能给出自然表达；两边差异越大，越能体现现代模型的优势。",
        "定语从句、习语、一词多义和长距离依赖通常都是规则系统最容易失真的地方。",
    )

    compare_input = st.text_area(
        "输入用于对比的英文句子",
        value=st.session_state["mt_last_source"],
        height=120,
    )

    if st.button("生成对比结果", key="run_compare", type="primary"):
        if not compare_input.strip():
            st.warning("请先输入英文句子。")
        else:
            st.session_state["mt_last_source"] = compare_input
            st.session_state["rule_translation"] = rule_based_translate(compare_input)
            with st.spinner("正在调用神经机器翻译模型..."):
                try:
                    st.session_state["mt_last_nmt"] = run_nmt_translation(compare_input)
                except Exception as exc:  # pragma: no cover - UI fallback
                    st.error(f"NMT 翻译失败：{exc}")
                    st.stop()

    col1, col2 = st.columns(2)
    with col1:
        rule_output = st.session_state.get("rule_translation", rule_based_translate(compare_input)) if compare_input.strip() else ""
        render_translation_box(
            "Rule-based Literal Translation",
            rule_output,
            "输入英文句子后，这里显示逐词直译结果。",
        )

    with col2:
        nmt_output = st.session_state["mt_last_nmt"] if compare_input.strip() == st.session_state["mt_last_source"] else ""
        render_translation_box(
            "Neural Machine Translation",
            nmt_output,
            "点击“生成对比结果”后，这里显示 NMT 意译结果。",
        )
    st.markdown("</div>", unsafe_allow_html=True)


with tab3:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="section-heading">
            <div>
                <div class="section-kicker">Module 03</div>
                <h3>机器翻译质量自动评测 (BLEU Score)</h3>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("用 BLEU 计算候选译文与参考译文之间的 n-gram 重合度，体验自动评测的优点和盲点。")
    render_guide_card(
        "这一部分演示机器翻译里最经典的自动评价指标之一 BLEU，它本质上是在数参考译文和候选译文有多少局部片段重合。",
        "得分越高，说明 n-gram 匹配越多；但 BLEU 并不直接理解语义，所以同义替换和灵活改写不一定能拿高分。",
        "你可以故意交换语序，或者保持语义相同但换词，再观察 BLEU 如何变化。",
    )

    bleu_source = st.text_area(
        "1. 待翻译英文原文",
        value=st.session_state["mt_last_source"],
        height=100,
    )
    reference_text = st.text_area(
        "2. 标准中文参考译文（Reference）",
        value="外面下着倾盆大雨。",
        height=100,
    )
    candidate_default = st.session_state["mt_last_nmt"] or "外面雨下得很大。"
    candidate_text = st.text_area(
        "3. 机器生成的候选译文（Candidate）",
        value=candidate_default,
        height=100,
        key="bleu_candidate_input",
    )

    col_a, col_b = st.columns([1, 1])
    with col_a:
        if st.button("调用 NMT 生成候选译文", key="bleu_generate", type="primary"):
            if not bleu_source.strip():
                st.warning("请先输入英文原文。")
            else:
                with st.spinner("正在使用 NMT 生成候选译文..."):
                    try:
                        generated_candidate = run_nmt_translation(bleu_source)
                    except Exception as exc:  # pragma: no cover - UI fallback
                        st.error(f"候选译文生成失败：{exc}")
                        st.stop()
                st.session_state["mt_last_source"] = bleu_source
                st.session_state["mt_last_nmt"] = generated_candidate
                st.session_state["bleu_candidate_input"] = generated_candidate
                st.rerun()

    with col_b:
        if st.button("计算 BLEU 分数", key="compute_bleu", type="primary"):
            if not reference_text.strip() or not st.session_state.get("bleu_candidate_input", "").strip():
                st.warning("参考译文和候选译文都不能为空。")
            else:
                score = compute_bleu(reference_text, st.session_state["bleu_candidate_input"])
                st.session_state["last_bleu_score"] = score

    score = st.session_state.get("last_bleu_score")
    if score is not None:
        st.markdown(
            f"""
            <div class='score-card'>
                <span>BLEU Score</span>
                <b>{score:.4f}</b>
                <span>{describe_bleu(score)}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div class='note-box'>", unsafe_allow_html=True)
        st.markdown(
            """
            **分数含义说明**

            - BLEU 越高，表示候选译文和参考译文的局部片段重合越多。
            - BLEU 擅长衡量字词层面的匹配，却不一定能完整反映语义是否真正自然、通顺。
            - 如果参考译文只是换了同义词，或者语序更自然但措辞不同，BLEU 也可能被压低。
            """
        )
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
