import importlib.util
import math
import random
import re
from collections import Counter
from contextlib import contextmanager
from typing import Dict, List, Sequence, Tuple

import nltk
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from nltk.corpus import reuters
from ui_theme import inject_iekg_theme, render_guide_card, render_hero


# Some local Python environments ship an incompatible torchvision build.
# This app only uses text pipelines, so we hide torchvision from transformers.
_original_find_spec = importlib.util.find_spec


def _safe_find_spec(name, *args, **kwargs):
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


def get_gpt2_resources_import():
    with suppress_torchvision_for_transformers():
        from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    return GPT2LMHeadModel, GPT2TokenizerFast


st.set_page_config(page_title="语言模型训练与对比分析平台", layout="wide")
inject_iekg_theme()


@st.cache_resource(show_spinner=False)
def ensure_nltk_resources() -> bool:
    resources = [
        ("corpora/reuters", "reuters"),
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
    ]
    for path, name in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name, quiet=True)
    return True


@st.cache_data(show_spinner=False)
def get_default_reuters_text() -> str:
    ensure_nltk_resources()
    words = reuters.words()
    sample = list(words[:5000])
    return " ".join(sample)


def tokenize_words(text: str) -> List[str]:
    ensure_nltk_resources()
    try:
        tokens = nltk.word_tokenize(text)
    except LookupError:
        tokens = re.findall(r"\b\w+\b|[^\w\s]", text)
    return [token.lower() for token in tokens if token.strip()]


def build_trigram_model(tokens: Sequence[str]) -> Dict[str, object]:
    padded = ["<s>", "<s>"] + list(tokens) + ["</s>"]
    trigram_counts: Counter = Counter()
    bigram_counts: Counter = Counter()
    unigram_counts: Counter = Counter(tokens)
    vocab = sorted(set(tokens))

    for i in range(2, len(padded)):
        context = (padded[i - 2], padded[i - 1])
        word = padded[i]
        trigram_counts[(context[0], context[1], word)] += 1
        bigram_counts[context] += 1

    return {
        "trigram_counts": trigram_counts,
        "bigram_counts": bigram_counts,
        "unigram_counts": unigram_counts,
        "vocab": vocab,
        "vocab_size": len(vocab) + 1,
    }


def trigram_sentence_probability(
    sentence: str,
    model: Dict[str, object],
    smoothing: bool = False,
) -> Tuple[float, List[Dict[str, float]]]:
    tokens = tokenize_words(sentence)
    if not tokens:
        return 0.0, []

    padded = ["<s>", "<s>"] + tokens + ["</s>"]
    trigram_counts: Counter = model["trigram_counts"]  # type: ignore[assignment]
    bigram_counts: Counter = model["bigram_counts"]  # type: ignore[assignment]
    vocab_size: int = model["vocab_size"]  # type: ignore[assignment]

    details = []
    joint_prob = 1.0
    for i in range(2, len(padded)):
        context = (padded[i - 2], padded[i - 1])
        word = padded[i]
        trigram_count = trigram_counts[(context[0], context[1], word)]
        bigram_count = bigram_counts[context]
        if smoothing:
            prob = (trigram_count + 1) / (bigram_count + vocab_size)
        else:
            prob = trigram_count / bigram_count if bigram_count else 0.0
        joint_prob *= prob
        details.append(
            {
                "context": f"({context[0]}, {context[1]}) -> {word}",
                "count": trigram_count,
                "context_count": bigram_count,
                "probability": prob,
            }
        )
    return joint_prob, details


class CharRNN(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x: torch.Tensor, hidden=None):
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded, hidden)
        logits = self.fc(output)
        return logits, hidden


def prepare_char_data(text: str):
    if len(text) < 2:
        raise ValueError("训练语料至少需要 2 个字符。")
    chars = sorted(set(text))
    stoi = {ch: idx for idx, ch in enumerate(chars)}
    itos = {idx: ch for ch, idx in stoi.items()}
    encoded = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)
    inputs = encoded[:-1].unsqueeze(0)
    targets = encoded[1:].unsqueeze(0)
    return chars, stoi, itos, inputs, targets


def train_char_rnn(
    corpus_text: str,
    hidden_size: int,
    epochs: int,
    learning_rate: float,
    progress_bar,
    chart_placeholder,
    status_placeholder,
):
    chars, stoi, itos, inputs, targets = prepare_char_data(corpus_text)
    model = CharRNN(vocab_size=len(chars), hidden_size=hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    losses: List[float] = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits, _ = model(inputs)
        loss = F.cross_entropy(logits.reshape(-1, len(chars)), targets.reshape(-1))
        loss.backward()
        optimizer.step()

        loss_value = float(loss.item())
        losses.append(loss_value)
        chart_placeholder.line_chart(pd.DataFrame({"loss": losses}))
        progress_bar.progress((epoch + 1) / epochs)
        status_placeholder.write(
            f"Epoch {epoch + 1}/{epochs} | Loss: {loss_value:.4f}"
        )

    return model, stoi, itos, losses


def generate_text(
    model: CharRNN,
    stoi: Dict[str, int],
    itos: Dict[int, str],
    seed: str,
    length: int = 50,
) -> str:
    if not seed:
        seed = random.choice(list(stoi.keys()))
    clean_seed = "".join(ch for ch in seed if ch in stoi)
    if not clean_seed:
        clean_seed = random.choice(list(stoi.keys()))

    model.eval()
    generated = clean_seed
    input_ids = torch.tensor([[stoi[ch] for ch in clean_seed]], dtype=torch.long)

    with torch.no_grad():
        _, hidden = model(input_ids)
        current_char = clean_seed[-1]
        for _ in range(length):
            current_input = torch.tensor([[stoi[current_char]]], dtype=torch.long)
            logits, hidden = model(current_input, hidden)
            probs = F.softmax(logits[0, -1], dim=-1)
            next_id = torch.multinomial(probs, num_samples=1).item()
            current_char = itos[next_id]
            generated += current_char
    return generated


@st.cache_resource(show_spinner=False)
def load_masked_lm_pipeline():
    hf_pipeline = get_pipeline_import()
    return hf_pipeline("fill-mask", model="bert-base-uncased")


@st.cache_resource(show_spinner=False)
def load_text_generation_pipeline():
    hf_pipeline = get_pipeline_import()
    return hf_pipeline("text-generation", model="gpt2")


@st.cache_resource(show_spinner=False)
def load_gpt2_ppl_resources():
    GPT2LMHeadModel, GPT2TokenizerFast = get_gpt2_resources_import()
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()
    return tokenizer, model


def compute_perplexity(sentences: Sequence[str]) -> pd.DataFrame:
    tokenizer, model = load_gpt2_ppl_resources()
    rows = []
    for sentence in sentences:
        stripped = sentence.strip()
        if not stripped:
            continue
        encodings = tokenizer(stripped, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**encodings, labels=encodings["input_ids"])
        loss = float(outputs.loss.item())
        ppl = math.exp(loss)
        rows.append(
            {
                "Sentence": stripped,
                "Cross-Entropy Loss": round(loss, 4),
                "Perplexity (PPL)": round(ppl, 4),
            }
        )
    return pd.DataFrame(rows)


render_hero(
    "语言模型训练与对比分析平台",
    "Language Modeling Lab",
    "沿用信息抽取系统的界面语言，把 n-gram、RNN、BERT/GPT-2 和困惑度评价放进统一的语言模型实验工作台。",
    [
        ("Step 01", "n-gram"),
        ("Step 02", "Char RNN"),
        ("Step 03", "BERT vs GPT-2"),
        ("Step 04", "Perplexity"),
    ],
)

ensure_nltk_resources()

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "模块 1：n-gram & Smoothing",
        "模块 2：Train your own RNN-LM",
        "模块 3：Masked LM vs. Causal LM",
        "模块 4：Perplexity",
    ]
)


with tab1:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="section-heading">
            <div>
                <div class="section-kicker">Module 01</div>
                <h3>n 元语言模型与数据平滑</h3>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("从最经典的 trigram 出发，看未平滑和加一平滑下句子概率如何变化。")
    render_guide_card(
        "这一部分演示基于词序列统计的传统语言模型，核心是 trigram 条件概率和零概率问题。",
        "表格里每一行对应一个 trigram，上方显示联合概率；如果出现零概率，再对比平滑前后结果。",
        "特别适合拿来讲为什么语料稀疏会让传统语言模型很脆弱。",
    )
    corpus_input = st.text_area(
        "基础英文语料",
        value=get_default_reuters_text(),
        height=220,
        help="默认使用 NLTK Reuters 语料的前 5000 个 token。",
    )
    sentence_input = st.text_input(
        "输入待计算概率的句子",
        value="The market is moving higher today .",
    )
    use_smoothing = st.checkbox("开启加一平滑（Laplace Smoothing）", value=False)

    tokens = tokenize_words(corpus_input)
    if len(tokens) < 3:
        st.warning("语料过短，至少需要 3 个 token 才能构建 Trigram 模型。")
    else:
        trigram_model = build_trigram_model(tokens)
        raw_prob, raw_details = trigram_sentence_probability(
            sentence_input, trigram_model, smoothing=False
        )
        smooth_prob, smooth_details = trigram_sentence_probability(
            sentence_input, trigram_model, smoothing=True
        )

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Token 数", len(tokens))
        col_b.metric("词汇表大小", len(trigram_model["vocab"]))  # type: ignore[arg-type]
        col_c.metric("唯一词频项", len(trigram_model["unigram_counts"]))  # type: ignore[arg-type]

        display_prob = smooth_prob if use_smoothing else raw_prob
        st.markdown(f"**当前设置下的联合概率：** `{display_prob:.12e}`")

        zero_events = [item for item in raw_details if item["probability"] == 0.0]
        if zero_events:
            st.warning("检测到未出现过的 Trigram，未平滑时会出现零概率。")
            compare_df = pd.DataFrame(
                {
                    "Trigram": [item["context"] for item in raw_details],
                    "Raw Probability": [item["probability"] for item in raw_details],
                    "Smoothed Probability": [
                        item["probability"] for item in smooth_details
                    ],
                }
            )
            st.dataframe(compare_df, use_container_width=True)
            st.markdown(
                f"未平滑联合概率：`{raw_prob:.12e}`  \n平滑后联合概率：`{smooth_prob:.12e}`"
            )
        else:
            st.success("当前句子的所有 Trigram 都在语料中出现过。")
            details_df = pd.DataFrame(raw_details if not use_smoothing else smooth_details)
            st.dataframe(details_df, use_container_width=True)

        top_words = pd.DataFrame(
            trigram_model["unigram_counts"].most_common(15),  # type: ignore[union-attr]
            columns=["Word", "Frequency"],
        )
        st.markdown("**高频词 Top-15**")
        st.dataframe(top_words, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


with tab2:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="section-heading">
            <div>
                <div class="section-kicker">Module 02</div>
                <h3>从零训练 RNN 语言模型</h3>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("直接在一个很小的字符级语料上训练 LSTM，观察 loss 下降和生成效果。")
    render_guide_card(
        "这一部分演示神经语言模型如何从字符序列中学习模式，并在训练后继续续写文本。",
        "先根据 loss 曲线判断训练是否收敛，再看 Seed 触发的生成结果是否保持了原语料的风格。",
        "字符级模型很轻，适合课堂上快速训练和现场展示。",
    )
    default_poem = (
        "Two roads diverged in a yellow wood,\n"
        "And sorry I could not travel both,\n"
        "And be one traveler, long I stood."
    )
    rnn_corpus = st.text_area("输入短语料", value=default_poem, height=180)
    hidden_size = st.slider("Hidden Size", min_value=16, max_value=128, value=64, step=8)
    epochs = st.slider("Epochs", min_value=10, max_value=200, value=60, step=10)
    learning_rate = st.slider(
        "Learning Rate", min_value=0.001, max_value=0.1, value=0.01, step=0.001
    )

    if "rnn_bundle" not in st.session_state:
        st.session_state.rnn_bundle = None

    if st.button("开始训练", type="primary"):
        if len(rnn_corpus) < 2:
            st.error("请输入至少两个字符的训练语料。")
        else:
            progress_bar = st.progress(0.0)
            chart_placeholder = st.empty()
            status_placeholder = st.empty()
            try:
                model, stoi, itos, losses = train_char_rnn(
                    rnn_corpus,
                    hidden_size,
                    epochs,
                    learning_rate,
                    progress_bar,
                    chart_placeholder,
                    status_placeholder,
                )
                st.session_state.rnn_bundle = {
                    "model": model,
                    "stoi": stoi,
                    "itos": itos,
                    "losses": losses,
                }
                st.success("训练完成，可以开始生成文本。")
            except ValueError as exc:
                st.error(str(exc))

    seed_text = st.text_input("起始字符（Seed）", value="T")
    if st.session_state.rnn_bundle is not None:
        if st.button("生成 50 字符文本"):
            bundle = st.session_state.rnn_bundle
            generated = generate_text(
                bundle["model"],
                bundle["stoi"],
                bundle["itos"],
                seed_text,
                length=50,
            )
            st.text_area("模型生成结果", value=generated, height=160)
    else:
        st.info("先完成一次训练，再根据 Seed 生成文本。")
    st.markdown("</div>", unsafe_allow_html=True)


with tab3:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="section-heading">
            <div>
                <div class="section-kicker">Module 03</div>
                <h3>预训练架构对比：BERT vs. GPT-2</h3>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("同一页对照 Masked LM 和 Causal LM 的推断方式。")
    render_guide_card(
        "这一部分对比双向掩码建模和自回归生成两种预训练范式在实际交互上的差异。",
        "左侧是填空式预测，右侧是续写式生成；一个补全局部空位，一个顺着前缀向后写。",
        "讲解时很适合顺手引出 encoder-only 和 decoder-only 的结构区别。",
    )
    left, right = st.columns(2)

    with left:
        st.markdown("### BERT: Masked Language Modeling")
        bert_input = st.text_input(
            "输入带 [MASK] 的句子",
            value="The man went to the [MASK] to buy some milk.",
        )
        if st.button("运行 BERT Top-5 预测"):
            if "[MASK]" not in bert_input:
                st.error("请输入包含 [MASK] 标记的句子。")
            else:
                with st.spinner("加载并运行 bert-base-uncased..."):
                    mask_filler = load_masked_lm_pipeline()
                    results = mask_filler(bert_input, top_k=5)
                bert_df = pd.DataFrame(
                    [
                        {
                            "Token": item["token_str"].strip(),
                            "Probability": round(float(item["score"]), 6),
                            "Completed Sentence": item["sequence"],
                        }
                        for item in results
                    ]
                )
                st.dataframe(bert_df, use_container_width=True)

    with right:
        st.markdown("### GPT-2: Causal Language Modeling")
        gpt_prompt = st.text_area(
            "输入前缀 Prompt",
            value="Once upon a time in a small town",
            height=120,
        )
        if st.button("运行 GPT-2 续写"):
            with st.spinner("加载并运行 gpt2..."):
                generator = load_text_generation_pipeline()
                generated = generator(
                    gpt_prompt,
                    max_new_tokens=20,
                    do_sample=True,
                    temperature=0.9,
                    top_k=50,
                    pad_token_id=50256,
                )
            st.text_area(
                "GPT-2 生成结果",
                value=generated[0]["generated_text"],
                height=220,
            )
    st.markdown("</div>", unsafe_allow_html=True)


with tab4:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="section-heading">
            <div>
                <div class="section-kicker">Module 04</div>
                <h3>语言模型评价：GPT-2 困惑度计算</h3>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("用 GPT-2 计算多句文本的交叉熵和困惑度，感受模型“惊讶程度”。")
    render_guide_card(
        "这一部分演示语言模型常用的自动评价指标 perplexity，以及它与交叉熵之间的关系。",
        "输出表格会同时列出每句文本的 Loss 和 PPL，数值越低通常表示模型越熟悉这类表达。",
        "可以把自然句和故意怪异的句子混在一起输入，对比会很明显。",
    )
    ppl_text = st.text_area(
        "输入多段测试句子（每行一句）",
        value=(
            "The stock market opened higher today.\n"
            "Colorless green ideas sleep furiously.\n"
            "This is a simple example for perplexity calculation."
        ),
        height=180,
    )
    if st.button("计算 PPL"):
        sentences = [line for line in ppl_text.splitlines() if line.strip()]
        if not sentences:
            st.error("请至少输入一句测试文本。")
        else:
            with st.spinner("使用 GPT-2 计算交叉熵与困惑度..."):
                ppl_df = compute_perplexity(sentences)
            if ppl_df.empty:
                st.warning("没有可计算的有效句子。")
            else:
                st.dataframe(ppl_df, use_container_width=True)
                st.caption("PPL = exp(Cross-Entropy Loss)，数值越低通常表示模型越不“惊讶”。")
    st.markdown("</div>", unsafe_allow_html=True)
