from __future__ import annotations

# 这个页面聚焦两类经典句法分析任务：
# 1. 依存句法分析：展示词与词之间的支配/修饰关系。
# 2. 成分句法分析：展示句子被划分为哪些短语，以及短语如何递归嵌套。
#
# 代码设计目标：
# - 页面可以直接放进现有的 Streamlit 合集里运行；
# - 如果本地还没有安装 spaCy / benepar / svgling 等库，尽量自动补齐；
# - 运行时避免在 Streamlit Cloud 上临时下载大型模型，优先通过 requirements 预装；
# - 即使个别依赖失败，页面也尽量给出可理解的提示，而不是直接崩溃。

import html
import importlib
import os
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import streamlit as st
import streamlit.components.v1 as components

from deploy_utils import ensure_named_cache_dir, get_nltk_data_dir
from ui_theme import inject_iekg_theme, render_guide_card, render_hero


st.set_page_config(
    page_title="句法分析平台",
    page_icon="🪢",
    layout="wide",
    initial_sidebar_state="expanded",
)


inject_iekg_theme(
    """
    .syntax-card {
        background: linear-gradient(180deg, rgba(255,255,255,0.97), rgba(248,251,255,0.92));
        border-radius: 22px;
        padding: 1rem 1.05rem;
        border: 1px solid rgba(15, 23, 42, 0.08);
        box-shadow: 0 14px 30px rgba(15, 23, 42, 0.06);
        margin-bottom: 1rem;
    }
    .syntax-card h4 {
        margin: 0 0 0.8rem;
        color: #0f172a;
        font-size: 1.08rem;
    }
    .syntax-caption {
        color: #475569;
        line-height: 1.7;
        margin-bottom: 0.8rem;
    }
    .tree-shell {
        background: rgba(255,255,255,0.88);
        border-radius: 18px;
        border: 1px solid rgba(148, 163, 184, 0.18);
        padding: 0.8rem;
        overflow-x: auto;
    }
    .tree-text {
        margin: 0;
        color: #1e293b;
        font-size: 0.96rem;
        line-height: 1.65;
        white-space: pre-wrap;
        font-family: 'IBM Plex Mono', 'Consolas', monospace;
    }
    .const-tree-shell {
        background: linear-gradient(180deg, rgba(239,252,252,0.92), rgba(248,251,255,0.96));
        border-radius: 26px;
        border: 1px solid rgba(125, 211, 252, 0.38);
        padding: 1.35rem;
        overflow-x: auto;
    }
    .const-tree {
        min-width: max-content;
        color: #0f172a;
        font-family: 'IBM Plex Sans', 'Microsoft YaHei', sans-serif;
        font-size: 0.95rem;
    }
    .const-card-node {
        position: relative;
        display: inline-flex;
        flex-direction: column;
        gap: 0.9rem;
        padding: 1.7rem 1rem 1rem;
        border-radius: 1.6rem;
        border: 1px solid rgba(125, 211, 252, 0.45);
        background: linear-gradient(180deg, rgba(255,255,255,0.92), rgba(240,249,255,0.82));
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.7);
    }
    .const-card-node.root {
        background: linear-gradient(180deg, rgba(248,253,253,0.98), rgba(239,252,252,0.9));
    }
    .const-card-children {
        display: flex;
        align-items: stretch;
        gap: 0.9rem;
    }
    .const-card-label {
        position: absolute;
        top: 0.7rem;
        left: 0.8rem;
        display: inline-block;
        padding: 0.18rem 0.56rem;
        border-radius: 999px;
        background: rgba(207, 250, 254, 0.95);
        color: #0b7ea4;
        font-size: 0.82rem;
        font-weight: 800;
        letter-spacing: 0.02em;
    }
    .const-terminal {
        min-width: 90px;
        padding: 0.7rem 0.8rem;
        border-radius: 1.2rem;
        border: 1px solid rgba(125, 211, 252, 0.42);
        background: rgba(255,255,255,0.95);
        display: inline-flex;
        flex-direction: column;
        align-items: flex-start;
        gap: 0.48rem;
    }
    .const-terminal-pos {
        display: inline-block;
        padding: 0.16rem 0.54rem;
        border-radius: 999px;
        background: rgba(207, 250, 254, 0.95);
        color: #0b7ea4;
        font-size: 0.8rem;
        font-weight: 800;
    }
    .const-terminal-word {
        display: inline-block;
        padding: 0.34rem 0.66rem;
        border-radius: 999px;
        background: rgba(236, 253, 255, 0.98);
        color: #0f4c5c;
        font-size: 0.98rem;
        font-weight: 700;
    }
    .analysis-box {
        border-radius: 18px;
        padding: 0.95rem 1rem;
        background: linear-gradient(145deg, rgba(15, 118, 110, 0.08), rgba(37, 99, 235, 0.08));
        border: 1px solid rgba(37, 99, 235, 0.12);
        color: #334155;
        line-height: 1.75;
    }
    .phrase-chip {
        display: inline-block;
        padding: 0.18rem 0.48rem;
        border-radius: 999px;
        color: white;
        font-size: 0.8rem;
        font-weight: 700;
        white-space: nowrap;
    }
    .dep-svg-shell {
        background: white;
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 18px;
        padding: 0.65rem;
        overflow-x: auto;
    }
    .dependency-hint {
        color: #475569;
        font-size: 0.92rem;
        line-height: 1.7;
        margin-top: 0.75rem;
    }
    """
)


render_hero(
    "句法分析平台",
    "Syntax Parsing Studio",
    "把依存句法和成分句法放进同一个教学工作台里：上半部分看词与词的依赖结构，下半部分看短语如何递归嵌套成整句。",
    [
        ("Step 01", "Dependency Parse"),
        ("Step 02", "Constituency Parse"),
        ("Step 03", "Phrase Explanation"),
    ],
)


render_guide_card(
    "输入一句英文句子后，系统会同时生成依存句法图和成分句法树，适合课堂演示两种句法理论的差异。",
    "依存句法强调“核心词 + 依存弧”，成分句法强调“短语 + 层级嵌套”；同一句话可以从两种视角解读结构。",
    "推荐优先尝试带介词短语歧义的句子，例如 I saw the man with a telescope.，这样更容易看到结构解释差异。",
)


# 常见依存关系标签解释表。
# spaCy 在不同模型版本里可能混用 Universal Dependencies 与旧标签，
# 所以这里同时兼容如 obj / dobj、pobj、prep 等常见写法。
DEP_LABEL_EXPLANATIONS = {
    "ROOT": "句子根节点，通常是谓语核心。",
    "root": "句子根节点，通常是谓语核心。",
    "nsubj": "名词性主语，表示动作或状态的发出者。",
    "nsubjpass": "被动主语，表示被动句中的承受者。",
    "csubj": "从句主语，由一个从句充当主语。",
    "obj": "宾语，表示动作直接作用的对象。",
    "dobj": "直接宾语，表示动作直接作用的对象。",
    "iobj": "间接宾语，表示动作受益者或接收者。",
    "pobj": "介词宾语，表示介词后面的核心对象。",
    "prep": "介词修饰语，表示由介词引出的修饰结构。",
    "nmod": "名词性修饰语，常用于介词结构或名词补充说明。",
    "amod": "形容词修饰语，描述名词性质。",
    "advmod": "副词修饰语，修饰动词、形容词或整个句子。",
    "aux": "助动词，辅助表达时态、语态或语气。",
    "auxpass": "被动助动词，帮助构成被动语态。",
    "det": "限定词，如冠词、指示词、数量限定词等。",
    "compound": "复合成分，常用于名词复合结构。",
    "compound:prt": "短语动词小品词，如 give up 中的 up。",
    "poss": "所属关系，表示所有者。",
    "case": "格标记，常表现为介词或格助词。",
    "cc": "并列连接词，如 and、or。",
    "conj": "并列成分，被连接的词或短语。",
    "attr": "表语，常见于系动词结构后。",
    "acomp": "形容词补语，常见于系动词后的形容词。",
    "xcomp": "开放补语，其主语通常受上层谓词控制。",
    "ccomp": "从句补语，作为上层谓词的内容补足。",
    "relcl": "关系从句修饰语，用于修饰先行词。",
    "acl": "名词附属从句，作为名词的从句修饰语。",
    "mark": "从属连词标记，如 that、because。",
    "punct": "标点符号。",
}


# 成分短语类型颜色/图标映射，用于短语列表展示。
PHRASE_STYLE = {
    "NP": {"color": "#15803d", "icon": "■"},
    "VP": {"color": "#2563eb", "icon": "■"},
    "PP": {"color": "#ea580c", "icon": "■"},
    "ADJP": {"color": "#7c3aed", "icon": "■"},
    "ADVP": {"color": "#0891b2", "icon": "■"},
    "S": {"color": "#475569", "icon": "■"},
    "SBAR": {"color": "#b91c1c", "icon": "■"},
}


def normalize_phrase_label(label: str) -> str:
    """把成分标签归一化为基础短语类型。

    例如：
    - NP-SBJ -> NP
    - VP-TPC -> VP
    - WHNP -> WHNP（保持原样）
    """

    return label.split("-")[0].split("=")[0].strip()


def ensure_python_package(package_name: str, import_name: str | None = None) -> Any | None:
    """确保某个 Python 包可导入。

    - 先尝试正常导入；
    - 如果失败，则通过 pip 自动安装；
    - 安装成功后再次导入并返回模块对象；
    - 任一步失败都返回 None，并在页面上给出提醒。
    """

    module_name = import_name or package_name
    try:
        return importlib.import_module(module_name)
    except ImportError:
        st.info(f"正在安装依赖 `{package_name}`，首次运行可能需要几十秒。")
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", package_name],
                check=True,
                capture_output=True,
                text=True,
            )
            return importlib.import_module(module_name)
        except Exception as exc:  # noqa: BLE001
            st.error(f"自动安装 `{package_name}` 失败：{exc}")
            return None


# 先准备基础依赖。这里使用“按需安装”的方式，避免让用户手动补环境。
spacy = ensure_python_package("spacy")
nltk = ensure_python_package("nltk")
benepar = ensure_python_package("benepar")
svgling = ensure_python_package("svgling")
pd = ensure_python_package("pandas")


SPACY_MODEL_NAME = "en_core_web_sm"
SPACY_MODEL_INSTALL_HINT = (
    "请把 spaCy 英文模型作为依赖预装，例如在 requirements.txt 中加入：\n"
    "en-core-web-sm @ "
    "https://github.com/explosion/spacy-models/releases/download/"
    "en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"
)
BENEPAR_MODEL_NAME = "benepar_en3"
BENEPAR_MODEL_URL = "https://github.com/nikitakit/self-attentive-parser/releases/download/models/benepar_en3.zip"
HF_CACHE_DIR = ensure_named_cache_dir("hf_cache_syntax")
NLTK_DATA_DIR = get_nltk_data_dir()
os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_CACHE_DIR))
os.environ.setdefault("NLTK_DATA", str(NLTK_DATA_DIR))


def ensure_spacy_model() -> bool:
    """确保 spaCy 英文小模型可用。"""

    if spacy is None:
        return False
    try:
        spacy.load(SPACY_MODEL_NAME)
        return True
    except Exception:
        st.error(
            "当前环境未检测到 spaCy 英文模型 `en_core_web_sm`，"
            "而 Streamlit Cloud 运行时通常不适合再临时下载模型。\n\n"
            f"{SPACY_MODEL_INSTALL_HINT}"
        )
        return False


def ensure_benepar_model() -> bool:
    """确保 Berkeley Neural Parser 英文模型可用。"""

    if benepar is None:
        return False
    if nltk is not None and str(NLTK_DATA_DIR) not in nltk.data.path:
        nltk.data.path.append(str(NLTK_DATA_DIR))
    try:
        benepar.Parser(BENEPAR_MODEL_NAME)
        return True
    except Exception:
        st.info("未检测到 `benepar_en3`，正在下载英文成分句法模型。首次运行可能需要几十秒。")
        try:
            download_ok = benepar.download(
                BENEPAR_MODEL_NAME,
                download_dir=str(NLTK_DATA_DIR),
                quiet=True,
                raise_on_error=True,
            )
            if not download_ok:
                raise RuntimeError("benepar 官方下载器未成功返回 benepar_en3 下载结果。")
            if str(NLTK_DATA_DIR) not in nltk.data.path:
                nltk.data.path.append(str(NLTK_DATA_DIR))
            benepar.Parser(BENEPAR_MODEL_NAME)
            return True
        except Exception as exc:  # noqa: BLE001
            try:
                models_dir = Path(NLTK_DATA_DIR) / "models"
                target_dir = models_dir / BENEPAR_MODEL_NAME
                archive_path = HF_CACHE_DIR / f"{BENEPAR_MODEL_NAME}.zip"
                models_dir.mkdir(parents=True, exist_ok=True)

                urllib.request.urlretrieve(BENEPAR_MODEL_URL, archive_path)
                with zipfile.ZipFile(archive_path) as zip_file:
                    zip_file.extractall(models_dir)

                if str(NLTK_DATA_DIR) not in nltk.data.path:
                    nltk.data.path.append(str(NLTK_DATA_DIR))
                if not target_dir.exists():
                    raise RuntimeError(f"已下载压缩包，但未找到目标目录：{target_dir}")
                benepar.Parser(BENEPAR_MODEL_NAME)
                return True
            except Exception as direct_exc:  # noqa: BLE001
                combined_error = (
                    f"benepar 官方下载器失败：{exc}\n"
                    f"官方模型直链下载失败：{direct_exc}"
                )
                st.warning(
                    "当前环境未能成功下载 benepar 英文成分句法模型 `benepar_en3`，"
                    "所以下半部分会暂时回退为提示信息。\n\n"
                    f"下载失败原因：{combined_error}"
                )
                return False


def ensure_transformers_t5_compat() -> None:
    """为较新的 transformers 版本补回 benepar 依赖的旧 T5Tokenizer 接口。"""

    try:
        from transformers import T5Tokenizer
    except Exception:
        return

    if hasattr(T5Tokenizer, "build_inputs_with_special_tokens"):
        return

    def _build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        eos = [self.eos_token_id] if self.eos_token_id is not None else []
        if token_ids_1 is None:
            return list(token_ids_0) + eos
        return list(token_ids_0) + eos + list(token_ids_1) + eos

    T5Tokenizer.build_inputs_with_special_tokens = _build_inputs_with_special_tokens


def ensure_nltk_ready() -> bool:
    """确保 NLTK 中的基础组件可用。"""

    if nltk is None:
        return False
    try:
        from nltk.tokenize import TreebankWordTokenizer  # noqa: F401

        return True
    except Exception as exc:  # noqa: BLE001
        st.error(f"NLTK 初始化失败：{exc}")
        return False


@st.cache_resource(show_spinner=False)
def load_spacy_pipeline():
    """加载 spaCy 管线，并缓存，避免每次交互都重新初始化模型。"""

    if not ensure_spacy_model():
        return None
    return spacy.load(SPACY_MODEL_NAME)


@st.cache_resource(show_spinner=False)
def load_benepar_parser():
    """加载 benepar 解析器，并缓存。"""

    ensure_transformers_t5_compat()
    if not ensure_benepar_model() or not ensure_nltk_ready():
        return None
    return benepar.Parser(BENEPAR_MODEL_NAME)


def phrase_badge(label: str) -> str:
    """根据短语类型返回彩色标签 HTML。"""

    style = PHRASE_STYLE.get(label, {"color": "#475569", "icon": "■"})
    return (
        f"<span class='phrase-chip' style='background:{style['color']};'>"
        f"{style['icon']} {html.escape(label)}</span>"
    )


def get_dependency_explanation(dep_label: str) -> str:
    """将依存标签映射为中文解释。"""

    return DEP_LABEL_EXPLANATIONS.get(dep_label, "其他修饰关系。")


def render_dependency_svg(doc) -> str:
    """使用 displaCy 将依存句法结果渲染为 SVG 字符串。"""

    options = {
        "compact": False,
        "bg": "#ffffff",
        "color": "#0f172a",
        "font": "IBM Plex Sans, Microsoft YaHei, sans-serif",
        "distance": 100,
        "arrow_spacing": 12,
    }
    svg = spacy.displacy.render(doc, style="dep", jupyter=False, options=options)
    return svg.replace("\n\n", "\n")


def get_head_text(token) -> str:
    """统一获取一个 token 的中心词文本。"""

    return "ROOT" if token.head == token else token.head.text


def build_dependency_table(doc):
    """构建依存关系解释表。"""

    rows = []
    for token in doc:
        rows.append(
            {
                "序号": token.i + 1,
                "词": token.text,
                "词性": token.pos_,
                "中心词": get_head_text(token),
                "依存标签": token.dep_,
                "标签解释": get_dependency_explanation(token.dep_),
            }
        )
    return pd.DataFrame(rows) if pd is not None else rows


def sentence_tokens(text: str) -> list[str]:
    """使用 NLTK 的 Treebank 规则做英文分词。

    这里不用 punkt 等额外语料，目的是减少首次运行时的依赖摩擦。
    """

    from nltk.tokenize import TreebankWordTokenizer

    tokenizer = TreebankWordTokenizer()
    return tokenizer.tokenize(text)


def parse_constituency(text: str):
    """生成成分句法树。

    返回值是 NLTK Tree；如果失败则返回 None。
    """

    parser = load_benepar_parser()
    if parser is None:
        return None
    tokens = sentence_tokens(text)
    if not tokens:
        return None
    return parser.parse(tokens)


def load_constituency_tree(text: str, doc):
    """仅使用 benepar 生成真实成分句法树。"""

    parser = load_benepar_parser()
    if parser is not None:
        tokens = sentence_tokens(text)
        if tokens:
            return parser.parse(tokens), "benepar"

    return None, "unavailable"


def leaf_positions(tree) -> list[tuple[str, str]]:
    """提取某个子树里所有“词 / POS”对。

    benepar 输出的是 NLTK Tree：
    - 非终结点是短语标签（NP/VP/PP...）
    - 倒数第二层通常是词性标签（NN/VBD/IN...）
    - 最底层才是真实词形
    """

    pairs: list[tuple[str, str]] = []
    if not hasattr(tree, "label"):
        return pairs

    for subtree in tree.subtrees():
        if len(subtree) == 1 and isinstance(subtree[0], str):
            pairs.append((subtree[0], subtree.label()))
    return pairs


def choose_phrase_head(label: str, subtree) -> str:
    """基于简单启发式，为一个短语选取“核心词”。

    这里不追求严格语言学定义，而是给教学展示一个可读、稳定的核心词。
    """

    tagged_leaves = leaf_positions(subtree)
    if not tagged_leaves:
        leaves = subtree.leaves() if hasattr(subtree, "leaves") else []
        return leaves[-1] if leaves else ""

    base_label = normalize_phrase_label(label)

    if base_label == "NP":
        preferred = ("NN", "NNS", "NNP", "NNPS", "PRP", "CD")
        for word, pos in reversed(tagged_leaves):
            if pos.startswith(preferred):
                return word
    if base_label == "VP":
        for word, pos in tagged_leaves:
            if pos.startswith("VB"):
                return word
    if base_label == "PP":
        for word, pos in tagged_leaves:
            if pos in {"IN", "TO", "RP"}:
                return word
    if base_label == "ADJP":
        for word, pos in reversed(tagged_leaves):
            if pos.startswith("JJ"):
                return word
    if base_label == "ADVP":
        for word, pos in reversed(tagged_leaves):
            if pos.startswith("RB"):
                return word

    return tagged_leaves[-1][0]


def extract_phrases(tree, min_leaves: int = 1) -> list[dict[str, str]]:
    """递归提取短语列表。

    只保留真正的短语层（如 NP/VP/PP/S），避免把 POS 预终结点也当成短语。
    """

    phrases: list[dict[str, str]] = []

    for subtree in tree.subtrees():
        if not hasattr(subtree, "label"):
            continue

        raw_label = subtree.label()
        label = normalize_phrase_label(raw_label)
        if not label or len(subtree.leaves()) < min_leaves:
            continue

        # 跳过预终结点，例如 (NN telescope) 这一层。
        if len(subtree) == 1 and isinstance(subtree[0], str):
            continue

        content = " ".join(subtree.leaves())
        head_word = choose_phrase_head(label, subtree)
        phrases.append(
            {
                "type": label,
                "raw_type": raw_label,
                "content": content,
                "head": head_word,
            }
        )

    return phrases


def render_phrase_table_markdown(phrases: list[dict[str, str]]) -> str:
    """把短语列表渲染成 Markdown 表格字符串。"""

    lines = [
        "| 短语类型 | 短语内容 | 核心词 |",
        "| --- | --- | --- |",
    ]
    for item in phrases:
        label = item["type"]
        style = PHRASE_STYLE.get(label, {"icon": "■"})
        decorated = f"{style['icon']} {label}"
        content = item["content"].replace("|", "\\|")
        head_word = item["head"].replace("|", "\\|")
        lines.append(f"| {decorated} | {content} | {head_word} |")
    return "\n".join(lines)


def tree_to_svg(tree) -> str | None:
    """如果安装了 svgling，就把成分树转成 SVG。"""

    if svgling is None:
        return None
    try:
        figure = svgling.draw_tree(tree)
        return figure._repr_svg_()
    except Exception:
        return None


def tree_to_pretty_text(tree) -> str:
    """把成分树转成多级缩进文本，作为图形化失败时的稳妥回退。"""

    if tree is None:
        return "暂无成分句法结果。"
    return tree.pformat(margin=80)


def tree_to_html(tree) -> str:
    """把 benepar 的真实解析树渲染为嵌套卡片树。"""

    def render_node(node, is_root: bool = False) -> str:
        if isinstance(node, str):
            return f"<span class='const-terminal-word'>{html.escape(node)}</span>"

        label = html.escape(str(node.label()))
        if len(node) == 1 and isinstance(node[0], str):
            word = html.escape(str(node[0]))
            return (
                "<div class='const-terminal'>"
                f"<span class='const-terminal-pos'>{label}</span>"
                f"<span class='const-terminal-word'>{word}</span>"
                "</div>"
            )

        children_html = "".join(render_node(child) for child in node)
        root_class = " root" if is_root else ""
        return (
            f"<div class='const-card-node{root_class}'>"
            f"<span class='const-card-label'>{label}</span>"
            f"<div class='const-card-children'>{children_html}</div>"
            "</div>"
        )

    return f"<div class='const-tree'>{render_node(tree, is_root=True)}</div>"


def generate_dynamic_explanations(doc, phrases: list[dict[str, str]]) -> list[str]:
    """根据当前解析结果自动生成简要学习说明。"""

    explanations: list[str] = []

    root = next((token for token in doc if token.dep_ in {"ROOT", "root"}), None)
    if root is not None:
        explanations.append(f"句子的核心谓词是 `{root.text}`，它作为整句的结构中心连接主要成分。")

    subject = next((token for token in doc if token.dep_ in {"nsubj", "nsubjpass", "csubj"}), None)
    if subject is not None and root is not None:
        explanations.append(f"`{subject.text}` 被识别为主语，并与核心谓词 `{root.text}` 构成句子的主干。")

    obj = next((token for token in doc if token.dep_ in {"obj", "dobj", "pobj", "attr"}), None)
    if obj is not None and root is not None:
        explanations.append(f"`{obj.text}` 是核心谓词 `{root.text}` 关联的重要补足成分，帮助说明动作或状态指向的对象。")

    prep_token = next((token for token in doc if token.dep_ == "prep"), None)
    if prep_token is not None:
        prep_object = next((child for child in prep_token.children if child.dep_ in {"pobj", "obj"}), None)
        governor = prep_token.head.text if prep_token.head != prep_token else prep_token.text
        if prep_object is not None:
            explanations.append(
                f"介词短语 `{prep_token.text} {prep_object.text}` 被解析为 `{governor}` 的修饰语，"
                f"通常用于补充工具、位置、方向或伴随信息。"
            )
        else:
            explanations.append(f"介词 `{prep_token.text}` 在句中引出附加说明，修饰中心词 `{governor}`。")

    # 成分句法层面的说明：优先寻找 VP/PP 组合，展示短语嵌套思路。
    phrase_types = {item["type"] for item in phrases}
    if "VP" in phrase_types and "PP" in phrase_types:
        explanations.append("从成分句法角度看，动词短语 VP 往往会继续嵌套介词短语 PP，用来补充动作发生的方式、地点或工具。")
    elif "NP" in phrase_types and "PP" in phrase_types:
        explanations.append("短语结构中出现了 NP 与 PP 的嵌套，这通常表示名词短语被进一步限定或补充说明。")

    if not explanations:
        explanations.append("当前句子结构较简单，模型主要识别出了主干谓词与少量修饰成分。")

    return explanations[:4]


def render_sidebar(explanations: list[str]) -> None:
    """渲染学习侧边栏。"""

    st.sidebar.markdown("## 学习侧边栏")
    st.sidebar.markdown("### 依存句法简介")
    st.sidebar.markdown(
        "- `核心词`：句法结构中的中心词，其他词围绕它组织。\n"
        "- `依存弧`：连接中心词与修饰词的关系线。\n"
        "- `nsubj`：名词性主语。\n"
        "- `obj / dobj`：宾语或直接宾语。\n"
        "- `prep`：介词修饰语，引出介词短语。\n"
        "- `pobj`：介词宾语。"
    )

    st.sidebar.markdown("### 成分句法简介")
    st.sidebar.markdown(
        "- `短语结构`：句子由词逐层组合为短语，再组合为更大结构。\n"
        "- `NP`：名词短语，常承担主语、宾语等功能。\n"
        "- `VP`：动词短语，通常围绕谓词组织。\n"
        "- `PP`：介词短语，常作修饰语。\n"
        "- `递归嵌套`：一个短语内部还可以继续包含其他短语。"
    )

    st.sidebar.markdown("### 当前句子的简要分析")
    for item in explanations:
        st.sidebar.markdown(f"- {item}")


example_sentence = "I saw the man with a telescope near the station."

sentence = st.text_input(
    "输入一句英文句子",
    value=example_sentence,
    help="当前页面使用英文句法模型，推荐输入英文句子，例如：I saw the man with a telescope.",
)


with st.expander("依赖与模型说明", expanded=False):
    st.code(
        "pip install streamlit 'spacy>=3.8,<3.9' benepar svgling nltk pandas\n"
        "pip install "
        "https://github.com/explosion/spacy-models/releases/download/"
        "en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl\n"
        "python -c \"import benepar; benepar.download('benepar_en3')\"",
        language="bash",
    )
    st.markdown(
        "为了让 Streamlit Cloud 更稳定，建议把 `en_core_web_sm` 在部署阶段预装到环境里，"
        "不要依赖页面运行时再下载。`benepar_en3` 如果缺失，页面会通过 benepar 官方模型索引自动下载到可写缓存目录；"
        "若下载失败，页面仍会保留依存句法分析，但不会再用规则树冒充成分句法结果，而是明确提示当前成分句法不可用。"
    )


# 只有在句子非空时才继续分析，避免无意义的模型调用。
if not sentence.strip():
    st.warning("请输入一句英文句子后再开始分析。")
    st.stop()


nlp = load_spacy_pipeline()
if nlp is None:
    st.error("spaCy 模型尚未就绪，当前无法进行依存句法分析。")
    st.stop()

doc = nlp(sentence.strip())

constituency_tree, constituency_mode = load_constituency_tree(sentence.strip(), doc)
phrase_rows = extract_phrases(constituency_tree) if constituency_tree is not None else []
dynamic_explanations = generate_dynamic_explanations(doc, phrase_rows)
render_sidebar(dynamic_explanations)


metric_col1, metric_col2, metric_col3 = st.columns(3)
metric_col1.metric("Token 数量", len([token for token in doc]))
metric_col2.metric("依存关系数", len([token for token in doc if token.dep_]))
metric_col3.metric("短语数量", len(phrase_rows))


st.markdown(
    """
    <div class="section-card">
        <div class="section-heading">
            <div>
                <div class="section-kicker">Upper Panel</div>
                <h3>依存句法分析</h3>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

dependency_svg = render_dependency_svg(doc)
st.markdown('<div class="syntax-card">', unsafe_allow_html=True)
st.markdown(
    '<div class="syntax-caption">上栏展示词与词之间的依存弧。弧线起点通常是中心词，终点是被支配或被修饰的词。</div>',
    unsafe_allow_html=True,
)
st.markdown('<div class="dep-svg-shell">', unsafe_allow_html=True)
components.html(dependency_svg, height=420, scrolling=True)
st.markdown("</div>", unsafe_allow_html=True)
st.markdown(
    '<div class="dependency-hint">提示：如果一个介词短语挂在动词上，它通常说明动作的工具、地点或方式；如果挂在名词上，则更像是在限定名词。</div>',
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)


st.markdown('<div class="syntax-card"><h4>依存关系解释表</h4>', unsafe_allow_html=True)
dependency_table = build_dependency_table(doc)
if pd is not None:
    st.dataframe(dependency_table, use_container_width=True, hide_index=True)
else:
    st.table(dependency_table)
st.markdown("</div>", unsafe_allow_html=True)


st.markdown(
    """
    <div class="section-card">
        <div class="section-heading">
            <div>
                <div class="section-kicker">Lower Panel</div>
                <h3>成分句法分析</h3>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="syntax-card">', unsafe_allow_html=True)
if constituency_mode == "benepar":
    st.markdown(
        '<div class="syntax-caption">下栏展示的是 benepar 实时生成的真实成分句法树。优先使用 SVG 图形树；下方同步给出同一棵解析树的文本表示，便于核对而不是额外伪造一棵树。</div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<div class="syntax-caption">当前环境未能加载 benepar 神经成分句法模型，因此下栏不会展示伪造或规则拼接的成分树。</div>',
        unsafe_allow_html=True,
    )

if constituency_tree is None:
    st.warning("当前无法生成真实成分句法树，请检查 benepar 及其模型是否安装成功。")
else:
    pretty_tree_text = tree_to_pretty_text(constituency_tree)
    st.markdown('<div class="const-tree-shell">', unsafe_allow_html=True)
    st.markdown(tree_to_html(constituency_tree), unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    with st.expander("查看 benepar 原始括号表示", expanded=False):
        st.markdown(
            f"<pre class='tree-text'>{html.escape(pretty_tree_text)}</pre>",
            unsafe_allow_html=True,
        )

st.markdown("</div>", unsafe_allow_html=True)


st.markdown('<div class="syntax-card"><h4>当前句子的简要分析说明</h4>', unsafe_allow_html=True)
st.markdown("<div class='analysis-box'>", unsafe_allow_html=True)
for item in dynamic_explanations:
    st.markdown(f"- {item}")
st.markdown("</div></div>", unsafe_allow_html=True)


with st.expander("短语列表展示", expanded=False):
    if phrase_rows:
        st.markdown(
            "下面的表格列出成分句法树中的主要短语、其覆盖内容，以及基于启发式规则估计的核心词。"
        )
        legend_html = " ".join(
            phrase_badge(label)
            for label in ["NP", "VP", "PP", "ADJP", "ADVP", "S", "SBAR"]
        )
        st.markdown(legend_html, unsafe_allow_html=True)
        st.markdown(render_phrase_table_markdown(phrase_rows))
    else:
        st.info("当前没有可展示的短语列表。")
