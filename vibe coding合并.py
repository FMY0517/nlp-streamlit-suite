from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import runpy
from urllib.parse import quote

import streamlit as st


APP_CONFIG = {
    "cn_tokenizer": {
        "label": "中文分词平台",
        "file": "中文分词平台.py",
    },
    "syntax_parser": {
        "label": "句法分析平台",
        "file": "句法分析平台.py",
    },
    "semantic": {
        "label": "词义消歧与语义角色标注",
        "file": "词义消歧与语义角色标注系统.py",
    },
    "semantic_lab": {
        "label": "语义分析综合测试平台",
        "file": "语义分析综合测试平台.py",
    },
    "discourse": {
        "label": "篇章分析综合平台",
        "file": "篇章分析综合平台.py",
    },
    "lm": {
        "label": "语言模型训练与对比分析",
        "file": "语言模型训练与对比分析平台.py",
    },
    "ie_kg": {
        "label": "信息抽取与知识图谱构建",
        "file": "信息抽取与知识图谱构建系统.py",
    },
    "mt_eval": {
        "label": "机器翻译机制与质量测评系统",
        "file": "机器翻译机制与质量测评系统.py",
    },
    "sentiment_dashboard": {
        "label": "情感分析与可视化仪表盘",
        "file": "情感分析与可视化仪表盘.py",
    },
}

DEFAULT_APP_KEY = "semantic"
BASE_DIR = Path(__file__).resolve().parent


st.set_page_config(
    page_title="NLP Web 应用合集",
    page_icon="🧩",
    layout="wide",
    initial_sidebar_state="auto",
)


def normalize_query_value(value: object) -> str:
    if isinstance(value, list):
        return str(value[0]) if value else ""
    return str(value or "")


def get_selected_app_key() -> str:
    selected = normalize_query_value(st.query_params.get("app", DEFAULT_APP_KEY))
    if selected not in APP_CONFIG:
        return DEFAULT_APP_KEY
    return selected


def render_navigation(selected_key: str) -> None:
    nav_items = []
    for key, config in APP_CONFIG.items():
        selected_class = " active" if key == selected_key else ""
        nav_items.append(
            (
                f'<a class="merge-nav-link{selected_class}" href="?app={quote(key)}">'
                f"{config['label']}</a>"
            )
        )

    st.markdown(
        f"""
        <style>
            .stApp .block-container {{
                padding-right: 17.5rem;
            }}
            #merge-nav-shell {{
                position: fixed;
                top: 5rem;
                right: 1rem;
                width: 15rem;
                z-index: 999;
            }}
            #merge-nav-card {{
                border-radius: 24px;
                padding: 1rem;
                background:
                    radial-gradient(circle at top right, rgba(191, 219, 254, 0.9), transparent 30%),
                    linear-gradient(135deg, rgba(255, 255, 255, 0.97), rgba(246, 250, 255, 0.94));
                border: 1px solid rgba(255, 255, 255, 0.6);
                box-shadow: 0 18px 40px rgba(15, 23, 42, 0.12);
                backdrop-filter: blur(10px);
            }}
            #merge-nav-title {{
                font-size: 0.92rem;
                font-weight: 700;
                color: #2563eb;
                margin-bottom: 0.8rem;
                text-transform: uppercase;
                letter-spacing: 0.08em;
            }}
            .merge-nav-links {{
                display: flex;
                flex-direction: column;
                gap: 0.55rem;
            }}
            .merge-nav-link {{
                display: block;
                padding: 0.8rem 0.9rem;
                border-radius: 16px;
                text-decoration: none;
                font-weight: 600;
                color: #334155;
                background: rgba(255, 255, 255, 0.72);
                border: 1px solid rgba(148, 163, 184, 0.18);
                transition: all 0.18s ease;
            }}
            .merge-nav-link:hover {{
                color: #0f172a;
                border-color: rgba(37, 99, 235, 0.35);
                box-shadow: 0 8px 20px rgba(37, 99, 235, 0.12);
            }}
            .merge-nav-link.active {{
                color: #ffffff;
                background: linear-gradient(135deg, #0f766e, #2563eb);
                border-color: transparent;
                box-shadow: 0 10px 22px rgba(37, 99, 235, 0.24);
            }}
            @media (max-width: 1100px) {{
                .stApp .block-container {{
                    padding-right: 1rem;
                    padding-bottom: 8rem;
                }}
                #merge-nav-shell {{
                    top: auto;
                    bottom: 1rem;
                    right: 1rem;
                    left: 1rem;
                    width: auto;
                }}
                .merge-nav-links {{
                    display: grid;
                    grid-template-columns: repeat(2, minmax(0, 1fr));
                }}
            }}
            @media (max-width: 680px) {{
                .merge-nav-links {{
                    grid-template-columns: 1fr;
                }}
            }}
        </style>
        <div id="merge-nav-shell">
            <div id="merge-nav-card">
                <div id="merge-nav-title">应用切换</div>
                <div class="merge-nav-links">
                    {''.join(nav_items)}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@contextmanager
def suppress_nested_page_config():
    original_set_page_config = st.set_page_config

    def _no_op_set_page_config(*args, **kwargs):
        return None

    st.set_page_config = _no_op_set_page_config
    try:
        yield
    finally:
        st.set_page_config = original_set_page_config


def run_selected_app(selected_key: str) -> None:
    app_file = BASE_DIR / APP_CONFIG[selected_key]["file"]

    if not app_file.exists():
        st.error(f"未找到应用文件：{app_file.name}")
        return

    with suppress_nested_page_config():
        runpy.run_path(str(app_file), run_name="__main__")


selected_app_key = get_selected_app_key()
render_navigation(selected_app_key)
run_selected_app(selected_app_key)
