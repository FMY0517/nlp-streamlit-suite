from __future__ import annotations

from pathlib import Path
import os


APP_CACHE_ENV = "NLP_APP_CACHE_ROOT"
APP_CACHE_NAME = "nlp_web_suite"


def get_cache_root() -> Path:
    """返回跨平台可写缓存根目录。

    优先读取环境变量，便于云端覆盖；
    否则默认写到当前用户家目录下的 `.cache/nlp_web_suite`。
    """

    env_value = os.environ.get(APP_CACHE_ENV)
    if env_value:
        root = Path(env_value).expanduser()
    else:
        root = Path.home() / ".cache" / APP_CACHE_NAME
    root.mkdir(parents=True, exist_ok=True)
    return root


def ensure_named_cache_dir(name: str) -> Path:
    """在统一缓存根目录下创建命名子目录。"""

    path = get_cache_root() / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_nltk_data_dir() -> Path:
    """返回 NLTK 数据目录。"""

    return ensure_named_cache_dir("nltk_data")


def find_available_chinese_font() -> str | None:
    """查找当前环境可用的中文字体文件。

    搜索顺序：
    1. 环境变量 `NLP_CHINESE_FONT`
    2. 仓库内 `assets/fonts`
    3. Linux 常见中文字体路径
    4. Windows 常见中文字体路径
    """

    env_font = os.environ.get("NLP_CHINESE_FONT")
    if env_font and Path(env_font).exists():
        return env_font

    candidates = [
        Path("assets/fonts/NotoSansCJK-Regular.ttc"),
        Path("assets/fonts/NotoSerifCJK-Regular.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"),
        Path("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"),
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\msyhbd.ttc"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
        Path(r"C:\Windows\Fonts\simsun.ttc"),
        Path(r"C:\Windows\Fonts\simkai.ttf"),
    ]

    for path in candidates:
        if path.exists():
            return str(path)
    return None
