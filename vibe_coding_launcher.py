from __future__ import annotations

import runpy
from pathlib import Path


TARGET = Path(__file__).with_name("vibe coding合并.py")


runpy.run_path(str(TARGET), run_name="__main__")
