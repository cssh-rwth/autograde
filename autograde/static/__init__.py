from pathlib import Path
from typing import Dict

# Globals and constants variables.
STATIC_DIR = Path(__file__).parent.absolute()
STATIC_FILES: Dict[str, Path] = {p.name: p for p in filter(lambda p: not p.name.startswith('__'), STATIC_DIR.glob('*'))}
INJECT_BEFORE = STATIC_FILES['inject_before.py'].read_text()
INJECT_AFTER = STATIC_FILES['inject_after.py'].read_text()
CSS = STATIC_FILES['style.css'].read_text()
FAVICON = STATIC_FILES['favicon.ico'].read_bytes()
