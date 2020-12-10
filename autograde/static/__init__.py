from pathlib import Path


def _load(*args, mode='rt'):
    with Path(__file__).parent.joinpath(*args).open(mode=mode if mode.startswith('r') else f'r{mode}') as f:
        return f.read().strip() + '\n' if mode.endswith('t') else f.read()


# Globals and constants variables.
INJECT_BEFORE = _load('inject_before.py')
INJECT_AFTER = _load('inject_after.py')
CSS = _load('basic.css')
FAVICON = _load('favicon.ico', mode='rb')
