# Standard library modules.
import pathlib

# Third party modules.

# Local modules


def _load(*args):
    with open(pathlib.Path(__file__).parent.joinpath(*args), mode='rt') as f:
        return f.read().strip() + '\n'


# Globals and constants variables.
INJECT_BEFORE = _load('inject_before.py')
INJECT_AFTER = _load('inject_after.py')
REPORT_TEMPLATE = _load('report.rst') + '\n'

