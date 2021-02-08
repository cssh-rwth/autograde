from importlib.metadata import version as version
from pathlib import Path

from autograde.helpers import assert_equal, assert_iter_eqal, assert_is, assert_isclose, \
    assert_raises
from autograde.notebook_test import NotebookTest

__version__ = version('jupyter-autograde')
__ag_root__ = Path(__file__).parent.parent.absolute()
