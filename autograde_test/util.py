import math
import shutil
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory

from autograde.cli import cli
from autograde.helpers import assert_iter_eqal
from autograde.util import project_root

_tmp = TemporaryDirectory()
TMP_ROOT = Path(_tmp.__enter__()).parent
_tmp.cleanup()

DEMO = project_root().joinpath('demo')


def load_demo_archive() -> bytes:
    """
    Run the demo and load results into memory. The result archive persists over
    several runs, i.e. should manually be deleted when the demo is modified
    """
    archive = TMP_ROOT.joinpath('demo_result.zip')

    if not archive.is_file():
        with TemporaryDirectory() as tmp:
            cli([
                'test', str(DEMO.joinpath('test.py')), str(DEMO),
                '--context', str(DEMO.joinpath('context')),
                '--target', str(tmp)
            ])

            shutil.move(next(Path(tmp).glob('*.zip')), archive)

    with archive.open(mode='rb') as f:
        return f.read()


def float_equal(a, b):
    return math.isclose(a, b) or (math.isnan(a) and math.isnan(b))


assert_floats_equal = partial(assert_iter_eqal, comp=float_equal)
