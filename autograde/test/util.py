import shutil
from collections import defaultdict
from contextlib import contextmanager
from functools import cache, partial
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, ContextManager

from autograde.cli import cli
from autograde.helpers import assert_iter_eqal
from autograde.util import project_root, float_equal

_tmp = TemporaryDirectory()
TMP_ROOT = Path(_tmp.__enter__()).parent
_tmp.cleanup()

DEMO = project_root().joinpath('demo')
EXAMPLES = project_root().joinpath('autograde', 'test', 'examples')


@cache
def load_demo_archive() -> bytes:
    """
    Run the demo and load results into memory. The result archive persists over
    several runs, i.e. should be manually deleted when the demo is modified.
    """
    archive = TMP_ROOT.joinpath('autograde_demo_result.zip')

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


@cache
def load_example_archives() -> Dict[str, Dict[str, bytes]]:
    """
    Run the examples and load results into memory. The result archive persist
    over several runs, i.e. should be manually deleted when examples are
    modified.
    """

    archives = defaultdict(lambda: {})

    for test, notebook in product(EXAMPLES.glob('test_*.py'), EXAMPLES.glob('solution_*.ipynb')):
        tid, nid = test.stem.split('_')[-1], notebook.stem.split('_')[-1]

        archive = TMP_ROOT.joinpath(f'autograde_example_result_{tid}_{nid}.zip')

        if not archive.is_file():
            with TemporaryDirectory() as tmp:
                cli(['test', str(test), str(notebook), '--target', str(tmp)])
                shutil.move(next(Path(tmp).glob('*.zip')), archive)

        with archive.open(mode='rb') as f:
            archives[tid][nid] = f.read()

    return dict(archives)


@contextmanager
def mount_demo_archive() -> ContextManager[Path]:
    """Mount demo archive to temporary directory"""
    with TemporaryDirectory() as temp:
        with Path(temp).joinpath('archive.zip').open(mode='wb') as f:
            f.write(load_demo_archive())

        yield Path(temp)


@contextmanager
def mount_example_archives() -> ContextManager[Path]:
    """Mount example archives to temporary directory"""
    with TemporaryDirectory() as temp:
        for tid, results in load_example_archives().items():
            test_path = Path(temp).joinpath(f'test_{tid}')
            test_path.mkdir()

            for nid, result in results.items():
                with test_path.joinpath(f'results_{nid}.zip').open(mode='wb') as f:
                    f.write(result)

        yield Path(temp)


assert_floats_equal = partial(assert_iter_eqal, comp=float_equal)
