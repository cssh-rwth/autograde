import subprocess
from pathlib import Path
from typing import Optional

from autograde.util import logger


# no typo, the word "test" caused some confusion with the test framework
def cmd_tset(test: Path, notebook: Path, target: Path, verbosity: int, context: Optional[Path] = None) -> int:
    assert test.is_file(), f'{test} is no regular file'
    assert notebook.is_file() or notebook.is_dir(), f'{notebook} is no regular file or directory'
    assert target.is_dir(), f'{target} is no regular directory'
    assert context is None or context.is_dir(), f'{context} is no regular directory'

    if notebook.is_file():
        notebooks = [notebook]
    else:
        notebooks = list(filter(
            lambda p: '.ipynb_checkpoints' not in p.parts,
            notebook.rglob('*.ipynb')
        ))

    def run(path_nb_):
        cmd = [
            'python', f'"{test}"',
            f'"{path_nb_}"',
            '--target', f'"{target}"',
            *(('--context', f'"{context}"') if context else ()),
            *(('-' + 'v' * verbosity,) if verbosity > 0 else ())
        ]

        logger.info(f'test: {path_nb_}')
        logger.debug('run' + ' '.join(cmd))

        return subprocess.call(' '.join(cmd), shell=True)

    return sum(map(run, notebooks))
