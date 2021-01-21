import os
import subprocess
from pathlib import Path
from typing import Optional

from autograde.cli.util import namespace_args
from autograde.util import logger


# no typo, the word "test" caused some confusion at the test framework
@namespace_args
def cmd_tst(test: str, notebook: str, target: str, tag: str, verbose: int, context: Optional[str] = None,
            backend: Optional[str] = None, **_) -> int:
    """Run autograde test script on jupyter notebook(s)"""
    path_tst = Path(test).expanduser().absolute()
    path_nbk = Path(notebook).expanduser().absolute()
    path_tgt = Path(target or Path.cwd()).expanduser().absolute()
    path_cxt = Path(context).expanduser().absolute() if context else None

    # TODO turn into Value Error
    assert path_tst.is_file(), f'{path_tst} is no regular file'
    assert path_nbk.is_file() or path_nbk.is_dir(), f'{path_nbk} is no regular file or directory'
    assert path_tgt.is_dir(), f'{path_tgt} is no regular directory'
    assert path_cxt is None or path_cxt.is_dir(), f'{path_cxt} is no regular directory'

    if path_nbk.is_file():
        notebooks = [path_nbk]
    else:
        notebooks = list(filter(
            lambda p: '.ipynb_checkpoints' not in p.parts,
            path_nbk.rglob('*.ipynb')
        ))

    def run(path_nb_):
        if backend is None:
            cmd = [
                'python', f'"{path_tst}"',
                f'"{path_nb_}"',
                '--target', f'"{path_tgt}"',
                *(('--context', f'"{path_cxt}"') if path_cxt else ()),
                *(('-' + 'v' * verbose,) if verbose > 0 else ())
            ]
        elif 'docker' in backend:
            cmd = [
                'docker', 'run',
                '-v', f'"{path_tst}:/autograde/test.py"',
                '-v', f'"{path_nb_}:/autograde/notebook.ipynb"',
                '-v', f'"{path_tgt}:/autograde/target"',
                *(('-v', f'"{path_cxt}:/autograde/context:ro"') if path_cxt else ()),
                *(('-u', str(os.geteuid())) if 'rootless' not in backend else ()),
                tag,
                *(('-' + 'v' * verbose,) if verbose > 0 else ())
            ]
        elif backend == 'podman':
            cmd = [
                'podman', 'run',
                '-v', f'"{path_tst}:/autograde/test.py:Z"',
                '-v', f'"{path_nb_}:/autograde/notebook.ipynb:Z"',
                '-v', f'"{path_tgt}:/autograde/target:Z"',
                *(('-v', f'"{path_cxt}:/autograde/context:Z"') if path_cxt else ()),
                tag,
                *(('-' + 'v' * verbose,) if verbose > 0 else ())
            ]
        else:
            raise ValueError(f'unknown backend: {backend}')

        logger.info(f'test: {path_nb_}')
        logger.debug('run' + ' '.join(cmd))

        if not backend:
            return subprocess.call(' '.join(cmd), shell=True)

        return subprocess.run(cmd).returncode

    return sum(map(run, notebooks))
