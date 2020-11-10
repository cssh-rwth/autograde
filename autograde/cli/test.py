import os
import subprocess
from pathlib import Path

from autograde.util import logger


# no typo, the word "test" caused some confusion at the test framework
def cmd_tst(args):
    """Run autograde test script on jupyter notebook(s)"""
    path_tst = Path(args.test).expanduser().absolute()
    path_nbk = Path(args.notebook).expanduser().absolute()
    path_tgt = Path(args.target or Path.cwd()).expanduser().absolute()
    path_cxt = Path(args.context).expanduser().absolute() if args.context else None

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
        if args.backend is None:
            cmd = [
                'python', f'"{path_tst}"',
                f'"{path_nb_}"',
                '-t', f'"{path_tgt}"',
                *(('-c', f'"{path_cxt}"') if path_cxt else ()),
                *(('-' + 'v' * args.verbose,) if args.verbose > 0 else ())
            ]
        elif 'docker' in args.backend:
            cmd = [
                'docker', 'run',
                '-v', f'"{path_tst}:/autograde/test.py"',
                '-v', f'"{path_nb_}:/autograde/notebook.ipynb"',
                '-v', f'"{path_tgt}:/autograde/target"',
                *(('-v', f'"{path_cxt}:/autograde/context:ro"') if path_cxt else ()),
                *(('-u', str(os.geteuid())) if 'rootless' not in args.backend else ()),
                args.tag,
                *(('-' + 'v' * args.verbose,) if args.verbose > 0 else ())
            ]
        elif args.backend == 'podman':
            cmd = [
                'podman', 'run',
                '-v', f'"{path_tst}:/autograde/test.py:Z"',
                '-v', f'"{path_nb_}:/autograde/notebook.ipynb:Z"',
                '-v', f'"{path_tgt}:/autograde/target:Z"',
                *(('-v', f'"{path_cxt}:/autograde/context:Z"') if path_cxt else ()),
                args.tag,
                *(('-' + 'v' * args.verbose,) if args.verbose > 0 else ())
            ]
        else:
            raise ValueError(f'unknown backend: {args.backend}')

        logger.info(f'test: {path_nb_}')
        logger.debug('run' + ' '.join(cmd))

        if not args.backend:
            return subprocess.call(' '.join(cmd), shell=True)

        return subprocess.run(cmd).returncode

    return sum(map(run, notebooks))
