import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional

from autograde.util import logger, project_root
from autograde.cli.util import namespace_args


@namespace_args
def cmd_build(tag: str, quiet: bool, backend: Optional[str] = None, requirements: Optional[str] = None, **_) -> int:
    """Build autograde container image for specified backend"""
    if backend is None:
        logger.warning('no backend specified')
        return 1

    if requirements:
        with Path(requirements).open(mode='rt', encoding='utf-8') as f:
            requirements = list(filter(lambda s: s, map(str.strip, f.readlines())))
    else:
        requirements = []

    with TemporaryDirectory() as tmp:
        logger.debug(f'copy source to {tmp}')
        shutil.copytree(project_root(), tmp, dirs_exist_ok=True)

        if requirements:
            logger.info(f'add additional requirements: {requirements}')
            with Path(tmp).joinpath('requirements.txt').open(mode='wt', encoding='utf-8') as f:
                logger.debug('add additional requirements: ' + ', '.join(requirements))
                f.write('\n'.join(requirements))

        if 'docker' in backend:
            cmd = ['docker', 'build', '-t', tag, tmp]
        elif backend == 'podman':
            cmd = ['podman', 'build', '-t', tag, '--cgroup-manager=cgroupfs', tmp]
        else:
            raise ValueError(f'unknown backend: {backend}')

        logger.debug('run: ' + ' '.join(cmd))
        return subprocess.run(cmd, capture_output=quiet).returncode
