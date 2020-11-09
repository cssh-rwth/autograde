import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

from autograde.util import logger


def cmd_build(args):
    """Build autograde container image for specified backend"""
    if args.backend is None:
        logger.warning('no backend specified')
        return 1

    if args.requirements:
        with Path(args.requirements).open(mode='rt') as f:
            requirements = list(filter(lambda s: s, map(str.strip, f.read().split('\n'))))
    else:
        requirements = []

    with TemporaryDirectory() as tmp:
        logger.debug(f'copy source to {tmp}')
        shutil.copytree('.', tmp, dirs_exist_ok=True)

        if requirements:
            logger.info(f'add additional requirements: {requirements}')
            with Path(tmp).joinpath('requirements.txt').open(mode='w') as f:
                logger.debug('add additional requirements: ' + ' '.join(requirements))
                f.write('\n'.join(requirements))

        if 'docker' in args.backend:
            cmd = ['docker', 'build', '-t', args.tag, tmp]
        elif args.backend == 'podman':
            cmd = ['podman', 'build', '-t', args.tag, '--cgroup-manager=cgroupfs', tmp]
        else:
            raise ValueError(f'unknown backend: {args.backend}')

        logger.debug('run: ' + ' '.join(cmd))
        return subprocess.run(cmd, capture_output=args.quiet).returncode
