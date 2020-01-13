#!/usr/bin/env python3
# Standard library modules.
import os
import sys
import argparse
import subprocess
from pathlib import Path

# Third party modules.

# Local modules
from autograde.util import project_root, cd

# Globals and constants variables.
CONTAINER_TAG = 'autograde'


def build(args):
    cmd = [args.backend, 'build', '-t', CONTAINER_TAG, '.']

    with cd(project_root()):
        return subprocess.run(cmd, capture_output=args.quiet).returncode


def execute(args):
    path_tst = Path(args.test).expanduser().absolute()
    path_nbk = Path(args.notebook).expanduser().absolute()
    path_tgt = Path(args.target or Path.cwd()).expanduser().absolute()
    path_cxt = Path(args.context).expanduser().absolute() if args.context else None

    assert path_tst.is_file(), f'{path_tst} is no regular file'
    assert path_nbk.is_file(), f'{path_nbk} is no regular file'
    assert path_tgt.is_dir(), f'{path_tgt} is no regular directory'
    assert path_cxt is None or path_cxt.is_dir(), f'{path_cxt} is no regular directory'

    cmd = [
        args.backend, 'run',
        '-v', f'{path_tst}:/autograde/test.py',
        '-v', f'{path_nbk}:/autograde/notebook.ipynb',
        '-v', f'{path_tgt}:/autograde/target',
        *(('-v', f'{path_cxt}:/autograde/context:ro') if path_cxt else ()),
        '-u', str(os.geteuid()),
        CONTAINER_TAG
    ]

    return subprocess.run(cmd).returncode


def main():
    parser = argparse.ArgumentParser(description='run tests on jupyter notebook')

    parser.add_argument('-e', '--backend', type=str, default='docker', choices=['docker', 'podman'], metavar='', help='backend to use')

    subparsers = parser.add_subparsers(help='sub command help')

    bld_parser = subparsers.add_parser('build')
    bld_parser.add_argument('-q', '--quiet', action='store_true', help='mute output')
    bld_parser.set_defaults(func=build)

    exe_parser = subparsers.add_parser('exec')
    exe_parser.add_argument('test', type=str, help='autograde test script')
    exe_parser.add_argument('notebook', type=str, help='the jupyter notebook to be tested')
    exe_parser.add_argument('-t', '--target', type=str, metavar='', help='where to store results')
    exe_parser.add_argument('-c', '--context', type=str, metavar='', help='context directory')
    exe_parser.set_defaults(func=execute)

    args = parser.parse_args()

    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
