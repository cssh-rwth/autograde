#!/usr/bin/env #!/usr/bin/env python3

import argparse
import os
import sys
from pathlib import Path

import autograde
from autograde.backend import Backend
from autograde.util import logger, loglevel, project_root


def cli(args=None):
    # environment variables
    verbosity = int(os.environ.get('AG_VERBOSITY', 0))
    container_backend = os.environ.get('AG_BACKEND', 'local')
    container_tag = os.environ.get('AG_TAG', 'autograde')

    # command line arguments
    parser = argparse.ArgumentParser(
        description=f'autograde {autograde.__version__}\n'
                    'toolbox for testing and grading jupyter notebooks',
        epilog='autograde on github: https://github.com/cssh-rwth/autograde',
        prog='autograde',
    )

    # global flags
    backends = ','.join(Backend.available)
    parser.add_argument('-v', '--verbosity', action='count', default=verbosity,
                        help='verbosity level')
    parser.add_argument('--backend', type=str, default=container_backend,
                        choices=list(Backend.available), metavar='',
                        help=f'container backend to use {{{backends}}}, default is "{container_backend}"')
    parser.add_argument('--tag', type=str, default=container_tag, metavar='',
                        help=f'container tag, default: "{container_tag}"')
    parser.set_defaults(cmd='version')

    subparsers = parser.add_subparsers(help='sub command help')

    # build sub command
    bld_parser = subparsers.add_parser('build', help=Backend.build.__doc__)
    bld_parser.add_argument('-r', '--requirements', type=Path, default=None,
                            help='additional requirements to install')
    bld_parser.set_defaults(cmd='build')

    # test sub command
    tst_parser = subparsers.add_parser('test', help=Backend.test.__doc__)
    tst_parser.add_argument('test', type=Path, help='autograde test script')
    tst_parser.add_argument('notebook', type=Path, help='the jupyter notebook(s) to be tested')
    tst_parser.add_argument('-t', '--target', type=Path, metavar='', help='where to store results')
    tst_parser.add_argument('-c', '--context', type=Path, metavar='', help='context directory')
    tst_parser.set_defaults(cmd='test')

    # patch sub command
    ptc_parser = subparsers.add_parser('patch', help=Backend.patch.__doc__)
    ptc_parser.add_argument('result', type=Path, help='result archive(s) to be patched')
    ptc_parser.add_argument('patch', type=Path, help='result archive(s) for patching')
    ptc_parser.set_defaults(cmd='patch')

    # audit sub command
    adt_parser = subparsers.add_parser('audit', help=Backend.audit.__doc__)
    adt_parser.add_argument('result', type=Path, help='result archive(s) to audit')
    adt_parser.add_argument('-b', '--bind', type=str, default='127.0.0.1', help='host to bind to')
    adt_parser.add_argument('-p', '--port', type=int, default=5000, help='port')
    adt_parser.set_defaults(cmd='audit')

    # report sub command
    rpt_parser = subparsers.add_parser('report', help=Backend.report.__doc__)
    rpt_parser.add_argument('result', type=Path, help='result archive(s) for creating the report')
    rpt_parser.set_defaults(cmd='report')

    # summary sub command
    sum_parser = subparsers.add_parser('summary', help=Backend.summary.__doc__)
    sum_parser.add_argument('result', type=Path, help='result archives to summarize')
    sum_parser.set_defaults(cmd='summary')

    # version sub command
    vrs_parser = subparsers.add_parser('version', help=Backend.version.__doc__)
    vrs_parser.set_defaults(cmd='version')

    args = parser.parse_args(args).__dict__

    logger.setLevel(loglevel(args['verbosity']))
    logger.debug(f'python: {sys.executable}')
    logger.debug(f'autograde: {project_root()}')
    logger.debug(f'default encoding: {sys.getdefaultencoding()}')
    logger.debug(f'supported backends: {set(Backend.supported)}')
    logger.debug(f'available backends: {set(Backend.available)}')
    logger.debug(f'args: {args}')

    backend = Backend.load(args.pop('backend'), tag=args.pop('tag'), verbosity=args.pop('verbosity'))
    command = getattr(backend, args.pop('cmd'))

    return command(**args) or 0


if __name__ == '__main__':
    sys.exit(cli())
