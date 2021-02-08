#!/usr/bin/env #!/usr/bin/env python3

import argparse
import os
import sys
from pathlib import Path

from autograde.cli.audit import cmd_audit
from autograde.cli.build import cmd_build
from autograde.cli.patch import cmd_patch
from autograde.cli.report import cmd_report
from autograde.cli.summary import cmd_summary
from autograde.cli.test import cmd_tst
from autograde.cli.version import cmd_version
from autograde.util import logger, loglevel, project_root


def cli(args=None):
    # environment variables
    verbosity = int(os.environ.get('AG_VERBOSITY', 0))
    container_backend = os.environ.get('AG_BACKEND', None)
    container_tag = os.environ.get('AG_TAG', 'autograde')

    # command line arguments
    parser = argparse.ArgumentParser(
        description='utility for grading jupyter notebooks',
        epilog='autograde on github: https://github.com/cssh-rwth/autograde',
        prog='autograde',
    )

    # global flags
    parser.add_argument('-v', '--verbose', action='count', default=verbosity,
                        help='verbosity level')
    parser.add_argument('--backend', type=str, default=container_backend,
                        choices=['docker', 'rootless-docker', 'podman'], metavar='',
                        help=f'container backend to use, default is {container_backend}')
    parser.add_argument('--tag', type=str, default=container_tag, metavar='',
                        help=f'container tag, default: "{container_tag}"')
    parser.set_defaults(func=cmd_version)

    subparsers = parser.add_subparsers(help='sub command help')

    # build sub command
    bld_parser = subparsers.add_parser('build', help=cmd_build.__doc__)
    bld_parser.add_argument('-r', '--requirements', type=Path, default=None,
                            help='additional requirements to install')
    bld_parser.add_argument('-q', '--quiet', action='store_true', help='mute output')
    bld_parser.set_defaults(func=cmd_build)

    # test sub command
    tst_parser = subparsers.add_parser('test', help=cmd_tst.__doc__)
    tst_parser.add_argument('test', type=str, help='autograde test script')
    tst_parser.add_argument('notebook', type=str, help='the jupyter notebook(s) to be tested')
    tst_parser.add_argument('-t', '--target', type=str, metavar='', help='where to store results')
    tst_parser.add_argument('-c', '--context', type=str, metavar='', help='context directory')
    tst_parser.set_defaults(func=cmd_tst)

    # patch sub command
    ptc_parser = subparsers.add_parser('patch', help=cmd_patch.__doc__)
    ptc_parser.add_argument('result', type=str, help='result archive(s) to be patched')
    ptc_parser.add_argument('patch', type=str, help='result archive(s) for patching')
    ptc_parser.set_defaults(func=cmd_patch)

    # audit sub command
    adt_parser = subparsers.add_parser('audit', help=cmd_audit.__doc__)
    adt_parser.add_argument('result', type=str, help='result archive(s) to audit')
    adt_parser.add_argument('-b', '--bind', type=str, default='127.0.0.1', help='host to bind to')
    adt_parser.add_argument('-p', '--port', type=int, default=5000, help='port')
    adt_parser.set_defaults(func=cmd_audit)

    # report sub command
    rpt_parser = subparsers.add_parser('report', help=cmd_report.__doc__)
    rpt_parser.add_argument('result', type=str, help='result archive(s) for creating the report')
    rpt_parser.set_defaults(func=cmd_report)

    # summary sub command
    sum_parser = subparsers.add_parser('summary', help=cmd_summary.__doc__)
    sum_parser.add_argument('result', type=str, help='result archives to summarize')
    sum_parser.set_defaults(func=cmd_summary)

    # version sub command
    vrs_parser = subparsers.add_parser('version', help=cmd_version.__doc__)
    vrs_parser.set_defaults(func=cmd_version)

    args = parser.parse_args(args)

    logger.setLevel(loglevel(args.verbose))
    logger.debug(f'python: {sys.executable}')
    logger.debug(f'autograde: {project_root()}')
    logger.debug(f'default encoding: {sys.getdefaultencoding()}')
    logger.debug(f'args: {args}')

    return args.func(args)


if __name__ == '__main__':
    sys.exit(cli())
