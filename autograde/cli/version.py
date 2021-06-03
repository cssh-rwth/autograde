import sys

import autograde


def cmd_version(_):
    """Display version of autograde"""
    print(f'autograde version {autograde.__version__}')
    print(f'python {sys.version.split()[0]} at {sys.executable}')
    print(f'default encoding {sys.getdefaultencoding()}')

    return 0
