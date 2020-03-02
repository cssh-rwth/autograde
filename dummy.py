#!/usr/bin/env python3
# Standard library modules.
import os
import sys
import subprocess

# Third party modules.

# Local modules
import autograde
from autograde.util import project_root

# Globals and constants variables.


if __name__ == '__main__':
    print(f'autograde version {autograde.__version__}\n', flush=True)

    if 'unittest' in list(map(str.lower, sys.argv)):
        test_dir = project_root().joinpath('test')
        sys.exit(subprocess.run(['python', '-m', 'nose', str(test_dir)]).returncode)

    sys.exit(subprocess.run(['python', '-m', 'autograde', 'test', '--help']).returncode)
