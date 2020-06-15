#!/usr/bin/env python3
# Standard library modules.
import sys
import subprocess

# Third party modules.

# Local modules
import autograde
from autograde.util import project_root

# Globals and constants variables.


if __name__ == '__main__':
    print(sys.version)
    print(f'autograde version {autograde.__version__}', flush=True)

    if 'unittest' in list(map(str.lower, sys.argv)):
        test_dir = project_root().joinpath('test')
        sys.exit(subprocess.run(['python', '-m', 'nose', str(test_dir)]).returncode)
    else:
        print('append `unittest` command to execute tests within the container')

    sys.exit(subprocess.run(['python', '-m', 'autograde', 'test', '--help']).returncode)
