#!/usr/bin/env python3
# Standard library modules.
import sys
import subprocess

# Third party modules.

# Local modules
import autograde

# Globals and constants variables.


if __name__ == '__main__':
    print(sys.version)
    print(f'autograde version {autograde.__version__}', flush=True)
    sys.exit(subprocess.run(['python', '-m', 'autograde', '--help']).returncode)
