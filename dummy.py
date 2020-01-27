#!/usr/bin/env python3
# Standard library modules.
import sys
import subprocess

# Third party modules.

# Local modules

# Globals and constants variables.


if __name__ == '__main__':
    sys.exit(subprocess.run(['python', '-m', 'autograde', 'test', '--help']).returncode)
