#!/usr/bin/env python3
import subprocess
import sys

import autograde

if __name__ == '__main__':
    print(sys.version)
    print(f'autograde version {autograde.__version__}', flush=True)
    sys.exit(subprocess.run(['python', '-m', 'autograde', '--help']).returncode)
