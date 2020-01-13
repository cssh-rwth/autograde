#!/usr/bin/env python3
# Standard library modules.
import os
import sys
import subprocess

# Third party modules.

# Local modules

# Globals and constants variables.


def main():
    print('-' * 80)
    print(os.getcwd(), *sys.argv)
    print('-' * 80)

    for d in ['.', 'src', 'target', 'context']:
        cp = subprocess.run(['ls', '-la', d], capture_output=True)
        print(f'files in "{d}"')
        print(cp.stdout.decode('utf-8'))
        print('-' * 80)


if __name__ == '__main__':
    sys.exit(main())
