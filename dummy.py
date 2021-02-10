#!/usr/bin/env python3
import sys

from autograde.cli import cli

if __name__ == '__main__':
    cli(['version'])
    print(flush=True)
    cli(['--help'])
    sys.exit(0)
