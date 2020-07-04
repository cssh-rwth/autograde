#!/usr/bin/env python3
# Standard library modules.
import sys

# Third party modules.

# Local modules
from autograde import NotebookTest

# Globals and constants variables.
nbt = NotebookTest('demo notebook test', cell_timeout=1., test_timeout=.1)


@nbt.register(target='square', label='t1')
def test_square(square):
    for i in range(10):
        assert i ** 2 == square(i)


@nbt.register(target='absolute', label='t2')
def test_absolute(absolute):
    for i in range(10):
        assert abs(i) == absolute(i)


@nbt.register(target='factorial', label='t3')
def test_factorial(factorial):
    for i, f in [(0, 1), (1, 1), (2, 2), (3, 6), (4, 24), (5, 120)]:
        assert f == factorial(i)


if __name__ == '__main__':
    sys.exit(nbt.execute())
