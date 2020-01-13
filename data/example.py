#!/usr/bin/env python3
# Standard library modules.
import sys

# Third party modules.

# Local modules
from autograde import NotebookTest

# Globals and constants variables.
nbt = NotebookTest()


@nbt.register(target='foo')
def test_foo(foo):
    print('foo')
    for i in range(-5, 5):
        assert i ** 2 == foo(i)


@nbt.register(target='bar', score=2.5, label='some label')
def test_nar(bar):
    print('bar', file=sys.stderr)
    for i in range(-5, 5):
        assert i ** 2 / 2 == bar(i)


@nbt.register(target='fnord', score=1.5, label='another label')
def test_nar(fnord):
    assert fnord() == 42  # this one fails


if __name__ == '__main__':
    sys.exit(nbt.execute())
