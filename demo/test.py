#!/usr/bin/env python3
# Standard library modules.
import sys

# Third party modules.

# Local modules
from autograde import NotebookTest

# Globals and constants variables.
nbt = NotebookTest(cell_timeout=1., test_timeout=.1)


# this test will succeed
@nbt.register(target='foo')
def test_foo(foo):
    print('foo')
    for i in range(-5, 5):
        assert i ** 2 == foo(i)


# as well as this one
@nbt.register(target='bar', score=2.5, label='some label')
def test_bar(bar):
    print('bar', file=sys.stderr)
    for i in range(-5, 5):
        assert i ** 2 / 2 == bar(i)


# multiple targets? no problem!
@nbt.register(target=('foo', 'bar'))
def test_foo_bar(foo, bar):
    assert foo(0) == bar(0)


# but this test fails
@nbt.register(target='fnord', score=1.5, label='another label')
def test_fnord(fnord):
    assert fnord() == 42


# this one will fail due to the global timeout
@nbt.register(target='sleep', score=1, label='global timeout')
def test_sleep_1(sleep):
    sleep(.2)


# specifying timeout here will overwrite global settings
@nbt.register(target='sleep', score=1, timeout_=.06, label='local timeout')
def test_sleep_2(sleep):
    sleep(.08)


# `execute` brings a simple comand line interface, e.g.:
# `$ test.py notebook.ipynb -c context/ -t /tmp/ -vvv`
if __name__ == '__main__':
    sys.exit(nbt.execute())
