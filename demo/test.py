#!/usr/bin/env python3
# Standard library modules.
import sys

# Third party modules.

# Local modules
from autograde import NotebookTest
from autograde.helpers import assert_raises

# Globals and constants variables.
nbt = NotebookTest('demo notebook test', cell_timeout=1., test_timeout=.1)

nbt.set_import_filter(r'networkx|requests', blacklist=True)


# this test will succeed
@nbt.register(target='foo')
def test_foo(foo):
    print('foo')
    for i in range(-5, 5):
        assert i ** 2 == foo(i)


# as well as this one (note the optional return message)
@nbt.register(target='bar', score=2.5, label='some label')
def test_bar(bar):
    print('bar', file=sys.stderr)
    for i in range(-5, 5):
        assert i ** 2 / 2 == bar(i)

    return 'well done'


# multiple targets? no problem!
@nbt.register(target=('foo', 'bar'))
def test_foo_bar(foo, bar):
    assert foo(0) == bar(0)


# but this test fails
@nbt.register(target='fnord', score=1.5, label='another label')
def test_fnord(fnord):
    assert fnord() == 42


# see if the import restrictions defined above work
@nbt.register(target='illegal_import', score=.5, timeout_=1., label='test import filter')
def test_illegal_import(illegal_import):
    with assert_raises(ImportError):
        illegal_import()


# this one will fail due to the global timeout
@nbt.register(target='sleep', score=1, label='global timeout')
def test_sleep_1(sleep):
    sleep(.2)


# specifying timeout here will overwrite global settings
@nbt.register(target='sleep', score=1, timeout_=.06, label='local timeout')
def test_sleep_2(sleep):
    sleep(.08)


# Sometimes, the textual cells of a notebook are also of interest and should be included into the
# report. However, other than regular test cases, textual tests cannot be passed and scored with NaN
# by default. This feature is intended to support manual inspection.
nbt.register_comment(target=r'\*A1:\*', score=4, label='Bob')
nbt.register_comment(target=r'\*A2:\*', score=1, label='Douglas')
nbt.register_comment(target=r'\*A3:\*', score=2.5)


# `execute` brings a simple comand line interface, e.g.:
# `$ test.py notebook.ipynb -c context/ -t /tmp/ -vvv`
if __name__ == '__main__':
    sys.exit(nbt.execute())
