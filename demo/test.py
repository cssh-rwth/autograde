#!/usr/bin/env python3
# Standard library modules.
import sys
import inspect

# Third party modules.

# Local modules
from autograde import NotebookTest
from autograde.helpers import assert_raises

# Globals and constants variables.
nbt = NotebookTest('demo notebook test', cell_timeout=1., test_timeout=.1)

nbt.set_import_filter(r'networkx|requests', blacklist=True)


# this test will succeed
@nbt.register(target='foo', label='test foo')
def test_foo(foo):
    print('foo')
    for i in range(-5, 5):
        assert i ** 2 == foo(i)


# as well as this one (note the optional return message)
@nbt.register(target='bar', label='test bar', score=2.5)
def test_bar(bar):
    print('bar', file=sys.stderr)
    for i in range(-5, 5):
        assert i ** 2 / 2 == bar(i)

    return 'well done'


# multiple targets? no problem!
@nbt.register(target=('foo', 'bar'), label='test foo & bar')
def test_foo_bar(foo, bar):
    assert foo(0) == bar(0)


# but this test fails
@nbt.register(target='fnord', label='test fnord', score=1.5)
def test_fnord(fnord):
    assert fnord() == 42


# see if the import restrictions defined above work
@nbt.register(target='illegal_import', label='test import filter', score=.5, timeout_=1.)
def test_illegal_import(illegal_import):
    with assert_raises(ImportError):
        illegal_import()


# this one will fail due to the global timeout
@nbt.register(target='sleep', label='test global timeout', score=1)
def test_sleep_1(sleep):
    sleep(.2)


# specifying timeout here will overwrite global settings
@nbt.register(target='sleep', label='test local timeout', score=1, timeout_=.06)
def test_sleep_2(sleep):
    sleep(.08)


# this test will succeed
@nbt.register(target='foo', label='inspect source')
def test_inspect_source(foo):
    print(inspect.getsource(foo))


# Sometimes, the textual cells of a notebook are also of interest and should be included into the
# report. However, other than regular test cases, textual tests cannot be passed and scored with NaN
# by default. This feature is intended to support manual inspection.
nbt.register_comment(target=r'\*A1:\*', label='Bob', score=4)
nbt.register_comment(target=r'\*A2:\*', label='Douglas', score=1)
nbt.register_comment(target=r'\*A3:\*', label='???', score=2.5)

# Similarly, one may include figures into the report. Currently, PNG and SVG files are supported.
nbt.register_figure(target='plot.png', label='polygon PNG')
nbt.register_figure(target='does_not_exist.png', label='file not found')


# One may also load raw files from artifacts for more advanced testing
@nbt.register(target='__ARTIFACTS__', label='raw artifact')
def test_raw_artifacts(artifacts):
    content = artifacts['fnord.txt'].decode('utf-8')
    return f'the following was read from a file: "{content}"'


# `execute` brings a simple comand line interface, e.g.:
# `$ test.py notebook.ipynb -c context/ -t /tmp/ -vvv`
if __name__ == '__main__':
    sys.exit(nbt.execute())
