#!/usr/bin/env python3
import sys

from autograde import NotebookTest
from autograde.helpers import assert_raises

nbt = NotebookTest('demo notebook test', cell_timeout=4., test_timeout=.1)

# an import filter can be any regular expression, e.g. r'autograde|networkx|typ.*'
nbt.set_import_filter(r'autograde', blacklist=True)


# this test will succeed
@nbt.register(target='square', label='test square')
def test_square(square):
    # Everything printed to stdout is included into the report
    print('F' + ('O' * 77))
    for i in range(-5, 5):
        assert i ** 2 == square(i)


# as well as this one (note the custom return message)
@nbt.register(target='cube', label='test cube', score=2.5)
def test_cube(cube):
    # Everything printed to stderr is included into the report
    print('cube', file=sys.stderr)
    for i in range(-5, 5):
        assert i ** 3 == cube(i)

    return 'well done 👌'


# this test only partially succeeds, returning a custom score
@nbt.register(target='abs_cube', label='test abs_cube', score=3.)
def test_abs_cube(abs_cube):
    score = 0.
    score += int(abs_cube(2) == 8)
    score += int(abs_cube(0) == 0)
    score += int(abs_cube(-2) == 8)  # fails
    return score  # returns 2.0


# here we test a constant, returning custom score and message
@nbt.register(target='SOME_CONSTANT', label='test constant', score=2.)
def test_constant(some_constant):
    score = 1.
    score += int(some_constant == 1337)

    if score < 2.:
        return score, 'at least you declared it 🥴'


# testing multiple targets? no problem!
@nbt.register(target=('square', 'cube'), label='test square & cube')
def test_square_cube(square, cube):
    assert square(-1) != cube(-1)
    assert square(0) == cube(0)
    assert square(1) == cube(1)
    assert square(2) != cube(2)


# but this test fails
@nbt.register(target='fail', label='test failure', score=1.5)
def test_fail(fail):
    fail()


# one may also return negative scores
@nbt.register(target='fail', label='test negative score')
def test_negative_score(fail):
    try:
        fail()
    except ValueError:
        return -0.5


# see if the import restrictions defined above work
@nbt.register(target='illegal_import', label='test import filter', score=.5, timeout=1.)
def test_illegal_import(illegal_import):
    with assert_raises(ImportError):
        illegal_import()


# this one will fail due to the global timeout
@nbt.register(target='sleep', label='test global timeout', score=1)
def test_sleep_1(sleep):
    sleep(.2)


# specifying the timeout locally will overwrite global settings
@nbt.register(target='sleep', label='test local timeout', score=1, timeout=.06)
def test_sleep_2(sleep):
    sleep(.08)


# inspecting source code of a function works as expected
@nbt.register(target='square', label='inspect source')
def test_inspect_source(square):
    import inspect
    print(inspect.getsource(square))


# Sometimes, the textual cells of a notebook are also of interest and should be included into the
# report. However, other than regular test cases, textual tests cannot be passed directly and are
# scored with NaN by default. Those test cases are supposed to be scored manually in audit mode.
nbt.register_comment(target=r'\*A1:\*', label='Bob', score=4)
nbt.register_comment(target=r'\*A2:\*', label='Douglas', score=1)
nbt.register_comment(target=r'\*A3:\*', label='???', score=2.5)

# Similarly, one may include figures into the report. Currently, PNG and SVG files are supported.
nbt.register_figure(target='plot.png', label='plot PNG')
nbt.register_figure(target='does_not_exist.png', label='file not found')


# One may also load raw files from artifacts for more advanced testing
@nbt.register(target='__ARTIFACTS__', label='raw artifact')
def test_raw_artifacts(artifacts):
    content = artifacts['fnord.txt'].decode('utf-8')
    return f'the following was read from a file: "{content}"'


# There are a few other special variables
@nbt.register(target=('__CONTEXT__', '__TEAM_MEMBERS__', '__COMMENTS__'), label='special variables')
def test_special_variables(context, team_members, comments):
    for member in team_members:
        print(f'Hello {member.first_name}')
    print()
    print(f'the tested notebook has {len(comments)} comments')
    assert context.is_dir()


# `execute` brings a simple comand line interface, e.g.:
# `$ autograde -vvv test demo/test.py demo/ --context demo/context --target /tmp`
if __name__ == '__main__':
    sys.exit(nbt.execute())
