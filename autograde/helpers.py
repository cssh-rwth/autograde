# Standard library modules.
import math
from contextlib import contextmanager

# Third party modules.

# Local modules

# Globals and constants variables.


def _msg(x, exp):
    return f'got "{str(x):.50}", expected "{str(exp):.50}"'


def assert_equal(x, exp, msg=None):
    assert x == exp, msg or _msg(x, exp)


def assert_is(x, exp, msg=None):
    assert x is exp, msg or _msg(x, exp)


def assert_isclose(x, exp, msg=None, **kwargs):
    assert math.isclose(x, exp, **kwargs), msg or _msg(x, exp)


@contextmanager
def assert_raises(*exceptions):
    failed = False
    exceptions = exceptions or (Exception,)

    try:
        yield None

    except exceptions:
        failed = True

    assert failed, f'none of {exceptions} were risen'

