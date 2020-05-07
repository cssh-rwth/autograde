# Standard library modules.
import re
import math
import builtins
from unittest import mock
from contextlib import contextmanager, ExitStack

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


@contextmanager
def import_hook(callback):
    def dummy(*_):
        raise ImportError(f'calling `import_module` is not allowed within `import_hook` context')

    with ExitStack() as es:
        es.enter_context(mock.patch('importlib.import_module', side_effect=dummy))
        es.enter_context(mock.patch('importlib.__import__', side_effect=callback))
        es.enter_context(mock.patch('builtins.__import__', side_effect=callback))
        yield None


@contextmanager
def import_filter(regex, flags=0, blacklist=False):
    pattern = re.compile(regex, flags)
    _import = builtins.__import__

    def callback(target, *args):
        matches = pattern.search(target) is not None

        if matches and blacklist:
            raise ImportError(f'\'{target}\' is blacklisted')

        if not matches and not blacklist:
            raise ImportError(f'\'{target}\' is not whitelisted')

        return _import(target, *args)

    with import_hook(callback):
        yield None

