import math
from contextlib import contextmanager
from typing import Any, Callable, Optional


def _msg(x: Any, exp: Any) -> str:
    return f'got "{str(x):.50}", expected "{str(exp):.50}"'


def assert_equal(x: Any, exp: Any, msg: Optional[str] = None):
    assert x == exp, msg or _msg(x, exp)


def assert_iter_eqal(x: Any, exp: Any, comp: Callable[[Any, Any], bool] = lambda a, b: a == b,
                     msg: Optional[str] = None):
    la = tuple(x)
    lb = tuple(exp)

    assert len(la) == len(lb), f'given iterables are of different size: len({la}) != len({lb})'

    for a, b in zip(la, lb):
        assert comp(a, b), msg or _msg(la, lb)


def assert_is(x: Any, exp: Any, msg: Optional[str] = None):
    assert x is exp, msg or _msg(x, exp)


def assert_isclose(x: Any, exp: Any, msg: Optional[str] = None, **kwargs):
    assert math.isclose(x, exp, **kwargs), msg or _msg(x, exp)


@contextmanager
def assert_raises(*exceptions: Exception):
    failed = False
    exceptions = exceptions or (Exception,)

    try:
        yield None
    except exceptions:
        failed = True

    assert failed, f'none of {exceptions} were risen'
