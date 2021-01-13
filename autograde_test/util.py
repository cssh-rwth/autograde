import math
from functools import partial

from autograde.helpers import assert_iter_eqal


def float_equal(a, b):
    return math.isclose(a, b) or (math.isnan(a) and math.isnan(b))


assert_floats_equal = partial(assert_iter_eqal, comp=float_equal)
