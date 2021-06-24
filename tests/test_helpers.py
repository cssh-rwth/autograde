from unittest import TestCase

from autograde.helpers import assert_equal, assert_iter_eqal, assert_is, assert_isclose, \
    assert_raises


class TestHelpers(TestCase):
    def test_assert_equal(self):
        assert_equal(42, 42)

        with self.assertRaises(AssertionError):
            assert_equal(13, 37)

    def test_assert_iter_equal(self):
        assert_iter_eqal((1, 2, 3), (1, 2, 3))
        assert_iter_eqal((1, 2, 3), [1, 2, 3])
        assert_iter_eqal((1, 2, 3), (1, 4, 3), comp=lambda _, __: True)

        with self.assertRaises(AssertionError):
            assert_iter_eqal((1, 2, 3), (1, 2, 3, 4))
            assert_iter_eqal((1, 2, 3), (1, 4, 3))
            assert_iter_eqal((1, 2, 3), (1, 2, 3), comp=lambda _, __: False)

    def test_assert_is(self):
        obj = object()
        assert_equal(obj, obj)

        with self.assertRaises(AssertionError):
            assert_is(object(), object())

    def test_assert_isclose(self):
        assert_isclose(10.09, 10, abs_tol=.1)

        with self.assertRaises(AssertionError):
            assert_isclose(10.11, 10, abs_tol=.1)

    def test_assert_raises(self):
        with assert_raises():
            assert False

        with assert_raises(AssertionError):
            assert False

        with self.assertRaises(AssertionError):
            with assert_raises():
                pass

        with self.assertRaises(ValueError):
            with assert_raises(AssertionError):
                raise ValueError
