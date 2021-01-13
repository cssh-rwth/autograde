import builtins
import re
from collections import defaultdict
from unittest import TestCase

from autograde.helpers import assert_equal, assert_iter_eqal, assert_is, assert_isclose, \
    assert_raises, import_hook, import_filter


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

    def test_import_hook(self):
        counts = defaultdict(lambda: 0)
        _import = builtins.__import__

        def count(target, *args):
            counts[target] += 1
            return _import(target, *args)

        with import_hook(count):
            import importlib
            import types as _
            from types import SimpleNamespace as _
            __import__('types')
            __import__('types', dict())
            __import__('types', dict(), dict())
            exec('import types')
            exec('import types', dict())
            exec('import types', dict(), dict())
            importlib.__import__('types')
            importlib.__import__('types', dict())
            importlib.__import__('types', dict(), dict())

            with self.assertRaises(ImportError):
                importlib.import_module('types')

        self.assertEqual(dict(types=11, importlib=1), dict(counts))

    def test_import_filter(self):
        with import_filter(r'type.*'):
            __import__('types')

        with self.assertRaises(ImportError):
            with import_filter(r'type.*'):
                __import__('typing')

        with self.assertRaises(ImportError):
            with import_filter(re.compile(r'type.*'), blacklist=True):
                __import__('types')

        with import_filter(re.compile(r'type.*'), blacklist=True):
            __import__('typing')
