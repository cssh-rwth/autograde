import importlib
import inspect
import io
import math
import string
import sys
import time
from collections import Counter
from functools import partial
from itertools import combinations
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional
from unittest import TestCase
from zipfile import ZipFile

from autograde.helpers import assert_iter_eqal
from autograde.util import parse_bool, loglevel, project_root, snake_case, \
    camel_case, prune_join, capture_output, cd, cd_zip, StopWatch, WatchDog, \
    import_filter, run_with_timeout, exec_with_timeout, shadow_exec


class TestUtil(TestCase):
    def test_loglevel(self):
        self.assertEqual(40, loglevel(-1))
        self.assertEqual(40, loglevel(0))
        self.assertEqual(30, loglevel(1))
        self.assertEqual(20, loglevel(2))
        self.assertEqual(10, loglevel(3))
        self.assertEqual(10, loglevel(4))

    def test_project_root(self):
        self.assertEqual(project_root(), Path(__file__).parent.parent)

    def test_capture_output(self):
        stdout = sys.stdout
        stderr = sys.stderr

        with io.StringIO() as tmp_stdout, io.StringIO() as tmp_stderr:
            sys.stdout = tmp_stdout
            sys.stderr = tmp_stderr

            with io.StringIO() as out_buffer, io.StringIO() as err_buffer:
                with capture_output() as (bo, be):
                    self.assertIs(bo, tmp_stdout)
                    self.assertIs(be, tmp_stderr)
                    print('<o1>')
                    print('<e1>', file=sys.stderr)

                with capture_output(out_buffer) as (bo, be):
                    self.assertIs(bo, tmp_stdout)
                    self.assertIs(be, tmp_stderr)
                    print('<o2>')
                    print('<e2>', file=sys.stderr)

                with capture_output(tmp_stderr=err_buffer) as (bo, be):
                    self.assertIs(bo, tmp_stdout)
                    self.assertIs(be, tmp_stderr)
                    print('<o3>')
                    print('<e3>', file=sys.stderr)

                with capture_output(out_buffer, err_buffer) as (bo, be):
                    self.assertIs(bo, tmp_stdout)
                    self.assertIs(be, tmp_stderr)
                    print('<o4>')
                    print('<e4>', file=sys.stderr)

                self.assertEqual('<o2>\n<o4>\n', out_buffer.getvalue())
                self.assertEqual('<e3>\n<e4>\n', err_buffer.getvalue())

            self.assertEqual('<o1>\n<o3>\n', tmp_stdout.getvalue())
            self.assertEqual('<e1>\n<e2>\n', tmp_stderr.getvalue())

        sys.stdout = stdout
        sys.stderr = stderr


class TestStringFunctions(TestCase):
    def test_parse_bool(self):
        for v in (False, 0, 'F', 'f', 'false', 'FALSE', 'FalsE', 'nO', 'n'):
            self.assertFalse(parse_bool(v))

        for v in (True, 1, 'T', 't', 'true', 'TRUE', 'TruE', 'yEs', 'y'):
            self.assertTrue(parse_bool(v))

        for v in ('ja', 'nein', 'fnord', Exception, io):
            with self.assertRaises(ValueError):
                parse_bool(v)

    def test_snake_case(self):
        self.assertEqual('', snake_case(''))
        self.assertEqual('foo_bar', snake_case('fOO+BaR'))
        self.assertEqual('fnord_foo42_bar', snake_case('FnORD&FOo42=bar'))
        self.assertEqual('hein_blöd', snake_case(' hein blöd'))

    def test_camel_case(self):
        self.assertEqual('', camel_case(''))
        self.assertEqual('FooBar', camel_case('fOO+BaR'))
        self.assertEqual('FnordFoo42Bar', camel_case('FnORD&FOo42=bar'))
        self.assertEqual('HeinBlöd', camel_case(' hein blöd'))

    def test_prune_join(self):
        dictionary = [string.ascii_uppercase[i - 1] * i for i in range(1, 9)]

        for words in combinations(dictionary, 3):
            for mw in range(15):
                if mw < 11:
                    with self.assertRaises(ValueError):
                        prune_join(words, max_width=mw)
                else:
                    joined = prune_join(words, max_width=mw)
                    self.assertLessEqual(len(joined), mw)

                    counts = Counter(joined)
                    self.assertEqual(2, counts[','])

                    for word in words:
                        self.assertIn(word[0], counts)


class TestCD(TestCase):
    def test_cd(self):
        cwd = Path('')

        with cd(cwd.parent):
            self.assertEqual(Path(''), cwd.parent)

        self.assertEqual(Path(''), cwd)

    def test_cd_zip(self):
        path = Path('fnord.zip')

        with TemporaryDirectory() as dir_, cd(dir_):
            # write empty archive
            with cd_zip(path):
                pass

            self.assertTrue(path.exists())

            # read only mode is not supported
            with self.assertRaises(ValueError), cd_zip(path, mode='r'):
                pass

            # create some contents
            with cd_zip(path):
                with open('foo', mode='wt') as f:
                    f.write('FOO')

            # see if changes persisted
            with ZipFile(path) as zipf:
                self.assertSetEqual(set(zipf.namelist()), {'foo'})

            # extend archive
            with cd_zip(path, mode='a'):
                with open('bar', mode='wb') as f:
                    f.write(b'BAR')

            with ZipFile(path) as zipf:
                self.assertSetEqual(set(zipf.namelist()), {'foo', 'bar'})

            # overwrite archive
            with cd_zip(path, mode='w'):
                with open('bar', mode='wb') as f:
                    f.write(b'BAR')

            # see if changes persisted
            with ZipFile(path) as zipf:
                self.assertSetEqual(set(zipf.namelist()), {'bar'})


class TestStopWatch(TestCase):
    def test(self):
        assert_list_eq = partial(assert_iter_eqal, comp=lambda a, b: math.isclose(a, b, abs_tol=5e-3))

        sw = StopWatch()

        assert_list_eq([0.], sw.duration_abs())
        assert_list_eq([0.], sw.duration_rel())

        time.sleep(1e-2)
        self.assertEqual(1, sw.capture())

        time.sleep(2e-2)
        with sw:
            time.sleep(3e-2)

        time.sleep(4e-2)
        self.assertEqual(4, sw.capture())

        assert_list_eq([0., 1e-2, 3e-2, 6e-2, 1e-1], sw.duration_abs())
        assert_list_eq([0., 1e-2, 2e-2, 3e-2, 4e-2], sw.duration_rel())


class TestImportFilter(TestCase):

    def test_import_keyword_success(self):
        with import_filter(r'typ*', blacklist=True):
            import math
            _ = dir(math)

        with import_filter(r'typ*'):
            import types
            _ = dir(types)

    def test_import_keyword_failure(self):
        with self.assertRaises(ImportError), import_filter(r'typ*'):
            import math
            _ = dir(math)

        with self.assertRaises(ImportError), import_filter(r'typ*', blacklist=True):
            import types
            _ = dir(types)

    def test_import_keyword_alias_success(self):
        with import_filter(r'typ*', blacklist=True):
            import math as foo
            _ = dir(foo)

        with import_filter(r'typ*'):
            import types as foo
            _ = dir(foo)

    def test_import_keyword_alias_failure(self):
        with self.assertRaises(ImportError), import_filter(r'typ*'):
            import math as foo
            _ = dir(foo)

        with self.assertRaises(ImportError), import_filter(r'typ*', blacklist=True):
            import types as foo
            _ = dir(foo)

    def test_from_keyword_success(self):
        with import_filter(r'typ*', blacklist=True):
            from math import isclose
            _ = dir(isclose)

        with import_filter(r'typ*'):
            from types import SimpleNamespace
            _ = dir(SimpleNamespace)

    def test_from_keyword_failure(self):
        with self.assertRaises(ImportError), import_filter(r'typ*'):
            from math import isclose
            _ = dir(isclose)

        with self.assertRaises(ImportError), import_filter(r'typ*', blacklist=True):
            from types import SimpleNamespace
            _ = dir(SimpleNamespace)

    def test___import___keyword_success(self):
        for args in [(), (dict(),), (dict(), dict())]:
            with import_filter(r'typ*', blacklist=True):
                __import__('math', *args)

            with import_filter(r'typ*'):
                __import__('types', *args)

    def test___import_keyword___failure(self):
        for args in [(), (dict(),), (dict(), dict())]:
            with self.assertRaises(ImportError), import_filter(r'typ*'):
                __import__('math', *args)

            with self.assertRaises(ImportError), import_filter(r'typ*', blacklist=True):
                __import__('types', *args)

    def test_exec_success(self):
        for args in [(), (dict(),), (dict(), dict())]:
            with import_filter(r'typ*', blacklist=True):
                exec('import math', *args)

            with import_filter(r'typ*'):
                exec('import types', *args)

    def test_exec_failure(self):
        for args in [(), (dict(),), (dict(), dict())]:
            with self.assertRaises(ImportError), import_filter(r'typ*'):
                exec('import math', *args)

            with self.assertRaises(ImportError), import_filter(r'typ*', blacklist=True):
                exec('import types', *args)

    def test_importlib___import___success(self):
        for args in [(), (dict(),), (dict(), dict())]:
            with import_filter(r'typ*', blacklist=True):
                importlib.__import__('math', *args)

            with import_filter(r'typ*'):
                importlib.__import__('types', *args)

    def test_importlib___import___failure(self):
        for args in [(), (dict(),), (dict(), dict())]:
            with self.assertRaises(ImportError), import_filter(r'typ*'):
                importlib.__import__('math', *args)

            with self.assertRaises(ImportError), import_filter(r'typ*', blacklist=True):
                importlib.__import__('types', *args)

    def test_importlib_import_module_success(self):
        with import_filter(r'typ*', blacklist=True):
            importlib.import_module('math')

        with import_filter(r'typ*'):
            importlib.import_module('types')

    def test_importlib_import_module_failure(self):
        with self.assertRaises(ImportError), import_filter(r'typ*'):
            importlib.import_module('math')

        with self.assertRaises(ImportError), import_filter(r'typ*', blacklist=True):
            importlib.import_module('types')


def some_function(return_value=None, block_for: float = 0, *, exception: Optional[Exception] = None):
    start_t = time.monotonic()
    while time.monotonic() - start_t < block_for:
        time.sleep(1e-3)
    if exception is not None:
        raise exception
    return return_value


class TestRunWithTimeout(TestCase):
    def test_success(self):
        self.assertEqual(run_with_timeout(some_function, 1, args=(42,)), 42)

    def test_exception(self):
        with self.assertRaises(AssertionError):
            run_with_timeout(some_function, 1, kwargs=dict(exception=AssertionError()))

    def test_timeout(self):
        with self.assertRaises(TimeoutError):
            run_with_timeout(some_function, 1e-4, kwargs=dict(block_for=1.))


class TestExecWithTimeout(TestCase):
    def test_success(self):
        state = dict()
        exec_with_timeout('foo = 42', state, timeout=1)
        self.assertEqual(state['foo'], 42)

    def test_exception(self):
        with self.assertRaises(AssertionError):
            exec_with_timeout('assert False', timeout=1)

    def test_timeout(self):
        with self.assertRaises(TimeoutError):
            exec_with_timeout('while True:\n    pass', timeout=1e-4)


class TestShadowExec(TestCase):
    def test_success(self):
        state = dict()
        source = 'def foo():\n\treturn 42'
        with shadow_exec(source, state) as path:
            with open(path, mode='rt') as f:
                shadow_source = f.read()

            self.assertEqual(f'{source}\n', inspect.getsource(state['foo']))
            self.assertEqual(f'{source}\n', shadow_source)

        self.assertEqual(42, state['foo']())

    def test_inspect(self):
        state = dict()
        source = 'def foo():\n    return 42'

        with shadow_exec(source, state) as path:
            with open(path, mode='rt') as f:
                shadow_source = f.read()

            self.assertEqual(f'{source}\n', inspect.getsource(state['foo']))
            self.assertEqual(f'{source}\n', shadow_source)

        with self.assertRaises(OSError):
            inspect.getsource(state['foo'])

    def test_exception(self):
        state = dict()
        with shadow_exec('def bar():\n\tassert False', state):
            pass

        with self.assertRaises(AssertionError):
            state['bar']()

    def test_timeout(self):
        with self.assertRaises(TimeoutError):
            with shadow_exec('while True:\n    pass', timeout=1e-4):
                pass


class TestWatchDog(TestCase):

    def test_init(self):
        with TemporaryDirectory() as tmp:
            f_1 = Path(tmp).joinpath('foo')
            f_1.touch()

            with self.assertRaises(ValueError):
                WatchDog(f_1)

    def test_file_presence(self):
        with TemporaryDirectory() as tmp:
            f_1 = Path(tmp).joinpath('foo')
            f_2 = Path(tmp).joinpath('bar', 'foo')

            wd = WatchDog(tmp)
            self.assertSetEqual(set(wd.list_changed()), set())
            self.assertSetEqual(set(wd.list_unchanged()), set())

            f_1.touch()
            self.assertSetEqual(set(wd.list_changed()), {f_1})
            self.assertSetEqual(set(wd.list_unchanged()), set())

            wd.reload()
            self.assertSetEqual(set(wd.list_changed()), set())
            self.assertSetEqual(set(wd.list_unchanged()), {f_1})

            f_2.parent.mkdir()
            f_2.touch()
            self.assertSetEqual(set(wd.list_changed()), {f_2})
            self.assertSetEqual(set(wd.list_unchanged()), {f_1})

    def test_file_access(self):
        with TemporaryDirectory() as tmp:
            f_1 = Path(tmp).joinpath('foo')
            f_2 = Path(tmp).joinpath('bar', 'foo')

            f_1.touch()
            f_2.parent.mkdir()
            f_2.touch()

            wd = WatchDog(tmp)

            self.assertSetEqual(set(wd.list_changed()), set())
            self.assertSetEqual(set(wd.list_unchanged()), {f_1, f_2})

            f_1.open(mode='r').close()

            self.assertSetEqual(set(wd.list_changed()), set())
            self.assertSetEqual(set(wd.list_unchanged()), {f_1, f_2})

            with f_1.open(mode='w') as f:
                f.write('hello world')

            self.assertSetEqual(set(wd.list_changed()), {f_1})
            self.assertSetEqual(set(wd.list_unchanged()), {f_2})
