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
from unittest import TestCase
from zipfile import ZipFile

from autograde.helpers import assert_iter_eqal
from autograde.util import parse_bool, loglevel, project_root, snake_case, \
    camel_case, prune_join, capture_output, cd, cd_zip, StopWatch, deadline, \
    WatchDog


class TestUtil(TestCase):
    def test_parse_bool(self):
        for v in (False, 0, 'F', 'f', 'false', 'FALSE', 'FalsE', 'nO', 'n'):
            self.assertFalse(parse_bool(v))

        for v in (True, 1, 'T', 't', 'true', 'TRUE', 'TruE', 'yEs', 'y'):
            self.assertTrue(parse_bool(v))

        for v in ('ja', 'nein', 'fnord', Exception, io):
            with self.assertRaises(ValueError):
                parse_bool(v)

    def test_loglevel(self):
        self.assertEqual(40, loglevel(-1))
        self.assertEqual(40, loglevel(0))
        self.assertEqual(30, loglevel(1))
        self.assertEqual(20, loglevel(2))
        self.assertEqual(10, loglevel(3))
        self.assertEqual(10, loglevel(4))

    def test_project_root(self):
        self.assertEqual(project_root(), Path(__file__).parent.parent)

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

    def test_capture_output(self):
        stdout = sys.stdout
        stderr = sys.stderr

        with io.StringIO() as tmpstdout, io.StringIO() as tmpstderr:
            sys.stdout = tmpstdout
            sys.stderr = tmpstderr

            with io.StringIO() as out_buffer, io.StringIO() as err_buffer:
                with capture_output() as (bo, be):
                    self.assertIs(bo, tmpstdout)
                    self.assertIs(be, tmpstderr)

                    print('<o1>')
                    print('<e1>', file=sys.stderr)

                with capture_output(out_buffer) as (bo, be):
                    self.assertIs(bo, tmpstdout)
                    self.assertIs(be, tmpstderr)

                    print('<o2>')
                    print('<e2>', file=sys.stderr)

                with capture_output(tmp_stderr=err_buffer) as (bo, be):
                    self.assertIs(bo, tmpstdout)
                    self.assertIs(be, tmpstderr)

                    print('<o3>')
                    print('<e3>', file=sys.stderr)

                with capture_output(out_buffer, err_buffer) as (bo, be):
                    self.assertIs(bo, tmpstdout)
                    self.assertIs(be, tmpstderr)

                    print('<o4>')
                    print('<e4>', file=sys.stderr)

                self.assertEqual('<o2>\n<o4>\n', out_buffer.getvalue())
                self.assertEqual('<e3>\n<e4>\n', err_buffer.getvalue())

            self.assertEqual('<o1>\n<o3>\n', tmpstdout.getvalue())
            self.assertEqual('<e1>\n<e2>\n', tmpstderr.getvalue())

        sys.stdout = stdout
        sys.stderr = stderr

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

    def test_stopwatch(self):
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

    def test_deadline(self):
        self.assertIsNone(sys.gettrace())

        with deadline(0):
            self.assertIsNone(sys.gettrace())
            time.sleep(.01)

        self.assertIsNone(sys.gettrace())

        with self.assertRaises(TimeoutError):
            with deadline(.01):
                self.assertIsNotNone(sys.gettrace())
                time.sleep(.02)

        self.assertIsNone(sys.gettrace())


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
