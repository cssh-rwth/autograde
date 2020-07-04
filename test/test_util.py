# Standard library modules.
import io
import sys
import time
import string
from pathlib import Path
from unittest import TestCase
from collections import Counter
from itertools import combinations
from tempfile import TemporaryDirectory

# Third party modules.

# Local modules
from autograde.util import parse_bool, loglevel, project_root, snake_case,\
    camel_case, prune_join, capture_output, cd, mount_tar, timeout

# Globals and constants variables.


class TestUtil(TestCase):
    def test_parse_bool(self):
        for v in (False, 0, 'F', 'f', 'false', 'FALSE',  'FalsE', 'nO', 'n'):
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
        dictionary = [string.ascii_uppercase[i-1] * i for i in range(1, 9)]

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
        cwd = Path('.')

        with cd(cwd.parent):
            self.assertEqual(Path('.'), cwd.parent)

        self.assertEqual(Path('.'), cwd)

    def test_tar_mount(self):
        path = Path('fnord.tar')

        with TemporaryDirectory() as dir, cd(dir):
            # reading non existing archive fails
            with self.assertRaises(FileNotFoundError):
                with mount_tar(path):
                    pass

            self.assertFalse(path.exists())

            # write empty archive
            with mount_tar(path, mode='w'):
                pass

            self.assertTrue(path.exists())

            # append some contents
            with mount_tar(path, mode='a') as tar, cd(tar):
                with open('foo', mode='wt') as f:
                    f.write('FOO')

            # see if changes persisted
            with mount_tar(path) as tar, cd(tar):
                self.assertTrue(Path('foo').exists())
                self.assertFalse(Path('bar').exists())

            # overwrite archive
            with mount_tar(path, mode='w') as tar, cd(tar):
                with open('bar', mode='wb') as f:
                    f.write(b'BAR')

            # see if changes persisted
            with mount_tar(path) as tar, cd(tar):
                self.assertFalse(Path('foo').exists())
                self.assertTrue(Path('bar').exists())

    def test_timeout(self):
        self.assertIsNone(sys.gettrace())

        with timeout(0):
            self.assertIsNone(sys.gettrace())
            time.sleep(.01)

        self.assertIsNone(sys.gettrace())

        with self.assertRaises(TimeoutError):
            with timeout(.01):
                self.assertIsNotNone(sys.gettrace())
                time.sleep(.02)

        self.assertIsNone(sys.gettrace())
