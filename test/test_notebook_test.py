# Standard library modules.
import io
import os
import json
import time
import tarfile
from hashlib import md5
from pathlib import Path
from unittest import TestCase
import importlib.util as import_util
from tempfile import TemporaryDirectory


# Third party modules.

# Local modules
import autograde
from autograde.util import project_root, cd
from autograde.notebook_test import as_py_comment, exec_notebook, NotebookTestCase, NotebookTest

# Globals and constants variables.
PROJECT_ROOT = project_root()


class TestFunctions(TestCase):
    def test_as_py_comment(self):
        self.assertEqual('', as_py_comment(''))
        self.assertEqual('# foo', as_py_comment('foo'))
        self.assertEqual('# foo\n# bar', as_py_comment('foo\nbar'))

    def test_exec_notebook(self):
        nb_path = PROJECT_ROOT.joinpath('demo', 'notebook.ipynb')
        with open(nb_path, mode='rt') as f:
            nb = f.read()

        with TemporaryDirectory() as path, cd(path):
            # forward errors raised in notebook
            with self.assertRaises(FileNotFoundError):
                with io.StringIO(nb) as nb_buffer, open(os.devnull, 'w') as stdout:
                    exec_notebook(nb_buffer, file=stdout)

            # cell timeout
            with self.assertRaises(TimeoutError):
                with io.StringIO(nb) as nb_buffer, open(os.devnull, 'w') as stdout:
                    exec_notebook(nb_buffer, file=stdout, cell_timeout=0.05)

            # ignore errors
            with io.StringIO(nb) as nb_buffer, io.StringIO() as stdout:
                state = exec_notebook(nb_buffer, file=stdout, ignore_errors=True)
                stdout = stdout.getvalue()

        self.assertIn('__IB_FLAG__', state)
        self.assertIn('__IA_FLAG__', state)
        self.assertEqual(state.get('SOME_CONSTANT'), 42)
        self.assertIn('this goes to stdout', stdout)


class TestNotebookTestCase(TestCase):
    def test_simple(self):
        def test(foo):
            self.assertEqual(foo, 42)

        tc = NotebookTestCase(test, target='foo')
        self.assertTupleEqual((1.0, 'ok'), tc(dict(foo=42)))

    def test_unknown_target(self):
        def test():
            pass

        tc = NotebookTestCase(test, target='foo')
        s, _ = tc({})
        self.assertEqual(s, 0.)

    def test_multi_target(self):
        def test(foo, bar):
            self.assertEqual(foo, 42)
            self.assertEqual(bar, 1337)

        tc = NotebookTestCase(test, target=('foo', 'bar'))
        self.assertTupleEqual((1.0, 'ok'), tc(dict(foo=42, bar=1337)))

    def test_score(self):
        def test(fail):
            assert not fail

        tc = NotebookTestCase(test, target='fail')
        s, _ = tc(dict(fail=False))
        self.assertEqual(s, 1.0)
        self.assertEqual(tc.score, 1.0)

        tc = NotebookTestCase(test, target='fail')
        s, _ = tc(dict(fail=True))
        self.assertEqual(s, 0.)
        self.assertEqual(tc.score, 1.0)

        tc = NotebookTestCase(test, target='fail', score=2)
        s, _ = tc(dict(fail=False))
        self.assertEqual(s, 2.0)
        self.assertEqual(tc.score, 2.0)

    def test_label(self):
        def test():
            pass

        tc = NotebookTestCase(test, target='foo')
        self.assertEqual('', tc.label)

        tc = NotebookTestCase(test, target='foo', label='bar')
        self.assertEqual('bar', tc.label)

    def test_timeout(self):
        def test(delay):
            time.sleep(delay)

        tc = NotebookTestCase(test, target='delay')
        s, _ = tc(dict(delay=.1))
        self.assertEqual(s, 1.)

        tc = NotebookTestCase(test, target='delay', timeout=.05)
        s, _ = tc(dict(delay=.1))
        self.assertEqual(s, 0.)


class TestNotebookTest(TestCase):
    def test_register(self):
        nbt = NotebookTest()

        def test(foo):
            self.assertEqual(42, foo)

        decorator = nbt.register(target='foo', score=2, label='bar', timeout_=1)
        self.assertEqual(0, len(nbt))

        case = decorator(test)
        self.assertEqual(1, len(nbt))
        self.assertEqual(case.label, 'bar')
        self.assertEqual(case.timout, 1)
        self.assertEqual(case.score, 2)
        self.assertTupleEqual(case.targets, ('foo',))

        self.assertTupleEqual((2.0, 'ok'), case(dict(foo=42)))

    def test_register_decorator(self):
        nbt = NotebookTest()
        self.assertEqual(0, len(nbt))

        @nbt.register(target='foo', score=2, label='bar', timeout_=1)
        def test(foo):
            self.assertEqual(42, foo)

        self.assertEqual(1, len(nbt))
        self.assertEqual(test.label, 'bar')
        self.assertEqual(test.timout, 1)
        self.assertEqual(test.score, 2)
        self.assertTupleEqual(test.targets, ('foo',))

        self.assertTupleEqual((2.0, 'ok'), test(dict(foo=42)))

    def test_grade_notebook(self):
        nb_path = PROJECT_ROOT.joinpath('demo', 'notebook.ipynb')
        t_path = PROJECT_ROOT.joinpath('demo', 'test.py')

        with open(nb_path, mode='rb') as f:
            md5_sum = md5(f.read()).hexdigest()

        # load test as module
        spec = import_util.spec_from_file_location('nbtest', t_path)
        nbtest = import_util.module_from_spec(spec)
        spec.loader.exec_module(nbtest)

        with TemporaryDirectory() as path, cd(path):
            nbtest.nbt.grade_notebook(nb_path, context=PROJECT_ROOT.joinpath('demo', 'context'))

            rpath, *_ = Path(path).glob('results_*.tar.xz')

            with tarfile.open(rpath, mode='r') as tar:
                self.assertListEqual(sorted(tar.getnames())[1:], [
                    'artifacts.tar.xz',
                    'code.py',
                    'notebook.ipynb',
                    'report.rst',
                    'test_results.csv',
                    'test_results.json'
                ])

                results = json.load(tar.extractfile(tar.getmember('test_results.json')))

        self.assertEqual(results['autograde_version'], autograde.__version__)

        self.assertEqual(results['checksum']['md5sum'], md5_sum)

        self.assertEqual(results['summary']['tests'], 6)
        self.assertEqual(results['summary']['passed'], 3)
        self.assertEqual(results['summary']['score'], 4.5)
        self.assertEqual(results['summary']['score_max'], 8.0)

        for key in ['orig_file', 'team_members', 'test_cases', 'results']:
            self.assertIn(key, results)
