# Standard library modules.
import io
import os
import re
import math
import time
import tarfile
from pathlib import Path
from hashlib import sha256
from functools import partial
from unittest import TestCase
from dataclasses import astuple
import importlib.util as import_util
from tempfile import TemporaryDirectory


# Third party modules.

# Local modules
import autograde
from autograde.util import project_root, cd
from autograde.helpers import assert_iter_eqal
from autograde.notebook_test import as_py_comment, exec_notebook, Result, Results, NotebookTestCase,\
    NotebookTest

# Globals and constants variables.
PROJECT_ROOT = project_root()


def float_equal(a, b):
    return math.isclose(a, b) or (math.isnan(a) and math.isnan(b))


assert_floats_equal = partial(assert_iter_eqal, comp=float_equal)


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


class TestResult(TestCase):
    def test_passed(self):
        for r in [Result(1, '', [], 0., 0., '', '', ''), Result(1, '', [], 1., 1., '', '', '')]:
            self.assertTrue(r.passed())
            self.assertFalse(r.failed())
            self.assertFalse(r.pending())

    def test_failed(self):
        for r in [Result(1, '', [], 0., 1., '', '', ''), Result(1, '', [], .5, 1., '', '', '')]:
            self.assertFalse(r.passed())
            self.assertTrue(r.failed())
            self.assertFalse(r.pending())

    def test_pending(self):
        for r in [Result(1, '', [], math.nan, 0., '', '', ''), Result(1, '', [], math.nan, 1., '', '', '')]:
            self.assertFalse(r.passed())
            self.assertFalse(r.failed())
            self.assertTrue(r.pending())


class TestResults(TestCase):
    def test_patch(self):
        results_a = Results('', '', '', [], [], [], [])
        results_b = Results('', '', '', [], [], [], [])
        assert_floats_equal((0, 0, 0, 0, 0., 0.), astuple(results_a.patch(results_b).summary()))

        results_a = Results('', '', '', [], [], [], [])
        results_b = Results('', '', '', [], [], [], [Result(1, '', [], 0., 1., '', '', '')])
        assert_floats_equal((1, 1, 0, 0, 0., 1.), astuple(results_a.patch(results_b).summary()))

        results_a = Results('', '', '', [], [], [], [Result(1, '', [], 0., 1., '', '', '')])
        results_b = Results('', '', '', [], [], [], [])
        assert_floats_equal((1, 1, 0, 0, 0., 1.), astuple(results_a.patch(results_b).summary()))

        results_a = Results('', '', '', [], [], [], [Result(1, '', [], 0., 1., '', '', '')])
        results_b = Results('', '', '', [], [], [], [Result(2, '', [], 1., 1., '', '', '')])
        assert_floats_equal((2, 1, 1, 0, 1., 2.), astuple(results_a.patch(results_b).summary()))

        results_a = Results('', '', '', [], [], [], [Result(1, '', [], 0., 1., '', '', '')])
        results_b = Results('', '', '', [], [], [], [Result(1, '', [], 1., 1., '', '', '')])
        assert_floats_equal((1, 0, 1, 0, 1., 1.), astuple(results_a.patch(results_b).summary()))

        results_a = Results('', '', '', [], [], [], [Result(1, '', [], 0., 1., '', '', '')])
        results_b = Results('', '', '', [], [], [], [Result(1, '', [], 0., 1., '', '', 'foo')])
        assert_floats_equal((1, 1, 0, 0, 0., 1.), astuple(results_a.patch(results_b).summary()))

        results_a = Results('', '', '', [], [], [], [Result(1, '', [], math.nan, 1., '', '', '')])
        results_b = Results('', '', '', [], [], [], [Result(1, '', [], 0., 1., '', '', '')])
        assert_floats_equal((1, 1, 0, 0, 0., 1.), astuple(results_a.patch(results_b).summary()))

        results_a = Results('', '', '', [], [], [], [Result(1, '', [], 0., 1., '', '', '')])
        results_b = Results('', '', '', [], [], [], [Result(1, '', [], math.nan, 1., '', '', '')])
        assert_floats_equal((1, 1, 0, 0, 0., 1.), astuple(results_a.patch(results_b).summary()))

        with self.assertRaises(ValueError):
            results_a = Results('', '', '0'*64, [], [], [], [])
            results_b = Results('', '', '1'*64, [], [], [], [])
            results_a.patch(results_b)

    def test_summary(self):
        results = Results('', '', '', [], [], [], [])
        assert_floats_equal((0, 0, 0, 0, 0, 0), astuple(results.summary()))

        results = Results('', '', '', [], [], [], [Result(1, '', [], 0., 1., '', '', '')])
        assert_floats_equal((1, 1, 0, 0, 0., 1.), astuple(results.summary()))

        results = Results('', '', '', [], [], [], [Result(2, '', [], 1., 1., '', '', '')])
        assert_floats_equal((1, 0, 1, 0, 1., 1.), astuple(results.summary()))

        results = Results('', '', '', [], [], [], [Result(3, '', [], math.nan, 1., '', '', '')])
        assert_floats_equal((1, 0, 0, 1, math.nan, 1.), astuple(results.summary()))


class TestNotebookTestCase(TestCase):
    def test_simple(self):
        def test(foo):
            self.assertEqual(foo, 42)

        tc = NotebookTestCase(test, target='foo')
        self.assertTupleEqual((1.0, 'ok'), tc(dict(foo=42)))

    def test_msg(self):
        tc = NotebookTestCase(lambda x: x, target='foo')
        self.assertTupleEqual((4.0, 'ok'), tc(dict(foo=4)))
        self.assertTupleEqual((4.2, 'ok'), tc(dict(foo=4.2)))
        self.assertTupleEqual((1.0, '42'), tc(dict(foo='42')))

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
        nbt = NotebookTest('')

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
        nbt = NotebookTest('')
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

    def test_set_import_filter(self):
        nbt = NotebookTest('')
        regex = r'networkx|requests'

        self.assertNotIn('IMPORT_FILTER', nbt._variables)

        nbt.set_import_filter(regex, blacklist=True)

        self.assertIn('IMPORT_FILTER', nbt._variables)
        self.assertEqual((re.compile(regex), True), nbt._variables['IMPORT_FILTER'])

    def test__grade_notebook(self):
        nb_path = PROJECT_ROOT.joinpath('demo', 'notebook.ipynb')
        t_path = PROJECT_ROOT.joinpath('demo', 'test.py')
        c_path = PROJECT_ROOT.joinpath('demo', 'context')

        with open(nb_path, mode='rb') as f:
            sha256_sum = sha256(f.read()).hexdigest()

        # load test as module
        spec = import_util.spec_from_file_location('nbtest', t_path)
        nbtest = import_util.module_from_spec(spec)
        spec.loader.exec_module(nbtest)

        with TemporaryDirectory() as path, cd(path):
            nbtest.nbt._grade_notebook(nb_path, context=c_path)

            rpath, *_ = Path(path).glob('results_*.tar.xz')

            with tarfile.open(rpath, mode='r') as tar:
                self.assertListEqual(sorted(tar.getnames())[1:], [
                    'artifacts',
                    'artifacts/bar.txt',
                    'artifacts/figures',
                    'artifacts/figures/fig_code_cell_3_1.png',
                    'artifacts/figures/fig_code_cell_8_1.png',
                    'artifacts/figures/fig_code_cell_8_2.png',
                    'artifacts/fnord.txt',
                    'code.py',
                    'notebook.ipynb',
                    'results.json'
                ])

                results = Results.from_json(tar.extractfile(tar.getmember('results.json')).read())

        self.assertEqual(results.version, autograde.__version__)

        self.assertEqual(results.checksum, sha256_sum)

        self.assertListEqual(results.excluded_artifacts, ['foo.txt'])

        assert_floats_equal(astuple(results.summary()), (10, 4, 4, 2, math.nan, 16))

    def test_execute(self):
        nb_path = PROJECT_ROOT.joinpath('demo', 'notebook.ipynb')
        t_path = PROJECT_ROOT.joinpath('demo', 'test.py')
        c_path = PROJECT_ROOT.joinpath('demo', 'context')

        # load test as module
        spec = import_util.spec_from_file_location('nbtest', t_path)
        nbtest = import_util.module_from_spec(spec)
        spec.loader.exec_module(nbtest)

        with TemporaryDirectory() as path, cd(path):
            self.assertEqual(4, nbtest.nbt.execute(args=(str(nb_path), '--context', str(c_path))))
