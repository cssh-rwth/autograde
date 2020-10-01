import importlib.util as import_util
import math
import os
import re
import tarfile
import time
from dataclasses import astuple
from functools import partial
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import autograde
from autograde.helpers import assert_iter_eqal
from autograde.notebook_test import Result, Results, NotebookTestCase, \
    NotebookTest
from autograde.util import project_root, cd, capture_output

PROJECT_ROOT = project_root()


def float_equal(a, b):
    return math.isclose(a, b) or (math.isnan(a) and math.isnan(b))


assert_floats_equal = partial(assert_iter_eqal, comp=float_equal)


class TestResult(TestCase):
    def test_passed(self):
        for r in [Result('1', '', [], 0., 0., [], '', ''), Result('1', '', [], 1., 1., [], '', '')]:
            self.assertTrue(r.passed())
            self.assertFalse(r.failed())
            self.assertFalse(r.pending())

    def test_failed(self):
        for r in [Result('1', '', [], 0., 1., [], '', ''), Result('1', '', [], .5, 1., [], '', '')]:
            self.assertFalse(r.passed())
            self.assertTrue(r.failed())
            self.assertFalse(r.pending())

    def test_pending(self):
        for r in [Result('1', '', [], math.nan, 0., [], '', ''), Result('1', '', [], math.nan, 1., [], '', '')]:
            self.assertFalse(r.passed())
            self.assertFalse(r.failed())
            self.assertTrue(r.pending())


class TestResults(TestCase):
    def test_patch(self):
        results_a = Results('', '', '', [], [], [], [])
        results_b = Results('', '', '', [], [], [], [])
        patch_result = results_a.patch(results_b)
        assert_floats_equal((0, 0, 0, 0, 0., 0.), astuple(patch_result.summary()))
        self.assertListEqual([('', results_b.timestamp, [])], patch_result.applied_patches)

        results_a = Results('', '', '', [], [], [], [])
        results_b = Results('', '', '', [], [], [], [Result('1', '', [], 0., 1., [], '', '')])
        patch_result = results_a.patch(results_b)
        assert_floats_equal((1, 1, 0, 0, 0., 1.), astuple(patch_result.summary()))
        self.assertListEqual([('', results_b.timestamp, ['1'])], patch_result.applied_patches)

        results_a = Results('', '', '', [], [], [], [Result('1', '', [], 0., 1., [], '', '')])
        results_b = Results('', '', '', [], [], [], [])
        patch_result = results_a.patch(results_b)
        assert_floats_equal((1, 1, 0, 0, 0., 1.), astuple(patch_result.summary()))
        self.assertListEqual([('', results_b.timestamp, [])], patch_result.applied_patches)

        results_a = Results('', '', '', [], [], [], [Result('1', '', [], 0., 1., [], '', '')])
        results_b = Results('', '', '', [], [], [], [Result('2', '', [], 1., 1., [], '', '')])
        patch_result = results_a.patch(results_b)
        assert_floats_equal((2, 1, 1, 0, 1., 2.), astuple(patch_result.summary()))
        self.assertListEqual([('', results_b.timestamp, ['2'])], patch_result.applied_patches)

        results_a = Results('', '', '', [], [], [], [Result('1', '', [], 0., 1., [], '', '')])
        results_b = Results('', '', '', [], [], [], [Result('1', '', [], 1., 1., [], '', '')])
        patch_result = results_a.patch(results_b)
        assert_floats_equal((1, 0, 1, 0, 1., 1.), astuple(patch_result.summary()))
        self.assertListEqual([('', results_b.timestamp, ['1'])], patch_result.applied_patches)

        results_a = Results('', '', '', [], [], [], [Result('1', '', [], 0., 1., [], '', '')])
        results_b = Results('', '', '', [], [], [], [Result('1', '', [], 0., 1., [], '', 'foo')])
        patch_result = results_a.patch(results_b)
        assert_floats_equal((1, 1, 0, 0, 0., 1.), astuple(patch_result.summary()))
        self.assertListEqual([('', results_b.timestamp, ['1'])], patch_result.applied_patches)

        results_a = Results('', '', '', [], [], [], [Result('1', '', [], math.nan, 1., [], '', '')])
        results_b = Results('', '', '', [], [], [], [Result('1', '', [], 0., 1., [], '', '')])
        patch_result = results_a.patch(results_b)
        assert_floats_equal((1, 1, 0, 0, 0., 1.), astuple(patch_result.summary()))
        self.assertListEqual([('', results_b.timestamp, ['1'])], patch_result.applied_patches)

        results_a = Results('', '', '', [], [], [], [Result('1', '', [], 0., 1., [], '', '')])
        results_b = Results('', '', '', [], [], [], [Result('1', '', [], math.nan, 1., [], '', '')])
        patch_result = results_a.patch(results_b)
        assert_floats_equal((1, 1, 0, 0, 0., 1.), astuple(patch_result.summary()))
        self.assertListEqual([('', results_b.timestamp, [])], patch_result.applied_patches)

        results_a = Results('a', '', '', [], [], [], [Result('1', '', [], 0., 1., [], '', '')])
        results_b = Results('b', '', '', [], [], [], [Result('2', '', [], 1., 1., [], '', '')])
        results_c = Results('c', '', '', [], [], [], [Result('3', '', [], 2., 4., [], '', ''),
                                                      Result('4', '', [], 4., 8., [], '', '')])
        patch_result = results_a.patch(results_b).patch(results_c)
        assert_floats_equal((4, 3, 1, 0, 7., 14.), astuple(patch_result.summary()))
        self.assertListEqual([('b', results_b.timestamp, ['2']),
                              ('c', results_c.timestamp, ['3', '4'])], patch_result.applied_patches)

        with self.assertRaises(ValueError):
            results_a = Results('', '', '0'*64, [], [], [], [])
            results_b = Results('', '', '1'*64, [], [], [], [])
            results_a.patch(results_b)

    def test_summary(self):
        results = Results('', '', '', [], [], [], [])
        assert_floats_equal((0, 0, 0, 0, 0, 0), astuple(results.summary()))

        results = Results('', '', '', [], [], [], [Result('1', '', [], 0., 1., [], '', '')])
        assert_floats_equal((1, 1, 0, 0, 0., 1.), astuple(results.summary()))

        results = Results('', '', '', [], [], [], [Result('2', '', [], 1., 1., [], '', '')])
        assert_floats_equal((1, 0, 1, 0, 1., 1.), astuple(results.summary()))

        results = Results('', '', '', [], [], [], [Result('3', '', [], math.nan, 1., [], '', '')])
        assert_floats_equal((1, 0, 0, 1, math.nan, 1.), astuple(results.summary()))


class TestNotebookTestCase(TestCase):
    def test_simple(self):
        def test(foo):
            self.assertEqual(foo, 42)

        tc = NotebookTestCase(test, target='foo', label='')
        self.assertTupleEqual((1.0, 'ok'), tc(dict(foo=42)))

    def test_msg(self):
        tc = NotebookTestCase(lambda x: x, target='foo', label='')
        self.assertTupleEqual((4.0, 'ok'), tc(dict(foo=4)))
        self.assertTupleEqual((4.2, 'ok'), tc(dict(foo=4.2)))
        self.assertTupleEqual((1.0, '42'), tc(dict(foo='42')))

    def test_unknown_target(self):
        def test():
            pass

        tc = NotebookTestCase(test, target='foo', label='')
        with open(os.devnull, 'w') as stderr, capture_output(tmp_stderr=stderr):
            s, _ = tc({})
        self.assertEqual(s, 0.)

    def test_multi_target(self):
        def test(foo, bar):
            self.assertEqual(foo, 42)
            self.assertEqual(bar, 1337)

        tc = NotebookTestCase(test, target=('foo', 'bar'), label='')
        self.assertTupleEqual((1.0, 'ok'), tc(dict(foo=42, bar=1337)))

    def test_score(self):
        def test(fail):
            assert not fail

        tc = NotebookTestCase(test, target='fail', label='')
        s, _ = tc(dict(fail=False))
        self.assertEqual(s, 1.0)
        self.assertEqual(tc.score, 1.0)

        tc = NotebookTestCase(test, target='fail', label='')
        with open(os.devnull, 'w') as stderr, capture_output(tmp_stderr=stderr):
            s, _ = tc(dict(fail=True))
        self.assertEqual(s, 0.)
        self.assertEqual(tc.score, 1.0)

        tc = NotebookTestCase(test, target='fail', label='', score=2)
        s, _ = tc(dict(fail=False))
        self.assertEqual(s, 2.0)
        self.assertEqual(tc.score, 2.0)

    def test_label(self):
        def test():
            pass

        tc = NotebookTestCase(test, target='foo', label='bar')
        self.assertEqual('bar', tc.label)

    def test_timeout(self):
        def test(delay):
            time.sleep(delay)

        tc = NotebookTestCase(test, target='delay', label='')
        s, _ = tc(dict(delay=.1))
        self.assertEqual(s, 1.)

        tc = NotebookTestCase(test, target='delay', label='', timeout=.05)
        with open(os.devnull, 'w') as stderr, capture_output(tmp_stderr=stderr):
            s, _ = tc(dict(delay=.1))
        self.assertEqual(s, 0.)


class TestNotebookTest(TestCase):
    def test_register(self):
        nbt = NotebookTest('')

        def test(foo):
            self.assertEqual(42, foo)

        decorator = nbt.register(target='foo', label='bar', score=2, timeout=1)
        self.assertEqual(0, len(nbt))

        case = decorator(test)
        self.assertEqual(1, len(nbt))
        self.assertEqual(case.label, 'bar')
        self.assertEqual(case.timout, 1)
        self.assertEqual(case.score, 2)
        self.assertTupleEqual(case.targets, ('foo',))

        self.assertTupleEqual((2.0, 'ok'), case(dict(foo=42)))

        with self.assertRaises(ValueError):
            nbt.register(target='foo', label='bar')(lambda: None)

    def test_register_decorator(self):
        nbt = NotebookTest('')
        self.assertEqual(0, len(nbt))

        @nbt.register(target='foo', label='bar', score=2, timeout=1)
        def test(foo):
            self.assertEqual(42, foo)

        self.assertEqual(1, len(nbt))
        self.assertEqual(test.label, 'bar')
        self.assertEqual(test.timout, 1)
        self.assertEqual(test.score, 2)
        self.assertTupleEqual(test.targets, ('foo',))

        self.assertTupleEqual((2.0, 'ok'), test(dict(foo=42)))

        with self.assertRaises(ValueError):
            @nbt.register(target='foo', label='bar')
            def test():
                pass

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
                    'artifacts/figures/fig_nb_3_1.png',
                    'artifacts/figures/fig_nb_8_1.png',
                    'artifacts/figures/fig_nb_8_2.png',
                    'artifacts/figures/fig_nb_9_1.png',
                    'artifacts/fnord.txt',
                    'artifacts/plot.png',
                    'code.py',
                    'notebook.ipynb',
                    'results.json'
                ])

                results = Results.from_json(tar.extractfile(tar.getmember('results.json')).read())

        self.assertEqual(results.version, autograde.__version__)

        self.assertEqual(results.checksum, sha256_sum)

        self.assertListEqual(results.excluded_artifacts, ['foo.txt'])

        assert_floats_equal(astuple(results.summary()), (14, 5, 6, 3, math.nan, 20))

    def test_execute(self):
        nb_path = PROJECT_ROOT.joinpath('demo', 'notebook.ipynb')
        t_path = PROJECT_ROOT.joinpath('demo', 'test.py')
        c_path = PROJECT_ROOT.joinpath('demo', 'context')

        # load test as module
        spec = import_util.spec_from_file_location('nbtest', t_path)
        nbtest = import_util.module_from_spec(spec)
        spec.loader.exec_module(nbtest)

        with TemporaryDirectory() as path, cd(path):
            self.assertEqual(5, nbtest.nbt.execute(args=(str(nb_path), '--context', str(c_path))))
