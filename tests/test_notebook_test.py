import importlib.util as import_util
import math
import os
import re
import time
from dataclasses import astuple
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from zipfile import ZipFile

import autograde
from autograde.notebook_test import UnitTest, \
    NotebookTest
from autograde.test_result import NotebookTestResult
from autograde.util import project_root, cd, capture_output
from tests.util import assert_floats_equal

PROJECT_ROOT = project_root()


class TestUnitTest(TestCase):
    def test_simple(self):
        def test_function(foo):
            assert foo == 42

        ut = UnitTest(test_function, target='foo', label='')
        self.assertTupleEqual((1.0, '✅ passed'), ut(dict(foo=42)))

    def test_msg_passed(self):
        ut = UnitTest(lambda x: x, target='foo', label='', score=4.)
        self.assertTupleEqual((4.0, '✅ passed'), ut(dict(foo=5)))

    def test_msg_partially_passed(self):
        ut = UnitTest(lambda x: x, target='foo', label='', score=4.)
        self.assertTupleEqual((3.0, '¯\\_(ツ)_/¯ partially passed'), ut(dict(foo=3.0)))

    def test_msg_custom_message(self):
        ut = UnitTest(lambda x: x, target='foo', label='', score=4.)
        self.assertTupleEqual((4.0, '42'), ut(dict(foo='42')))

    def test_unknown_target(self):
        def test_function(*_):
            pass

        ut = UnitTest(test_function, target='foo', label='')
        with open(os.devnull, 'w') as stderr, capture_output(tmp_stderr=stderr):
            s, msg = ut({})

        self.assertEqual(s, 0.)
        self.assertEqual(msg, '❌ NameError: "foo"')

    def test_multiple_targets(self):
        def test_function(foo, bar):
            assert foo == 42
            assert bar == 1337

        ut = UnitTest(test_function, target=('foo', 'bar'), label='')
        self.assertTupleEqual((1.0, '✅ passed'), ut(dict(foo=42, bar=1337)))

        with open(os.devnull, 'w') as stderr, capture_output(tmp_stderr=stderr):
            self.assertTupleEqual((0, '❌ AssertionError: ""'), ut(dict(foo=42, bar=0)))
            self.assertTupleEqual((0, '❌ NameError: "bar"'), ut(dict(foo=42)))

    def test_score_fail(self):
        def test_function(_):
            pass

        ut = UnitTest(test_function, target='foo', label='')
        s, _ = ut(dict(foo=42))
        self.assertEqual(s, 1.0)
        self.assertEqual(ut.score, 1.0)

    def test_score_pass(self):
        def test_function(_):
            raise AssertionError

        ut = UnitTest(test_function, target='foo', label='')
        with open(os.devnull, 'w') as stderr, capture_output(tmp_stderr=stderr):
            s, _ = ut(dict(foo=42))
        self.assertEqual(s, 0.)
        self.assertEqual(ut.score, 1.0)

    def test_score_pass_higher_score(self):
        def test_function(_):
            pass

        ut = UnitTest(test_function, target='foo', label='', score=2)
        s, _ = ut(dict(foo=42))
        self.assertEqual(s, 2.0)
        self.assertEqual(ut.score, 2.0)

    def test_score_partially_pass(self):
        def test_function(*_):
            return .5

        ut = UnitTest(test_function, target='foo', label='')
        s, _ = ut(dict(foo=42))
        self.assertEqual(s, .5)

    def test_label(self):
        def test_function():
            pass

        ut = UnitTest(test_function, target='foo', label='bar')
        self.assertEqual('bar', ut.label)

    def test_timeout(self):
        def test_function(delay):
            time.sleep(delay)

        ut = UnitTest(test_function, target='delay', label='')
        s, _ = ut(dict(delay=.1))
        self.assertEqual(s, 1.)

        ut = UnitTest(test_function, target='delay', label='', timeout=.05)
        with open(os.devnull, 'w') as stderr, capture_output(tmp_stderr=stderr):
            s, _ = ut(dict(delay=.1))
        self.assertEqual(s, 0.)


class TestNotebookTest(TestCase):
    def test_register(self):
        nbt = NotebookTest('')

        def test_function(foo):
            self.assertEqual(42, foo)

        decorator = nbt.register(target='foo', label='bar', score=2, timeout=1)
        self.assertEqual(0, len(nbt))

        unit_test = decorator(test_function)
        self.assertEqual(1, len(nbt))
        self.assertEqual(unit_test.label, 'bar')
        self.assertEqual(unit_test.timout, 1)
        self.assertEqual(unit_test.score, 2)
        self.assertTupleEqual(unit_test.targets, ('foo',))

        self.assertTupleEqual((2.0, '✅ passed'), unit_test(dict(foo=42)))

        with self.assertRaises(ValueError):
            nbt.register(target='foo', label='bar')(lambda: None)

    def test_register_decorator(self):
        nbt = NotebookTest('')
        self.assertEqual(0, len(nbt))

        @nbt.register(target='foo', label='bar', score=2, timeout=1)
        def test_function(foo):
            self.assertEqual(42, foo)

        self.assertEqual(1, len(nbt))
        self.assertEqual(test_function.label, 'bar')
        self.assertEqual(test_function.timout, 1)
        self.assertEqual(test_function.score, 2)
        self.assertTupleEqual(test_function.targets, ('foo',))

        self.assertTupleEqual((2.0, '✅ passed'), test_function(dict(foo=42)))

        with self.assertRaises(ValueError):
            @nbt.register(target='foo', label='bar')
            def test_function():
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

            rpath, *_ = Path(path).glob('results_*.zip')

            with ZipFile(rpath, mode='r') as zipf:
                self.assertListEqual(sorted(zipf.namelist()), [
                    'artifacts/bar.txt',
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

                results = NotebookTestResult.from_json(zipf.open('results.json').read())

        self.assertEqual(results.version, autograde.__version__)
        self.assertEqual(results.checksum, sha256_sum)
        self.assertListEqual(results.excluded_artifacts, ['foo.txt'])
        assert_floats_equal(astuple(results.summarize()), (18, 6, 7, 2, 3, math.nan, 27))

    def test_execute(self):
        nb_path = PROJECT_ROOT.joinpath('demo', 'notebook.ipynb')
        t_path = PROJECT_ROOT.joinpath('demo', 'test.py')
        c_path = PROJECT_ROOT.joinpath('demo', 'context')

        # load test as module
        spec = import_util.spec_from_file_location('nbtest', t_path)
        nbtest = import_util.module_from_spec(spec)
        spec.loader.exec_module(nbtest)

        with TemporaryDirectory() as path, cd(path):
            self.assertEqual(6, nbtest.nbt.execute(args=(str(nb_path), '--context', str(c_path))))
