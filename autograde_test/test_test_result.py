import io
import math
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import astuple
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.case import TestCase
from zipfile import ZipFile

import nbformat

from autograde.test_result import UnitTestResult, NotebookTestResult, NotebookTestResultArchive
from autograde.util import cd
from autograde_test.util import assert_floats_equal, load_demo_archive


class TestUnitTestResult(TestCase):
    def test_passed(self):
        for r in [UnitTestResult('1', '', [], 1., 1., [], '', ''), UnitTestResult('1', '', [], 2., 2., [], '', '')]:
            self.assertTrue(r.passed())
            self.assertFalse(r.partially_passed())
            self.assertFalse(r.failed())
            self.assertFalse(r.pending())

    def test_partially_passed(self):
        for r in [UnitTestResult('1', '', [], .5, 1., [], '', ''), UnitTestResult('1', '', [], 1., 2., [], '', '')]:
            self.assertFalse(r.passed())
            self.assertTrue(r.partially_passed())
            self.assertFalse(r.failed())
            self.assertFalse(r.pending())

    def test_failed(self):
        for r in [UnitTestResult('1', '', [], 0., 1., [], '', ''), UnitTestResult('1', '', [], 0., 2., [], '', '')]:
            self.assertFalse(r.passed())
            self.assertFalse(r.partially_passed())
            self.assertTrue(r.failed())
            self.assertFalse(r.pending())

    def test_pending(self):
        for r in [UnitTestResult('1', '', [], math.nan, 0., [], '', ''),
                  UnitTestResult('1', '', [], math.nan, 1., [], '', '')]:
            self.assertFalse(r.passed())
            self.assertFalse(r.partially_passed())
            self.assertFalse(r.failed())
            self.assertTrue(r.pending())


class TestNotebookTestResult(TestCase):
    def test_patch(self):
        results_a = NotebookTestResult('', '', '', [], [], [], [])
        results_b = NotebookTestResult('', '', '', [], [], [], [])
        patch_result = results_a.patch(results_b)
        assert_floats_equal(astuple(patch_result.summarize()), (0, 0, 0, 0, 0., 0.))
        self.assertListEqual([('', results_b.timestamp, [])], patch_result.applied_patches)

        results_a = NotebookTestResult('', '', '', [], [], [], [])
        results_b = NotebookTestResult('', '', '', [], [], [], [UnitTestResult('1', '', [], 0., 1., [], '', '')])
        patch_result = results_a.patch(results_b)
        assert_floats_equal(astuple(patch_result.summarize()), (1, 1, 0, 0, 0., 1.))
        self.assertListEqual([('', results_b.timestamp, ['1'])], patch_result.applied_patches)

        results_a = NotebookTestResult('', '', '', [], [], [], [UnitTestResult('1', '', [], 0., 1., [], '', '')])
        results_b = NotebookTestResult('', '', '', [], [], [], [])
        patch_result = results_a.patch(results_b)
        assert_floats_equal(astuple(patch_result.summarize()), (1, 1, 0, 0, 0., 1.))
        self.assertListEqual([('', results_b.timestamp, [])], patch_result.applied_patches)

        results_a = NotebookTestResult('', '', '', [], [], [], [UnitTestResult('1', '', [], 0., 1., [], '', '')])
        results_b = NotebookTestResult('', '', '', [], [], [], [UnitTestResult('2', '', [], 1., 1., [], '', '')])
        patch_result = results_a.patch(results_b)
        assert_floats_equal(astuple(patch_result.summarize()), (2, 1, 1, 0, 1., 2.))
        self.assertListEqual([('', results_b.timestamp, ['2'])], patch_result.applied_patches)

        results_a = NotebookTestResult('', '', '', [], [], [], [UnitTestResult('1', '', [], 0., 1., [], '', '')])
        results_b = NotebookTestResult('', '', '', [], [], [], [UnitTestResult('1', '', [], 1., 1., [], '', '')])
        patch_result = results_a.patch(results_b)
        assert_floats_equal(astuple(patch_result.summarize()), (1, 0, 1, 0, 1., 1.))
        self.assertListEqual([('', results_b.timestamp, ['1'])], patch_result.applied_patches)

        results_a = NotebookTestResult('', '', '', [], [], [], [UnitTestResult('1', '', [], 0., 1., [], '', '')])
        results_b = NotebookTestResult('', '', '', [], [], [], [UnitTestResult('1', '', [], 0., 1., [], '', 'foo')])
        patch_result = results_a.patch(results_b)
        assert_floats_equal(astuple(patch_result.summarize()), (1, 1, 0, 0, 0., 1.))
        self.assertListEqual([('', results_b.timestamp, ['1'])], patch_result.applied_patches)

        results_a = NotebookTestResult('', '', '', [], [], [], [UnitTestResult('1', '', [], math.nan, 1., [], '', '')])
        results_b = NotebookTestResult('', '', '', [], [], [], [UnitTestResult('1', '', [], 0., 1., [], '', '')])
        patch_result = results_a.patch(results_b)
        assert_floats_equal(astuple(patch_result.summarize()), (1, 1, 0, 0, 0., 1.))
        self.assertListEqual([('', results_b.timestamp, ['1'])], patch_result.applied_patches)

        results_a = NotebookTestResult('', '', '', [], [], [], [UnitTestResult('1', '', [], 0., 1., [], '', '')])
        results_b = NotebookTestResult('', '', '', [], [], [], [UnitTestResult('1', '', [], math.nan, 1., [], '', '')])
        patch_result = results_a.patch(results_b)
        assert_floats_equal(astuple(patch_result.summarize()), (1, 1, 0, 0, 0., 1.))
        self.assertListEqual([('', results_b.timestamp, [])], patch_result.applied_patches)

        results_a = NotebookTestResult('a', '', '', [], [], [], [UnitTestResult('1', '', [], 0., 1., [], '', '')])
        results_b = NotebookTestResult('b', '', '', [], [], [], [UnitTestResult('2', '', [], 1., 1., [], '', '')])
        results_c = NotebookTestResult('c', '', '', [], [], [], [UnitTestResult('3', '', [], 2., 4., [], '', ''),
                                                                 UnitTestResult('4', '', [], 4., 8., [], '', '')])
        patch_result = results_a.patch(results_b).patch(results_c)
        assert_floats_equal(astuple(patch_result.summarize()), (4, 1, 1, 0, 7., 14.))
        self.assertListEqual([('b', results_b.timestamp, ['2']),
                              ('c', results_c.timestamp, ['3', '4'])], patch_result.applied_patches)

        with self.assertRaises(ValueError):
            results_a = NotebookTestResult('', '', '0' * 64, [], [], [], [])
            results_b = NotebookTestResult('', '', '1' * 64, [], [], [], [])
            results_a.patch(results_b)

    def test_summary(self):
        results = NotebookTestResult('', '', '', [], [], [], [])
        assert_floats_equal(astuple(results.summarize()), (0, 0, 0, 0, 0, 0))

        results = NotebookTestResult('', '', '', [], [], [], [UnitTestResult('1', '', [], 0., 1., [], '', '')])
        assert_floats_equal(astuple(results.summarize()), (1, 1, 0, 0, 0., 1.))

        results = NotebookTestResult('', '', '', [], [], [], [UnitTestResult('2', '', [], 1., 1., [], '', '')])
        assert_floats_equal(astuple(results.summarize()), (1, 0, 1, 0, 1., 1.))

        results = NotebookTestResult('', '', '', [], [], [], [UnitTestResult('3', '', [], math.nan, 1., [], '', '')])
        assert_floats_equal(astuple(results.summarize()), (1, 0, 0, 1, math.nan, 1.))


class TestNotebookTestResultArchive(TestCase):
    def setUp(self) -> None:
        self._exit_stack = ExitStack().__enter__()

        tmp = self._exit_stack.enter_context(TemporaryDirectory())
        self._exit_stack.enter_context(cd(tmp))

        with Path(tmp).joinpath('archive.zip').open(mode='wb') as f:
            f.write(load_demo_archive())

        self.file_list = {
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
        }

    def tearDown(self) -> None:
        self._exit_stack.close()

    def test_init(self):
        with self.assertRaises(ValueError):
            with NotebookTestResultArchive('', mode='ö'):
                pass

        with NotebookTestResultArchive('archive.zip'):
            pass

        with ZipFile('archive.zip', mode='w'):
            pass

        with self.assertRaises(KeyError):
            with NotebookTestResultArchive('archive.zip'):
                pass

    def test_hash(self):
        with NotebookTestResultArchive('archive.zip', mode='a') as archive:
            h_0 = hash(archive)

            archive.inject_patch(deepcopy(archive.results))

            h_1 = hash(archive)
            self.assertNotEqual(h_0, h_1)

            h_2 = hash(archive)
            self.assertEqual(h_1, h_2)

        with NotebookTestResultArchive('archive.zip', mode='a') as archive:
            h_3 = hash(archive)
            self.assertNotEqual(h_2, h_3)

    def test_filename(self):
        with NotebookTestResultArchive('archive.zip', mode='a') as archive:
            self.assertEqual(archive.filename, 'archive.zip')

        Path('archive.zip').rename('foobar.zip')

        with NotebookTestResultArchive('foobar.zip', mode='a') as archive:
            self.assertEqual(archive.filename, 'foobar.zip')

    def test_files(self):
        with NotebookTestResultArchive('archive.zip', mode='a') as archive:
            hsh = hash(archive)
            _ = archive.files
            self.assertEqual(hsh, hash(archive))

            self.assertSetEqual(set(archive.files), self.file_list)
            archive.inject_report()
            self.assertNotEqual(set(archive.files), self.file_list)

    def test_load_file(self):
        for m in ['a', 'r']:
            with NotebookTestResultArchive('archive.zip', mode=m) as archive:
                hsh = hash(archive)

                self.assertIsInstance(archive.load_file('code.py'), bytes)
                self.assertIsInstance(archive.load_file('code.py', encoding='utf-8'), str)

                archive.load_file('code.py', encoding='utf-8').startswith('__IB_FLAG__')

                with self.assertRaises(KeyError):
                    archive.load_file('foo')

                with self.assertRaises(ValueError):
                    archive.load_file('artifacts/plot.png', encoding='utf-8')

                self.assertEqual(hsh, hash(archive))

    def test_patch_count(self):
        with NotebookTestResultArchive('archive.zip', mode='a') as archive:
            hsh = hash(archive)
            _ = archive.patch_count
            self.assertEqual(hsh, hash(archive))

            for i in range(3):
                self.assertEqual(i, archive.patch_count)
                archive.inject_patch(deepcopy(archive.results))

            for i in range(3):
                self.assertEqual(i + 3, archive.patch_count)
                archive.inject_patch(deepcopy(archive.results))

    def test_report_count(self):
        with NotebookTestResultArchive('archive.zip', mode='a') as archive:
            hsh = hash(archive)
            _ = archive.report_count
            self.assertEqual(hsh, hash(archive))

            for i in range(3):
                self.assertEqual(i, archive.report_count)
                archive.inject_report()

            for i in range(3):
                self.assertEqual(i + 3, archive.report_count)
                archive.inject_report()

    def test_inject_patch(self):
        with self.assertRaises(ValueError):
            with NotebookTestResultArchive('archive.zip', mode='r') as archive:
                archive.inject_patch(deepcopy(archive.results))

        with NotebookTestResultArchive('archive.zip', mode='a') as archive:
            self.assertSetEqual(set(archive.files) - self.file_list, set())

            hsh = hash(archive)
            archive.inject_patch(deepcopy(archive.results))
            self.assertNotEqual(hash(archive), hsh)
            self.assertSetEqual(
                set(archive.files) - self.file_list,
                {'results_patch_01.json'}
            )

            hsh = hash(archive)
            archive.inject_patch(deepcopy(archive.results))
            self.assertNotEqual(hash(archive), hsh)
            self.assertSetEqual(
                set(archive.files) - self.file_list,
                {'results_patch_01.json', 'results_patch_02.json'}
            )

            hsh = hash(archive)
            archive.inject_report()
            archive.inject_patch(deepcopy(archive.results))
            self.assertNotEqual(hash(archive), hsh)
            self.assertSetEqual(
                set(archive.files) - self.file_list,
                {'results_patch_01.json', 'results_patch_02.json', 'results_patch_03.json',
                 'report.html', 'report_rev_01.html'}
            )

        # check if changes persist
        with NotebookTestResultArchive('archive.zip', mode='r') as archive:
            self.assertSetEqual(
                set(archive.files) - self.file_list,
                {'results_patch_01.json', 'results_patch_02.json', 'results_patch_03.json',
                 'report.html', 'report_rev_01.html'}
            )

    def test_inject_report(self):
        with self.assertRaises(ValueError):
            with NotebookTestResultArchive('archive.zip', mode='r') as archive:
                archive.inject_report()

        with NotebookTestResultArchive('archive.zip', mode='a') as archive:
            self.assertSetEqual(set(archive.files) - self.file_list, set())

            hsh = hash(archive)
            archive.inject_report()
            self.assertNotEqual(hash(archive), hsh)
            self.assertSetEqual(
                set(archive.files) - self.file_list,
                {'report.html'}
            )

            hsh = hash(archive)
            archive.inject_report()
            self.assertNotEqual(hash(archive), hsh)
            self.assertSetEqual(
                set(archive.files) - self.file_list,
                {'report.html', 'report_rev_01.html'}
            )

            hsh = hash(archive)
            archive.inject_report()
            self.assertNotEqual(hash(archive), hsh)
            self.assertSetEqual(
                set(archive.files) - self.file_list,
                {'report.html', 'report_rev_01.html', 'report_rev_02.html'}
            )

        # check if changes persist
        with NotebookTestResultArchive('archive.zip', mode='a') as archive:
            self.assertSetEqual(
                set(archive.files) - self.file_list,
                {'report.html', 'report_rev_01.html', 'report_rev_02.html'}
            )

    def test_results(self):
        with NotebookTestResultArchive('archive.zip', mode='a') as archive:
            hsh = hash(archive)
            _ = archive.results
            self.assertEqual(hsh, hash(archive))

            patch = deepcopy(archive.results)
            for result in patch.unit_test_results:
                result.score = result.score_max / 2

            archive.inject_patch(patch)
            summary = archive.results.summarize()

            self.assertSetEqual(set(archive.files) - self.file_list, {'results_patch_01.json'})
            self.assertAlmostEqual(summary.score, summary.score_max / 2)

    def test_code(self):
        with NotebookTestResultArchive('archive.zip', mode='r') as archive:
            hsh = hash(archive)
            self.assertIn('__IB_FLAG__', archive.code)
            self.assertEqual(hsh, hash(archive))

    def test_notebook(self):
        with NotebookTestResultArchive('archive.zip', mode='r') as archive:
            hsh = hash(archive)
            with io.StringIO(archive.notebook) as f:
                nbformat.read(f, 4)
            self.assertEqual(hsh, hash(archive))

    def test_report(self):
        with NotebookTestResultArchive('archive.zip', mode='r') as archive:
            hsh = hash(archive)
            self.assertIn('<!DOCTYPE html>', archive.report)
            self.assertEqual(hsh, hash(archive))
