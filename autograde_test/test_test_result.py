import math
from dataclasses import astuple
from unittest.case import TestCase

from autograde.test_result import UnitTestResult, NotebookTestResult
from autograde_test.util import assert_floats_equal


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
