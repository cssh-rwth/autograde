import os
import tarfile
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import pandas as pd

from autograde.__main__ import list_results, cli
from autograde.helpers import assert_isclose
from autograde.util import project_root, cd

PROJECT_ROOT = project_root()
EXAMPLES = PROJECT_ROOT.joinpath('test').joinpath('examples')


class TestWorkflow(TestCase):
    def test_scenatio_1(self):
        with TemporaryDirectory() as temp, cd(temp), warnings.catch_warnings():
            warnings.simplefilter('ignore')

            os.mkdir('results_1')
            os.mkdir('results_2')

            # run tests
            cli(['test', str(EXAMPLES.joinpath('test_1.py')), str(EXAMPLES), '-t', 'results_1'])
            cli(['test', str(EXAMPLES.joinpath('test_2.py')), str(EXAMPLES), '-t', 'results_2'])

            for path in list_results():
                with tarfile.open(path, mode='r') as tar:
                    self.assertListEqual(sorted(tar.getnames())[1:], [
                        'artifacts',
                        'code.py',
                        'notebook.ipynb',
                        'results.json'
                    ])

            # create reports for test 2 results
            cli(['report', 'results_2'])

            for path in list_results('results_2'):
                with tarfile.open(path, mode='r') as tar:
                    self.assertTrue('report.html' in tar.getnames())

            # create test summaries
            cli(['summary', 'results_1'])
            cli(['summary', 'results_2'])

            summary_1 = pd.read_csv(Path('results_1', 'summary.csv'))
            summary_2 = pd.read_csv(Path('results_2', 'summary.csv'))

            assert_isclose(10., summary_1['score'].sum())
            assert_isclose(8., summary_2['score'].sum())
            assert_isclose(12., summary_1['max_score'].sum())
            assert_isclose(12., summary_2['max_score'].sum())
            self.assertEqual(2, sum(summary_1['duplicate']))
            self.assertEqual(2, sum(summary_2['duplicate']))

            # patch test 1 results and re-compute report + summary
            cli(['patch', 'results_1', 'results_2'])
            cli(['report', 'results_1'])
            cli(['summary', 'results_1'])

            for path in list_results('results_1'):
                with tarfile.open(path, mode='r') as tar:
                    self.assertTrue('report.html' in tar.getnames())

            summary_1 = pd.read_csv(Path('results_1', 'summary.csv'))

            assert_isclose(8., summary_1['score'].sum())
            assert_isclose(12., summary_1['max_score'].sum())
            self.assertEqual(2, sum(summary_1['duplicate']))

            # compute global summary
            cli(['summary', '.'])

            summary = pd.read_csv('summary.csv')

            assert_isclose(16., summary['score'].sum())
            assert_isclose(24., summary['max_score'].sum())
            self.assertTrue(all(summary['duplicate']))
