from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from autograde.cli import cli
from autograde.util import cd
from tests.util import DEMO, EXAMPLES


class TestCMD(TestCase):

    def test_test_none(self):
        with TemporaryDirectory() as temp:
            cli(['test', str(DEMO.joinpath('test.py')), str(temp)])

    def test_test_single(self):
        with TemporaryDirectory() as temp, cd(temp):
            cli([
                'test',
                str(EXAMPLES.joinpath('test_1.py')),
                str(EXAMPLES.joinpath('solution_c.ipynb'))
            ])
            self.assertEqual(len(list(Path(temp).glob('results_*.zip'))), 1)

    def test_test_multi(self):
        with TemporaryDirectory() as temp, cd(temp):
            cli([
                'test',
                str(EXAMPLES.joinpath('test_2.py')),
                str(EXAMPLES)
            ])
            self.assertEqual(len(list(Path(temp).glob('results_*.zip'))), 3)

    def test_test_with_target(self):
        with TemporaryDirectory() as temp:
            cli([
                'test',
                str(EXAMPLES.joinpath('test_2.py')),
                str(EXAMPLES.joinpath('solution_b.ipynb')),
                '--target', str(temp)
            ])
            self.assertEqual(len(list(Path(temp).glob('results_*.zip'))), 1)

    def test_test_with_context(self):
        with TemporaryDirectory() as temp, cd(temp):
            cli([
                'test',
                str(DEMO.joinpath('test.py')),
                str(DEMO.joinpath('notebook.ipynb')),
                '--context', str(DEMO.joinpath('context'))
            ])
            self.assertEqual(len(list(Path(temp).glob('results_*.zip'))), 1)
