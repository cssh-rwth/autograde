from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from autograde.cli import cli
from tests.util import mount_example_archives


class TestCMD(TestCase):

    def test_summarize_none(self):
        with TemporaryDirectory() as temp:
            cli(['summary', str(temp)])
            files = set(map(lambda p: p.name, Path(temp).glob('*')))
            self.assertSetEqual(files, {'summary.html', 'summary.csv', 'raw.csv'})

    def test_summary(self):
        with mount_example_archives() as examples:
            results = examples.joinpath('test_2')
            cli(['summary', str(results)])

            files = set(map(lambda p: p.name, Path(results).glob('*')))
            self.assertTrue(set.issubset({'summary.html', 'summary.csv', 'raw.csv'}, files))
