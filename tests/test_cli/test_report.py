from tempfile import TemporaryDirectory
from unittest import TestCase

from autograde.cli import cli
from autograde.test_result import NotebookTestResultArchive
from tests.util import mount_example_archives


class TestCMD(TestCase):

    def test_report_none(self):
        with TemporaryDirectory() as temp:
            cli(['report', str(temp)])

    def test_report_single(self):
        with mount_example_archives() as examples:
            result = examples.joinpath('test_1', 'results_a.zip')

            for i in range(3):
                with NotebookTestResultArchive(result) as archive:
                    self.assertEqual(archive.report_count, i)

                cli(['report', str(result)])

                with NotebookTestResultArchive(result) as archive:
                    self.assertEqual(archive.report_count, i + 1)

    def test_report_multi(self):
        with mount_example_archives() as examples:
            results = examples.joinpath('test_2')

            for result in results.glob('*.zip'):
                with NotebookTestResultArchive(result) as archive:
                    self.assertEqual(archive.report_count, 0)

            cli(['report', str(results)])

            for result in results.glob('*.zip'):
                with NotebookTestResultArchive(result) as archive:
                    self.assertEqual(archive.report_count, 1)
