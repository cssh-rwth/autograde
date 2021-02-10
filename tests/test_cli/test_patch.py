from tempfile import TemporaryDirectory
from unittest import TestCase

from autograde.cli import cli
from autograde.test_result import NotebookTestResultArchive
from tests.util import mount_example_archives


class TestCMD(TestCase):

    def test_patch_none(self):
        with TemporaryDirectory() as temp:
            cli(['patch', str(temp), str(temp)])

    def test_patch_single(self):
        with mount_example_archives() as examples:
            result = examples.joinpath('test_1', 'results_c.zip')
            patch = examples.joinpath('test_2', 'results_c.zip')

            for i in range(3):
                with NotebookTestResultArchive(result) as archive:
                    self.assertEqual(archive.patch_count, i)

                cli(['patch', str(result), str(patch)])

                with NotebookTestResultArchive(result) as archive:
                    self.assertEqual(archive.patch_count, i + 1)

    def test_patch_multi(self):
        with mount_example_archives() as examples:
            results = examples.joinpath('test_1')
            patches = examples.joinpath('test_2')

            for result in results.glob('*.zip'):
                with NotebookTestResultArchive(result) as archive:
                    self.assertEqual(archive.patch_count, 0)

            cli(['patch', str(results), str(patches)])

            for result in results.glob('*.zip'):
                with NotebookTestResultArchive(result) as archive:
                    self.assertEqual(archive.patch_count, 1)

    def test_patch_partial(self):
        with mount_example_archives() as examples:
            results = examples.joinpath('test_1')
            patches = examples.joinpath('test_2')

            results.joinpath('results_a.zip').unlink()
            patches.joinpath('results_c.zip').unlink()

            with NotebookTestResultArchive(results.joinpath('results_b.zip')) as archive:
                self.assertEqual(archive.patch_count, 0)
            with NotebookTestResultArchive(results.joinpath('results_c.zip')) as archive:
                self.assertEqual(archive.patch_count, 0)

            cli(['patch', str(results), str(patches)])

            with NotebookTestResultArchive(results.joinpath('results_b.zip')) as archive:
                self.assertEqual(archive.patch_count, 1)

            with NotebookTestResultArchive(results.joinpath('results_c.zip')) as archive:
                self.assertEqual(archive.patch_count, 0)
