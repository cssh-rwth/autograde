import io
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Type
from unittest import TestCase

import autograde
from autograde.backend import Backend
from autograde.backend.local import Local
from autograde.test_result import NotebookTestResultArchive
from autograde.util import capture_output
from tests.util import DEMO, EXAMPLES, mount_example_archives


class TestBackend(TestCase):
    backend: Backend = None
    backend_cls: Type[Backend] = Local

    @classmethod
    def setUpClass(cls):
        cls.backend = cls.backend_cls(tag='autograde-test', verbosity=3)

    # Test Patch Command
    def test_patch_none(self):
        with TemporaryDirectory() as temp:
            temp = Path(temp)
            self.backend.patch(temp, temp)

    def test_patch_single(self):
        with mount_example_archives() as examples:
            result = examples.joinpath('test_1', 'results_c.zip')
            patch = examples.joinpath('test_2', 'results_c.zip')

            for i in range(3):
                with NotebookTestResultArchive(result) as archive:
                    self.assertEqual(archive.patch_count, i)

                self.backend.patch(result, patch)

                with NotebookTestResultArchive(result) as archive:
                    self.assertEqual(archive.patch_count, i + 1)

    def test_patch_multi(self):
        with mount_example_archives() as examples:
            results = examples.joinpath('test_1')
            patches = examples.joinpath('test_2')

            for result in results.glob('*.zip'):
                with NotebookTestResultArchive(result) as archive:
                    self.assertEqual(archive.patch_count, 0)

            self.backend.patch(results, patches)

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

            self.backend.patch(results, patches)

            with NotebookTestResultArchive(results.joinpath('results_b.zip')) as archive:
                self.assertEqual(archive.patch_count, 1)

            with NotebookTestResultArchive(results.joinpath('results_c.zip')) as archive:
                self.assertEqual(archive.patch_count, 0)

    # Test Report Command
    def test_report_none(self):
        with TemporaryDirectory() as temp:
            self.backend.report(Path(temp))

    def test_report_single(self):
        with mount_example_archives() as examples:
            result = examples.joinpath('test_1', 'results_a.zip')

            for i in range(3):
                with NotebookTestResultArchive(result) as archive:
                    self.assertEqual(archive.report_count, i)

                self.backend.report(result)

                with NotebookTestResultArchive(result) as archive:
                    self.assertEqual(archive.report_count, i + 1)

    def test_report_multi(self):
        with mount_example_archives() as examples:
            results = examples.joinpath('test_2')

            for result in results.glob('*.zip'):
                with NotebookTestResultArchive(result) as archive:
                    self.assertEqual(archive.report_count, 0)

            self.backend.report(results)

            for result in results.glob('*.zip'):
                with NotebookTestResultArchive(result) as archive:
                    self.assertEqual(archive.report_count, 1)

    # Test Summarize Command
    def test_summarize_none(self):
        with TemporaryDirectory() as temp:
            temp = Path(temp)
            self.backend.summary(temp)
            files = set(map(lambda p: p.name, temp.glob('*')))
            self.assertSetEqual(files, {'summary.html', 'summary.csv', 'raw.csv'})

    def test_summarize(self):
        with mount_example_archives() as examples:
            results = examples.joinpath('test_2')
            self.backend.summary(results)

            files = set(map(lambda p: p.name, Path(results).glob('*')))
            self.assertTrue(set.issubset({'summary.html', 'summary.csv', 'raw.csv'}, files))

    # Test Test Command
    def test_test_none(self):
        with TemporaryDirectory() as temp:
            self.backend.test(DEMO.joinpath('test.py'), DEMO, Path(temp))

    def test_test_single(self):
        with TemporaryDirectory() as temp:
            temp = Path(temp)
            self.backend.test(
                EXAMPLES.joinpath('test_1.py'),
                EXAMPLES.joinpath('solution_c.ipynb'),
                temp
            )
            self.assertEqual(len(list(temp.glob('results_*.zip'))), 1)

    def test_test_multi(self):
        with TemporaryDirectory() as temp:
            temp = Path(temp)
            self.backend.test(
                EXAMPLES.joinpath('test_2.py'),
                EXAMPLES,
                temp
            )
            self.assertEqual(len(list(temp.glob('results_*.zip'))), 3)

    def test_test_with_target(self):
        with TemporaryDirectory() as temp:
            temp = Path(temp)
            self.backend.test(
                EXAMPLES.joinpath('test_2.py'),
                EXAMPLES.joinpath('solution_b.ipynb'),
                temp
            )
            self.assertEqual(len(list(temp.glob('results_*.zip'))), 1)

    def test_test_with_context(self):
        with TemporaryDirectory() as temp:
            temp = Path(temp)
            self.backend.test(
                DEMO.joinpath('test.py'),
                DEMO.joinpath('notebook.ipynb'),
                temp,
                DEMO.joinpath('context')
            )
            self.assertEqual(len(list(temp.glob('results_*.zip'))), 1)

    # Test Version Command
    def test_version(self):
        with io.StringIO() as stdout, io.StringIO() as stderr:
            with capture_output(stdout, stderr):
                self.backend.version()

            stdout = stdout.getvalue()
            stderr = stderr.getvalue()

        self.assertIn(autograde.__version__, stdout)
        self.assertIn(sys.version.split()[0], stdout)
        self.assertIn(sys.getdefaultencoding(), stdout)
        self.assertEqual(stderr, '')
