from base64 import b64decode
from unittest import TestCase

from autograde.backend.local.util import b64str, find_archives, traverse_archives, merge_results, \
    summarize_results, plot_score_distribution
from autograde.helpers import assert_isclose
from tests.util import mount_example_archives


class TestFunctions(TestCase):

    def test_b64str(self):
        self.assertEqual(b64decode(b64str(b'Fnord')), b'Fnord')

    def test_find_archives(self):
        with mount_example_archives() as examples:
            self.assertEqual(len(find_archives(examples)), 6)
            self.assertEqual(find_archives(examples), sorted(find_archives(examples)))
            self.assertEqual(len(find_archives(examples, prefix='foo')), 0)

            empty = examples.joinpath('empty')
            empty.mkdir()

            self.assertEqual(len(find_archives(empty)), 0)

    def test_traverse_archives(self):
        with mount_example_archives() as examples:
            for archive in traverse_archives(find_archives(examples.joinpath('test_1'))):
                with self.assertRaises(ValueError):
                    archive.inject_report()

            for archive in traverse_archives(find_archives(examples.joinpath('test_2')), mode='a'):
                archive.inject_report()

    def test_merge_results_empty(self):
        df = merge_results(traverse_archives([]))
        self.assertEqual(len(df), 0)

    def test_merge_results(self):
        with mount_example_archives() as examples:
            df = merge_results(traverse_archives(find_archives(examples.joinpath('test_1'))))
            self.assertEqual(len(df), 12)
            assert_isclose(df['score'].sum(), 10)
            assert_isclose(df['max_score'].sum(), 12)

    def test_summarize_results_empty(self):
        summary = summarize_results(merge_results(traverse_archives([])))
        self.assertEqual(len(summary), 0)

    def test_summarize_results(self):
        with mount_example_archives() as examples:
            summary = summarize_results(merge_results(traverse_archives(find_archives(examples.joinpath('test_2')))))
            assert_isclose(summary['score'].sum(), 8)
            assert_isclose(summary['max_score'].sum(), 12)
            assert_isclose(summary['duplicate'].sum(), 2)

    def test_plot_score_distribution_empty(self):
        summary = summarize_results(merge_results(traverse_archives([])))
        self.assertIsNone(plot_score_distribution(summary))

    def test_plot_score_distribution(self):
        with mount_example_archives() as examples:
            summary = summarize_results(merge_results(traverse_archives(find_archives(examples.joinpath('test_1')))))
            self.assertIsNotNone(plot_score_distribution(summary))
