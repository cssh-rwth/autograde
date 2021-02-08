from contextlib import ExitStack
from getpass import getuser
from pathlib import Path
import math
from unittest import TestCase

from autograde.cli import cli
from autograde.cli.audit import AuditSettings, AuditState
from autograde.test.test_test_result import utr_dummy
from autograde.test.util import mount_demo_archive, mount_example_archives
from autograde.util import cd

USER = getuser()


class TestAuditSettings(TestCase):
    def test_update(self):
        settings = AuditSettings()
        self.assertTrue(settings.selector.match('lknjhsavhkljv'))
        self.assertEqual(settings.auditor, USER)
        self.assertFalse(settings.show_identities)

        settings.update(selector='a+', auditor='FooBar', show_identities=True)
        self.assertTrue(settings.selector.match('aaa'))
        self.assertFalse(settings.selector.match('lknjhsavhkljv'))
        self.assertEqual(settings.auditor, 'FooBar')

        settings.update()
        self.assertTrue(settings.selector.match('lknjhsavhkljv'))
        self.assertEqual(settings.auditor, USER)
        self.assertFalse(settings.show_identities)

    def test_select(self):
        settings = AuditSettings()
        self.assertTrue(settings.select(utr_dummy(label='ljködjfsdjd')))

        settings.update(selector='a+')
        self.assertFalse(settings.select(utr_dummy(label='ljködjfsdjd')))
        self.assertTrue(settings.select(utr_dummy(label='aaa')))

    def test_filter_results(self):
        settings = AuditSettings()
        utrs = [
            utr_dummy(label='aaa'),
            utr_dummy(label='bbb'),
            utr_dummy(label='ccc')
        ]

        self.assertListEqual(list(settings.filter_results(utrs)), utrs)

        settings.update(selector='[ab]+')
        self.assertListEqual(list(settings.filter_results(utrs)), utrs[:2])

        settings.update(selector='c+')
        self.assertListEqual(list(settings.filter_results(utrs)), utrs[2:])

        settings.update(selector='vsaihsfjhsfd')
        self.assertListEqual(list(settings.filter_results(utrs)), [])

    def test_format_comment(self):
        settings = AuditSettings()
        self.assertEqual(settings.format_comment('my message'), f'{USER}: my message')

        settings = AuditSettings(auditor='Alice')
        self.assertEqual(settings.format_comment('my message'), 'Alice: my message')


class TestAuditState(TestCase):

    def setUp(self) -> None:
        self._exit_stack = ExitStack().__enter__()

        temp = self._exit_stack.enter_context(mount_example_archives())
        self._exit_stack.enter_context(cd(temp))

        Path(temp).joinpath('test_empty').mkdir()

    def tearDown(self) -> None:
        self._exit_stack.close()

    def test_init_empty(self):
        with AuditState(Path('test_empty')) as state:
            self.assertEqual(state.settings, AuditSettings())
            self.assertEqual(len(state.archives), 0)
            self.assertSetEqual(state.patched, set())

    def test_init_nonempty(self):
        for path in map(Path, ['test_1', 'test_2']):
            with AuditState(path) as state:
                self.assertEqual(state.settings, AuditSettings())
                self.assertEqual(len(state.archives), 3)
                self.assertSetEqual(state.patched, set())

    def test_prev_id(self):
        with AuditState(Path('test_1')) as state:
            aids = list(state.archives)

            paid = state.prev_id(aids[-1])
            self.assertEqual(paid, aids[1])

            paid = state.prev_id(paid)
            self.assertEqual(paid, aids[0])

            paid = state.prev_id(paid)
            self.assertIsNone(paid)

    def test_next_id(self):
        with AuditState(Path('test_1')) as state:
            aids = list(state.archives)

            naid = state.next_id(aids[0])
            self.assertEqual(naid, aids[1])

            naid = state.next_id(naid)
            self.assertEqual(naid, aids[2])

            naid = state.next_id(naid)
            self.assertIsNone(naid)

    def test_parse_empty_form(self):
        scores, comments = AuditState._parse_form({})
        self.assertTrue(len(scores) == 0)
        self.assertTrue(len(comments) == 0)

    def test_parse_form_correct(self):
        scores, comments = AuditState._parse_form({
            'score:foo': 42,
            'score:bar': '13.37',
            'score:fnord': '',
            'comment:foo': 'foo',
            'comment:bar': 'bar',
            'comment:fnord': '',
        })

        self.assertEqual(scores['foo'], 42.)
        self.assertEqual(scores['bar'], 13.37)
        self.assertTrue(math.isnan(scores['fnord']))
        self.assertEqual(comments['foo'], 'foo')
        self.assertEqual(comments['bar'], 'bar')
        self.assertIsNone(comments.get('fnord'))

    def test_parse_form_incorrect(self):
        with self.assertRaises(TypeError):
            AuditState._parse_form({'score:foo': None})
        with self.assertRaises(ValueError):
            AuditState._parse_form({'score:foo': 'abc'})

    def test_patch_empty(self):
        with mount_example_archives() as path, AuditState(path.joinpath('test_2')) as state:
            aid = next((id for id, a in state.archives.items() if 'results_a' in a.filename))
            self.assertEqual(state.archives[aid].patch_count, 0)

            state.patch(aid)
            self.assertEqual(state.archives[aid].patch_count, 0)

    def test_patch_no_change(self):
        with mount_example_archives() as path, AuditState(path.joinpath('test_2')) as state:
            aid = next((id for id, a in state.archives.items() if 'results_a' in a.filename))
            self.assertEqual(state.archives[aid].patch_count, 0)

            form = {
                **{f'score:{r.id}': r.score for r in state.archives[aid].results.unit_test_results},
                **{f'comment:{r.id}': '' for r in state.archives[aid].results.unit_test_results}
            }

            state.patch(aid, **form)
            self.assertEqual(state.archives[aid].patch_count, 0)

    def test_patch_changed_scores(self):
        with mount_example_archives() as path, AuditState(path.joinpath('test_2')) as state:
            aid = next((id for id, a in state.archives.items() if 'results_a' in a.filename))
            self.assertEqual(state.archives[aid].patch_count, 0)

            form = {f'score:{r.id}': r.score / 2 for r in state.archives[aid].results.unit_test_results}

            state.patch(aid, **form)
            self.assertEqual(state.archives[aid].patch_count, 1)

    def test_patch_changed_comments(self):
        with mount_example_archives() as path, AuditState(path.joinpath('test_2')) as state:
            aid = next((id for id, a in state.archives.items() if 'results_a' in a.filename))
            self.assertEqual(state.archives[aid].patch_count, 0)

            form = {f'comment:{r.id}': 'foo' for r in state.archives[aid].results.unit_test_results}

            state.patch(aid, **form)
            self.assertEqual(state.archives[aid].patch_count, 1)


def live_audit_demo():
    with mount_demo_archive() as path:
        cli(['-vvv', 'audit', str(path)])


def live_audit_example_1():
    with mount_example_archives() as path:
        cli(['-vvv', 'audit', str(path.joinpath('test_1'))])


def live_audit_example_2():
    with mount_example_archives() as path:
        cli(['-vvv', 'audit', str(path.joinpath('test_2'))])
