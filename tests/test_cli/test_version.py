import io
import sys
from unittest import TestCase

import autograde
from autograde.cli import cli
from autograde.util import capture_output


class TestCMD(TestCase):

    def test_version(self):
        with io.StringIO() as stdout, io.StringIO() as stderr:
            with capture_output(stdout, stderr):
                cli(['version'])

            stdout = stdout.getvalue()
            stderr = stderr.getvalue()

        self.assertIn(autograde.__version__, stdout)
        self.assertIn(sys.version.split()[0], stdout)
        self.assertIn(sys.getdefaultencoding(), stdout)
        self.assertEqual(stderr, '')
