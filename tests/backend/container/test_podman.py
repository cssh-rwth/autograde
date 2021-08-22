import os
from unittest import skip, skipUnless

from autograde.backend import Backend
from autograde.backend.container import ContainerBackend, Podman
from autograde.util import parse_bool
from tests.backend.test_backend import TestBackend


@skipUnless('podman' in Backend.available, 'Podman is not available on this system')
class TestPodmanBackend(TestBackend):
    backend: ContainerBackend = None
    backend_cls = Podman

    @classmethod
    def setUpClass(cls):
        super(TestPodmanBackend, cls).setUpClass()
        cls.backend.build(from_source=parse_bool(os.getenv('AG_TEST_BUILD_IMAGE_FROM_SOURCE', False)))

    @classmethod
    def tearDownClass(cls) -> None:
        cls.backend.untag()

    @skip
    def test_version(self):
        # capturing stdout/stderr would require quite some tweaking of the
        # `Command` class - an effort not quite worth in this case
        self.fail()


# prevent backend tests from being scheduled a second time
del TestBackend
