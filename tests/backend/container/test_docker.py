from unittest import skip, skipUnless

from autograde.backend import Backend
from autograde.backend.container import ContainerBackend, Docker
from tests.backend.container.config import SKIP_CONTAINER, BUILD_IMAGE_FROM_SOURCE
from tests.backend.test_backend import TestBackend


@skipUnless('docker' in Backend.available and not SKIP_CONTAINER, 'Skip testing with Docker backend')
class TestDockerBackend(TestBackend):
    backend: ContainerBackend = None
    backend_cls = Docker

    @classmethod
    def setUpClass(cls):
        super(TestDockerBackend, cls).setUpClass()
        cls.backend.build(from_source=BUILD_IMAGE_FROM_SOURCE)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.backend.untag()

    @skip
    def test_version(self):
        # capturing stdout/stderr would require quite some tweaking of the
        # `Command` class - an effort not quite worth in this case
        self.fail()


# prevent tests from being scheduled for execution a second time
del TestBackend
