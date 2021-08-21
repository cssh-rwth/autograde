import json

from autograde.backend.container.common import ContainerCommand, ContainerBackend
from autograde.util import logger


class DockerCommand(ContainerCommand):
    def __init__(self, *command: str):
        super().__init__('docker', *command)


class Docker(ContainerBackend):
    @staticmethod
    def new_container_command(subcommand: str) -> ContainerCommand:
        return DockerCommand(subcommand)

    @classmethod
    def is_available(cls) -> bool:
        try:
            cmd = cls.new_container_command('info')
            cmd.named_parameter('--format', '{{json .}}')
            cp = cmd.run(capture_output=True)
            assert cp.returncode == 0

            info = json.loads(cp.stdout.decode('utf-8'))
            if 'name=rootless' in info['SecurityOptions']:
                return True

            logger.warning('Docker is available but does not run in rootless mode!')
            return False

        except (FileNotFoundError, AssertionError, KeyError, json.decoder.JSONDecodeError):
            return False
