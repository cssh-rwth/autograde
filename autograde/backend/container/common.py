import shutil
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from subprocess import CompletedProcess
from tempfile import TemporaryDirectory
from typing import Any, Optional

from autograde.backend.base import Backend
from autograde.util import cd, logger, project_root, render


class Command(list):
    def __init__(self, *command: str):
        super().__init__(command)

    def __repr__(self):
        return ' '.join(self)

    def __add__(self, other: 'Command') -> 'Command':
        return Command(*list.__add__(self, other))

    def __iadd__(self, other: 'Command') -> 'Command':
        self.extend(other)
        return self

    def parameter(self, value: Any):
        self.append(str(value))

    def named_parameter(self, name: str, value: Any):
        self.append(name)
        self.append(str(value))

    def run(self, **kwargs) -> CompletedProcess:
        logger.debug(f'> {self}')
        cmd = ' '.join(self) if (shell := kwargs.get('shell')) and shell else self
        return subprocess.run(cmd, **kwargs)


class ContainerCommand(Command):
    mounts = Path('/autograde/mounts')
    mount_flags = {'ro', 'z', 'Z'}
    mount_flags_default = {'Z'}

    def mount(self, path: Path, mount_point: str, *flags: str) -> Path:
        flags = set(flags).union(self.mount_flags_default)
        assert set.issubset(flags, self.mount_flags)

        mount_point = self.mounts.joinpath(mount_point)
        flags = f':{",".join(flags)}' if flags else ''
        self.named_parameter('--volume', f'{path.absolute()}:{mount_point}{flags}')
        return mount_point

    def bind(self, bind: str, host_port: int, container_port):
        self.named_parameter('--publish', f'{bind}:{host_port}:{container_port}')


class PodmanCommand(ContainerCommand):
    def __init__(self, *command: str):
        super().__init__('podman', *command)


class DockerCommand(ContainerCommand):
    def __init__(self, *command: str):
        super().__init__('docker', *command)


class AutogradeCommand(Command):
    def verbosity(self, verbosity: int):
        if verbosity > 0:
            vs = 'v' * verbosity
            self.parameter(f'-{vs}')


class ContainerBackend(ABC, Backend):
    @staticmethod
    @abstractmethod
    def new_container_command(subcommand: str) -> ContainerCommand:
        raise NotImplementedError

    def new_autograde_command(self, subcommand: str) -> AutogradeCommand:
        cmd = AutogradeCommand(self.tag)
        cmd.verbosity(self.verbosity)
        cmd.parameter(subcommand)
        return cmd

    def audit(self, result: Path, bind: str, port: int) -> int:
        # container command
        c_cmd = self.new_container_command('run')
        mp_result = c_cmd.mount(result, result.name)
        c_cmd.bind(bind, port, 8000)
        # autograde command
        ag_cmd = self.new_autograde_command('audit')
        ag_cmd.named_parameter('--bind', '0.0.0.0')
        ag_cmd.named_parameter('--port', 8000)
        ag_cmd.parameter(mp_result)
        return (c_cmd + ag_cmd).run().returncode

    def build(self, requirements: Optional[Path] = None, from_source: bool = False) -> int:
        # container command
        with TemporaryDirectory() as tmp, cd(tmp):
            kwargs = {}
            tmp = Path(tmp)
            dockerfile = tmp.joinpath('Dockerfile')
            source_dir = tmp.joinpath('src')
            if requirements:
                requirements_copy = tmp.joinpath('requirements.txt')
                requirements_copy.write_bytes(requirements.read_bytes())
                kwargs['requirements'] = requirements_copy.relative_to(tmp)
            if from_source:
                shutil.copytree(project_root(), source_dir)
                kwargs['source_dir'] = source_dir.relative_to(tmp)
            dockerfile.write_text(render('container.dockerfile', minify=False, **kwargs))
            cmd = self.new_container_command('build')
            cmd.named_parameter('--file', dockerfile)
            cmd.named_parameter('--tag', self.tag)
            cmd.parameter('.')
            return cmd.run().returncode

    def untag(self):
        cmd = self.new_container_command('image')
        cmd.parameter('untag')
        cmd.parameter(self.tag)
        return cmd.run().returncode

    def patch(self, result: Path, patch: Path) -> int:
        # container command
        c_cmd = self.new_container_command('run')
        mp_result = c_cmd.mount(result, f'patchee_{result.name}')
        mp_patch = c_cmd.mount(patch, f'patch_{patch.name}')
        # autograde command
        ag_cmd = self.new_autograde_command('patch')
        ag_cmd.parameter(mp_result)
        ag_cmd.parameter(mp_patch)
        return (c_cmd + ag_cmd).run().returncode

    def report(self, result: Path) -> int:
        # container command
        c_cmd = self.new_container_command('run')
        mp_result = c_cmd.mount(result, result.name)
        # autograde command
        ag_cmd = self.new_autograde_command('report')
        ag_cmd.parameter(mp_result)
        return (c_cmd + ag_cmd).run().returncode

    def summary(self, result: Optional[Path] = None) -> int:
        # container command
        c_cmd = self.new_container_command('run')
        mp_result = c_cmd.mount(result or Path('.'), 'result')
        # autograde command
        ag_cmd = self.new_autograde_command('summary')
        ag_cmd.parameter(mp_result)
        return (c_cmd + ag_cmd).run().returncode

    def test(self, test: Path, notebook: Path, target: Path, context: Optional[Path] = None) -> int:
        # container command
        c_cmd = self.new_container_command('run')
        mp_test = c_cmd.mount(test, 'test.py')
        mp_notebook = c_cmd.mount(notebook, 'notebook.ipynb')
        mp_target = c_cmd.mount(target, 'target')
        mp_context = c_cmd.mount(context, 'context') if context else None
        # autograde command
        ag_cmd = self.new_autograde_command('test')
        ag_cmd.parameter(mp_test)
        ag_cmd.parameter(mp_notebook)
        ag_cmd.named_parameter('--target', mp_target)
        if mp_context:
            ag_cmd.named_parameter('--context', mp_context)
        return (c_cmd + ag_cmd).run().returncode

    def version(self) -> int:
        # container command
        c_cmd = self.new_container_command('run')
        # autograde command
        ag_cmd = self.new_autograde_command('version')
        return (c_cmd + ag_cmd).run().returncode
