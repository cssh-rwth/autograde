import subprocess
from abc import ABCMeta, abstractmethod
from pathlib import Path
from shlex import quote as shell_escape
from subprocess import CompletedProcess
from typing import Dict, Optional, Any

from pkg_resources import iter_entry_points

from autograde.util import loglevel, logger


class ShellCommand(list):
    def __init__(self, *command: str):
        super().__init__(command)

    def __repr__(self):
        return ' '.join(self)

    def __add__(self, other: 'ShellCommand') -> 'ShellCommand':
        return ShellCommand(*list.__add__(self, other))

    def __iadd__(self, other: 'ShellCommand') -> 'ShellCommand':
        self.extend(other)
        return self

    def parameter(self, value: Any):
        self.append(str(value))

    def named_parameter(self, name: str, value: Any):
        self.append(name)
        self.append(str(value))

    def run(self, **kwargs) -> CompletedProcess:
        logger.debug(f'> {self}')
        cmd = ' '.join(map(shell_escape, self)) if kwargs.get('shell', False) else self
        return subprocess.run(cmd, **kwargs)


class AutogradeCommand(ShellCommand):
    def verbosity(self, verbosity: int):
        if verbosity > 0:
            vs = 'v' * verbosity
            self.parameter(f'-{vs}')


class Backend(metaclass=ABCMeta):
    supported: Dict[str, type] = {}
    available: Dict[str, type] = {}

    def __init__(self, tag: str, verbosity: int):
        self.tag = tag
        self.verbosity = max(0, min(verbosity, 3))
        self.log_level = loglevel(self.verbosity)

    @classmethod
    @abstractmethod
    def is_available(cls) -> bool:
        raise NotImplementedError

    @classmethod
    def load(cls, name: str, **kwargs) -> 'Backend':
        """Instantiate a backend by name"""
        if backend := cls.available.get(name):
            return backend(**kwargs)
        raise ValueError(f'"{name}" backend is not available, chose from {set(cls.available)}')

    @abstractmethod
    def audit(self, result: Path, bind: str, port: int) -> int:
        """Launch a web interface for manually auditing test archives"""
        raise NotImplementedError

    @abstractmethod
    def build(self, requirements: Optional[Path] = None, from_source: bool = False) -> int:
        """Build autograde container image"""
        raise NotImplementedError

    @abstractmethod
    def patch(self, result: Path, patch: Path) -> int:
        """Patch result archive(s) with results from another run"""
        raise NotImplementedError

    @abstractmethod
    def report(self, result: Path) -> int:
        """Inject a human readable report (standalone HTML) into result archive(s)"""
        raise NotImplementedError

    @abstractmethod
    def summary(self, result: Path) -> int:
        """Generate human & machine readable summaries of given result archives"""
        raise NotImplementedError

    @abstractmethod
    def test(self, test: Path, notebook: Path, target: Path, context: Optional[Path] = None) -> int:
        """Run autograde test script on jupyter notebook(s)"""
        raise NotImplementedError

    @abstractmethod
    def version(self) -> int:
        """Display version of autograde"""
        raise NotImplementedError


Backend.supported = {e.name: e.load() for e in iter_entry_points('ag_backends')}
Backend.available = {n: b for n, b in Backend.supported.items() if b.is_available()}
