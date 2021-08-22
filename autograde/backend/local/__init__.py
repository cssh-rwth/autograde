import sys
from pathlib import Path
from typing import Optional

import autograde
from autograde.backend.base import Backend
from autograde.backend.local.audit import cmd_audit
from autograde.backend.local.patch import cmd_patch
from autograde.backend.local.report import cmd_report
from autograde.backend.local.summary import cmd_summary
from autograde.backend.local.test import cmd_tset


class Local(Backend):
    @classmethod
    def is_available(cls) -> bool:
        return True

    def audit(self, result: Path, bind: str, port: int) -> int:
        return cmd_audit(result, bind, port)

    def build(self, requirements: Optional[Path] = None, from_source: bool = False) -> int:
        raise ValueError('command is not supported by this backend')

    def patch(self, result: Path, patch: Path) -> int:
        return cmd_patch(result, patch)

    def report(self, result: Path) -> int:
        return cmd_report(result)

    def summary(self, result: Path) -> int:
        return cmd_summary(result)

    def test(self, test: Path, notebook: Path, target: Path, context: Optional[str] = None) -> int:
        return cmd_tset(test, notebook, target, self.verbosity, context)

    def version(self) -> int:
        print(f'autograde version {autograde.__version__}')
        print(f'python {sys.version.split()[0]} at {sys.executable}')
        print(f'default encoding {sys.getdefaultencoding()}')
        return 0
