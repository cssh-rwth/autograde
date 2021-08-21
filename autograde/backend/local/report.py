from pathlib import Path

from autograde.backend.local.util import find_archives, traverse_archives
from autograde.util import logger


def cmd_report(result: Path) -> int:
    for archive in traverse_archives(find_archives(result), mode='a'):
        logger.info(f'render report for {archive.filename}')
        archive.inject_report()

    return 0
