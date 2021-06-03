from pathlib import Path

from autograde.cli.util import namespace_args, find_archives, traverse_archives
from autograde.util import logger


@namespace_args
def cmd_report(result: str, **_) -> int:
    """Inject a human readable report (standalone HTML) into result archive(s)"""

    for archive in traverse_archives(find_archives(Path(result)), mode='a'):
        logger.info(f'render report for {archive.filename}')
        archive.inject_report()

    return 0
