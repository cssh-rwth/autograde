from pathlib import Path

from autograde.cli.util import namespace_args, list_results
from autograde.test_result import NotebookTestResultArchive
from autograde.util import logger


@namespace_args
def cmd_report(result: str, **_) -> int:
    """Inject a human readable report (standalone HTML) into result archive(s)"""

    for path in list_results(Path(result)):
        logger.info(f'render report for {path}')
        with NotebookTestResultArchive(path, mode='a') as archive:
            archive.inject_report()

    return 0
