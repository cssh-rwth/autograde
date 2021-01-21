from pathlib import Path

from autograde.cli.util import namespace_args, list_results
from autograde.test_result import NotebookTestResultArchive
from autograde.util import logger


@namespace_args
def cmd_patch(patch: str, result: str, **_) -> int:
    """Patch result archive(s) with results from a different run"""
    patch = Path(patch)
    result = Path(result)

    # load patches
    patches = dict()
    for path in list_results(patch):
        with NotebookTestResultArchive(path) as archive:
            patches[archive.results.checksum] = archive.results

    # inject patches
    for path in list_results(result):
        with NotebookTestResultArchive(path, mode='a') as archive:
            if patch := patches.get(archive.results.checksum):
                archive.inject_patch(patch)
            else:
                logger.warn(f'no patch for {path} found')

    return 0
