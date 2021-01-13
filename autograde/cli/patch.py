from autograde.test_result import NotebookTestResultArchive
from autograde.util import logger


def cmd_patch(args):
    """Patch result archive(s) with results from a different run"""

    from autograde.cli.util import list_results

    # load patches
    patches = dict()
    for path in list_results(args.patch):
        with NotebookTestResultArchive(path) as archive:
            patches[archive.results.checksum] = archive.results

    # inject patches
    for path in list_results(args.result):
        with NotebookTestResultArchive(path, mode='a') as archive:
            if patch := patches.get(archive.results.checksum):
                archive.inject_patch(patch)
            else:
                logger.warn(f'no patch for {path} found')

    return 0
