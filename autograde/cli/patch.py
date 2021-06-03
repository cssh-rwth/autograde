from pathlib import Path

from autograde.cli.util import namespace_args, find_archives, traverse_archives
from autograde.util import logger


@namespace_args
def cmd_patch(patch: str, result: str, **_) -> int:
    """Patch result archive(s) with results from a different run"""
    patch = Path(patch)
    result = Path(result)

    # load patches
    patches = {a.results.checksum: a.results for a in traverse_archives(find_archives(patch))}

    # inject patches
    for archive in traverse_archives(find_archives(result), mode='a'):
        if patch := patches.get(archive.results.checksum):
            archive.inject_patch(patch)
        else:
            logger.warning(f'no patch for {archive.filename} found')

    return 0
