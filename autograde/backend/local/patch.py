from pathlib import Path

from autograde.backend.local.util import find_archives, traverse_archives
from autograde.util import logger


def cmd_patch(result: Path, patch: Path) -> int:
    # load patches
    patches = {a.results.checksum: a.results for a in traverse_archives(find_archives(patch))}

    # inject patches
    for archive in traverse_archives(find_archives(result), mode='a'):
        if patch := patches.get(archive.results.checksum):
            archive.inject_patch(patch)
        else:
            logger.warning(f'no patch for {archive.filename} found')

    return 0
