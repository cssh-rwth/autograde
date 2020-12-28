from zipfile import ZipFile

from autograde.util import logger


def cmd_patch(args):
    """Patch result archive(s) with results from a different run"""

    from autograde.cli.util import load_patched, list_results, inject_patch

    # load & index all patches
    patches = dict()
    for path in list_results(args.patch):
        with ZipFile(path) as zipf:
            patch = load_patched(zipf)
            patches[patch.checksum] = patch

    # inject patches
    for path in list_results(args.result):
        with ZipFile(path, mode='a') as zipf:
            result = load_patched(zipf)
            if result.checksum in patches:
                inject_patch(patches[result.checksum], zipf)
            else:
                logger.warn(f'no patch for {path} found')

    return 0
