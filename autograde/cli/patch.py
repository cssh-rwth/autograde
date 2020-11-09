from autograde.util import logger, cd, mount_tar


def cmd_patch(args):
    """Patch result archive(s) with results from a different run"""

    from autograde.cli.util import load_patched, list_results, inject_patch

    # load & index all patches
    patches = dict()
    for path in list_results(args.patch):
        with mount_tar(path) as tar:
            patch = load_patched(tar)
            patches[patch.checksum] = patch

    # inject patches
    for path in list_results(args.result):
        with mount_tar(path, mode='a') as tar, cd(tar):
            result = load_patched()
            if result.checksum in patches:
                inject_patch(patches[result.checksum])
            else:
                logger.warn(f'no patch for {path} found')

    return 0
