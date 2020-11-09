from autograde.util import logger, cd, mount_tar


def cmd_report(args):
    """Inject a human readable report (standalone HTML) into result archive(s)"""
    from autograde.cli.util import load_patched, render, list_results

    for path in list_results(args.result):
        logger.info(f'render report for {path}')
        with mount_tar(path, mode='a') as tar, cd(tar):
            results = load_patched()
            with open('report.html', mode='wt') as f:
                f.write(render('report.html', title='report', id=results.checksum,
                               results={results.checksum: results}, summary=results.summary()))

    return
