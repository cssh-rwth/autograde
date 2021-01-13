from autograde.util import logger


def cmd_report(args):
    """Inject a human readable report (standalone HTML) into result archive(s)"""
    from autograde.cli.util import list_results
    from autograde.test_result import NotebookTestResultArchive

    for path in list_results(args.result):
        logger.info(f'render report for {path}')
        with NotebookTestResultArchive(path, mode='a') as archive:
            # rendering happens implicitly
            _ = archive.report

    return 0
