from zipfile import ZipFile

from autograde.util import logger


def cmd_report(args):
    """Inject a human readable report (standalone HTML) into result archive(s)"""
    from autograde.cli.util import load_patched, render, list_results

    for path in list_results(args.result):
        logger.info(f'render report for {path}')
        with ZipFile(path, mode='a') as zipf:
            results = load_patched(zipf)
            with zipf.open('report.html', mode='w') as f:
                f.write(render(
                    'report.html',
                    title='report',
                    id=results.checksum,
                    results={results.checksum: results}, summary=results.summary()
                ).encode('utf-8'))

    return
