from pathlib import Path

from autograde.test_result import NotebookTestResultArchive
from autograde.util import logger, render


def cmd_summary(args):
    """Generate human & machine readable summary of results"""

    from autograde.cli.util import list_results, merge_results, b64str, plot_score_distribution, summarize_results

    path = Path(args.result or Path.cwd()).expanduser().absolute()
    assert path.is_dir(), f'{path} is no regular directory'

    def reader():
        for archive_path in list_results(path):
            logger.debug(f'read {archive_path}')

            with NotebookTestResultArchive(archive_path, mode='r') as archive:
                yield archive

    # merge results
    results_df = merge_results(reader())
    logger.debug('store raw.csv')
    results_df.to_csv(path.joinpath('raw.csv'), index=False)

    # summarize results
    summary = summarize_results(results_df)
    logger.debug('store summary.csv')
    summary.to_csv(path.joinpath('summary.csv'), index=False)

    plots = [
        dict(
            title='Score Distribution',
            data=b64str(plot_score_distribution(summary))
        ),
    ]

    logger.info('render summary.html')
    with open(path.joinpath('summary.html'), mode='wt') as f:
        f.write(render('summary.html', title='summary', summary=summary, plots=plots))

    return 0
