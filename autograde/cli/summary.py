from pathlib import Path
from typing import Optional

from autograde.cli.util import namespace_args
from autograde.test_result import NotebookTestResultArchive
from autograde.util import logger, render


@namespace_args
def cmd_summary(result: Optional[str] = None, **_) -> int:
    """Generate human & machine readable summary of results"""

    from autograde.cli.util import list_results, merge_results, b64str, plot_score_distribution, summarize_results

    path = Path(result or Path.cwd()).expanduser().absolute()
    assert path.is_dir(), f'{path} is no regular directory'

    def reader():
        for archive_path in list_results(path):
            logger.debug(f'mount {archive_path}')
            with NotebookTestResultArchive(archive_path, mode='r') as archive:
                yield archive

    logger.info('render raw.csv')
    results_df = merge_results(reader())
    results_df.to_csv(path.joinpath('raw.csv'), index=False)

    logger.info('render summary.csv')
    summary = summarize_results(results_df)
    summary.to_csv(path.joinpath('summary.csv'), index=False)

    plots = [
        dict(
            title='Score Distribution',
            data=b64str(plot_score_distribution(summary))
        ),
    ]

    logger.info('render summary.html')
    with path.joinpath('summary.html').open(mode='wt', encoding='utf-8') as f:
        f.write(render('summary.html', title='summary', summary=summary, plots=plots))

    return 0
