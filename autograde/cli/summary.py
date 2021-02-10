from pathlib import Path
from typing import Optional

from autograde.cli.util import namespace_args, find_archives, traverse_archives, merge_results, b64str, \
    plot_score_distribution, summarize_results
from autograde.util import logger, render


@namespace_args
def cmd_summary(result: Optional[str] = None, **_) -> int:
    """Generate human & machine readable summary of results"""

    path = Path(result or Path.cwd()).expanduser().absolute()
    assert path.is_dir(), f'{path} is no regular directory'

    logger.info('render raw.csv')
    results_df = merge_results(traverse_archives(find_archives(path)))
    results_df.to_csv(path.joinpath('raw.csv'), index=False)

    logger.info('render summary.csv')
    summary = summarize_results(results_df)
    summary.to_csv(path.joinpath('summary.csv'), index=False)
    score_distribution = plot_score_distribution(summary)

    plots = [
        dict(
            title='Score Distribution',
            data=b64str(score_distribution) if score_distribution else None
        ),
    ]

    logger.info('render summary.html')
    with path.joinpath('summary.html').open(mode='wt', encoding='utf-8') as f:
        f.write(render('summary.html', title='summary', summary=summary, plots=plots))

    return 0
