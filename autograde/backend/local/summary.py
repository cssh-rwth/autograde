from pathlib import Path

from autograde.backend.local.util import find_archives, traverse_archives, merge_results, b64str, \
    plot_score_distribution, summarize_results
from autograde.util import logger, render


def cmd_summary(result: Path) -> int:
    assert result.is_dir(), f'{result} is no regular directory'

    logger.info('render raw.csv')
    results_df = merge_results(traverse_archives(find_archives(result)))
    results_df.to_csv(result.joinpath('raw.csv'), index=False)

    logger.info('render summary.csv')
    summary = summarize_results(results_df)
    summary.to_csv(result.joinpath('summary.csv'), index=False)
    score_distribution = plot_score_distribution(summary)

    plots = [
        dict(
            title='Score Distribution',
            data=b64str(score_distribution) if score_distribution else None
        ),
    ]

    logger.info('render summary.html')
    with result.joinpath('summary.html').open(mode='wt', encoding='utf-8') as f:
        f.write(render('summary.html', title='summary', summary=summary, plots=plots))

    return 0
