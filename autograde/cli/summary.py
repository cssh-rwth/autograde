from pathlib import Path

from autograde.util import logger, cd, mount_tar


def cmd_summary(args):
    """Generate human & machine readable summary of results"""

    from autograde.cli.util import load_patched, render, list_results, merge_results, b64str, plot_fraud_matrix, \
        plot_score_distribution, summarize_results

    path = Path(args.result or Path.cwd()).expanduser().absolute()
    assert path.is_dir(), f'{path} is no regular directory'

    results = list()
    sources = dict()
    for path_ in list_results(path):
        logger.debug(f'read {path_}')

        with mount_tar(path_) as tar, cd(tar):
            r = load_patched()
            results.append(r)

            with open('code.py', mode='rt') as f:
                sources[r.checksum] = f.read()

    # merge results
    results_df = merge_results(results)
    logger.debug('store raw.csv')
    results_df.to_csv(path.joinpath('raw.csv'), index=False)

    # summarize results
    summary_df = summarize_results(results)
    logger.debug('store summary.csv')
    summary_df.to_csv(path.joinpath('summary.csv'), index=False)

    plots = dict(
        distribution=b64str(plot_score_distribution(summary_df)),
        similarities=b64str(plot_fraud_matrix(sources))
    )

    logger.info('render summary.html')
    with open(path.joinpath('summary.html'), mode='wt') as f:
        f.write(render('summary.html', title='summary', summary=summary_df, plots=plots))

    return 0
