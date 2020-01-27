#!/usr/bin/env python3
# ensure matplotlib uses the right backend (this has to be done before import of pyplot!)
import matplotlib as mpl
mpl.use('Agg')

# Standard library modules.
import os
import sys
import json
import tarfile
import argparse
import subprocess
from pathlib import Path
from functools import reduce
from itertools import combinations
from difflib import SequenceMatcher

# Third party modules.
import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns

# Local modules
from autograde.util import logger, loglevel, project_root, cd

# Globals and constants variables.
CONTAINER_TAG = 'autograde'


def build(args):
    cmd = [args.backend, 'build', '-t', CONTAINER_TAG, '.']

    with cd(project_root()):
        return subprocess.run(cmd, capture_output=args.quiet).returncode


def test(args):
    path_tst = Path(args.test).expanduser().absolute()
    path_nbk = Path(args.notebook).expanduser().absolute()
    path_tgt = Path(args.target or Path.cwd()).expanduser().absolute()
    path_cxt = Path(args.context).expanduser().absolute() if args.context else None

    assert path_tst.is_file(), f'{path_tst} is no regular file'
    assert path_nbk.is_file() or path_nbk.is_dir(), f'{path_nbk} is no regular file or directory'
    assert path_tgt.is_dir(), f'{path_tgt} is no regular directory'
    assert path_cxt is None or path_cxt.is_dir(), f'{path_cxt} is no regular directory'

    if path_nbk.is_file():
        notebooks = [path_nbk]

    else:
        notebooks = list(filter(
            lambda p: '.ipynb_checkpoints' not in p.parts,
            path_nbk.rglob('*.ipynb')
        ))

    def run(path_nb_):
        cmd = [
            args.backend, 'run',
            '-v', f'{path_tst}:/autograde/test.py',
            '-v', f'{path_nb_}:/autograde/notebook.ipynb',
            '-v', f'{path_tgt}:/autograde/target',
            *(('-v', f'{path_cxt}:/autograde/context:ro') if path_cxt else ()),
            '-u', str(os.geteuid()),
            CONTAINER_TAG
        ]

        logger.info(f'test: {path_nb_}')
        logger.debug(' '.join(cmd))

        return subprocess.run(cmd).returncode

    return reduce(max, map(run, notebooks))


def summary(args):
    root = Path(args.location or Path.cwd()).expanduser().absolute()

    assert root.is_dir(), f'{root} is no regular directory'

    logger.info(f'summarize results in: {root}')

    # extract results
    results = []
    code = {}
    for path in root.rglob('results_*.tar.xz'):
        logger.debug(f'read {path}')
        with tarfile.open(path, mode='r') as tar:
            try:
                r = json.load(tar.extractfile(tar.getmember('test_results.json')))
                results.append(r)

                c = tar.extractfile(tar.getmember('code.py')).read().decode('utf-8')
                code[r['checksum']['blake2bsum']] = c

            except KeyError as error:
                logger.warning(f'{path} does not contain {error}, skip')
                continue

    # consistency check
    max_scores = {r['summary']['score_max'] for r in results}
    assert len(max_scores) == 1, 'found different max scores in results'
    max_score = max_scores.pop()

    # summary
    header = ['student_id', 'last_name', 'first_name', 'score', 'max_score', 'checksum']

    def row_factory():
        for r in results:
            for member in r['team_members']:
                yield (
                    member['student_id'],
                    member['last_name'],
                    member['first_name'],
                    r['summary']['score'],
                    max_score,
                    r['checksum']['blake2bsum']
                )

    df = pd.DataFrame(row_factory(), columns=header).sort_values(by='score')
    df['multiple_submissions'] = df['student_id'].duplicated(keep=False)

    # csv export
    logger.debug('save summary')
    df.to_csv(root.joinpath('summary.csv'), index=False)

    # plot score distributions
    logger.debug('plot score distributions')
    plt.clf()
    ax = sns.distplot(df[~df['student_id'].duplicated(keep='first')]['score'], rug=True, fit=norm, bins=int(max_score))
    ax.set_xlim(0, max_score)
    ax.set_xlabel('score')
    ax.set_ylabel('share')
    ax.set_title('score distribution without duplicates (take lower score)')

    plt.tight_layout()
    plt.savefig(root.joinpath('score_distribution.pdf'))

    # basic fraud detection
    logger.debug('apply fraud detection')
    hashes = sorted(code)
    diffs = pd.DataFrame(np.NaN, index=hashes, columns=hashes)

    for h in hashes:
        diffs.loc[h][h] = 1.

    for (ha, ca), (hb, cb) in combinations(code.items(), 2):
        diffs.loc[ha][hb] = diffs.loc[hb][ha] = SequenceMatcher(a=ca, b=cb).ratio()

    plt.clf()
    ax = sns.heatmap(diffs, vmin=0., vmax=1., xticklabels=True, yticklabels=True)
    ax.set_title('similarity of notebook code')

    plt.tight_layout()
    plt.savefig(root.joinpath('similarities.pdf'))

    logger.debug('done')
    return 0


def main():
    parser = argparse.ArgumentParser(description='run tests on jupyter notebook', prog='autograde')

    parser.add_argument('-v', '--verbose', action='count', default=0, help='verbosity level')
    parser.add_argument('-e', '--backend', type=str, default='docker', choices=['docker', 'podman'],
                        metavar='', help='backend to use')

    subparsers = parser.add_subparsers(help='sub command help')

    bld_parser = subparsers.add_parser('build')
    bld_parser.add_argument('-q', '--quiet', action='store_true', help='mute output')
    bld_parser.set_defaults(func=build)

    exe_parser = subparsers.add_parser('test')
    exe_parser.add_argument('test', type=str, help='autograde test script')
    exe_parser.add_argument(
        'notebook', type=str,
        help='the jupyter notebook to be tested or a directory to be searched for notebooks'
    )
    exe_parser.add_argument('-t', '--target', type=str, metavar='', help='where to store results')
    exe_parser.add_argument('-c', '--context', type=str, metavar='', help='context directory')
    exe_parser.set_defaults(func=test)

    sum_parser = subparsers.add_parser('summary')
    sum_parser.add_argument('location', type=str, help='location with result files to summarize')
    sum_parser.set_defaults(func=summary)

    args = parser.parse_args()

    logger.setLevel(loglevel(args.verbose))
    logger.debug(f'args: {args}')

    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
