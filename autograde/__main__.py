#!/usr/bin/env python3
# ensure matplotlib uses the right backend (this has to be done before import of pyplot!)
import matplotlib as mpl
mpl.use('Agg')

# Standard library modules.
import io
import os
import sys
import json
import math
import base64
import argparse
import subprocess
from typing import List
from pathlib import Path
from contextlib import ExitStack
from itertools import combinations
from difflib import SequenceMatcher
from collections import OrderedDict

# Third party modules.
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.linalg import LinAlgError
from jinja2 import Environment, PackageLoader, select_autoescape

# Local modules
import autograde
from autograde.util import logger, timestamp_utc_iso, loglevel, project_root, cd, mount_tar
from autograde.static import CSS
from autograde.notebook_test import Results

# Globals and constants variables.
CONTAINER_TAG = 'autograde'
JINJA_ENV = Environment(
    loader=PackageLoader('autograde', 'templates'),
    autoescape=select_autoescape(['html', 'xml']),
    trim_blocks=True,
    lstrip_blocks=True
)


def list_results(path='.', prefix='results') -> List[Path]:
    path = Path(path).expanduser().absolute()

    if path.is_file():
        return [path]

    return list(path.rglob(f'{prefix}_*.tar.xz'))


def inject_patch(results, path='.', prefix='results') -> Path:
    path = Path(path)
    patch_count = len(list(path.glob(f'{prefix}_patch*.json')))
    path = path.joinpath(f'{prefix}_patch_{patch_count+1}.json')

    with path.open(mode='wt') as f:
        json.dump(results.to_dict(), f, indent=4)

    return path


def load_patched(path='.', prefix='results') -> Results:
    path = Path(path)

    with path.joinpath(f'{prefix}.json').open(mode='rt') as f:
        results = Results.from_json(f.read())

    for patch_path in sorted(path.glob(f'{prefix}_patch*.json')):
        with patch_path.open(mode='rt') as f:
            results = results.patch(Results.from_json(f.read()))

    return results


def render(template, **kwargs):
    return JINJA_ENV.get_template(template).render(
        autograde=autograde,
        css=CSS,
        timestamp=timestamp_utc_iso(),
        **kwargs
    )


def build(args):
    """Build autograde container image for specified backend"""
    if args.backend is None:
        logger.warning('no backend specified')
        return 1

    cmd = [args.backend, 'build', '-t', CONTAINER_TAG, '.']

    with cd(project_root()):
        return subprocess.run(cmd, capture_output=args.quiet).returncode


def test(args):
    """Run autograde test script on jupyter notebook(s)"""
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
        if args.backend is None:
            cmd = [
                'python', str(path_tst),
                str(path_nb_),
                '-t', str(path_tgt),
                *(('-c', str(path_cxt)) if path_cxt else ()),
                *(('-' + 'v' * args.verbose,) if args.verbose > 0 else ())
            ]
        elif args.backend == 'docker':
            cmd = [
                args.backend, 'run',
                '-v', f'{path_tst}:/autograde/test.py',
                '-v', f'{path_nb_}:/autograde/notebook.ipynb',
                '-v', f'{path_tgt}:/autograde/target',
                *(('-v', f'{path_cxt}:/autograde/context:ro') if path_cxt else ()),
                '-u', str(os.geteuid()),
                CONTAINER_TAG,
                *(('-' + 'v' * args.verbose,) if args.verbose > 0 else ())
            ]
        elif args.backend == 'podman':
            cmd = [
                args.backend, 'run',
                '-v', f'{path_tst}:/autograde/test.py:Z',
                '-v', f'{path_nb_}:/autograde/notebook.ipynb:Z',
                '-v', f'{path_tgt}:/autograde/target:Z',
                *(('-v', f'{path_cxt}:/autograde/context:Z') if path_cxt else ()),
                CONTAINER_TAG,
                *(('-' + 'v' * args.verbose,) if args.verbose > 0 else ())
            ]
        else:
            raise ValueError(f'unknown backend: {args.backend}')

        logger.info(f'test: {path_nb_}')
        logger.debug(' '.join(cmd))

        return subprocess.run(cmd).returncode

    return sum(map(run, notebooks))


def patch(args):
    """Patch result archive(s) with results from a different run"""
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


def audit(args):
    """Launch a web interface for manually auditing test results"""
    import logging
    from flask import Flask, redirect, request
    import flask.cli as flask_cli
    from werkzeug.exceptions import HTTPException, InternalServerError

    with ExitStack() as exit_stack:
        # mount & index all results
        mounts = OrderedDict()
        for path in list_results(args.result):
            mount = exit_stack.enter_context(mount_tar(path, mode='a'))
            mounts[load_patched(mount).checksum] = Path(mount)

        patched = set()
        next_ids = dict(zip(mounts, list(mounts)[1:]))

        # create actual flask application
        app = Flask('autograde - audit')

        # monkey patching for nicer cli output
        flask_cli.show_server_banner = lambda *_, **__: logger.debug('suppress flask banner')
        app.logger = logger
        logging.root = logger

        @app.errorhandler(Exception)
        def handle_error(error):
            logger.warning(error)
            error = error if isinstance(error, HTTPException) else InternalServerError()
            return render('error.html', title='Oooops', error=error), error.code

        @app.route('/')
        def route_root():
            logger.debug('route index, re-direct')
            return redirect('/audit')

        @app.route('/audit', strict_slashes=False)
        @app.route('/audit/<string:id>')
        def route_audit(id=None):
            logger.debug('route audit')
            path = mounts.get(id)
            return render('audit.html', title='audit', mounts=mounts, next_id=next_ids.get(id),
                          results=load_patched(path) if path else None, patched=patched)

        @app.route('/patch', methods=('POST',))
        def route_patch():
            logger.debug('route patch')

            if (id := request.form.get('id')) and (mount := mounts.get(id)):
                scores = dict()
                comments = dict()
                results = load_patched(mount)

                results.title = 'manual audit'
                results.timestamp = timestamp_utc_iso()

                # extract form data
                for key, value in request.form.items():
                    if key.startswith('score:'):
                        scores[key.split(':')[-1]] = math.nan if value == '' else float(value)
                    elif key.startswith('comment:'):
                        comments[key.split(':')[-1]] = value

                # update results
                for result in results.results:
                    if not math.isclose((score := scores.get(result.id)), result.score):
                        logger.debug(f'update score of result {result.id[:8]}')
                        result.score = score

                    if comment := comments.get(result.id):
                        logger.debug(f'update messages of result {result.id[:8]}')
                        result.messages.append(comment.strip())

                # patch
                inject_patch(results, mount)

                # flag results as patched
                patched.add(id)

                # update report if it exists
                if path.joinpath('report.html').exists():
                    logger.debug(f'update report for {id}')
                    render('report.html', title='report', results=results, summary=results.summary())

                if next_id := next_ids.get(id):
                    return redirect(f'/audit/{next_id}')

            return redirect('/audit')

        @app.route('/report/<string:id>')
        def route_report(id):
            results = load_patched(mounts[id])
            return render('report.html', title='report (preview)', results=results, summary=results.summary())

        app.run(host=args.bind, port=args.port)


def report(args):
    """Inject a human readable report (standalone HTML) into result archive(s)"""
    for path in list_results(args.result):
        logger.info(f'render report for {path}')
        with mount_tar(path, mode='a') as tar, cd(tar):
            results = load_patched()
            with open('report.html', mode='wt') as f:
                f.write(render('report.html', title='report', results=results, summary=results.summary()))

    return 0


def summary(args):
    """Generate humand & machine readable summary of results"""
    root = Path(args.result or Path.cwd()).expanduser().absolute()

    assert root.is_dir(), f'{root} is no regular directory'

    logger.info(f'summarize results in: {root}')

    # extract results
    results = list()
    code = dict()
    paths = dict()
    for path in list_results(root):
        logger.debug(f'read {path}')

        with mount_tar(path) as tar, cd(tar):
            r = load_patched()
            results.append(r)

            paths[r.checksum] = path

            with open('code.py', mode='rt') as f:
                code[r.checksum] = f.read()

    # consistency check
    max_scores = {r.summary().score_max for r in results}
    assert len(max_scores) == 1, 'found different max scores in results'
    max_score = max_scores.pop()

    # summary
    header = ['student_id', 'last_name', 'first_name', 'score', 'max_score', 'checksum', 'path']

    def row_factory():
        for r in results:
            for member in r.team_members:
                yield (
                    member.student_id,
                    member.last_name,
                    member.first_name,
                    r.summary().score,
                    max_score,
                    r.checksum,
                    paths[r.checksum].relative_to(root)
                )

    df = pd.DataFrame(row_factory(), columns=header).sort_values(by='last_name')
    df['multiple_submissions'] = df['student_id'].duplicated(keep=False)

    # csv export
    logger.debug('save summary')
    df.drop(columns=['path']).to_csv(root.joinpath('summary.csv'), index=False)

    # plotting
    plots = dict()
    # plot score distributions
    logger.debug('plot score distributions')
    plt.clf()
    ax = plt.gca()
    try:
        sns.distplot(
            df[~df['student_id'].duplicated(keep='first')]['score'], rug=True, fit=norm,
            bins=int(max_score), ax=ax
        )
    except LinAlgError:
        logger.warning('unable to plot score distribution')
    ax.set_xlim(0, max_score)
    ax.set_xlabel('score')
    ax.set_ylabel('share')
    ax.set_title('score distribution without duplicates (takes lower score)')

    with io.BytesIO() as buffer:
        plt.savefig(buffer, format='svg', transparent=False)
        plots['score_distribution'] = base64.b64encode(buffer.getvalue()).decode('utf-8')

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

    with io.BytesIO() as buffer:
        plt.savefig(buffer, format='svg', transparent=False)
        plots['similarities'] = base64.b64encode(buffer.getvalue()).decode('utf-8')

    logger.info('render summary.html')
    with open(root.joinpath('summary.html'), mode='wt') as f:
        f.write(render('summary.html', title='summary', summary=df, plots=plots))

    logger.debug('done')
    return 0


def version(_):
    """Display version of autograde"""
    print(f'autograde {autograde.__version__}')
    return 0


def main(args=None):
    parser = argparse.ArgumentParser(
        description=f'utility for grading jupyter notebooks',
        epilog=f'autograde on github: https://github.com/cssh-rwth/autograde',
        prog='autograde',
    )

    # global flags
    parser.add_argument('-v', '--verbose', action='count', default=0, help='verbosity level')
    parser.add_argument('-e', '--backend', type=str, default=None, choices=['docker', 'podman'],
                        metavar='', help='container backend to use')

    subparsers = parser.add_subparsers(help='sub command help')

    # build sub command
    bld_parser = subparsers.add_parser('build', help=build.__doc__)
    bld_parser.add_argument('-q', '--quiet', action='store_true', help='mute output')
    bld_parser.set_defaults(func=build)

    # test sub command
    tst_parser = subparsers.add_parser('test', help=test.__doc__)
    tst_parser.add_argument('test', type=str, help='autograde test script')
    tst_parser.add_argument('notebook', type=str, help='the jupyter notebook(s) to be tested')
    tst_parser.add_argument('-t', '--target', type=str, metavar='', help='where to store results')
    tst_parser.add_argument('-c', '--context', type=str, metavar='', help='context directory')
    tst_parser.set_defaults(func=test)

    # patch sub command
    ptc_parser = subparsers.add_parser('patch', help=patch.__doc__)
    ptc_parser.add_argument('result', type=str, help='result archive(s) to be patched')
    ptc_parser.add_argument('patch', type=str, help='result archive(s) for patching')
    ptc_parser.set_defaults(func=patch)

    # audit sub command
    adt_parser = subparsers.add_parser('audit', help=report.__doc__)
    adt_parser.add_argument('result', type=str, help='result archive(s) to audit')
    adt_parser.add_argument('-b', '--bind', type=str, default='127.0.0.1', help='host to bind to')
    adt_parser.add_argument('-p', '--port', type=int, default=5000, help='port')
    adt_parser.set_defaults(func=audit)

    # report sub command
    rpt_parser = subparsers.add_parser('report', help=report.__doc__)
    rpt_parser.add_argument('result', type=str, help='result archive(s) for creating the report')
    rpt_parser.set_defaults(func=report)

    # summary sub command
    sum_parser = subparsers.add_parser('summary', help=summary.__doc__)
    sum_parser.add_argument('result', type=str, help='result archives to summarize')
    sum_parser.set_defaults(func=summary)

    # version sub command
    vrs_parser = subparsers.add_parser('version', help=version.__doc__)
    vrs_parser.set_defaults(func=version)

    args = parser.parse_args(args)

    logger.setLevel(loglevel(args.verbose))
    logger.debug(f'args: {args}')

    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
