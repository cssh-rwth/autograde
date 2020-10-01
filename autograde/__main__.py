#!/usr/bin/env python3
# ensure matplotlib uses the right backend (this has to be done before import of pyplot!)
import matplotlib as mpl
mpl.use('Agg')

import argparse
import base64
import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
from collections import OrderedDict
from contextlib import ExitStack
from copy import deepcopy
from difflib import SequenceMatcher
from itertools import combinations
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Dict, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from jinja2 import Environment, PackageLoader, select_autoescape
from scipy.linalg import LinAlgError
from scipy.stats import norm

import autograde
from autograde.notebook_test import Result, Results
from autograde.static import CSS, FAVICON
from autograde.util import logger, parse_bool, timestamp_utc_iso, loglevel, cd, \
    mount_tar

# Globals and constants variables.
FAVICON = base64.b64encode(FAVICON).decode('utf-8')
JINJA_ENV = Environment(
    loader=PackageLoader('autograde', 'templates'),
    autoescape=select_autoescape(['html', 'xml']),
    trim_blocks=True,
    lstrip_blocks=True
)


def b64str(data) -> str:
    """Convert bytes like in base64 encoded utf-8 string"""
    return base64.b64encode(data).decode('utf-8')


def list_results(path='.', prefix='results') -> List[Path]:
    """List all results archives at given location"""
    path = Path(path).expanduser().absolute()

    if path.is_file():
        return [path]

    return list(path.rglob(f'{prefix}_*.tar.xz'))


def inject_patch(results: Results, path='.', prefix: str = 'results') -> Path:
    """Store results as patch in mounted results archive"""
    path = Path(path)
    ct = len(list(path.glob(f'{prefix}_patch*.json')))

    with cd(path):
        with open(f'{prefix}_patch_{ct + 1:02d}.json', mode='wt') as f:
            json.dump(results.to_dict(), f, indent=4)

        # update report if it exists
        if Path('report.html').exists():
            results = load_patched()
            logger.debug(f'update report for {results.checksum}')
            with open('report.html', mode='wt') as f:
                f.write(render('report.html', title='report', id=results.checksum,
                               results={results.checksum: results}, summary=results.summary()))

    return path


def load_patched(path='.', prefix: str = 'results') -> Results:
    """Load results and apply patches from mounted results archive"""
    path = Path(path)

    with path.joinpath(f'{prefix}.json').open(mode='rt') as f:
        results = Results.from_json(f.read())

    for patch_path in sorted(path.glob(f'{prefix}_patch*.json')):
        with patch_path.open(mode='rt') as f:
            results = results.patch(Results.from_json(f.read()))

    return results


def render(template, **kwargs):
    """Render template with default values set"""
    return JINJA_ENV.get_template(template).render(
        autograde=autograde,
        css=CSS,
        favicon=FAVICON,
        timestamp=timestamp_utc_iso(),
        **kwargs
    )


def cmd_build(args):
    """Build autograde container image for specified backend"""
    if args.backend is None:
        logger.warning('no backend specified')
        return 1

    if args.requirements:
        with Path(args.requirements).open(mode='rt') as f:
            requirements = list(filter(lambda s: s, map(str.strip, f.read().split('\n'))))
    else:
        requirements = []

    with TemporaryDirectory() as tmp:
        logger.debug(f'copy source to {tmp}')
        shutil.copytree('.', tmp, dirs_exist_ok=True)

        if requirements:
            logger.info(f'add additional requirements: {requirements}')
            with Path(tmp).joinpath('requirements.txt').open(mode='w') as f:
                logger.debug('add additional requirements: ' + ' '.join(requirements))
                f.write('\n'.join(requirements))

        if 'docker' in args.backend:
            cmd = ['docker', 'build', '-t', args.tag, tmp]
        elif args.backend == 'podman':
            cmd = ['podman', 'build', '-t', args.tag, '--cgroup-manager=cgroupfs', tmp]
        else:
            raise ValueError(f'unknown backend: {args.backend}')

        logger.debug('run: ' + ' '.join(cmd))
        return subprocess.run(cmd, capture_output=args.quiet).returncode


def cmd_test(args):
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
        elif 'docker' in args.backend:
            cmd = [
                'docker', 'run',
                '-v', f'{path_tst}:/autograde/test.py',
                '-v', f'{path_nb_}:/autograde/notebook.ipynb',
                '-v', f'{path_tgt}:/autograde/target',
                *(('-v', f'{path_cxt}:/autograde/context:ro') if path_cxt else ()),
                *(('-u', str(os.geteuid())) if 'rootless' not in args.backend else ()),
                args.tag,
                *(('-' + 'v' * args.verbose,) if args.verbose > 0 else ())
            ]
        elif args.backend == 'podman':
            cmd = [
                'podman', 'run',
                '-v', f'{path_tst}:/autograde/test.py:Z',
                '-v', f'{path_nb_}:/autograde/notebook.ipynb:Z',
                '-v', f'{path_tgt}:/autograde/target:Z',
                *(('-v', f'{path_cxt}:/autograde/context:Z') if path_cxt else ()),
                args.tag,
                *(('-' + 'v' * args.verbose,) if args.verbose > 0 else ())
            ]
        else:
            raise ValueError(f'unknown backend: {args.backend}')

        logger.info(f'test: {path_nb_}')
        logger.debug('run' + ' '.join(cmd))

        if not args.backend:
            return subprocess.call(' '.join(cmd), shell=True)

        return subprocess.run(cmd).returncode

    return sum(map(run, notebooks))


def cmd_patch(args):
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


def merge_results(results) -> pd.DataFrame:
    header = ['student_id', 'last_name', 'first_name', 'notebook_id', 'task_id', 'score', 'max_score']

    def row_factory():
        for r in results:
            for member in r.team_members:
                for t in r:
                    yield (
                        member.student_id,
                        member.last_name,
                        member.first_name,
                        r.checksum,
                        t.id,
                        t.score,
                        t.score_max
                    )

    return pd.DataFrame(row_factory(), columns=header)


def summarize_results(results) -> pd.DataFrame:
    logger.debug(f'summarize {len(results)} results')
    header = ['student_id', 'last_name', 'first_name', 'score', 'max_score', 'patches', 'checksum']

    def row_factory():
        for r in results:
            for member in r.team_members:
                s = r.summary()
                yield (
                    member.student_id,
                    member.last_name,
                    member.first_name,
                    s.score,
                    s.score_max,
                    len(r.applied_patches),
                    r.checksum
                )

    summary_df = pd.DataFrame(row_factory(), columns=header).sort_values(by='score')
    summary_df['duplicate'] = summary_df['student_id'].duplicated(keep=False)

    if not math.isclose(summary_df['max_score'].std(), 0):
        logger.warning('max scores seem not to be consistent!')

    return summary_df.sort_values(by='last_name')


def plot_score_distribution(summary_df: pd.DataFrame):
    logger.debug('plot score distributions')

    summary_df = summary_df.sort_values(by='score')
    max_score = summary_df['max_score'].max()

    plt.clf()
    ax = plt.gca()
    try:
        sns.distplot(
            summary_df[~summary_df['student_id'].duplicated(keep='first')]['score'], rug=True, fit=norm,
            bins=int(max_score), ax=ax
        )
    except LinAlgError as error:
        logger.warning(f'unable to plot score distribution: {error}')

    ax.set_xlim(0, max_score)
    ax.set_xlabel('score')
    ax.set_ylabel('share')
    ax.set_title('score distribution without duplicates (takes lower score)')
    plt.tight_layout()

    with io.BytesIO() as buffer:
        plt.savefig(buffer, format='svg', transparent=False)
        return buffer.getvalue()


def plot_fraud_matrix(sources: Dict[str, str]) -> bytes:
    logger.debug('apply fraud detection')
    hashes = sorted(sources)
    diffs = pd.DataFrame(np.NaN, index=hashes, columns=hashes)

    for h in hashes:
        diffs.loc[h][h] = 1.

    for (ha, ca), (hb, cb) in combinations(sources.items(), 2):
        diffs.loc[ha][hb] = diffs.loc[hb][ha] = SequenceMatcher(a=ca, b=cb).ratio()

    plt.clf()
    ax = sns.heatmap(diffs, vmin=0., vmax=1., xticklabels=True, yticklabels=True)
    ax.set_title('similarity of notebook code')

    with io.BytesIO() as buffer:
        plt.savefig(buffer, format='svg', transparent=False)
        return buffer.getvalue()


class AuditSettings:
    def __init__(self, **kwargs):
        self.update(**kwargs)

    def update(self, selector=None, auditor=None):
        self.selector = re.compile(selector or '')
        self.auditor = auditor or ''

    def select(self, result: Result) -> bool:
        return bool(self.selector.search(result.label))

    def filter_results(self, results: Iterable[Result]) -> Iterable[Result]:
        return filter(self.select, results)

    def format_comment(self, comment):
        if self.auditor:
            return f'{self.auditor}: {comment.strip()}'
        return comment.strip()


def cmd_audit(args):
    """Launch a web interface for manually auditing test results"""
    import logging
    from flask import Flask, redirect, request
    import flask.cli as flask_cli
    from werkzeug.exceptions import HTTPException, InternalServerError

    with ExitStack() as exit_stack:
        # settings
        settings = AuditSettings()

        # mount & index all results
        mounts = OrderedDict()
        sources = dict()
        results = dict()
        for path in list_results(args.result):
            mount = Path(exit_stack.enter_context(mount_tar(path, mode='a')))

            r = load_patched(mount)
            results[r.checksum] = r
            mounts[r.checksum] = mount

            with mount.joinpath('code.py').open(mode='rt') as f:
                sources[r.checksum] = f.read()

        patched = set()
        next_ids = dict(zip(mounts, list(mounts)[1:]))
        prev_ids = dict(((b, a) for a, b in next_ids.items()))

        # create actual flask application
        app = Flask('autograde - audit')

        # monkey patching for nicer cli output
        flask_cli.show_server_banner = lambda *_, **__: logger.debug('suppress flask banner')
        app.logger = logger
        logging.root = logger

        @app.errorhandler(Exception)
        def handle_error(error):
            logger.warning(type(error), error)
            error = error if isinstance(error, HTTPException) else InternalServerError()
            return render('error.html', title='Oooops', error=error), error.code

        @app.route('/')
        def route_root():
            return redirect('/audit')

        @app.route('/settings', methods=('POST',))
        def route_settings():
            settings.update(**request.form)
            return redirect(request.referrer)

        @app.route('/audit', strict_slashes=False)
        @app.route('/audit/<string:id>')
        def route_audit(id=None):
            return render('audit.html', title='audit', settings=settings, results=results, id=id,
                          prev_id=prev_ids.get(id), next_id=next_ids.get(id), patched=patched,
                          mounts=mounts)

        @app.route('/patch', methods=('POST',))
        def route_patch():
            if (id := request.form.get('id')) and (mount := mounts.get(id)):
                scores = dict()
                comments = dict()
                r = deepcopy(results[id])

                r.title = 'manual audit'
                r.timestamp = timestamp_utc_iso()

                # extract form data
                for key, value in request.form.items():
                    if key.startswith('score:'):
                        scores[key.split(':')[-1]] = math.nan if value == '' else float(value)
                    elif key.startswith('comment:'):
                        comments[key.split(':')[-1]] = value

                # update results
                modification_flag = False
                for result in r.results:
                    score = scores.get(result.id)
                    if score is not None and not math.isclose(score, result.score):
                        logger.debug(f'update score of result {result.id[:8]}')
                        result.score = score
                        modification_flag = True

                    if comment := comments.get(result.id):
                        logger.debug(f'update messages of result {result.id[:8]}')
                        result.messages.append(settings.format_comment(comment))
                        modification_flag = True

                # patch results back
                if modification_flag:
                    # update state & persist patch
                    inject_patch(r, mount)
                    results[id] = results[id].patch(r)
                    patched.add(id)
                else:
                    logger.debug('no modifications were made')

                if next_id := next_ids.get(id):
                    return redirect(f'/audit/{next_id}#edit')

            return redirect('/audit')

        @app.route('/report/<string:id>')
        def route_report(id):
            return render('report.html', title='report (preview)', id=id, results=results,
                          summary=results[id].summary())

        @app.route('/source/<string:id>')
        def route_source(id):
            return render('source_view.html', title='source view', source=sources.get(id, 'None'),
                          id=id)

        @app.route('/summary', strict_slashes=False)
        def route_summary():
            summary_df = summarize_results(results.values())

            plot_distribution = parse_bool(request.args.get('distribution', 'f')) and 2 < len(summary_df)
            plot_similarities = parse_bool(request.args.get('similarities', 'f')) and 1 < len(summary_df)

            plots = dict(
                distribution=b64str(plot_score_distribution(summary_df)) if plot_distribution else None,
                similarities=b64str(plot_fraud_matrix(sources)) if plot_similarities else None
            )

            return render('summary.html', title='summary', summary=summary_df, plots=plots)

        @app.route('/stop')
        def route_stop():
            if func := request.environ.get('werkzeug.server.shutdown'):
                logger.debug('shutdown werkzeug server')
                func()
            else:
                logger.debug('not running with werkzeug server')
                return redirect('/audit')

            return render('message.html', title='stop server', message='ciao kakao :)')

        app.run(host=args.bind, port=args.port)


def cmd_report(args):
    """Inject a human readable report (standalone HTML) into result archive(s)"""
    for path in list_results(args.result):
        logger.info(f'render report for {path}')
        with mount_tar(path, mode='a') as tar, cd(tar):
            results = load_patched()
            with open('report.html', mode='wt') as f:
                f.write(render('report.html', title='report', id=results.checksum,
                               results={results.checksum: results}, summary=results.summary()))

    return 0


def cmd_summary(args):
    """Generate human & machine readable summary of results"""
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


def version(_):
    """Display version of autograde"""
    print(f'autograde version {autograde.__version__}')
    return 0


def cli(args=None):
    # environment variables
    verbosity = int(os.environ.get('AG_VERBOSITY', 0))
    container_backend = os.environ.get('AG_BACKEND', None)
    container_tag = os.environ.get('AG_TAG', 'autograde')

    # command line arguments
    parser = argparse.ArgumentParser(
        description='utility for grading jupyter notebooks',
        epilog='autograde on github: https://github.com/cssh-rwth/autograde',
        prog='autograde',
    )

    # global flags
    parser.add_argument('-v', '--verbose', action='count', default=verbosity,
                        help='verbosity level')
    parser.add_argument('--backend', type=str, default=container_backend,
                        choices=['docker', 'rootless-docker', 'podman'], metavar='',
                        help=f'container backend to use, default is {container_backend}')
    parser.add_argument('--tag', type=str, default=container_tag, metavar='',
                        help=f'container tag, default: "{container_tag}"')
    parser.set_defaults(func=version)

    subparsers = parser.add_subparsers(help='sub command help')

    # build sub command
    bld_parser = subparsers.add_parser('build', help=cmd_build.__doc__)
    bld_parser.add_argument('-r', '--requirements', type=Path, default=None,
                            help='additional requirements to install')
    bld_parser.add_argument('-q', '--quiet', action='store_true', help='mute output')
    bld_parser.set_defaults(func=cmd_build)

    # test sub command
    tst_parser = subparsers.add_parser('test', help=cmd_test.__doc__)
    tst_parser.add_argument('test', type=str, help='autograde test script')
    tst_parser.add_argument('notebook', type=str, help='the jupyter notebook(s) to be tested')
    tst_parser.add_argument('-t', '--target', type=str, metavar='', help='where to store results')
    tst_parser.add_argument('-c', '--context', type=str, metavar='', help='context directory')
    tst_parser.set_defaults(func=cmd_test)

    # patch sub command
    ptc_parser = subparsers.add_parser('patch', help=cmd_patch.__doc__)
    ptc_parser.add_argument('result', type=str, help='result archive(s) to be patched')
    ptc_parser.add_argument('patch', type=str, help='result archive(s) for patching')
    ptc_parser.set_defaults(func=cmd_patch)

    # audit sub command
    adt_parser = subparsers.add_parser('audit', help=cmd_audit.__doc__)
    adt_parser.add_argument('result', type=str, help='result archive(s) to audit')
    adt_parser.add_argument('-b', '--bind', type=str, default='127.0.0.1', help='host to bind to')
    adt_parser.add_argument('-p', '--port', type=int, default=5000, help='port')
    adt_parser.set_defaults(func=cmd_audit)

    # report sub command
    rpt_parser = subparsers.add_parser('report', help=cmd_report.__doc__)
    rpt_parser.add_argument('result', type=str, help='result archive(s) for creating the report')
    rpt_parser.set_defaults(func=cmd_report)

    # summary sub command
    sum_parser = subparsers.add_parser('summary', help=cmd_summary.__doc__)
    sum_parser.add_argument('result', type=str, help='result archives to summarize')
    sum_parser.set_defaults(func=cmd_summary)

    # version sub command
    vrs_parser = subparsers.add_parser('version', help=version.__doc__)
    vrs_parser.set_defaults(func=version)

    args = parser.parse_args(args)

    logger.setLevel(loglevel(args.verbose))
    logger.debug(f'default encoding: {sys.getdefaultencoding()}')
    logger.debug(f'args: {args}')

    return args.func(args)


if __name__ == '__main__':
    sys.exit(cli())
