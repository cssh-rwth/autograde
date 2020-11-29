import math
import re
from collections import OrderedDict
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import dataclass
from getpass import getuser
from pathlib import Path
from typing import Iterable

from autograde.cli.util import load_patched, render, list_results, summarize_results, b64str, plot_fraud_matrix, \
    plot_score_distribution, inject_patch
from autograde.notebook_test import Result


@dataclass
class AuditSettings:
    selector: re.Pattern = re.compile('')
    auditor: str = str(getuser())
    show_identities: bool = False

    def update(self, selector=None, auditor=None, show_identities=False):
        self.selector = re.compile(selector or '')
        self.auditor = auditor or str(getuser())
        self.show_identities = bool(show_identities)

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

    from autograde.util import logger, parse_bool, timestamp_utc_iso, mount_tar

    with ExitStack() as exit_stack:
        # settings
        settings = AuditSettings()

        # mount & index all results
        mounts = OrderedDict()
        sources = dict()
        results = dict()
        for path in list_results(args.result):
            mount_path = Path(exit_stack.enter_context(mount_tar(path, mode='a')))

            r = load_patched(mount_path)
            results[r.checksum] = r
            mounts[r.checksum] = mount_path

            with mount_path.joinpath('code.py').open(mode='rt') as f:
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
            logger.warning(f'{type(error)}: {error}')
            error = error if isinstance(error, HTTPException) else InternalServerError()
            return render('error.html', title='Oooops', error=error), error.code

        @app.route('/')
        def route_root():
            return redirect('/audit')

        @app.route('/settings', methods=('POST',))
        def route_settings():
            settings.update(**request.form)
            logger.debug(f'update settings: {settings}')
            return redirect(request.referrer)

        @app.route('/audit', strict_slashes=False)
        @app.route('/audit/<string:id>')
        def route_audit(id=None):
            return render('audit.html', title='audit', settings=settings, results=results, id=id,
                          prev_id=prev_ids.get(id), next_id=next_ids.get(id), patched=patched,
                          mounts=mounts)

        @app.route('/patch', methods=('POST',))
        def route_patch():
            if (rid := request.form.get('id')) and (mount := mounts.get(rid)):
                scores = dict()
                comments = dict()
                r = deepcopy(results[rid])

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
                    results[rid] = results[rid].patch(r)
                    patched.add(rid)
                else:
                    logger.debug('no modifications were made')

                if next_id := next_ids.get(rid):
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
