import math
import io
import re
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import dataclass
from getpass import getuser
from pathlib import Path
from typing import Dict, Iterable, Optional, Set

from autograde.cli.util import logger, summarize_results, b64str, plot_score_distribution, \
    list_results
from autograde.test_result import UnitTestResult, NotebookTestResultArchive
from autograde.util import parse_bool, render, timestamp_utc_iso


@dataclass
class AuditSettings:
    selector: re.Pattern = re.compile('')
    auditor: str = str(getuser())
    show_identities: bool = False

    def update(self, selector=None, auditor=None, show_identities=False):
        self.selector = re.compile(selector or '')
        self.auditor = auditor or str(getuser())
        self.show_identities = bool(show_identities)

    def select(self, result: UnitTestResult) -> bool:
        return bool(self.selector.search(result.label))

    def filter_results(self, results: Iterable[UnitTestResult]) -> Iterable[UnitTestResult]:
        return filter(self.select, results)

    def format_comment(self, comment):
        if self.auditor:
            return f'{self.auditor}: {comment.strip()}'
        return comment.strip()


class AuditState:
    def __init__(self, path: Path):
        self._exit_stack = ExitStack().__enter__()

        self.settings: AuditSettings = AuditSettings()
        self.archives: Dict[str, NotebookTestResultArchive] = dict()

        # load archives
        for path in list_results(path):
            archive = self._exit_stack.enter_context(NotebookTestResultArchive(path, mode='a'))
            self.archives[archive.results.checksum] = archive

        # build index
        self._next_ids = dict(zip(self.archives, list(self.archives)[1:]))
        self._prev_ids = dict(((b, a) for a, b in self._next_ids.items()))

        self.patched: Set[str] = set()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._exit_stack.__exit__(exc_type, exc_val, exc_tb)

    def prev_id(self, aid: str) -> Optional[str]:
        return self._prev_ids.get(aid)

    def next_id(self, aid: str) -> Optional[str]:
        return self._next_ids.get(aid)

    def patch(self, id: str, **kwargs):
        aid = id
        scores = dict()
        comments = dict()

        archive = self.archives[aid]
        patch = deepcopy(archive.results)

        patch.title = 'manual audit'
        patch.timestamp = timestamp_utc_iso()
        if auditor := self.settings.auditor:
            patch.title += f' by {auditor}'

        # extract form data
        for key, value in kwargs.items():
            if key.startswith('score:'):
                scores[key.split(':')[-1]] = math.nan if value == '' else float(value)
            elif key.startswith('comment:'):
                comments[key.split(':')[-1]] = value
            else:
                logger.warning(f'unknown form item: "{key}" (ignore)')

        # update archives
        modification_flag = False
        for result in patch.unit_test_results:
            score = scores.get(result.id)
            if score is not None and not math.isclose(score, result.score):
                logger.debug(f'update score of unit test result {result.id}')
                result.score = score
                modification_flag = True

            if comment := comments.get(result.id):
                logger.debug(f'update messages of unit test result {result.id}')
                result.messages.append(self.settings.format_comment(comment))
                modification_flag = True

        # patch archives back
        if modification_flag:
            # update state & persist patch
            archive.inject_patch(patch)
            self.patched.add(aid)
        else:
            logger.debug('no modifications were made')

        return self.next_id(aid)


def cmd_audit(args):
    """Launch a web interface for manually auditing test archives"""
    import logging
    from flask import Flask, redirect, request, send_file
    import flask.cli as flask_cli
    from werkzeug.exceptions import HTTPException, InternalServerError

    with AuditState(args.result) as state:
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
            state.settings.update(**request.form)
            logger.debug(f'update settings: {state.settings}')
            return redirect(request.referrer)

        @app.route('/audit', strict_slashes=False)
        @app.route('/audit/<string:id>')
        def route_audit(id: Optional[str] = None):
            return render('audit.html', title='audit', state=state, id=id)

        @app.route('/patch', methods=('POST',))
        def route_patch():
            if next_id := state.patch(**request.form):
                return redirect(f'/audit/{next_id}')
            return redirect('/audit')

        @app.route('/report/<string:id>')
        def route_report(id):
            return state.archives[id].report

        @app.route('/download/<string:id>/<path:path>')
        def route_download(id, path):
            return send_file(
                io.BytesIO(state.archives[id].load_file(path)),
                attachment_filename=str(path).split('/')[-1],
                as_attachment=True
            )

        @app.route('/summary', strict_slashes=False)
        def route_summary():
            summary = summarize_results(a.results for a in state.archives.values())
            plot_distribution = parse_bool(request.args.get('distribution', 'f')) and 2 < len(summary)
            plots = [
                dict(
                    title='Score Distribution',
                    data=b64str(plot_score_distribution(summary)) if plot_distribution else None
                ),
            ]
            return render('summary.html', title='summary', summary=summary, plots=plots)

        @app.route('/stop')
        def route_stop():
            if func := request.environ.get('werkzeug.server.shutdown'):
                logger.debug('stop werkzeug server')
                func()
            else:
                logger.debug('not running with werkzeug server')
                return redirect('/audit')

            return render('message.html', title='stop server', message='ciao kakao :)')

        app.run(host=args.bind, port=args.port)
