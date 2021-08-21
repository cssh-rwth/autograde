import io
import logging
import math
import re
from contextlib import ExitStack
from dataclasses import dataclass
from getpass import getuser
from pathlib import Path
from typing import Dict, Iterable, Optional, Set, Tuple

import flask.cli as flask_cli
from flask import Flask, redirect, request, send_file
from werkzeug.exceptions import HTTPException, InternalServerError

from autograde.backend.local.util import merge_results, summarize_results, b64str, plot_score_distribution, \
    find_archives
from autograde.test_result import UnitTestResult, NotebookTestResultArchive
from autograde.util import now, logger, parse_bool, render


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

    def format_comment(self, comment: str):
        if self.auditor:
            return f'{self.auditor}: {comment.strip()}'
        return comment.strip()


class AuditState:
    def __init__(self, path: Path):
        self._exit_stack = ExitStack().__enter__()

        self.settings: AuditSettings = AuditSettings()
        self.archives: Dict[str, NotebookTestResultArchive] = dict()

        # load archives
        for path in find_archives(path):
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

    @staticmethod
    def _parse_form(form: Dict) -> Tuple[Dict[str, float], Dict[str, str]]:
        scores = dict()
        comments = dict()

        for key, value in form.items():
            if key.startswith('score:'):
                scores[key.split(':')[-1]] = math.nan if value == '' else float(value)
            elif key.startswith('comment:'):
                if value:
                    comments[key.split(':')[-1]] = str(value)
            else:
                logger.warning(f'ignore unknown form item: "{key}"')

        return scores, comments

    def patch(self, id: str, **kwargs):
        aid = id
        scores, comments = self._parse_form(kwargs)

        archive = self.archives[aid]
        patch = archive.results.copy()

        # update archives
        for result in patch.unit_test_results:
            if (score := scores.get(result.id)) is not None and not math.isclose(score, result.score):
                logger.debug(f'update score of unit test result {result.id}')
                result.score = score

            if comment := comments.get(result.id):
                logger.debug(f'update messages of unit test result {result.id}')
                result.messages.append(self.settings.format_comment(comment))

        # apply patch if there are changes
        if patch != archive.results:
            # update remaining attributes of patch
            patch.title = 'Audit'
            patch.timestamp = now()
            if auditor := self.settings.auditor:
                patch.title += f' by {auditor}'

            # persist patch & update state
            archive.inject_patch(patch)
            self.patched.add(aid)

        else:
            logger.debug('no modifications were made')

        return self.next_id(aid)


def cmd_audit(result: Path, bind: str, port: int, **_) -> int:
    with AuditState(result) as state:
        # create actual flask application
        app = Flask('autograde - audit')

        # monkey patching for nicer cli output
        flask_cli.show_server_banner = lambda *_, **__: logger.debug('suppress flask banner')
        app.logger = logger
        logging.root = logger

        if app.debug:
            # In debug mode, flask may reload code dynamically, e.g. on changes. This is not problem if the application
            # does not hold any state, a common best practice we're breaking with here!
            logger.warning('the application is running in debug mode which may corrupt your files!')

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
            summary = summarize_results(merge_results(state.archives.values()))
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

        app.run(host=bind, port=port, threaded=False, processes=1)

    return 0
