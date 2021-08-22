from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
import re
import shutil
import sys
import traceback
from collections import OrderedDict
from contextlib import ExitStack
from hashlib import sha256
from math import isclose
from pathlib import Path
from typing import Callable, Dict, Optional, Union, Iterable, Generator, Tuple

from autograde.notebook_executor import exec_notebook
from autograde.test_result import TeamMember, UnitTestResult, NotebookTestResult
from autograde.util import capture_output, cd, cd_zip, import_filter, logger, loglevel, run_with_timeout, WatchDog

Target = Union[str, Iterable[str]]
TestResult = Union[None, float, str, Tuple[float, str]]
TestFunction = Callable[..., TestResult]


class UnitTest:
    def __init__(self, test_function: TestFunction, target: Target, label: str, score: float = 1., timeout: float = -1):
        self._id = sha256(label.lower().strip().encode('utf-8')).hexdigest()
        self._test_function = test_function
        self._targets = (target,) if isinstance(target, str) else tuple(target)
        self._label = label
        self._score = score
        self._timeout = timeout

    def __call__(self, state, *args, **kwargs) -> Tuple[float, str]:
        try:
            # extract targets
            try:
                targets = [state[t] for t in self._targets]
            except KeyError as err:
                raise NameError(err.args[0])

            # run actual test
            if (t := self._timeout) > 0:
                result = run_with_timeout(self._test_function, args=(*targets, *args), kwargs=kwargs, timeout=t)
            else:
                result = self._test_function(*targets, *args, **kwargs)

            # interpret results
            if isinstance(result, (int, float)):
                score = min(float(result), self.score)
            else:
                score = self.score

            if isinstance(result, str):
                msg = result
            else:
                msg = '✅ passed' if isclose(score, self.score) else '¯\\_(ツ)_/¯ partially passed'

            if isinstance(result, tuple):
                score, msg = result

            return score, str(msg)

        except Exception as err:
            print('Test failed:', file=sys.stderr)
            traceback.print_exception(type(err), err, err.__traceback__, file=sys.stderr)
            return 0, f'❌ {type(err).__name__}: "{err}"'

    def __str__(self):
        timeout = f'{self._timeout:.2f}s' if self._timeout is not None else '∞'
        return f'{self.__class__.__name__}(target={self.targets}, score={self.score}, ' \
               f'timeout={timeout}, label="{self.label}")'

    id = property(fget=lambda self: self._id)
    targets = property(fget=lambda self: self._targets)
    label = property(fget=lambda self: self._label)
    score = property(fget=lambda self: self._score)
    timout = property(fget=lambda self: self._timeout)


TestDecorator = Callable[[TestFunction], UnitTest]


class NotebookTest:
    def __init__(self, title: str, cell_timeout: float = 0., test_timeout: float = 0.):
        self._title = title
        self._unit_tests = OrderedDict()
        self._cell_timeout = cell_timeout
        self._test_timeout = test_timeout
        self._variables = OrderedDict()

    def __len__(self):
        return len(self._unit_tests)

    def __str__(self):
        return f'{type(self).__name__}({len(self._unit_tests)} unit tests)'

    def __repr__(self):
        return f'{type(self).__name__}({self._unit_tests})'

    def register(self, target: Target, label: str, score: float = 1., timeout: float = 0.) -> TestDecorator:
        """
        Decorator for registering a new unit test for given target.

        :param target: can be anything from the notebooks scope, be it a variable, function, class
            or module
        :param label: label for identification of the unit test
        :param score: the weight for the unit test, default is 1.0
        :param timeout: how many seconds to wait before aborting the test
        :return: decorator wrapping the original function
        """

        def decorator(func: TestFunction) -> UnitTest:
            unit_test = UnitTest(func, target, label, score, timeout or self._test_timeout)
            if unit_test.id in self._unit_tests:
                raise ValueError('A unit test with same id was already registered. Consider using a different label!')
            self._unit_tests[unit_test.id] = unit_test
            return unit_test

        return decorator

    def register_comment(self, target: Union[str, Path], label: str, score: float = 1.) -> UnitTest:
        """
        Register a special unit test that looks for markdown comments in the notebook using the
        given regular expression. If no such comment is found, the test fails. In all other cases,
        the comment texts are included into the report. NOTE: a "passed" test is scored with NaN
        (not a number) as its output is intended for further, manual inspection.

        :param target: (compiled) regular expression that is searched for in markdown comments
        :param label: label for identification of the unit test
        :param score: the weight for the unit test, default is 1.0
        """
        pattern = re.compile(target) if isinstance(target, str) else target

        def search_comment(comments):
            comments = list(filter(pattern.search, comments))

            assert len(comments) > 0, 'no matching comments found'

            msg = ''
            for i, comment in enumerate(comments, start=1):
                msg += f'[MATCH {i}]:\n'
                msg += f'{comment.strip()}\n\n'

            return math.nan, msg

        return self.register('__COMMENTS__', label, score)(search_comment)

    def register_figure(self, target: Union[str, Path], label: str, score: float = 1) -> UnitTest:
        """
        Register a special unit test that loads an base64 encoded PNG or SVG image from artifacts.
        If the image does not exist, the test fails. In all other cases, the image is included into
        the report. NOTE: a "passed" test is scored with NaN (not a number) as its output is
        intended for further, manual inspection.

        :param target: name or (relative) path of figure to load
        :param label: label for identification of the unit test
        :param score: the weight for the unit test, default is 1.0
        """
        target = Path(target)
        prefixes = {
            'png': 'data:image/png;base64,',
            'svg': 'data:image/svg+xml;base64,'
        }

        prefix = prefixes.get(target.suffix[1:])

        if prefix is None:
            raise ValueError(f'Extension is not supported! Use one of {set(prefixes)}')

        def load_plot(artifacts):
            return math.nan, prefix + base64.b64encode(artifacts[target]).decode('utf8')

        return self.register('__ARTIFACTS__', label, score)(load_plot)

    def set_import_filter(self, regex: Union[str, re.Pattern], blacklist: bool = False):
        """
        Set an import filter for the current notebook test. It will be activated for all cells
        executed (except the one injected before notebook) as well as all test cases. By default,
        the import filter checks whether the given regex pattern is found in the import name and
        raises an error if not. This behaviour can be negated via `blacklist` flag.

        :param regex: regular expression, e.g. `r"networkx|requests"`
        :param blacklist: whether the regex is used for white- or blacklisting, default is false
        """
        self._variables['IMPORT_FILTER'] = (
            re.compile(regex) if isinstance(regex, str) else regex,
            bool(blacklist)
        )

    def _apply_unit_tests(self, state: Dict) -> Generator[UnitTestResult, None, None]:
        if_regex, if_blacklist = self._variables.get('IMPORT_FILTER', (None, None))

        for i, unit_test in enumerate(self._unit_tests.values(), start=1):
            logger.debug(f'[{i}/{len(self._unit_tests)}] execute {unit_test}')

            with io.StringIO() as stdout, io.StringIO() as stderr:
                with ExitStack() as stack:
                    if if_regex is not None:
                        stack.enter_context(import_filter(if_regex, blacklist=if_blacklist))
                    stack.enter_context(capture_output(stdout, stderr))
                    score, msg = unit_test(state)

                yield UnitTestResult(
                    id=unit_test.id,
                    label=unit_test.label,
                    target=unit_test.targets,
                    score=score,
                    score_max=unit_test.score,
                    messages=[msg],
                    stdout=stdout.getvalue(),
                    stderr=stderr.getvalue()
                )

        logger.debug('testing completed')

    def _grade_notebook(self, nb_path: Path, target_dir: Optional[Path] = None, context: Optional[Path] = None):
        target_dir = target_dir or os.getcwd()

        # prepare notebook
        with nb_path.open(mode='rb') as f:
            nb_data = f.read()

        nb_hash = sha256(nb_data).hexdigest()
        nb_hash_short = nb_hash[:8]

        with cd(target_dir):
            archive = Path(f'results_{nb_hash_short}.zip')

            if archive.exists():
                logger.debug(f'remove existing {archive}')
                archive.unlink()

            with ExitStack() as stack:
                # ensure all artifacts created in this context are persisted in the results archive
                stack.enter_context(cd_zip(archive))

                # store copy of notebook
                logger.debug('dump copy of original notebook')
                with open('notebook.ipynb', mode='wb') as f:
                    f.write(nb_data)

                # prepare execution context and execute notebook
                with open('code.py', mode='wt', encoding='utf-8') as code, cd('artifacts', mkdir=True):
                    # prepare execution context in file system
                    if context is not None:
                        logger.debug(f'copy context files from: {context}')
                        shutil.copytree(context, '.', dirs_exist_ok=True)

                    # build index of all files known before execution
                    wd = WatchDog(Path('.'))

                    # actual notebook execution, resulting in the final state we'll run tests on
                    try:
                        logger.debug('execute notebook')
                        state = stack.enter_context(exec_notebook(
                            io.StringIO(nb_data.decode('utf-8')),
                            file=code,
                            ignore_errors=True,
                            cell_timeout=self._cell_timeout,
                            variables=self._variables
                        ))

                    except ValueError as err:
                        logger.warning(str(err))
                        state = {}

                    # collect artifacts and remove files that haven't changed
                    artifacts = list(wd.list_changed())
                    excluded_artifacts = list(wd.list_unchanged())

                    for path in excluded_artifacts:
                        path.unlink()

                # infer meta information
                try:
                    group = list(map(lambda m: TeamMember(**m), state.get('team_members', [])))
                except TypeError:
                    group = []

                if not group:
                    logger.warning(f'Couldn\'t find valid information about members in {nb_path}')

                # enrich state with meta information
                state['__TEAM_MEMBERS__'] = group.copy()
                state['__CONTEXT__'] = context

                # run unit tests
                logger.debug('apply unit tests')
                results = NotebookTestResult(
                    title=self._title,
                    checksum=nb_hash,
                    team_members=group,
                    artifacts=sorted(map(str, artifacts)),
                    excluded_artifacts=sorted(map(str, excluded_artifacts)),
                    unit_test_results=list(self._apply_unit_tests(state)),
                )

                # store results as json
                logger.debug('store results')
                with open('results.json', mode='wt', encoding='utf-8') as f:
                    json.dump(results.to_dict(), fp=f, indent=4)

                # infer new, better readable name
                names = results.format_members(separator='')
                archive_name_new = Path(f'results-{names}-{nb_hash_short}.zip')

            if archive_name_new.exists():
                logger.debug(f'remove existing {archive_name_new}')
                archive_name_new.unlink()

            logger.debug(f'rename results archive to {archive_name_new}')
            archive.rename(archive_name_new)

        return results

    def execute(self, args=None):
        """
        Commandline interface for notebook test. Call with `--help` flag to get further information.

        :param args: optional arguments, uses `sys.argv` by default
        :return: number of failed tests
        """
        parser = argparse.ArgumentParser(description='run tests on jupyter notebook')

        parser.add_argument('notebook', type=str, help='the jupyter notebook to test')
        parser.add_argument('-t', '--target', type=str, metavar='', help='where to store results')
        parser.add_argument('-c', '--context', type=str, metavar='', help='context directory')
        parser.add_argument('-v', '--verbose', action='count', default=0, help='verbosity level')

        args = parser.parse_args(args)

        logger.setLevel(loglevel(args.verbose))
        logger.debug(f'args: {args}')

        results = self._grade_notebook(
            Path(args.notebook).absolute(),
            target_dir=Path(args.target).absolute() if args.target else None,
            context=Path(args.context).absolute() if args.context else None
        )

        return results.summarize().failed
