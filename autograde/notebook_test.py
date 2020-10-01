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
from copy import deepcopy
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Dict, List, Tuple, Union, Iterable

from dataclasses_json import dataclass_json

import autograde
from autograde.helpers import import_filter
from autograde.notebook_executor import exec_notebook
from autograde.util import logger, timestamp_utc_iso, loglevel, camel_case, \
    prune_join, capture_output, cd, mount_tar, timeout as timeout_context

T_TARGET = Union[str, Iterable[str]]


@dataclass_json
@dataclass
class TeamMember:
    first_name: str
    last_name: str
    student_id: str


@dataclass_json
@dataclass
class Result:
    id: str
    label: str
    target: List[str]
    score: float
    score_max: float
    messages: List[str]
    stdout: str
    stderr: str

    def passed(self) -> bool:
        return math.isclose(self.score, self.score_max)

    def failed(self) -> bool:
        return (not math.isnan(self.score)) and (not math.isclose(self.score, self.score_max))

    def pending(self) -> bool:
        return math.isnan(self.score)


@dataclass_json
@dataclass
class Results:
    title: str
    notebook: str
    checksum: str
    team_members: List[TeamMember]
    artifacts: List[str]
    excluded_artifacts: List[str]
    results: List[Result]
    applied_patches: List[Tuple[str, str, List[str]]] = field(default_factory=lambda: [])
    version: str = field(default_factory=lambda: autograde.__version__)
    timestamp: str = field(default_factory=timestamp_utc_iso)

    def __iter__(self):
        return iter(self.results)

    def format_members(self, *args, **kwargs):
        last_names = sorted((m.last_name for m in self.team_members))
        return prune_join(map(camel_case, last_names), *args, **kwargs)

    def patch(self, patch: Results) -> Results:
        """
        Create a copy of self and patch results of given results object into it. NOTE that pending
        results are ignored.

        :param patch: results to be patched into self
        :return: patched copy
        """
        patched = deepcopy(self)
        results = {r.id: r for r in self.results}

        if not patched.checksum == patch.checksum:
            raise ValueError('patch must not have a different origin aka checksum!')

        change_list = []
        for result in patch.results:
            if result != results.get(result.id) and not result.pending():
                results[result.id] = result
                change_list.append(result.id)

        patched.results = list(results.values())
        patched.applied_patches.append((patch.title, patch.timestamp, change_list))

        return patched

    def summary(self) -> ResultSummary:
        return ResultSummary(self)


@dataclass_json
@dataclass(init=False)
class ResultSummary:
    tests: int
    failed: int
    passed: int
    pending: int
    score: float
    score_max: float

    def __init__(self, results: Results):
        self.tests = len(results.results)
        self.failed = sum(r.failed() for r in results.results)
        self.passed = sum(r.passed() for r in results.results)
        self.pending = sum(r.pending() for r in results.results)
        self.score = sum(r.score for r in results.results)
        self.score_max = sum(r.score_max for r in results.results)


class NotebookTestCase:
    def __init__(self, test_function, target: T_TARGET, label: str, score: float = 1., timeout: float = 0.):
        self._id = sha256(label.lower().strip().encode('utf-8')).hexdigest()
        self._test_func = test_function
        self._targets = (target,) if isinstance(target, str) else tuple(target)
        self._label = label
        self._score = score
        self._timeout = timeout

    def __call__(self, state, *args, **kwargs):
        try:
            # extract targets
            try:
                targets = [state[t] for t in self._targets]
            except KeyError as err:
                raise NameError(err)

            # apply actual test
            with timeout_context(self._timeout):
                result = self._test_func(*targets, *args, **kwargs)

            # interpret results
            msg = result if isinstance(result, str) else 'ok'
            score = result if isinstance(result, (int, float)) else self.score
            score, msg = result if isinstance(result, tuple) else (score, msg)

            return float(score), str(msg)

        except Exception as err:
            print('Test failed:', file=sys.stderr)
            traceback.print_exception(type(err), err, err.__traceback__, file=sys.stderr)
            return 0, f'{type(err).__name__}: "{err}"'

    def __str__(self):
        timeout = f'{self._timeout:.2f}s' if self._timeout is not None else '∞'
        return f'{self.__class__.__name__}(target={self.targets}, score={self.score}, ' \
               f'timeout={timeout}, label="{self.label}")'

    id = property(fget=lambda self: self._id)
    targets = property(fget=lambda self: self._targets)
    label = property(fget=lambda self: self._label)
    score = property(fget=lambda self: self._score)
    timout = property(fget=lambda self: self._timeout)


class NotebookTest:
    def __init__(self, title, cell_timeout: float = 0., test_timeout: float = 0.):
        self._title = title
        self._cases = OrderedDict()
        self._cell_timeout = cell_timeout
        self._test_timeout = test_timeout
        self._variables = OrderedDict()

    def __len__(self):
        return len(self._cases)

    def __str__(self):
        return f'{type(self).__name__}({len(self._cases)} cases)'

    def __repr__(self):
        return f'{type(self).__name__}({self._cases})'

    def register(self, target: T_TARGET, label: str, score: float = 1., timeout: float = 0.):
        """
        Decorator for registering a new test case for given target.

        :param target: can be anything from the notebooks scope, be it a variable, function, class
            or module
        :param label: label for identification of the test case
        :param score: the weight for the test case, default is 1.0
        :param timeout: how many seconds to wait before aborting the test
        :return: decorator wrapping the original function
        """
        def decorator(func):
            case = NotebookTestCase(func, target, label, score, timeout or self._test_timeout)
            if case.id in self._cases:
                raise ValueError('A case with same id was already registered. Consider using a different label!')
            self._cases[case.id] = case
            return case

        return decorator

    def register_comment(self, target: Union[str, re.Pattern], label: str, score: float = 1.):
        """
        Register a special test case that looks for markdown comments in the notebook by a given
        regular expression. If no such comment is found, the test fails. In all other cases,
        the comment texts are included into the report. NOTE: a "passed" test is scored with NaN
        (not a number) as its output is intended for further, manual inspection.

        :param target: (compiled) regular expression that is searched for in markdown comments
        :param label: label for identification of the test case
        :param score: the weight for the test case, default is 1.0
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

        self.register('__COMMENTS__', label, score)(search_comment)

    def register_figure(self, target: Union[str, Path], label: str, score: float = 1,):
        """
        Register a special test case that loads an base64 encoded PNG or SVG image from artifacts.
        If the image does not exist, the test fails. In all other cases, the image is included into
        the report. NOTE: a "passed" test is scored with NaN (not a number) as its output is
        intended for further, manual inspection.

        :param target: name or (relative) path of figure to load
        :param label: label for identification of the test case
        :param score: the weight for the test case, default is 1.0
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

        self.register('__ARTIFACTS__', label, score)(load_plot)

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

    def _apply_cases(self, state: Dict) -> List[Result]:
        state = state.copy()
        results = []

        # prepare import filter
        if_regex, if_blacklist = self._variables.get('IMPORT_FILTER', (None, None))

        for i, case in enumerate(self._cases.values(), start=1):
            logger.debug(f'[{i}/{len(self._cases)}] execute {case}')

            with io.StringIO() as stdout, io.StringIO() as stderr:
                with ExitStack() as es:
                    if if_regex is not None:
                        es.enter_context(import_filter(if_regex, blacklist=if_blacklist))
                    es.enter_context(capture_output(stdout, stderr))

                    achieved, msg = case(state)

                results.append(Result(
                    id=case.id,
                    label=case.label,
                    target=case.targets,
                    score=achieved,
                    score_max=case.score,
                    messages=[msg],
                    stdout=stdout.getvalue(),
                    stderr=stderr.getvalue()
                ))

        logger.debug('testing completed')
        return results

    def _grade_notebook(self, nb_path, target_dir=None, context=None):
        target_dir = target_dir or os.getcwd()

        # prepare notebook
        with open(nb_path, mode='rb') as f:
            nb_data = f.read()

        nb_hash = sha256(nb_data).hexdigest()
        nb_hash_short = nb_hash[:8]

        with cd(target_dir):
            archive = Path(f'results_{nb_hash_short}.tar.xz')

            if archive.exists():
                logger.debug(f'remove existing {archive}')
                archive.unlink()

            with ExitStack() as exec_test_stack:
                tar = exec_test_stack.enter_context(mount_tar(archive, mode='w:xz'))
                exec_test_stack.enter_context(cd(tar))

                # store copy of notebook
                logger.debug('dump copy of original notebook')
                with open('notebook.ipynb', mode='wb') as f:
                    f.write(nb_data)

                # prepare context and execute notebook
                with open('code.py', mode='wt') as c, cd('artifacts', mkdir=True):
                    # prepare execution context in file system
                    if context is not None:
                        logger.debug(f'copy context files from: {context}')
                        shutil.copytree(context, '.', dirs_exist_ok=True)

                    # build index of all files known before execution
                    index = set()
                    for path in Path('.').glob('**/*'):
                        if path.is_file():
                            with path.open(mode='rb') as f:
                                index.add(sha256(f.read()).hexdigest())

                    # actual notebook execution
                    try:
                        logger.debug('execute notebook')
                        state = exec_test_stack.enter_context(exec_notebook(
                            io.StringIO(nb_data.decode('utf-8')),
                            file=c,
                            ignore_errors=True,
                            cell_timeout=self._cell_timeout,
                            variables=self._variables
                        ))

                    except ValueError:
                        state = {}

                    # remove files that haven't changed
                    artifacts = []
                    artifacts_excluded = []
                    for path in Path('.').glob('**/*'):
                        if path.is_file():
                            delete_flag = False
                            with path.open(mode='rb') as f:
                                if sha256(f.read()).hexdigest() in index:
                                    artifacts_excluded.append(str(path))
                                    delete_flag = True
                                else:
                                    artifacts.append(str(path))

                            if delete_flag:
                                path.unlink()

                # infer meta information
                group = list(map(lambda m: TeamMember(**m), state.get('team_members', [])))

                if not group:
                    logger.warning(f'Couldn\'t find valid information about team members in "{nb_path}"')

                # execute tests
                logger.debug('execute tests')
                results = Results(
                    title=self._title,
                    notebook=str(nb_path),
                    checksum=nb_hash,
                    team_members=group,
                    artifacts=sorted(artifacts),
                    excluded_artifacts=sorted(artifacts_excluded),
                    results=self._apply_cases(state)
                )

                # store results as json
                logger.debug('dump results as json')
                with open('results.json', mode='wt') as f:
                    json.dump(results.to_dict(), fp=f, indent=4)

                # infer new, more readable name
                names = results.format_members(separator=',')
                archive_name_new = Path(f'results_[{names}]_{nb_hash_short}.tar.xz')

            if archive_name_new.exists():
                logger.debug(f'remove existing {archive_name_new}')
                archive_name_new.unlink()

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

        return results.summary().failed
