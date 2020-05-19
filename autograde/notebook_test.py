# Standard library modules.
import io
import os
import re
import csv
import sys
import json
import argparse
import warnings
import traceback
from pathlib import Path
from copy import deepcopy
from datetime import datetime
from distutils import dir_util
from hashlib import md5, sha256
from contextlib import ExitStack
from collections import OrderedDict

# Third party modules.
from tabulate import tabulate
from nbformat import read, NotebookNode
from IPython.core.interactiveshell import InteractiveShell

# Local modules
import autograde
from autograde.helpers import import_filter
from autograde.templates import INJECT_BEFORE, INJECT_AFTER, REPORT_TEMPLATE
from autograde.util import logger, loglevel, camel_case, capture_output, cd, cd_tar, timeout

# Globals and constants variables.


def as_py_comment(s):
    if not s:
        return ''
    return '\n'.join(f'# {l}' for l in s.strip().split('\n'))


def exec_notebook(buffer, file=sys.stdout, ignore_errors=False, cell_timeout=0, variables=None):
    """
    Extract source code from jupyter notebook and execute it.

    :param buffer: file like with notebook data
    :param file: where to send stdout
    :param ignore_errors: whether or not errors will be forwarded or ignored
    :param cell_timeout: timeout for cell execution 0=∞
    :param variables: variables to be inserted into initial state
    :return: the state mutated by executed code
    """
    state = dict()
    variables = variables or {}
    state.update(deepcopy(variables))

    try:
        logger.debug('parse notebook')

        # when executed within a docker container, some minor warnings occur that we filter here
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')

            notebook = read(buffer, 4)
            shell = InteractiveShell.instance()

        _code_cells = filter(lambda c: c.cell_type == 'code', notebook.cells)

        cells = [
            ('injected from template', INJECT_BEFORE, 0),
            *((f'nb cell {i}', c, cell_timeout) for i, c in enumerate(_code_cells, start=1)),
            ('injected from template', INJECT_AFTER, 0)
        ]

    except Exception as error:
        logger.error(f'unable to parse notebook: {error}')
        raise ValueError(error)

    # prepare import filter
    if_regex, if_blacklist = variables.get('IMPORT_FILTER', (None, None))

    # the log is supposed to be a valid, standalone python script
    print('#!/usr/bin/env python3', file=file)

    # actual code execution
    for i, (label, code, timeout_) in enumerate(cells, start=1):
        with io.StringIO() as stdout, io.StringIO() as stderr:
            logger.debug(f'[{i}/{len(cells)}] execute cell ("{label}")')

            try:
                with capture_output(stdout, stderr):
                    # convert to valid python when origin is a notebook
                    if isinstance(code, NotebookNode):
                        code = shell.input_transformer_manager.transform_cell(code.source)

                    # actual execution that extends state
                    with ExitStack() as es:
                        if if_regex is not None and i > 1:
                            es.enter_context(import_filter(if_regex, blacklist=if_blacklist))
                        es.enter_context(timeout(timeout_))

                        exec(code, state)

            except Exception as error:
                # extend log with some meaningful error message
                traceback.print_exception(type(error), error, error.__traceback__, file=stderr)

                if not ignore_errors:
                    raise error

            finally:
                # log code and output
                with capture_output(file):
                    label = f' CODE CELL {label} '
                    print(f'# {label:-^78}')
                    print(str(code).strip())

                    stdout_s = stdout.getvalue()
                    if stdout_s:
                        print(f'\n# STDOUT')
                        print(as_py_comment(stdout_s))

                    stderr_s = stderr.getvalue()
                    if stderr_s:
                        print(f'\n# STDERR')
                        print(as_py_comment(stderr_s))

                    print('\n')

                # TODO clear plt state + condition to plot only if plt state has changed
                #   inject intermediate cells
                # exec(f'plt.savefig("cell_{label}.png");plt.close()', state)

    logger.debug('execution completed')
    return state


class NotebookTestCase:
    def __init__(self, test_function, target, score=1., label=None, timeout=0):
        self._test_func = test_function
        self._targets = (target,) if isinstance(target, str) else target
        self._score = float(score)
        self._label = str(label) if label else ''
        self._timeout = timeout

    def __call__(self, state, *args, **kwargs):
        try:
            # extract targets
            try:
                targets = [state[t] for t in self._targets]
            except KeyError as err:
                raise NameError(err)

            # apply actual test
            with timeout(self._timeout):
                msg = self._test_func(*targets, *args, **kwargs)

            return self.score, str(msg) if msg else 'ok'

        except Exception as err:
            msg_buffer = io.StringIO()
            traceback.print_exception(type(err), err, err.__traceback__, file=msg_buffer)
            return 0, msg_buffer.getvalue()

    def __str__(self):
        timeout_ = f'{self._timeout:.2f}s' if self._timeout is not None else '∞'
        return f'{self.__class__.__name__}(target={self.targets}, score={self.score}, ' \
               f'timeout={timeout_}, label="{self.label}")'

    targets = property(fget=lambda self: self._targets)
    score = property(fget=lambda self: self._score)
    label = property(fget=lambda self: self._label)
    timout = property(fget=lambda self: self._timeout)


class NotebookTest:
    def __init__(self, cell_timeout=0, test_timeout=0):
        self._cases = []
        self._cell_timeout = cell_timeout
        self._test_timeout = test_timeout
        self._variables = OrderedDict()

    def __len__(self):
        return len(self._cases)

    def __str__(self):
        return f'{type(self).__name__}({len(self._cases)} cases)'

    def __repr__(self):
        return f'{type(self).__name__}({self._cases})'

    def register(self, target, score=1., label='', timeout_=0):
        """
        Decorator for registering a new test case for given target.

        :param target: can be anything from the notebooks scope, be it a variable, function, class
            or module
        :param score: the weight for the test case, default is 1.0
        :param label: an optional label for easier identification of the test case
        :param timeout_: how many seconds to wait before aborting the test
        :return: decorator wrapping the original function
        """
        def decorator(func):
            case = NotebookTestCase(func, target, score, label, timeout_ or self._test_timeout)
            self._cases.append(case)
            return case

        return decorator

    def set_import_filter(self, regex, blacklist=False):
        """
        Set an import filter for the current notebook test. It will be activated for all cells
        executed (except the one injected before notebook) as well as all test cases. By default,
        the import filter checks whether the given regex pattern is found in the import name and
        raises an error if not. This behaviour can be negated via `blacklist` flag.

        :param regex: regular expression, e.g. `r"networkx|requests"`
        :param blacklist: whether the regex is used for white- or blacklisting, default is false
        :return:
        """
        self._variables['IMPORT_FILTER'] = (
            re.compile(regex) if isinstance(regex, str) else regex,
            bool(blacklist)
        )

    @staticmethod
    def _summarize_results(results):
        return OrderedDict(
            tests=len(results),
            passed=sum(r['score'] == r['score_max'] for r in results.values()),
            score=sum(r['score'] for r in results.values()),
            score_max=sum(r['score_max'] for r in results.values())
        )

    def _apply_tests(self, state):
        state = state.copy()
        results = OrderedDict()

        # prepare import filter
        if_regex, if_blacklist = self._variables.get('IMPORT_FILTER', (None, None))

        for i, case in enumerate(self._cases, start=1):
            logger.debug(f'[{i}/{len(self._cases)}] execute {case}')

            with io.StringIO() as stdout, io.StringIO() as stderr:
                with ExitStack() as es:
                    if if_regex is not None:
                        es.enter_context(import_filter(if_regex, blacklist=if_blacklist))
                    es.enter_context(capture_output(stdout, stderr))

                    achieved, msg = case(state)

                results[i] = OrderedDict(
                    label=case.label,
                    target=case.targets,
                    score=achieved,
                    score_max=case.score,
                    message=msg,
                    stdout=stdout.getvalue(),
                    stderr=stderr.getvalue()
                )

        logger.debug('testing completed')
        return results, self._summarize_results(results)

    def _grade_notebook(self, nb_path, target_dir=None, context=None):
        target_dir = target_dir or os.getcwd()

        # prepare notebook
        with open(nb_path, mode='rb') as f:
            nb_data = f.read()

        nb_hash_md5 = md5(nb_data).hexdigest()
        nb_hash_sha256 = sha256(nb_data).hexdigest()
        nb_hash_sha256_short = nb_hash_sha256[:8]

        with cd(target_dir):
            archive = Path(f'results_{nb_hash_sha256_short}.tar.xz')

            if archive.exists():
                logger.debug(f'remove existing {archive}')
                archive.unlink()

            with cd_tar(archive, mode='w:xz'):
                # prepare context and execute notebook
                with open('code.py', mode='wt') as c, cd_tar('artifacts.tar.xz', mode='w:xz'):
                    # prepare execution context in file system
                    if context is not None:
                        logger.debug(f'copy context files from: {context}')
                        dir_util.copy_tree(context, '.')

                    # build index of all files known before execution
                    index = set()
                    for p in os.listdir('.'):
                        with open(p, mode='rb') as f:
                            index.add(md5(f.read()).hexdigest())

                    # actual notebook execution
                    try:
                        logger.debug('execute notebook')
                        state = exec_notebook(
                            io.StringIO(nb_data.decode('utf-8')),
                            file=c,
                            ignore_errors=True,
                            cell_timeout=self._cell_timeout,
                            variables=self._variables
                        )

                    except ValueError:
                        state = {}

                    # remove files that haven't changed
                    for p in os.listdir('.'):
                        with open(p, mode='rb') as f:
                            if md5(f.read()).hexdigest() in index:
                                os.remove(p)

                # infer meta information
                group = state.get('team_members', {})

                if not group:
                    logger.warning(f'Couldn\'t find valid information about team members in "{nb_path}"')

                # execute tests
                logger.debug('execute tests')
                results, summary = self._apply_tests(state)
                enriched_results = OrderedDict(
                    autograde_version=autograde.__version__,
                    orig_file=str(nb_path),
                    checksum=dict(
                        md5sum=nb_hash_md5,
                        sha256sum=nb_hash_sha256
                    ),
                    team_members=group,
                    test_cases=list(map(str, self._cases)),
                    results=results,
                    summary=summary,
                )

                # store copy of notebook
                logger.debug('write copy of notebook')
                with open(f'notebook.ipynb', mode='wb') as f:
                    f.write(nb_data)

                # store results as csv file
                logger.debug('write results to csv')
                with open('test_results.csv', 'w', newline='') as csv_file:
                    csv_writer = csv.DictWriter(csv_file, fieldnames=next(iter(results.values())))
                    csv_writer.writeheader()
                    csv_writer.writerows(results.values())

                # store results as json
                logger.debug('write results to json')
                with open('test_results.json', mode='wt') as f:
                    json.dump(enriched_results, fp=f, indent=4)

                # create a human readable report
                logger.debug('write report')
                with open('report.rst', mode='wt') as f:
                    def _results():
                        ignore = {'stdout', 'stderr'}
                        for i, r in results.items():
                            yield {'nr': i, **{k: v for k, v in r.items() if k not in ignore}}

                    f.write(REPORT_TEMPLATE.format(
                        version=autograde.__version__,
                        timestamp=datetime.now().strftime("%m-%d-%Y, %H:%M:%S"),
                        team=tabulate(group, headers='keys', tablefmt='grid'),
                        results=tabulate(_results(), headers='keys', tablefmt='grid'),
                        summary=tabulate(summary.items(), tablefmt='grid'),
                    ))

                # create alternative, more readable name
                names = ','.join(map(camel_case, sorted(m['last_name'] for m in group)))
                archive_name_alt = Path(f'results_[{names}]_{nb_hash_sha256_short}.tar.xz')

            if archive_name_alt.exists():
                logger.debug(f'remove existing {archive_name_alt}')
                archive_name_alt.unlink()

            archive.rename(archive_name_alt)

        return enriched_results

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

        return len(self._cases) - results['summary'].get('passed', 0)
