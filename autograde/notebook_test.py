# Standard library modules.
import io
import os
import csv
import sys
import json
import argparse
import warnings
import traceback
from pathlib import Path
from datetime import datetime
from distutils import dir_util
from collections import OrderedDict
from hashlib import blake2b, md5, sha256

# Third party modules.
from tabulate import tabulate
from nbformat import read, NotebookNode
from IPython.core.interactiveshell import InteractiveShell

# Local modules
from autograde.util import logger, loglevel, capture_output, cd, cd_tar, timeout

# Globals and constants variables.
NOTEBOOK_TEST = OrderedDict()

REPORT_TEMPLATE = """
======
REPORT
======
**created {timestamp}**


TEAM
----

{team}


RESULTS
-------

{results}


SUMMARY
-------

{summary}

""".strip() + '\n\n'

INJECT_BEFORE = """
# ensure matplotlib works on a headless backend

def dummy_show(*args, **kwargs):
    print('`pyplot.show` does not display plots in test mode')

try:
    import matplotlib as mpl
    mpl.use('Agg')
    print("use 'Agg' backend")

    import matplotlib.pyplot as plt
    plt.show = dummy_show

except ImportError:
    print("'matplotlib' not found")

    from types import SimpleNamespace
    __dummy = lambda *args, **kwargs: None
    
    plt = SimpleNamespace(savefig=_dummy, cla=_dummy, clf=_dummy, close=_dummy, show=_dummy)

FIG_PREV = None
""".strip()

INJECT_AFTER = """
print('done')
""".strip()


def as_py_comment(s):
    if not s:
        return ''
    return '\n'.join(f'# {l}' for l in s.strip().split('\n'))


def exec_notebook(buffer, file=sys.stdout, ignore_errors=False, cell_timeout=0):
    error = None
    state = dict()

    try:
        logger.debug('parse notebook')

        # when executed within a docker container, some minor warnings occur that we filter here
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')

            notebook = read(buffer, 4)
            shell = InteractiveShell.instance()

        _code_cells = filter(lambda c: c.cell_type == 'code', notebook.cells)

        cells = [
            ('injected by test', INJECT_BEFORE),
            *((f'c{i}', c) for i, c in enumerate(_code_cells, start=1)),
            ('injected by test', INJECT_AFTER)
        ]

    except Exception as err:
        logger.error(f'unable to parse notebook: {err}')
        return err, state

    # the log is supposed to be a valid, standalone python script
    print('#!/usr/bin/env python3', file=file)

    # actual code execution
    for i, (label, code) in enumerate(cells, start=1):
        with io.StringIO() as stdout, io.StringIO() as stderr:
            logger.debug(f'[{i}/{len(cells)}] execute cell ("{label}")')

            try:
                with capture_output(stdout, stderr):
                    # convert to valid python when origin is a notebook
                    if isinstance(code, NotebookNode):
                        code = shell.input_transformer_manager.transform_cell(code.source)

                    # actual execution that extends state
                    with timeout(cell_timeout):
                        exec(code, state)

            except Exception as err:
                # extend log with some meaningful error message
                traceback.print_exception(type(err), err, err.__traceback__, file=stderr)

                if not ignore_errors:
                    error = err
                    break

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
                # exec(f'plt.savefig("cell_{label}.png");plt.close()', state)

    logger.debug('execution completed')
    return error, state


class NotebookTestCase:
    def __init__(self, test_function, target, score=1., label=None, timeout_=0):
        assert callable(test_function), '`test_function` has to be callable'

        self._test_func = test_function
        self._target = str(target)
        self._score = float(score)
        self._label = str(label) if label else ''
        self._timeout = timeout_

    def __call__(self, state, *args, **kwargs):
        try:
            target = state.get(self.target)
            assert target is not None, f'Target "{self.target}" is not specified!'

            with timeout(self._timeout):
                self._test_func(target, *args, **kwargs)

            return self.score, 'ok'

        except Exception as err:
            msg_buffer = io.StringIO()
            traceback.print_exception(type(err), err, err.__traceback__, file=msg_buffer)
            return 0, msg_buffer.getvalue()

    def __str__(self):
        timeout_ = f'{self._timeout:.2f}s' if self._timeout is not None else 'None'
        return f'{self.__class__.__name__}(target={self.target}, score={self.score}, ' \
               f'timeout={timeout_}, label="{self.label}")'

    target = property(fget=lambda self: self._target)
    score = property(fget=lambda self: self._score)
    label = property(fget=lambda self: self._label)


class NotebookTest:
    def __init__(self, cell_timeout=0, test_timeout=0):
        self._cases = []
        self._cell_timeout = cell_timeout
        self._test_timeout = test_timeout

    def __str__(self):
        return f'{type(self).__name__}({len(self._cases)} cases)'

    def __repr__(self):
        return f'{type(self).__name__}({self._cases})'

    def register(self, target, score=1., label='', timeout_=0):
        def decorator(func):
            case = NotebookTestCase(func, target, score, label, timeout_ or self._test_timeout)
            self._cases.append(case)
            return case

        return decorator

    @staticmethod
    def summarize_results(results):
        return OrderedDict(
            tests=len(results),
            passed=sum(r['score'] == r['score_max'] for r in results.values()),
            score=sum(r['score'] for r in results.values()),
            score_max=sum(r['score_max'] for r in results.values())
        )

    def apply_tests(self, state):
        state = state.copy()

        results = OrderedDict()
        for i, case in enumerate(self._cases, start=1):
            logger.debug(f'[{i}/{len(self._cases)}] execute {case}')

            with io.StringIO() as stdout, io.StringIO() as stderr:
                # with capture_output(stdout, stderr):
                with capture_output(stdout, stderr):
                    achieved, msg = case(state)

                results[i] = OrderedDict(
                    label=case.label,
                    target=case.target,
                    score=achieved,
                    score_max=case.score,
                    message=msg,
                    stdout=stdout.getvalue(),
                    stderr=stderr.getvalue()
                )

        logger.debug('testing completed')
        return results, self.summarize_results(results)

    def grade_notebook(self, nb_path, target_dir=None, context=None):
        target_dir = target_dir or os.getcwd()

        # prepare notebook
        with open(nb_path, mode='rb') as f:
            nb_data = f.read()

        nb_hash = blake2b(nb_data, digest_size=4).hexdigest()

        with cd(target_dir), cd_tar(f'results_{nb_hash}.tar.xz', mode='w:xz'):
            # prepare context and execute notebook
            with open('code.py', mode='wt') as c, cd_tar('artifacts.tar.xz', mode='w:xz'):
                if context is not None:
                    logger.debug(f'copy context files from: {context}')
                    dir_util.copy_tree(context, '.')

                logger.debug('execute notebook')
                error, state = exec_notebook(
                    io.StringIO(nb_data.decode('utf-8')),
                    file=c,
                    ignore_errors=True,
                    cell_timeout=self._cell_timeout
                )

            if error is not None:
                logger.debug(f'An error occurred when executing "{nb_path}": {error}')

            # infer meta information
            group = state.get('team_members', {})

            if not group:
                logger.warning(f'Couldn\'t find valid information about team members in "{nb_path}"')

            # execute tests
            logger.debug('execute tests')
            results, summary = self.apply_tests(state)
            enriched_results = OrderedDict(
                orig_file=str(nb_path),
                checksum=dict(
                    md5sum=md5(nb_data).hexdigest(),
                    sha256sum=sha256(nb_data).hexdigest(),
                    blake2bsum=nb_hash
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
                    timestamp=datetime.now().strftime("%m-%d-%Y, %H:%M:%S"),
                    team=tabulate(group, headers='keys', tablefmt='grid'),
                    results=tabulate(_results(), headers='keys', tablefmt='grid'),
                    summary=tabulate(summary.items(), tablefmt='grid'),
                ))

        return enriched_results

    def execute(self):
        parser = argparse.ArgumentParser(description='run tests on jupyter notebook')

        parser.add_argument('notebook', type=str, help='the jupyter notebook to test')
        parser.add_argument('-t', '--target', type=str, metavar='', help='where to store results')
        parser.add_argument('-c', '--context', type=str, metavar='', help='context directory')
        parser.add_argument('-v', '--verbose', action='count', default=0, help='verbosity level')

        args = parser.parse_args()

        logger.setLevel(loglevel(args.verbose))
        logger.debug(f'args: {args}')

        self.grade_notebook(
            Path(args.notebook).absolute(),
            target_dir=Path(args.target).absolute() if args.target else None,
            context=Path(args.context).absolute() if args.context else None
        )

        return 0
