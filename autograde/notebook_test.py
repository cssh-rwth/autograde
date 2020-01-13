# Standard library modules.
import io
import os
import csv
import sys
import json
import argparse
import traceback
from pathlib import Path
from warnings import warn
from distutils import dir_util
from collections import OrderedDict
from hashlib import blake2b, md5, sha256

# Third party modules.
from nbformat import read, NotebookNode
from IPython.core.interactiveshell import InteractiveShell

# Local modules
from autograde.util import capture_output, cd, cd_tar

# Globals and constants variables.
NOTEBOOK_TEST = OrderedDict()

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
    plt = SimpleNamespace(savefig=lambda *args, **kwargs: None)

FIG_PREV = None
""".strip()


def as_py_comment(s):
    if not s:
        return ''
    return '\n'.join(f'# {l}' for l in s.strip().split('\n'))


def exec_notebook(buffer, file=sys.stdout, ignore_errors=False):
    error = None
    state = dict()

    try:
        notebook = read(buffer, 4)
        shell = InteractiveShell.instance()
        cells = (
            ('injected by test', INJECT_BEFORE),
            *enumerate(filter(lambda c: c.cell_type == 'code', notebook.cells), start=1)
        )

    except Exception as err:
        return err, state

    # the log is supposed to be a valid, standalone python script
    with capture_output(file):
        print('#!/usr/bin/env python3')

    # actual code execution
    for label, code in cells:
        with io.StringIO() as stdout, io.StringIO() as stderr:
            try:
                with capture_output(stdout, stderr):
                    # convert to valid python when origin is a notebook
                    if isinstance(code, NotebookNode):
                        code = shell.input_transformer_manager.transform_cell(code.source)

                    # actual execution that extends state
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

    return error, state


class NotebookTestCase:
    def __init__(self, test_function, target, score=1., label=None, timeout=None):
        assert callable(test_function), '`func` has to be callable'

        self._test_func = test_function
        self._target = str(target)
        self._score = float(score)
        self._label = str(label) if label else ''
        self._timeout = timeout

    def __call__(self, state, *args, **kwargs):
        try:
            target = state.get(self.target)
            assert target is not None, f'Target "{self.target}" is not specified!'

            self._test_func(target, *args, **kwargs)
            return self.score, 'ok'

        # except AssertionError as error:
        #     return 0, error.args[0] if len(error.args) > 0 else ''

        except Exception as err:
            msg_buffer = io.StringIO()
            traceback.print_exception(type(err), err, err.__traceback__, file=msg_buffer)
            return 0, msg_buffer.getvalue()

    def __str__(self):
        timeout = f'{self._timeout:.2f}s' if self._timeout is not None else 'None'
        return f'{self.__class__.__name__}(target={self.target}, label="{self.label}", ' \
               f'score={self.score}, timeout={timeout})'

    target = property(fget=lambda self: self._target)
    score = property(fget=lambda self: self._score)
    label = property(fget=lambda self: self._label)


# TODO add before execution hook (pip install, download)
class NotebookTest:
    registry = []

    def __init__(self):
        self._cases = []

        self.registry.append(self)

    def __str__(self):
        return f'{type(self).__name__}({len(self._cases)} cases)'

    def __repr__(self):
        return f'{type(self).__name__}({self._cases})'

    def register(self, target, score=1., label='', timeout=None):
        def decorator(func):
            case = NotebookTestCase(func, target, score, label, timeout)
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
                    dir_util.copy_tree(context, '.')

                error, state = exec_notebook(
                    io.StringIO(nb_data.decode('utf-8')),
                    file=c,
                    ignore_errors=True
                )

            if error is not None:
                warn(f'An error occurred when executing "{nb_path}", continue anyway: {error}')

            # infer meta information
            group = state.get('team_members', {})

            if not group:
                warn(f'Couldn\'t find valid information about team members in "{nb_path}"')

            # execute tests
            results, summary = self.apply_tests(state)
            enriched_results = OrderedDict(
                checksum=dict(
                    md5sum=md5(nb_data).hexdigest(),
                    sha256sum=sha256(nb_data).hexdigest(),
                    blake2bsum=nb_hash
                ),
                team_members=group,
                test_cases=list(map(str, self._cases)),
                summary=summary,
                **results
            )

            # store copy of notebook
            with open(f'notebook.ipynb', mode='wb') as f:
                f.write(nb_data)

            # store results as csv file
            with open('test_results.csv', 'w', newline='') as csv_file:
                csv_writer = csv.DictWriter(csv_file, fieldnames=next(iter(results.values())))
                csv_writer.writeheader()
                csv_writer.writerows(results.values())

            # and store results as json also
            with open('test_results.json', mode='wt') as f:
                json.dump(enriched_results, fp=f, indent=4)

        return enriched_results

    def execute(self):
        parser = argparse.ArgumentParser(description='run tests on jupyter notebook')

        parser.add_argument('notebook', type=str, help='the jupyter notebook to test')
        parser.add_argument('-t', '--target', type=str, metavar='', help='where to store results')
        parser.add_argument('-c', '--context', type=str, metavar='', help='context directory')

        args = parser.parse_args()

        self.grade_notebook(
            Path(args.notebook).absolute(),
            target_dir=Path(args.target).absolute() if args.target else None,
            context=Path(args.context).absolute() if args.context else None
        )

        return 0
