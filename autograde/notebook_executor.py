import io
import sys
import traceback
import warnings
from contextlib import contextmanager, ExitStack
from copy import deepcopy
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Dict, TextIO

from IPython.core.interactiveshell import InteractiveShell
from nbformat import read

from autograde.helpers import import_filter
from autograde.static import INJECT_BEFORE, INJECT_AFTER
from autograde.util import logger, capture_output, deadline, StopWatch


def as_py_comment(s: str, indentation: int = 0):
    """
    Escape string as python comment

    :param s: any string
    :param indentation: spaces to ident, default is 0
    :return: escaped string
    """
    indentation = ' ' * indentation
    if not s:
        return ''
    return '\n'.join(f'# {indentation}{line}' for line in s.strip().split('\n'))


class ArtifactLoader:
    """Helper class that provides a dict like interface for accessing artifact files"""

    def __init__(self, root='artifacts'):
        self._root = Path(root)

    def __getitem__(self, path) -> bytes:
        with self._root.joinpath(path).open(mode='rb') as f:
            return f.read()


@contextmanager
def shadow_exec(source: str, *args, **kwargs):
    """
    Wrapper for builtin `exec` that accepts source code as input and executes it while preserving
    a copy of it in the file system. That's particularly useful when inspecting the state created
    by source execution later on.

    :param source: source code to be executed
    :param args: positional arguments for `exec`
    :param kwargs: key word arguments for `exec`
    :return: path of the source file
    """
    source = f'{source}\n'
    with NamedTemporaryFile(mode='wt', encoding='utf-8') as shadow_copy:
        shadow_copy.write(source)
        shadow_copy.flush()

        compiled_source = compile(source, shadow_copy.name, mode='exec')
        exec(compiled_source, *args, **kwargs)

        yield shadow_copy.name


@contextmanager
def exec_notebook(notebook, file: TextIO = sys.stdout, cell_timeout: float = 0.,
                  ignore_errors: bool = False, variables: Dict = None):
    """
    Extract source code from jupyter notebook and execute it.

    :param notebook: file like with notebook data
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

            notebook = read(notebook, 4)
            shell = InteractiveShell.instance()

        # extract comment cells
        md_cells = [c.source for c in filter(lambda c: c.cell_type == 'markdown', notebook.cells)]

        # prepare code cells for execution
        def _code_cells():
            yield 'injected: setup', INJECT_BEFORE, 0

            for i, cell in enumerate(filter(lambda c: c.cell_type == 'code', notebook.cells)):
                # render code
                source = shell.input_transformer_manager.transform_cell(cell.source)
                yield (
                    f'nb-{i + 1}',
                    f'{source.strip()}\n\n# injected by test\ndump_figure()',
                    cell_timeout
                )

            yield 'injected: teardown', INJECT_AFTER, 0

        code_cells = list(_code_cells())

    except Exception as error:
        logger.error(f'unable to parse notebook: {error}')
        raise ValueError(error)

    # prepare import filter
    if_regex, if_blacklist = variables.get('IMPORT_FILTER', (None, None))

    # the log is supposed to be a valid, standalone python script
    print('#!/usr/bin/env python3', file=file)

    # actual code execution
    with ExitStack() as shadow_stack:
        for i, (label, code_snippet, timeout) in enumerate(code_cells, start=1):
            state.update({'__LABEL__': deepcopy(label), '__PLOT_REGISTRY__': []})

            with io.StringIO() as stdout, io.StringIO() as stderr:
                logger.debug(f'[{i}/{len(code_cells)}] execute cell ("{label}")')
                stopwatch = StopWatch()

                # actual execution that mutates the state
                try:
                    with capture_output(stdout, stderr):
                        with ExitStack() as es:
                            if if_regex is not None and i > 1:
                                es.enter_context(import_filter(if_regex, blacklist=if_blacklist))
                            es.enter_context(deadline(timeout))
                            es.enter_context(stopwatch)

                            shadow_stack.enter_context(shadow_exec(code_snippet, state))

                # extend log with a meaningful error message
                except Exception as error:
                    traceback.print_exception(type(error), error, error.__traceback__, file=stderr)

                    if not ignore_errors:
                        print('abort due to previous error', file=stderr)
                        raise error

                # log code cell and outputs
                finally:
                    with capture_output(file):
                        _label = f' CODE CELL {label} '
                        print(f'# {_label:-^78}')
                        print(str(code_snippet).strip())

                        print()
                        print(f"# EXECUTED IN {stopwatch.duration_rel()[-1]:.3}s")

                        if stdout_s := stdout.getvalue():
                            print('# STDOUT')
                            print(as_py_comment(stdout_s, 4))

                        if stderr_s := stderr.getvalue():
                            print('# STDERR')
                            print(as_py_comment(stderr_s, 4))

                        print()

        # add special items to state
        state['__COMMENTS__'] = md_cells
        state['__ARTIFACTS__'] = ArtifactLoader()

        logger.debug('execution completed')
        yield state
