import io
import sys
import traceback
import warnings
from contextlib import contextmanager, ExitStack
from copy import deepcopy
from functools import wraps
from pathlib import Path
from typing import Any, Dict, Optional, TextIO, Type, Union

from IPython.core.interactiveshell import InteractiveShell
from nbformat import read, reads, NotebookNode

from autograde.static import INJECT_BEFORE, INJECT_AFTER
from autograde.util import logger, capture_output, shadow_exec, StopWatch, import_filter


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
        return self._root.joinpath(path).read_bytes()


class Shell(InteractiveShell):
    """Autograde's default shell for code parsing (not execution)"""

    @wraps(InteractiveShell.__init__)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Special commands are patched after initialization since IPython relies on them
        self.run_line_magic = self._run_line_magic
        self.run_cell_magic = self._run_cell_magic
        self.system_raw = self._system_raw
        self.system_piped = self._system_piped
        self.system = self._system_piped

    def enable_gui(self, gui=None):
        pass

    _run_line_magic = InteractiveShell.run_line_magic
    _run_cell_magic = InteractiveShell.run_cell_magic
    _system_raw = InteractiveShell.system_raw
    _system_piped = InteractiveShell.system_piped


class PedanticShell(Shell):
    """A shell that strictly prohibits usage of IPython special commands (starting with `%` or `!`)"""

    def _run_line_magic(self, magic_name, line, _stack_depth=1):
        raise PermissionError(
            f"Using IPython magic command is not allowed! run_line_magic({magic_name=}, {line=}, {_stack_depth=})"
        )

    def _run_cell_magic(self, magic_name, line, cell):
        raise PermissionError(
            f"Using IPython magic command is not allowed! run_cell_magic({magic_name=}, {line=}, {cell=})"
        )

    def _system_raw(self, cmd):
        raise PermissionError(f"fUsing IPython system commands is not allowed! system_raw({cmd=})")

    def _system_piped(self, cmd):
        raise PermissionError(f"fUsing IPython system commands is not allowed! system_piped({cmd=})")


class ForgivingShell(Shell):
    """A shell that ignores IPython special commands (starting with `%` or `!`)"""

    def _run_line_magic(self, magic_name, line, _stack_depth=1):
        print(f"Ignore IPython magic command: run_line_magic({magic_name=}, {line=}, {_stack_depth=})", file=sys.stderr)

    def _run_cell_magic(self, magic_name, line, cell):
        print(f"Ignore IPython magic command: run_cell_magic({magic_name=}, {line=}, {cell=})", file=sys.stderr)

    def _system_raw(self, cmd):
        print(f"Ignore IPython system command: system_raw({cmd=})", file=sys.stderr)

    def _system_piped(self, cmd):
        print(f"Ignore IPython system command: system_piped({cmd=})", file=sys.stderr)


@contextmanager
def exec_notebook(notebook: Union[NotebookNode, str, TextIO], file: TextIO = sys.stdout, cell_timeout: float = 0.,
                  ignore_errors: bool = False, shell_cls: Type[Shell] = Shell,
                  variables: Optional[Dict[str, Any]] = None):
    """
    Extract source code from jupyter notebook and execute it.

    :param notebook: the notebook to be executed
    :param file: where to send stdout
    :param ignore_errors: whether or not errors will be forwarded or ignored
    :param cell_timeout: timeout for cell execution 0=∞
    :param shell_cls: which shell to use
    :param variables: variables to be inserted into initial state
    :return: the state resulting from notebook execution
    """
    state = dict()
    variables = variables or {}

    try:
        logger.debug('parse notebook')

        # When executed within a container, some minor warnings occur that we filter here
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            if isinstance(notebook, NotebookNode):
                pass
            elif isinstance(notebook, str):
                notebook = reads(notebook, 4)
            else:
                notebook = read(notebook, 4)
            shell = shell_cls()

        # Extract comment cells
        md_cells = [c.source for c in filter(lambda c: c.cell_type == 'markdown', notebook.cells)]

        # Prepare code cells for execution
        def transform_code_cells():
            yield 'SETUP', INJECT_BEFORE, 0
            for i, cell in enumerate(filter(lambda c: c.cell_type == 'code', notebook.cells), start=1):
                yield f'{i}', shell.input_transformer_manager.transform_cell(cell.source).strip(), cell_timeout
                yield f'{i} CLEAN', 'auto_save_figure()', cell_timeout
            yield 'TEARDOWN', INJECT_AFTER, 0

        code_cells = list(transform_code_cells())

    except Exception as error:
        logger.error(f'unable to parse notebook: {error}')
        raise ValueError(error)

    # Prepare state
    state.update(deepcopy(variables))
    state['get_ipython'] = lambda: shell

    # Prepare import filter
    if_regex, if_blacklist = variables.get('IMPORT_FILTER', (None, None))

    # The log is supposed to be a valid, standalone python script
    print('#!/usr/bin/env python3', file=file)

    # Actual code execution
    # The shadow context determines the lifetime of code copies in the file system which are used for code inspection,
    # i.e. `inspect.getsource`
    with ExitStack() as shadow_context:
        for i, (label, code_cell, timeout) in enumerate(code_cells, start=1):
            state.update({'__LABEL__': deepcopy(label), '__PLOTS__': []})

            with io.StringIO() as stdout, io.StringIO() as stderr:
                logger.debug(f'[{i}/{len(code_cells)}] execute cell ("{label}")')
                stopwatch = StopWatch()

                # Execute cell
                try:
                    with capture_output(stdout, stderr), ExitStack() as execution_context:
                        if if_regex is not None and i > 1:
                            execution_context.enter_context(import_filter(if_regex, blacklist=if_blacklist))
                        execution_context.enter_context(stopwatch)
                        shadow_context.enter_context(shadow_exec(code_cell, state, timeout=timeout))

                # Extend log with a meaningful error message
                except Exception as error:
                    traceback.print_exception(type(error), error, error.__traceback__, file=stderr)

                    if not ignore_errors:
                        print('abort due to previous error', file=stderr)
                        raise error

                # Log code cell and outputs
                finally:
                    with capture_output(file):
                        _label = f' CODE CELL {label} '
                        print(f'# {_label:-^78}')
                        print(f'{str(code_cell).strip()}\n')
                        print(f"# EXECUTED IN {stopwatch.duration_rel()[-1]:.3}s")

                        if stdout_s := stdout.getvalue():
                            print('# STDOUT')
                            print(as_py_comment(stdout_s, 4))

                        if stderr_s := stderr.getvalue():
                            print('# STDERR')
                            print(as_py_comment(stderr_s, 4))

                        print()

        # Add special items to state
        state['__COMMENTS__'] = md_cells
        state['__ARTIFACTS__'] = ArtifactLoader()

        logger.debug('execution completed')
        yield state
