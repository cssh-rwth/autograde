import inspect
import io
import os
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterable
from unittest import TestCase

import nbformat.v4 as nbf
from nbformat import NotebookNode

from autograde.notebook_executor import as_py_comment, shadow_exec, ArtifactLoader, Shell, PedanticShell, \
    ForgivingShell, exec_notebook
from autograde.util import project_root, cd

PROJECT_ROOT = project_root()


class TestArtifactLoader(TestCase):
    def test_default(self):
        with TemporaryDirectory() as temp, cd(temp):
            os.mkdir('artifacts')

            with Path('artifacts').joinpath('foo').open(mode='wb') as f:
                f.write(b'FOO')

            loader = ArtifactLoader()

            self.assertEqual(b'FOO', loader['foo'])
            with self.assertRaises(FileNotFoundError):
                _ = loader['bar']

    def test_custom_root(self):
        with TemporaryDirectory() as temp, cd(temp):
            os.mkdir('root')

            with Path('root').joinpath('foo').open(mode='wb') as f:
                f.write(b'FOO')

            loader = ArtifactLoader('root')

            self.assertEqual(b'FOO', loader['foo'])
            with self.assertRaises(FileNotFoundError):
                _ = loader['bar']


class TestShell(TestCase):
    @staticmethod
    def nb_from_code_snippets(snippets: Iterable[str]) -> NotebookNode:
        nb = nbf.new_notebook()
        nb.cells.extend(map(nbf.new_code_cell, snippets))
        return nb

    def setUp(self) -> None:
        self.nb_line_magic = self.nb_from_code_snippets(["%time 1 ** 32\n"])
        self.nb_cell_magic = self.nb_from_code_snippets(["%%sh\n\nwhoami\n"])
        self.nb_system = self.nb_from_code_snippets(["!whoami\n"])

    def test_exec_line_magic(self):
        with open(os.devnull, 'w') as devnull, exec_notebook(self.nb_line_magic, file=devnull, shell_cls=Shell):
            pass

    def test_exec_cell_magic(self):
        with open(os.devnull, 'w') as devnull, exec_notebook(self.nb_cell_magic, file=devnull, shell_cls=Shell):
            pass

    def test_system(self):
        with open(os.devnull, 'w') as devnull, exec_notebook(self.nb_system, file=devnull, shell_cls=Shell):
            pass


class TestPedanticShell(TestShell):
    def test_exec_line_magic(self):
        with self.assertRaises(PermissionError), open(os.devnull, 'w') as devnull:
            with exec_notebook(self.nb_line_magic, file=devnull, shell_cls=PedanticShell):
                pass

    def test_exec_cell_magic(self):
        with self.assertRaises(PermissionError), open(os.devnull, 'w') as devnull:
            with exec_notebook(self.nb_cell_magic, file=devnull, shell_cls=PedanticShell):
                pass

    def test_system(self):
        with self.assertRaises(PermissionError), open(os.devnull, 'w') as devnull:
            with exec_notebook(self.nb_system, file=devnull, shell_cls=PedanticShell):
                pass


class TestForgivingShell(TestShell):
    def test_exec_line_magic(self):
        with io.StringIO() as buffer:
            with exec_notebook(self.nb_line_magic, file=buffer, shell_cls=ForgivingShell):
                pass
            self.assertIn("Ignore IPython magic command", buffer.getvalue())

    def test_exec_cell_magic(self):
        with io.StringIO() as buffer:
            with exec_notebook(self.nb_cell_magic, file=buffer, shell_cls=ForgivingShell):
                pass
            self.assertIn("Ignore IPython magic command", buffer.getvalue())

    def test_system(self):
        with io.StringIO() as buffer:
            with exec_notebook(self.nb_system, file=buffer, shell_cls=ForgivingShell):
                pass
            self.assertIn("Ignore IPython system command", buffer.getvalue())


class TestFunctions(TestCase):
    def test_as_py_comment(self):
        self.assertEqual('', as_py_comment(''))
        self.assertEqual('# foo', as_py_comment('foo'))
        self.assertEqual('# foo\n# bar', as_py_comment('foo\nbar').strip())
        self.assertEqual('#     foo', as_py_comment('foo', 4))

    def test_shadowed_exec(self):
        state = dict()
        source = 'def foo():\n\treturn 42'
        with shadow_exec(source, state) as path:
            with open(path, mode='rt') as f:
                shadow_source = f.read()

            self.assertEqual(f'{source}\n', inspect.getsource(state['foo']))
            self.assertEqual(f'{source}\n', shadow_source)

        self.assertEqual(42, state['foo']())

        with self.assertRaises(OSError):
            inspect.getsource(state['foo'])

        with shadow_exec('def bar():\n\tassert False', state):
            pass

        with self.assertRaises(AssertionError):
            state['bar']()

    def test_exec_notebook(self):
        nb_path = PROJECT_ROOT.joinpath('demo', 'notebook.ipynb')
        with open(nb_path, mode='rt') as f:
            nb = f.read()

        with TemporaryDirectory() as path, cd(path):
            shutil.copytree(PROJECT_ROOT.joinpath('demo', 'context'), '.', dirs_exist_ok=True)

            # forward errors raised in notebook
            with self.assertRaises(AssertionError):
                with io.StringIO(nb) as nb_buffer, open(os.devnull, 'w') as f:
                    with exec_notebook(nb_buffer, file=f):
                        pass

            # cell timeout
            with self.assertRaises(TimeoutError):
                with io.StringIO(nb) as nb_buffer, open(os.devnull, 'w') as f:
                    with exec_notebook(nb_buffer, file=f, cell_timeout=0.05):
                        pass

            # ignore errors
            with io.StringIO(nb) as nb_buffer, io.StringIO() as f:
                with exec_notebook(nb_buffer, file=f, ignore_errors=True) as state:
                    pass

        self.assertIn('__IMPORT_FILTER__', state)
        self.assertIn('__PLOTS__', state)
        self.assertIn('__LABEL__', state)
        self.assertEqual(state.get('SOME_CONSTANT'), 42)
