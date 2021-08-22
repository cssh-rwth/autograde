import base64
import builtins
import importlib
import logging
import math
import os
import re
import sys
import threading
import time
from contextlib import contextmanager, ExitStack
from ctypes import pythonapi, c_long, py_object
from datetime import datetime
from functools import partial, wraps
from pathlib import Path
from queue import Queue
from tempfile import NamedTemporaryFile, TemporaryDirectory
from threading import Thread
from typing import Any, Callable, ContextManager, Dict, Generator, Iterable, List, Optional, TextIO, Tuple, Union
from unittest import mock
from zipfile import ZipFile

from htmlmin.minify import html_minify
from jinja2 import Environment, PackageLoader, select_autoescape

import autograde
from autograde.static import CSS, FAVICON

DateTime = datetime
ALPHA_NUMERIC = re.compile(r'[^\w]')
FAVICON = base64.b64encode(FAVICON).decode('utf-8')
JINJA_ENV = Environment(
    loader=PackageLoader('autograde', 'templates'),
    autoescape=select_autoescape(['html', 'xml']),
    trim_blocks=True,
    lstrip_blocks=True
)

_formatter = logging.Formatter(
    '{asctime} [{levelname}] {processName}:  {message}',
    datefmt='%Y-%m-%d %H:%M:%S',
    style='{'
)

_stream_handler = logging.StreamHandler(sys.stdout)
_stream_handler.setFormatter(_formatter)

logger = logging.getLogger('autograde')
logger.addHandler(_stream_handler)


def parse_bool(s: Any) -> bool:
    s = str(s).lower()
    if s in {'0', 'false', 'f', 'no', 'n'}:
        return False
    elif s in {'1', 'true', 't', 'yes', 'y'}:
        return True
    raise ValueError(f'cannot parse "{s}" as boolean')


def now() -> DateTime:
    return DateTime.now().replace(microsecond=0)


def loglevel(x: int) -> int:
    return max(10, 40 - max(x, 0) * 10)


def project_root() -> Path:
    import autograde
    return Path(autograde.__file__).parent.parent


def float_equal(a: float, b: float) -> bool:
    return math.isclose(a, b) or (math.isnan(a) and math.isnan(b))


def _alpha_numeric_split(string: str) -> Generator[str, None, None]:
    yield from filter(lambda s: s, ALPHA_NUMERIC.split(string.strip()))


def snake_case(string: str) -> str:
    return '_'.join(map(str.lower, _alpha_numeric_split(string)))


def camel_case(string: str) -> str:
    return ''.join(f'{s[0].upper()}{s[1:].lower()}' for s in _alpha_numeric_split(string))


def prune_join(words: Iterable[str], separator: str = ',', max_width: Union[int, float] = float('inf')) -> str:
    """
    Join strings by given separator. Optionally, the strings are pruned in order
    make the resulting string matching the maximum width criteria.
    """
    words = list(words)
    max_words_width = max_width - (len(words) - 1) * len(separator)

    if max_words_width < len(words) * 3:
        raise ValueError(f'cannot fit given words into a string of width {max_width}')

    if not math.isinf(max_width):
        for _ in range(int(sum(map(len, words)) - max_words_width)):
            idx = words.index(max(words, key=len))
            words[idx] = f'{words[idx][:-3]}..'

    return separator.join(words)


@contextmanager
def capture_output(tmp_stdout: Optional[TextIO] = None, tmp_stderr: Optional[TextIO] = None) -> \
        ContextManager[Tuple[TextIO, TextIO]]:
    """temporary redirect stdout and/or stderr to custom streams"""
    stdout = sys.stdout
    stderr = sys.stderr

    sys.stdout = tmp_stdout or sys.stdout
    sys.stderr = tmp_stderr or sys.stderr

    try:
        yield stdout, stderr
    finally:
        sys.stdout = stdout
        sys.stderr = stderr


@contextmanager
def cd(path: Union[Path, str], mkdir: bool = False) -> ContextManager[Path]:
    """Change directory"""
    path = Path(path)
    cwd = Path(os.getcwd())

    if mkdir:
        logger.debug(f'create directories: {path}')
        os.makedirs(path, exist_ok=True)

    logger.debug(f'change directory: {path}')
    os.chdir(path)

    try:
        yield cwd

    finally:
        logger.debug(f'change directory: {cwd}')
        os.chdir(cwd)


@contextmanager
def cd_zip(path: Union[Path, str], mode: str = 'w') -> ContextManager[ZipFile]:
    """Change to a temporary directory and write all contents to zip file when context is left"""
    if mode not in set('wax'):
        raise ValueError('supported modes: "w", "a", "x"')

    with ZipFile(Path(path), mode=mode) as zipf, TemporaryDirectory() as tmp, cd(tmp):
        tmp = Path(tmp)
        yield zipf

        # add all files in temporary directory to zipfile
        for file in filter(Path.is_file, tmp.rglob('*')):
            zipf.write(file, file.relative_to(tmp))


class StopWatch:
    """Measure durations"""

    def __init__(self):
        self._captures = [time.monotonic()]

    def capture(self) -> int:
        """Store current time and return index of capture"""
        self._captures.append(time.monotonic())
        return len(self._captures) - 1

    def __enter__(self) -> int:
        return self.capture()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.capture()

    @property
    def captures(self):
        return self._captures

    def duration_abs(self) -> List[float]:
        """Compute absolute durations with respect to stop watch instantiation"""
        return list(map(lambda t: t - self.captures[0], self.captures))

    def duration_rel(self) -> List[float]:
        """Compute relative durations with respect to previous capture"""
        return list(map(lambda ts: ts[1] - ts[0], zip([self.captures[0]] + self.captures, self.captures)))


class KillableThread(Thread):
    """
    A thread implementation that can be forcibly terminated from the outside.
    Stolen from: blog.finxter.com/how-to-kill-a-thread-in-python/#Method_4_Using_ctypes_To_Raise_Exceptions_In_A_Thread
    """

    @wraps(Thread.__init__)
    def __init__(self, group=None, target=None, name=None, args=(), kwargs=None, *, daemon=None):
        self._result_queue = Queue()
        wrapper = target
        if target:
            def wrapper(*args, **kwargs):
                try:
                    self._result_queue.put(target(*args, **kwargs))
                except Exception as e:
                    self._result_queue.put(e)
        super().__init__(group, wrapper, name, args, kwargs, daemon=daemon)

    @property
    def thread_id(self) -> Optional[int]:
        """
        :return: unique thread id or none if thread has not been started
        """
        attr_name = '_thread_id'

        if (tid := getattr(self, attr_name, None)) is not None:
            return tid

        for tid, thread in threading._active.items():
            if thread is self:
                setattr(self, attr_name, tid)
                return tid

        return None

    def terminate(self):
        """
        raises SystemExit in the context of the given thread, which should
        cause the thread to exit silently (unless caught)
        """
        if not self.is_alive():
            raise RuntimeError('threads can only be killed when alive')

        if pythonapi.PyThreadState_SetAsyncExc(c_long(self.thread_id), py_object(SystemExit)) > 1:
            # if it returns a number greater than one, you're in trouble,
            # and you should call it again with exc=NULL to revert the effect
            pythonapi.PyThreadState_SetAsyncExc(self.thread_id, 0)
            raise SystemError("PyThreadState_SetAsyncExc failed")

    def result(self) -> Any:
        result = self._result_queue.get(block=True)
        if isinstance(result, BaseException):
            raise result
        return result


def run_with_timeout(target: Callable[..., Any], timeout: float, *, args: Tuple = (),
                     kwargs: Optional[Dict[str, Any]] = None) -> Any:
    """
    Run a given callable and raise a `TimeoutError` when the timeout limit is exceeded.

    :param target: callable object to be run
    :param timeout: timeout limit
    :param args: positional arguments for the target invocation
    :param kwargs: keyword arguments for the target invocation
    :return: return value of target invocation
    """
    worker = KillableThread(target=target, args=args, kwargs=kwargs or dict())
    worker.start()
    worker.join(timeout=timeout)
    if worker.is_alive():
        worker.terminate()
        worker.join()  # this may still block if the job catches SystemExit!
        raise TimeoutError(f'code execution took longer than {timeout or -1.:.3f}s to terminate')
    return worker.result()


def exec_with_timeout(*args, timeout: float):
    """
    Wrapper for builtin `exec` that aborts execution when given timeout limit is exceeded.

    :param args: positional arguments for `exec`
    :param timeout: timeout limit
    """
    run_with_timeout(lambda: exec(*args), timeout=timeout)


@contextmanager
def shadow_exec(source: str, *args, timeout: Optional[float] = None) -> ContextManager[Path]:
    """
    Wrapper for builtin `exec` that accepts source code as input and executes it while preserving
    a copy of it in the file system. That's particularly useful when inspecting the state created
    by source execution later on.

    :param source: source code to be executed
    :param args: positional arguments for `exec`
    :param timeout: optional, results in using `exec_with_timeout` internally when non negative
    :return: path of the source file
    """
    source = f'{source}\n'
    exec_function = partial(exec_with_timeout, timeout=timeout) if timeout and timeout > 0 else exec
    with NamedTemporaryFile(mode='wt', encoding='utf-8') as shadow_copy:
        shadow_copy.write(source)
        shadow_copy.flush()
        exec_function(compile(source, shadow_copy.name, mode='exec'), *args)
        yield Path(shadow_copy.name)


class WatchDog:
    """Observe directory for file changes without looking into files"""

    @staticmethod
    def _hash_stat(stat: os.stat_result) -> int:
        return hash(stat.st_ctime) ^ hash(stat.st_mtime_ns) ^ hash(stat.st_size)

    def __init__(self, path: Union[Path, str]):
        self._path = Path(path)

        if not self._path.is_dir():
            raise ValueError(f'{self._path} is no valid directory')

        self._index = dict()
        self.reload()

    def _list(self):
        yield from ((p, self._hash_stat(p.stat())) for p in self._path.rglob('*') if p.is_file())

    def reload(self):
        """Re-build index (discards all changes that have been registered)"""
        self._index = dict(self._list())

    def list_changed(self) -> Generator[Path, None, None]:
        """List files that have been modified since the index was built"""
        yield from (path for path, hsh in self._list() if hsh != self._index.get(path))

    def list_unchanged(self) -> Generator[Path, None, None]:
        """List files that have NOT been modified since the index was built"""
        yield from (path for path, hsh in self._list() if hsh == self._index.get(path))


def render(template: str, minify: bool = True, **kwargs) -> str:
    """Render template with default values set"""
    html = JINJA_ENV.get_template(template).render(
        autograde=autograde,
        css=CSS,
        favicon=FAVICON,
        timestamp=now().isoformat(),
        **kwargs
    )
    if minify:
        return html_minify(html)
    return html


@contextmanager
def import_filter(regex: Union[str, re.Pattern], flags: int = 0, blacklist: bool = False) -> ContextManager[None]:
    """Temporarily filter imports with a regex"""
    pattern = re.compile(regex, flags) if isinstance(regex, str) else regex
    builtins_import = builtins.__import__
    importlib_import = importlib.__import__
    importlib_import_module = importlib.import_module

    def make_guard(protected_function):
        @wraps(protected_function)
        def guard(target, *args, **kwargs):
            matches = pattern.search(target) is not None
            if (matches and blacklist) or (not matches and not blacklist):
                raise ImportError(f'usage of "{target}" is not permitted')
            return protected_function(target, *args, **kwargs)
        return guard

    with ExitStack() as stack:
        stack.enter_context(mock.patch('importlib.import_module', side_effect=make_guard(importlib_import_module)))
        stack.enter_context(mock.patch('importlib.__import__', side_effect=make_guard(importlib_import)))
        stack.enter_context(mock.patch('builtins.__import__', side_effect=make_guard(builtins_import)))
        yield None
