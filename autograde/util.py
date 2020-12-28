import datetime
import logging
import math
import os
import re
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import ContextManager, Iterable, Union, List
from zipfile import ZipFile

import pytz

ALPHA_NUMERIC = re.compile(r'[^\w]')

_formatter = logging.Formatter(
    '{asctime} [{levelname}] {processName}:  {message}',
    datefmt='%Y-%m-%d %H:%M:%S',
    style='{'
)

_stream_handler = logging.StreamHandler(sys.stdout)
_stream_handler.setFormatter(_formatter)

logger = logging.getLogger('autograde')
logger.addHandler(_stream_handler)


def parse_bool(s):
    s = str(s).lower()
    if s in {'0', 'false', 'f', 'no', 'n'}:
        return False
    elif s in {'1', 'true', 't', 'yes', 'y'}:
        return True
    raise ValueError(f'cannot parse "{s}" as boolean')


def timestamp_utc_iso():
    return datetime.datetime.now(pytz.utc).replace(microsecond=0).isoformat()


def loglevel(x):
    return max(10, 40 - max(x, 0) * 10)


def project_root():
    import autograde
    return Path(autograde.__file__).parent.parent


def _alpha_numeric_split(s):
    for w in ALPHA_NUMERIC.split(s.strip()):
        if w:
            yield w


def snake_case(s):
    return '_'.join(map(str.lower, _alpha_numeric_split(s)))


def camel_case(s):
    if not s:
        return ''
    return ''.join(f'{ss[0].upper()}{ss[1:].lower()}' for ss in _alpha_numeric_split(s))


def prune_join(words: Iterable[str], separator: str = ',', max_width: Union[int, float] = float('inf')):
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
def capture_output(tmp_stdout=None, tmp_stderr=None):
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
    """Change to a temporary directory and write all files to zip file when context is left"""
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

    def __enter__(self):
        self.capture()
        return self

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


@contextmanager
def timeout(timeout_):
    start = time.time()

    # trace callbacks
    def _globaltrace(frame, event, arg):
        return _localtrace if event == 'call' else None

    def _localtrace(frame, event, arg):
        if time.time() - start >= timeout_ and event == 'line':
            raise TimeoutError(f'code execution took longer than {timeout_:.3f}s to terminate')

    # activate tracing only in case timeout was actually set
    if timeout_:
        sys.settrace(_globaltrace)

    try:
        yield start

    finally:
        sys.settrace(None)
