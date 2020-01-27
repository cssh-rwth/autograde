# Standard library modules.
import os
import re
import sys
import time
import logging
import tarfile
from pathlib import Path
from contextlib import contextmanager
from tempfile import TemporaryDirectory

# Third party modules.

# Local modules

# Globals and constants variables.
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


def loglevel(x):
    return max(10, 40 - max(x, 0) * 10)


def project_root():
    import autograde
    return Path(autograde.__file__).parent.parent


def snake_case(s):
    return '_'.join(map(str.lower, ALPHA_NUMERIC.split(s.strip())))


def camel_case(s):
    return ''.join(f'{ss[0].upper()}{ss[1:].lower()}' for ss in ALPHA_NUMERIC.split(s.strip()))


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
def cd(tmp_cwd, mkdir=False):
    if mkdir:
        os.makedirs(tmp_cwd, exist_ok=True)

    cwd = os.getcwd()
    os.chdir(tmp_cwd)

    try:
        yield tmp_cwd

    finally:
        os.chdir(cwd)


@contextmanager
def cd_tar(name, mode='w'):
    assert mode.startswith('w') and '|' not in mode

    with TemporaryDirectory() as tempdir:
        try:
            with cd(tempdir):
                yield tempdir

        finally:
            with tarfile.open(name, mode=mode) as tf:
                tf.add(tempdir, arcname='')


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
