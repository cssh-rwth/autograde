# Standard library modules.
import os
import re
import sys
import pytz
import time
import logging
import tarfile
import datetime
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


def timestamp_utc_iso():
    return datetime.datetime.now(pytz.utc).replace(microsecond=0).isoformat()


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
def cd(path, mkdir=False):
    cwd = os.getcwd()

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
def mount_tar(path, mode='r'):
    prefix = mode[0]

    # TODO use ValueError instead
    assert prefix in ['r', 'w', 'a'], f'unknown prefix {prefix}'
    assert '|' not in mode, 'streaming is not supported'

    with TemporaryDirectory() as tempdir:
        logger.debug(f'create mount point at {tempdir} in mode "{mode}"')
        try:
            if prefix in ['r', 'a']:
                logger.debug(f'extract files from {path}')
                with tarfile.open(path, mode='r'+mode[1:]) as tar:
                    tar.extractall(tempdir)

            yield tempdir

        finally:
            if prefix in ['w', 'a']:
                logger.debug(f'write changes to {path}')
                with tarfile.open(path, mode='w'+mode[1:]) as tar:
                    tar.add(tempdir, arcname='')


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
