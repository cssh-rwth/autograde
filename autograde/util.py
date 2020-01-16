# Standard library modules.
import os
import sys
import logging
import tarfile
from pathlib import Path
from contextlib import contextmanager
from tempfile import TemporaryDirectory

# Third party modules.

# Local modules

# Globals and constants variables.


_formatter = logging.Formatter(
    '{asctime} [{levelname}] {processName}:\t{message}',
    datefmt='%Y-%m-%d %H:%M:%S',
    style='{'
)

_stream_handler = logging.StreamHandler(sys.stdout)
_stream_handler.setFormatter(_formatter)

logger = logging.getLogger('autograde')
logger.addHandler(_stream_handler)


def loglevel(x):
    return max(10, 40 - x * 10)


def project_root():
    import autograde
    return Path(autograde.__file__).parent.parent


@contextmanager
def capture_output(out_buffer=sys.stdout, err_buffer=sys.stderr):
    orig_stdout = sys.stdout
    orig_stderr = sys.stderr

    sys.stdout = out_buffer
    sys.stderr = err_buffer

    try:
        yield orig_stdout, orig_stderr

    finally:
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr


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
