#!/usr/bin/env python3
# Standard library modules.
import re
from pathlib import Path

# Third party modules.
from setuptools import setup, find_packages

# Local modules.

# Globals and constants variables.
BASEDIR = Path(__file__).parent

with open(BASEDIR.joinpath('autograde', '__init__.py'), 'r') as f:
    VERSION = re.search(r"__version__ = '(.*?)'", f.read()).group(1)

with open(BASEDIR.joinpath('README.rst'), 'r') as f:
    LONG_DESCRIPTION = f.read()

PACKAGES = find_packages()

INSTALL_REQUIRES = ['ipykernel', 'jupyter', 'matplotlib', 'seaborn', 'numpy', 'pandas', 'tabulate']
EXTRAS_REQUIRE = {'develop': ['nose', 'sphinx']}

ENTRY_POINTS = {}

setup(
    name='autograde',
    version=VERSION,
    url='https://git.rwth-aachen.de/cssh/autograde',
    description='util',
    author='Lukas Ochse',
    author_email='lukas.ochse@rwth-aachen.de',
    license="GPL v3",
    classifiers=[
        'Natural Language :: English',
        'Programming Language :: Python',
        'Operating System :: OS Independent'
    ],
    packages=PACKAGES,
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    entry_points=ENTRY_POINTS,
    test_suite='nose.collector',
)

