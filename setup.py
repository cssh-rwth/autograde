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
EXTRAS_REQUIRE = {'develop': ['nose', 'setuptools', 'wheel', 'twine',]}

ENTRY_POINTS = {}

setup(
    name='jupyter-autograde',
    author='0b11001111',
    maintainer='Chair for Computational Social Sciences and Humanities at RWTH Aachen University',
    maintainer_email='admin@cssh.rwth-aachen.de',
    description='automatic grading of jupyter notebooks',
    long_description=LONG_DESCRIPTION,
    license="MIT",
    url='https://github.com/cssh-rwth/autograde',
    keywords=['jupyter', 'teaching', 'unit test'],
    classifiers=[
        'Natural Language :: English',
        'Programming Language :: Python',
        'Operating System :: OS Independent'
    ],
    packages=PACKAGES,
    include_package_data=True,
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    python_requires='>=3.7',
    entry_points=ENTRY_POINTS,
    test_suite='nose.collector',
    version=VERSION,
)

