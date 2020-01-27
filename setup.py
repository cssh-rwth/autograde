#!/usr/bin/env python3
# Standard library modules.
import os

# Third party modules.
from setuptools import setup, find_packages

# Local modules.

# Globals and constants variables.
BASEDIR = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(BASEDIR, 'README.rst'), 'r') as fp:
    LONG_DESCRIPTION = fp.read()

PACKAGES = find_packages()

INSTALL_REQUIRES = ['ipykernel', 'jupyter', 'matplotlib', 'seaborn', 'numpy', 'pandas', 'tabulate']
EXTRAS_REQUIRE = {'develop': ['nose', 'sphinx']}

ENTRY_POINTS = {}

setup(
    name='autograde',
    version='0',
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

