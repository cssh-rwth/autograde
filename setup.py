#!/usr/bin/env python3
import re
from pathlib import Path

from setuptools import setup, find_packages

BASEDIR = Path(__file__).parent

with open(BASEDIR.joinpath('autograde', '__init__.py'), 'r') as f:
    VERSION = re.search(r"__version__ = '(.*?)'", f.read()).group(1)

with open(BASEDIR.joinpath('README.rst'), 'r') as f:
    LONG_DESCRIPTION = f.read()

PACKAGES = find_packages()

with BASEDIR.joinpath('requirements.txt').open(mode='rt') as f:
    INSTALL_REQUIRES = f.read().split('\n')

EXTRAS_REQUIRE = {'develop': ['nose', 'setuptools', 'wheel', 'twine', 'flake8']}

ENTRY_POINTS = {
    'console_scripts': ['autograde=autograde.__main__:cli'],
}

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
    python_requires='>=3.8',
    entry_points=ENTRY_POINTS,
    test_suite='nose.collector',
    version=VERSION,
)
