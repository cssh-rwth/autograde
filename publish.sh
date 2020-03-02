#!/usr/bin/env bash

# ensure no older builds are present
rm -rf build dist

python setup.py sdist bdist_wheel
python -m twine upload --repository-url https://upload.pypi.org/legacy/ dist/* -u cssh-rwth

# cleanup
rm -rf build dist

