# autograde

[![autograde test](https://github.com/cssh-rwth/autograde/workflows/test%20autograde/badge.svg)](https://github.com/cssh-rwth/autograde/actions)
[![autograde on PyPI](https://img.shields.io/pypi/v/jupyter-autograde?color=blue&label=jupyter-autograde)](https://pypi.org/project/jupyter-autograde)

*autograde* is a toolbox for testing *Jupyter* notebooks. Its features include execution of notebooks (optionally
isolated via docker/podman) with consecutive unit testing of the final notebook state. An audit mode allows for refining
results (e.g. grading plots by hand). Eventually, *autograde* can summarize these results in human and machine-readable
formats.

## setup

Install _autograde_ from _PyPI_ using _pip_ like this

```shell
pip install jupyter-autograde
```

Alternatively, _autograde_ can be set up from source code by cloning this repository and installing it
using [poetry](https://python-poetry.org/docs/)

```shell
git clone https://github.com/cssh-rwth/autograde.git && cd autograde
poetry install
```

If you intend to use autograde in a sandboxed environment
ensure [rootless docker](docs.docker.com/engine/security/rootless/) or [podman](podman.io/getting-started/installation)
are available on your system. So far, only rootless mode is supported!

## usage

Once installed, *autograde* can be invoked via the`autograde` command. If you are using a virtual environment (which
poetry does implicitly) you may have to activate it first. Alternative methods:

- `path/to/python -m autograde` runs *autograde* with a specific python binary, e.g. the one of your virtual
  environment.
- `poetry run autograde` if you've installed *autograde* from source

### testing

*autograde* comes with some example files located in the `demo/`
subdirectory that we will use for now to illustrate the workflow. Run

```shell
python -m autograde test demo/test.py demo/notebook.ipynb --target /tmp --context demo/context
```

What happened? Let's first have a look at the arguments of *autograde*:

- `demo/test.py` a script with test cases we want to apply
- `demo/notebook.ipynb` is the a notebook to be tested (here you may also specify a directory to be recursively searched
  for notebooks)
- The optional flag `--target` tells *autograde* where to store results, `/tmp` in our case, and the current working
  directory by default.
- The optional flag `--context` specifies a directory that is mounted into the sandbox and may contain arbitrary files
  or subdirectories. This is useful when the notebook expects some external files to be present such as data sets.

The output is a compressed archive that is named something like
`results_[Lastname1,Lastname2,...]_XXXXXXXX.zip` and which has the following contents:

- `artifacts/`: directory with all files that where created or modified by the tested notebook as well as rendered
  matplotlib plots.
- `code.py`: code extracted from the notebook including
  `stdout`/`stderr` as comments
- `notebook.ipynb`: an identical copy of the tested notebook
- `restults.json`: test results

### reports

The `report` sub command creates human readable HTML reports from test results:

```shell
python -m autograde report path/to/result(s)
```

The report is added to the results archive inplace.

### patching

Results from multiple test runs can be merged via the `patch` sub command:

```shell
python -m autograde patch path/to/result(s) /path/to/patch/result(s)
```

### summarize results

In a typical scenario, test cases are not just applied to one notebook but many at a time. Therefore, *autograde* comes
with a summary feature, that aggregates results, shows you a score distribution and has some very basic fraud detection.
To create a summary, simply run:

```shell
python -m autograde summary path/to/results
```

Two new files will appear in the result directory:

- `summary.csv`: aggregated results
- `summary.html`: human readable summary report

### help

To get an overview of all available commands and their usage, run

```shell
python -m autograde [sub command] --help
```
