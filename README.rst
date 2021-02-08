.. _auto-grade:

=========
autograde
=========

.. image:: https://github.com/cssh-rwth/autograde/workflows/test%20autograde/badge.svg
   :alt: autograde test
   :target: https://github.com/cssh-rwth/autograde/actions

.. image:: https://img.shields.io/pypi/v/jupyter-autograde?color=blue&label=jupyter-autograde
   :alt: autograde on PyPI
   :target: https://pypi.org/project/jupyter-autograde

*autograde* is a tool for testing *Jupyter* notebooks. Its features include execution of notebooks (optionally isolated via docker/podman) with consecutive unit testing of the final notebook state. On top of that, an audit mode allows for refining results (e.g. grading plots by hand). Eventually, *autograde* can summarize these results in human and machine readable formats.

setup
-----

Before installing *autograde* and in case you want to use it with a container backend, ensure `docker <https://www.docker.com/>`_ **or** `podman <https://podman.io/>`_ is available on your system.
We recommend podman as it runs rootless.

Now, in order to install *autograde*, run :code:`pip install jupyter-autograde`.
Alternatively, you can install *autograde* from source by cloning this repository and runing :code:`poetry install` within it.
This requires `poetry <https://python-poetry.org/docs/>`_ to be installed on your system!

Eventually, build the respective container image: :code:`python -m autograde build`.
**Note:** in order to build a container image, *autograde* must not be installed via *PyPI* but from source code!

usage
-----

testing
```````

*autograde* comes with some example files located in the :code:`demo/` subdirectory that we will use for now to illustrate the workflow. Run:

::

    python -m autograde test demo/test.py demo/notebook.ipynb --target /tmp --context demo/context

What happened? Let's first have a look at the arguments of *autograde*:

* :code:`demo/test.py` a script with test cases we want apply
* :code:`demo/notebook.ipynb` is the a notebook to be tested (here you may also specify a directory to be recursively searched for notebooks)
* The optional flag :code:`--target` tells *autograde* where to store results, :code:`/tmp` in our case, and the current working directory by default.
* The optional flag :code:`--context` specifies a directory that is mounted into the sandbox and may contain arbitrary files or subdirectories.
  This is useful when the notebook expects some external files to be present such as data sets.

The output is a compressed archive that is named something like :code:`results_[Lastname1,Lastname2,...]_XXXXXXXX.zip` and which has the following contents:

* :code:`artifacts/`: directory with all files that where created or modified by the tested notebook as well as rendered matplotlib plots.
* :code:`code.py`: code extracted from the notebook including :code:`stdout`/:code:`stderr` as comments
* :code:`notebook.ipynb`: an identical copy of the tested notebook
* :code:`test_restults.json`: test results


reports
```````

The :code:`report` sub command creates human readable HTML reports from test results:

::

    python -m autograde report path/to/result(s)

The respective report is added to the results archive inplace.


patching
````````

Results from multiple test runs can be merged via the :code:`patch` sub command:

::

    python -m autograde patch path/to/result(s) /path/to/patch/result(s)


summarize results
`````````````````

In a typical scenario, test cases are not just applied to one notebook but many at a time.
Therefore, *autograde* comes with a summary feature, that aggregates results, shows you a score distribution and has some very basic fraud detection.
To create a summary, simply run:

::

    python -m autograde summary path/to/results

Two new files will appear in the result directory:

* :code:`summary.csv`: aggregated results
* :code:`summary.html`: human readable summary report


help
````

To get an overview of all available commands and their usage, run

::

    python -m autograde [sub command] --help

