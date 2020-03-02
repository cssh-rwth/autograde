
.. _auto-grade:

=========
AUTOGRADE
=========

*autograde* is a tool that lets you run tests on *Jupyter notebooks* in an isolated environment and creates both, human and machine readable reports.


setup
-----

Before installing *autograde*, ensure `docker <https://www.docker.com/>`_ or `podman <https://podman.io/>`_ is installed on your system.

Now, in order to install *autograde*, run :code:`pip install jupyter-autograde`. Alternatively, you can install *autograde* from source by cloning this repository and runing :code:`pip install -e .` within it (if your're developing *autograde*, run :code:`pip install -e .[develop]` instead).

Eventually build the respective container image: :code:`python -m autograde build`

.. NOTE::
    When installing *autograde* via *PyPI*, *docker* support is not yet implemented. If you want to use docker, clone the directory and install the package from source.


usage
-----

apply tests
```````````

*autograde* comes with some example files located in the :code:`demo/` subdirectory that we will use for now to illustrate the workflow. Run:

::

    python -m autograde test demo/test.py demo/notebook.ipynb --target /tmp --context demo/context

What happened? Let's first have a look at the arguments of *autograde*:

* :code:`demo/test.py` contains the a script with test cases we want apply
* :code:`demo/notebook.ipynb` is the a notebook to be tested (here you may also specify a directory to be recursively searched for notebooks)
* The optional flag :code:`--target` tells *autograde* where to store results, :code:`/tmp` in our case and the current working directory by default.
* The optional flag :code:`--context` specifies a directory that is mounted into the sandbox and may arbitrary files or subdirectories. This is useful when the notebook expects some external files to be present.

The output is a compressed archive that is named something like :code:`results_[Lastname1,Lastname2,...]_XXXXXXXX.tar.xz` and which has the following contents:

* :code:`artifacts.tar.xz`: all files that where created by or visible to the tested notebook
* :code:`code.py`: code extracted from the notebook including :code:`stdout`/:code:`stderr` as comments
* :code:`notebook.ipynb`: an identical copy of the tested notebook
* :code:`test_results.csv`: test results
* :code:`test_restults.json`: test results, enriched with participant credentials and a summary
* :code:`report.rst`: human readable report

summarize results
`````````````````

In a typical scenario, test cases are not just applied to one notebook but many at a time. Therefore, *autograde* comes with a summary feature, that aggregates results, shows you a score distribution and has some very basic fraud detection. To create a summary, simply run:

::

    python -m autograde summary path/to/results

Three new files will appear in the result directory:

* :code:`summary.csv`: aggregated results
* :code:`score_distribution.pdf`: a score distribution (without duplicates)
* :code:`similarities.pdf`: similarity heatmap of all notebooks

