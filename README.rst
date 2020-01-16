
.. _auto-grade:

==========
AUTO GRADE
==========

*autograde* is a tool that lets you run test cases on *Jupyter notebooks* in an isolated environment and creates both, human and machine readable reports.


setup
-----

Before installing *autograde*, ensure `docker <https://www.docker.com/>`_ or `podman <https://podman.io/>`_ is installed on your system. We recommend the latter one.

Now, in order to install *autograde*, clone this repository and run :code:`pip install -e .` within it.

Eventually build the respective docker image: :code:`python -m autograde build`


usage
-----

Before we run our first test, we build the respective container image, which is done by running :code:`python -m autograde build` which may take some time depending on your internet connection.

*autograde* comes with some example files located in the :code:`demo/` subdirectory that we will use for now to illustrate the workflow. Run:

::

    python -m autograde test demo/test.py demo/notebook.ipynb --target /tmp --context demo/context

What happened? Let's first have a look at the arguments of *autograde*:

* :code:`demo/test.py` contains the test cases we want to apply
* :code:`demo/notebook.ipynb` is the respective notebook we want to test
* The optional flag :code:`--target` tells *autograde* where to store results, :code:`/tmp` in our case and the current working directory by default.
* The optional flag :code:`--context` specifies a directory that is mounted into the sandbox and may arbitrary files or subdirectories. This is useful when the notebook expects some external files to be present.

The output is a compressed archive that is named something like :code:`results_XXXXXXXX.tar.xz` and which has the following contents:

* :code:`artifacts.tar.xz`: all files that where created by or visible to the tested notebook
* :code:`code.py`: code extracted from the notebook including :code:`stdout`/:code:`stderr` as comments
* :code:`notebook.ipynb`: an identical copy of the tested notebook
* :code:`test_results.csv`: test results
* :code:`test_restults.json`: test results, enriched with participant credentials and a summary
* :code:`report.rst`: human readable report
