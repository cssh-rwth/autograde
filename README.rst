
.. _auto-grade:

==========
AUTO GRADE
==========

test multiple notebooks using docker

::

    podman container run auto_grade test.py --target=./submissions --provide=./data


usage
-----

Build docker image::

    python -m build


Execute test on notebook::

    python -m autograde exec test.py notebook.ipynb                                                                                            ~/workspace/autograde
