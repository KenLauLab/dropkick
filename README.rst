|Dropkick Logo|

Automated cell filtering for single-cell RNA sequencing data.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

|Latest Version|

----

``dropkick`` works primarily with |scanpy|_'s ``AnnData`` objects, and accepts input files in ``.h5ad`` or flat (``.csv``, ``.tsv``) format. It also writes outputs to ``.h5ad`` files when called from the terminal.

Installation via ``pip`` or from source requires a Fortran compiler (``brew install gcc`` for Mac users).

Install from PyPI:
^^^^^^^^^^^^^^^^^^
.. code:: bash

   pip install dropkick

Or compile from source:
^^^^^^^^^^^^^^^^^^^^^^^
.. code:: bash

   git clone https://github.com/KenLauLab/dropkick.git
   cd dropkick
   python setup.py install

----

``dropkick`` can be run as a command line tool or interactively with the
|scanpy|_ single-cell analysis suite.

Usage from command line:
^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: bash

   dropkick run path/to/counts.h5ad

Output will be saved in a new ``.h5ad`` file containing cell probability
scores, labels, and model parameters.

You can also run the ``dropkick.qc`` module from terminal for a quick
look at the total UMI distribution and ambient genes, saved as
``*_qc.png``:

.. code:: bash

   dropkick qc path/to/counts.h5ad

See |dropkick_tutorial.ipynb|_ for an
interactive walkthrough of the ``dropkick`` pipeline and its outputs.

.. |Dropkick Logo| image:: https://github.com/KenLauLab/dropkick/blob/master/data/dropkick_logo.png

.. |Latest Version| image:: https://img.shields.io/pypi/v/dropkick
   :target: https://pypi.python.org/pypi/dropkick/

.. |scanpy| replace:: ``scanpy``
.. _scanpy: https://icb-scanpy.readthedocs-hosted.com/en/stable/

.. |dropkick_tutorial.ipynb| replace:: ``dropkick_tutorial.ipynb``
.. _dropkick_tutorial.ipynb: dropkick_tutorial.ipynb
