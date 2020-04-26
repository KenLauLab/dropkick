![alt text](data/dropkick_logo.png)

### Automated cell filtering for single-cell RNA sequencing data.

[![Latest Version][pypi-image]][pypi-url]

---
`dropkick` works primarily with [**Scanpy**](https://icb-scanpy.readthedocs-hosted.com/en/stable/)'s `AnnData` objects, and accepts input files in `.h5ad` or flat (`.csv`, `.tsv`) format. It also writes outputs to `.h5ad` files when called from the terminal.

Installation via `pip` or from source requires a Fortran compiler (`brew install gcc` for Mac users).

#### Install from PyPI:
```bash
pip install dropkick
```

#### Or compile from source:
```bash
git clone https://github.com/KenLauLab/dropkick.git
cd dropkick
python setup.py install
```

---
`dropkick` can be run as a command line tool or interactively with the [`scanpy`](https://icb-scanpy.readthedocs-hosted.com/en/stable/) single-cell analysis suite.

#### Usage from command line:
```bash
dropkick run path/to/counts.h5ad
```

Output will be saved in a new `.h5ad` file containing __dropkick__ scores, labels, and model parameters.

You can also run the `dropkick.qc` module from terminal for a quick look at the total UMI distribution and ambient genes, saved as `*_qc.png`:
```bash
dropkick qc path/to/counts.h5ad
```

See [`dropkick_tutorial.ipynb`](dropkick_tutorial.ipynb) for an interactive walkthrough of the `dropkick` pipeline and its outputs.

[pypi-image]: https://img.shields.io/pypi/v/dropkick
[pypi-url]: https://pypi.python.org/pypi/dropkick/
