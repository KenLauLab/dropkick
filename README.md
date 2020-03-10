# dropkick
Automated cell filtering for single-cell RNA sequencing data.

`dropkick` works primarily with [**Scanpy**](https://icb-scanpy.readthedocs-hosted.com/en/stable/)'s `AnnData` objects, and accepts input files in `.h5ad` or flat (`.csv`, `.tsv`) format. It also writes outputs to `.h5ad` files when called from the command line.

#### Install from PyPI:
```bash
pip install -i https://test.pypi.org/simple/ dropkick  # testing package index
```

#### Usage from command line:
```bash
python -m dropkick path/to/counts.h5ad
```

Output will be saved in a new `.h5ad` file containing __dropkick__ scores, labels, and model parameters.

See [`dropkick_tutorial.ipynb`](dropkick_tutorial.ipynb) for an interactive walkthrough of the `dropkick` pipeline and its outputs.
