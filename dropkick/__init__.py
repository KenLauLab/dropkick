# -*- coding: utf-8 -*-
"""
package initialization

@author: C Heiser
"""
from .api import (
    dropkick,
    recipe_dropkick,
    plot_thresh_obs,
    coef_inventory,
    coef_plot,
    score_plot,
)
from .qc import qc_summary

__all__ = [
    "dropkick",
    "recipe_dropkick",
    "plot_thresh_obs",
    "coef_inventory",
    "coef_plot",
    "score_plot",
    "qc_summary",
]

from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions
