# -*- coding: utf-8 -*-
"""
package initialization

@author: C Heiser
"""
from .api import recipe_dropkick, dropkick, plot_thresh_obs, coef_inventory, coef_plot

__all__ = [
    "dropkick",
    "recipe_dropkick",
    "plot_thresh_obs",
    "coef_inventory",
    "coef_plot",
]

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
