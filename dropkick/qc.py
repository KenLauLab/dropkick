# -*- coding: utf-8 -*-
"""
Automated ambient gene testing and counts data QC

@author: C Heiser
"""
import argparse
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
from matplotlib import gridspec

from .api import recipe_dropkick


def dropout_plot(adata, show=False, ax=None):
    """
    plot dropout rates for all genes

    Parameters:
        adata (anndata.AnnData): object containing unfiltered scRNA-seq data
        show (bool): show plot or return object
        ax (matplotlib.axes.Axes): axes object for plotting

    Returns:
        plot of gene dropout rates
    """
    if not ax:
        fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(adata.var.dropout_rate[np.argsort(adata.var.dropout_rate)].values)
    ax.text(x=1, y=0.97, s="Ambient Genes:", fontweight="bold", fontsize=10)
    if adata.var.ambient.sum() < 14:
        # plot all ambient gene names if they'll fit
        [
            ax.text(
                x=1,
                y=0.9 - 0.05 * x,
                s=adata.var_names[adata.var.ambient][x],
                fontsize=10,
            )
            for x in range(adata.var.ambient.sum())
        ]
    else:
        # otherwise, plot first ten, with indicator that there's more
        [
            ax.text(
                x=1, y=0.9 - 0.05 * x, s=a.var_names[adata.var.ambient][x], fontsize=10
            )
            for x in range(10)
        ]
        ax.text(x=1, y=0.4, s=". . .", fontweight="bold", fontsize=10)
    ax.set_xscale("log")
    ax.set_ylabel("Dropout Rate")
    ax.set_xlabel("Ranked Genes")
    if not show:
        return ax


def counts_plot(adata, show=False, ax=None):
    """
    plot total counts for all barcodes

    Parameters:
        adata (anndata.AnnData): object containing unfiltered scRNA-seq data
        show (bool): show plot or return object
        ax (matplotlib.axes.Axes): axes object for plotting

    Returns:
        log-log plot of total counts and total genes per barcode,
        with percent ambient and mitochondrial counts on secondary axis
    """
    if not ax:
        fig, ax = plt.subplots(figsize=(9, 5))
    # plot total counts left y-axis
    ax.set_xlabel("Ranked Barcodes")
    ax.set_ylabel("Total Counts/Genes")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.plot(
        adata.obs.total_counts[np.argsort(adata.obs.total_counts)[::-1]].values,
        linewidth=3.5,
        color="g",
        alpha=0.8,
        label="Counts",
    )
    ax.scatter(
        list(range(adata.n_obs)),
        adata.obs.n_genes_by_counts[np.argsort(adata.obs.total_counts)[::-1]].values,
        s=18,
        color="r",
        alpha=0.3,
        edgecolors="none",
        label="Genes",
    )
    ax.legend(loc="lower left")

    # plot percent ambient counts on right y-axis
    ax2 = ax.twinx()
    ax2.set_ylabel("% Counts")
    ax2.scatter(
        list(range(adata.n_obs)),
        adata.obs.pct_counts_ambient[np.argsort(adata.obs.total_counts)[::-1]].values,
        s=18,
        alpha=0.3,
        edgecolors="none",
        label="% Ambient",
    )
    ax2.scatter(
        list(range(adata.n_obs)),
        adata.obs.pct_counts_mito[np.argsort(adata.obs.total_counts)[::-1]].values,
        s=18,
        alpha=0.3,
        edgecolors="none",
        label="% Mito",
    )
    ax2.set_xscale("log")
    ax2.tick_params(axis="y")
    ax2.legend(loc="upper right")

    ax.set_zorder(ax2.get_zorder() + 1)  # put ax in front of ax2
    ax.patch.set_visible(False)  # hide the 'canvas'
    if not show:
        return ax


def summary_plot(adata, show=True):
    """
    plot summary of counts distribution and ambient genes

    Parameters:
        adata (anndata.AnnData): object containing unfiltered scRNA-seq data
        show (bool): show plot or return object

    Returns:
        counts_plot(), sc.pl.highest_expr_genes(), and dropout_plot() in single figure
    """
    fig = plt.figure(figsize=(10, 10))
    # arrange axes as subplots
    gs = gridspec.GridSpec(2, 2, figure=fig)
    ax1 = plt.subplot(gs[0, :])
    ax2 = plt.subplot(gs[1, 0])
    ax3 = plt.subplot(gs[1, 1])
    # add plots to axes
    counts_plot(adata, ax=ax1, show=False)
    sc.pl.highest_expr_genes(adata, ax=ax2, show=False, n_top=20)
    dropout_plot(adata, ax=ax3, show=False)
    fig.tight_layout()
    # return
    if not show:
        return fig
