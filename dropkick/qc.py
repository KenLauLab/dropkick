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
    ax.plot(
        adata.var.pct_dropout_by_counts[
            np.argsort(adata.var.pct_dropout_by_counts)
        ].values
    )
    # get range of values for positioning text
    val_max = adata.var.pct_dropout_by_counts.max()
    val_range = val_max - adata.var.pct_dropout_by_counts.min()
    ax.text(
        x=1,
        y=(val_max - (0.03 * val_range)),
        s="Ambient Genes:",
        fontweight="bold",
        fontsize=10,
    )
    if adata.var.ambient.sum() < 14:
        # plot all ambient gene names if they'll fit
        [
            ax.text(
                x=1,
                y=((val_max - (0.10 * val_range)) - ((0.05 * val_range) * x)),
                s=adata.var_names[adata.var.ambient][x],
                fontsize=10,
            )
            for x in range(adata.var.ambient.sum())
        ]
    else:
        # otherwise, plot first ten, with indicator that there's more
        [
            ax.text(
                x=1,
                y=((val_max - (0.10 * val_range)) - ((0.05 * val_range) * x)),
                s=adata.var_names[adata.var.ambient][x],
                fontsize=10,
            )
            for x in range(10)
        ]
        ax.text(
            x=1,
            y=(val_max - (0.60 * val_range)),
            s=". . .",
            fontweight="bold",
            fontsize=10,
        )
    ax.set_xscale("log")
    ax.set_ylabel("Dropout Rate (%)")
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


def qc_summary(adata, save_to=None, verbose=True):
    """
    plot summary of counts distribution and ambient genes

    Parameters:
        adata (anndata.AnnData): object containing unfiltered scRNA-seq data
        save_to (str): path to .png file for saving figure; returns figure by default
        verbose (bool): print updates to console

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
    if save_to is not None:
        if verbose:
            print("Saving QC plot to {}".format(save_to))
        fig.savefig(save_to)
    else:
        return fig
