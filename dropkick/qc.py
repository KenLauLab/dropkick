# -*- coding: utf-8 -*-
"""
Automated ambient gene testing and counts data QC

@author: C Heiser
"""
import argparse
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
from matplotlib import cm, gridspec

from .api import recipe_dropkick


def dropout_plot(adata, show=False, ax=None):
    """
    plot dropout rates for all genes

    Parameters:
        adata (anndata.AnnData): object containing unfiltered scRNA-seq data
        show (bool): show plot or return object
        ax (matplotlib.axes.Axes): axes object for plotting. if None, create new.

    Returns:
        plot of gene dropout rates
    """
    if not ax:
        _, ax = plt.subplots(figsize=(4, 4))
    ax.plot(
        adata.var.pct_dropout_by_counts[
            np.argsort(adata.var.pct_dropout_by_counts)
        ].values,
        color="k",
        linewidth=2,
    )
    # get range of values for positioning text
    val_max = adata.var.pct_dropout_by_counts.max()
    val_range = val_max - adata.var.pct_dropout_by_counts.min()
    ax.text(
        x=1,
        y=(val_max - (0.03 * val_range)),
        s="Ambient Genes:",
        fontweight="bold",
        fontsize=12,
    )
    if adata.var.ambient.sum() < 14:
        # plot all ambient gene names if they'll fit
        [
            ax.text(
                x=1,
                y=((val_max - (0.10 * val_range)) - ((0.05 * val_range) * x)),
                s=adata.var_names[adata.var.ambient][x],
                fontsize=12,
                fontstyle="italic",
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
                fontsize=12,
                fontstyle="italic",
            )
            for x in range(10)
        ]
        ax.text(
            x=1,
            y=(val_max - (0.60 * val_range)),
            s=". . .",
            fontweight="bold",
            fontsize=12,
        )
    ax.set_xscale("log")
    ax.set_ylabel("Dropout Rate (%)", fontsize=12)
    ax.set_xlabel("Ranked Genes", fontsize=12)
    ax.tick_params(axis="both", which="major", labelsize=12)
    if not show:
        return ax


def counts_plot(adata, show=False, genes=True, ambient=True, mito=True, ax=None):
    """
    plot total counts for all barcodes

    Parameters:
        adata (anndata.AnnData): object containing unfiltered scRNA-seq data
        show (bool): show plot or return object
        genes (bool): show n_genes detected as points
        ambient (bool): show pct_counts_ambient as points
        mito (bool): show pct_counts_mito as points
        ax (matplotlib.axes.Axes): axes object for plotting. if None, create new.

    Returns:
        log-log plot of total counts and total genes per ranked barcode,
        with percent ambient and mitochondrial counts on secondary axis if desired
    """
    cmap = cm.get_cmap("coolwarm")
    if not ax:
        _, ax = plt.subplots(figsize=(7, 5))
    # plot total counts left y-axis
    ax.set_xlabel("Ranked Barcodes", fontsize=12)
    if genes:
        ax.set_ylabel("Total Counts/Genes", fontsize=12)
    else:
        ax.set_ylabel("Total Counts", fontsize=12)
    ax.set_xscale("log")
    ax.set_yscale("log")
    counts_ln = ax.plot(
        adata.obs.total_counts[np.argsort(adata.obs.total_counts)[::-1]].values,
        linewidth=2,
        color="k",
        alpha=0.8,
        label="Counts",
    )
    # start list of legend entries
    leg_entries = counts_ln
    if genes:
        genes_pts = ax.scatter(
            list(range(adata.n_obs)),
            adata.obs.n_genes_by_counts[
                np.argsort(adata.obs.total_counts)[::-1]
            ].values,
            s=18,
            color="g",
            alpha=0.3,
            edgecolors="none",
            label="Genes",
        )
        # append list of legend entries
        leg_entries = leg_entries + [genes_pts]
    ax.tick_params(axis="both", which="major", labelsize=12)

    if ambient or mito:
        # plot percent ambient counts on right y-axis
        ax2 = ax.twinx()
        ax2.set_ylabel("% Counts", fontsize=12)
        if ambient:
            ambient_pts = ax2.scatter(
                list(range(adata.n_obs)),
                adata.obs.pct_counts_ambient[
                    np.argsort(adata.obs.total_counts)[::-1]
                ].values,
                s=18,
                color=cmap(0.0),
                alpha=0.3,
                edgecolors="none",
                label="% Ambient",
            )
            # append list of legend entries
            leg_entries = leg_entries + [ambient_pts]
        if mito:
            mito_pts = ax2.scatter(
                list(range(adata.n_obs)),
                adata.obs.pct_counts_mito[
                    np.argsort(adata.obs.total_counts)[::-1]
                ].values,
                s=18,
                color=cmap(1.0),
                alpha=0.3,
                edgecolors="none",
                label="% Mito",
            )
            # append list of legend entries
            leg_entries = leg_entries + [mito_pts]
        ax2.set_xscale("log")
        ax2.tick_params(axis="y", which="major", labelsize=12)
        ax.set_zorder(ax2.get_zorder() + 1)  # put ax in front of ax2

    # add legend from entries
    labs = [l.get_label() for l in leg_entries]
    ax.legend(leg_entries, labs, loc="upper right", fontsize=12)

    ax.patch.set_visible(False)  # hide the 'canvas'
    if not show:
        return ax


def qc_summary(
    adata, mito=True, ambient=True, genes=True, fig=None, save_to=None, verbose=True
):
    """
    plot summary of counts distribution and ambient genes

    Parameters:
        adata (anndata.AnnData): object containing unfiltered scRNA-seq data
        mito (bool): show pct_counts_mito as points
        fig (matplotlib.figure): figure object for plotting. if None, create new.
        save_to (str): path to .png file for saving figure; returns figure by default
        verbose (bool): print updates to console

    Returns:
        fig (matplotlib.figure): counts_plot(), sc.pl.highest_expr_genes(), and
            dropout_plot() in single figure
    """
    if not fig:
        fig = plt.figure(figsize=(14, 7))
    # arrange axes as subplots
    gs = gridspec.GridSpec(nrows=3, ncols=3, figure=fig)
    ax1 = plt.subplot(gs[0:4, 0:2])
    ax2 = plt.subplot(gs[0:2, 2])
    # add plots to axes
    counts_plot(adata, ax=ax1, show=False, genes=genes, ambient=ambient, mito=mito)
    dropout_plot(adata, ax=ax2, show=False)
    gs.tight_layout(figure=fig, w_pad=1.8)
    # return
    if save_to is not None:
        if verbose:
            print("Saving QC plot to {}".format(save_to))
        fig.savefig(save_to, dpi=200)
    else:
        return fig
