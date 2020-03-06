# -*- coding: utf-8 -*-
"""
Automated testing of cell filtering labels

@author: C Heiser

usage: qc_test.py [-h] -c COUNTS -l LABELS [LABELS ...]
                  [-m METRICS [METRICS ...]] [--mito-names MITO_NAMES]
                  [--output-dir [OUTPUT_DIR]] [--cnmf]

optional arguments:
  -h, --help            show this help message and exit
  -c COUNTS, --counts COUNTS
                        Input (cell x gene) counts matrix as .h5ad file
  -l LABELS [LABELS ...], --labels LABELS [LABELS ...]
                        Labels defining cell sets to compare
  -m METRICS [METRICS ...], --metrics METRICS [METRICS ...]
                        Heuristics for comparing. Several can be specified
                        with e.g. '--metrics arcsinh_total_counts
                        arcsinh_n_genes_by_counts pct_counts_mito'
  --mito-names MITO_NAMES
                        Substring or regex defining mitochondrial genes
  --output-dir [OUTPUT_DIR]
                        Output directory. All output will be placed in
                        [output-dir]/[name]...
  --cnmf                Are cNMF usages available for testing?
"""
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import seaborn as sns
from matplotlib import gridspec
from scipy.stats import mannwhitneyu
from dropkick import recipe_dropkick, coef_inventory


def set_diff(adata, labels, metrics=None):
    """
    return number of cells different between two labels

    Parameters:
        adata (anndata.AnnData): object with cell labels in .obs
        labels (list of str): two labels (columns of .obs) to compare the cell sets
            1 = real cell, 0 = empty or dead

    Returns:
        prints results
    """
    if len(labels) != 2:
        raise ValueError("Please provide exactly two cell labels.")
    unique_0 = len(
        set(adata.obs_names[adata.obs[labels[0]] == 1]).difference(
            set(adata.obs_names[adata.obs[labels[1]] == 1])
        )
    )
    unique_1 = len(
        set(adata.obs_names[adata.obs[labels[1]] == 1]).difference(
            set(adata.obs_names[adata.obs[labels[0]] == 1])
        )
    )
    print(
        "{} cells in {} - {} unique".format(
            adata.obs[labels[0]].sum(), labels[0], unique_0
        )
    )
    if metrics is not None:
        for m in metrics:
            print(
                "\t{}: {:0.3}".format(
                    m, round(adata.obs.loc[adata.obs[labels[0]] == 1, m].mean(), 3)
                )
            )
    print(
        "{} cells in {} - {} unique".format(
            adata.obs[labels[1]].sum(), labels[1], unique_1
        )
    )
    if metrics is not None:
        for m in metrics:
            print(
                "\t{}: {:0.3}".format(
                    m, round(adata.obs.loc[adata.obs[labels[1]] == 1, m].mean(), 3)
                )
            )


def plot_set_obs(
    adata,
    labels,
    metrics=["arcsinh_total_counts", "arcsinh_n_genes_by_counts", "pct_counts_mito"],
    bins=40,
    show=True,
):
    """
    plot distribution of metrics in adata.obs for different labeled cell populations

    Parameters:
        adata (anndata.AnnData): object with cell labels and metrics in .obs
        labels (list of str): two labels (columns of .obs) to compare the cell sets
        metrics (list of str): .obs columns to plot distributions of
        bins (int): number of bins for histogram
        show (bool): show plot or return object

    Returns:
        plot of distributions of obs_cols split by cell labels
    """
    fig, axes = plt.subplots(ncols=len(metrics), nrows=1, figsize=(len(metrics) * 4, 4))
    axes[0].set_ylabel("cells")
    for i in range(len(metrics)):
        axes[i].hist(
            adata.obs.loc[adata.obs_names[adata.obs[labels[0]] == 1], metrics[i]],
            alpha=0.5,
            label=labels[0],
            bins=bins,
        )
        axes[i].hist(
            adata.obs.loc[adata.obs_names[adata.obs[labels[1]] == 1], metrics[i]],
            alpha=0.5,
            label=labels[1],
            bins=bins,
        )
        axes[i].hist(
            adata.obs.loc[
                set(adata.obs_names[adata.obs[labels[0]] == 1]).difference(
                    set(adata.obs_names[adata.obs[labels[1]] == 1])
                ),
                metrics[i],
            ],
            alpha=0.5,
            label="{} unique".format(labels[0]),
            bins=bins,
        )
        axes[i].hist(
            adata.obs.loc[
                set(adata.obs_names[adata.obs[labels[1]] == 1]).difference(
                    set(adata.obs_names[adata.obs[labels[0]] == 1])
                ),
                metrics[i],
            ],
            alpha=0.5,
            label="{} unique".format(labels[1]),
            bins=bins,
        )
        axes[i].set_title(metrics[i])
    axes[i].legend()
    fig.tight_layout()
    if show:
        plt.show()
    else:
        return fig


def cnmf_usage_test(adata, labels):
    """
    return significantly different GEP usages between two labels by Mann-Whitney test

    Parameters:
        adata (anndata.AnnData): object with cell labels in .obs[obs_col]
        labels (list of str): two labels (columns of .obs) to compare the cell sets
            1 = real cell, 0 = empty or dead

    Returns:
        prints results
    """
    # generate obs column with comparison of two labels for visualization
    adata.obs["compare"] = "same"
    adata.obs.loc[
        (adata.obs[labels[0]] == 1) & (adata.obs[labels[1]] == 0), "compare"
    ] = labels[0]
    adata.obs.loc[
        (adata.obs[labels[1]] == 1) & (adata.obs[labels[0]] == 0), "compare"
    ] = labels[1]
    # determine Mann-Whitney p-values
    sig = []
    p = []
    insig = []
    for gep in list(adata.obs.columns[adata.obs.columns.str.contains("usage_")]):
        u = mannwhitneyu(
            adata.obs.loc[adata.obs["compare"] == labels[0], gep],
            adata.obs.loc[adata.obs["compare"] == labels[1], gep],
            alternative="two-sided",
        )
        if u.pvalue <= 0.05:
            print("Significant result for {}: p-value = {:0.3e}".format(gep, u.pvalue))
            sig.append(gep)
            p.append(u.pvalue)
        else:
            insig.append(gep)
    # plot violins of significant results
    fig, axes = plt.subplots(ncols=len(sig), nrows=1, figsize=(len(sig) * 4, 4))
    for i in range(len(sig)):
        sns.violinplot(data=adata.obs, x="compare", y=sig[i], ax=axes[i])
        axes[i].annotate(
            "p={:0.3e}".format(p[i]), xy=(0.2, 0.92 * axes[i].get_ylim()[1])
        )
        axes[i].set_xlabel(None)
    fig.tight_layout()

    return fig, sig, insig


def rank_genes(
    adata,
    attr="varm",
    keys="cnmf_spectra",
    indices=None,
    labels=None,
    color="black",
    n_points=20,
    log=False,
    show=None,
    figsize=(7, 7),
):
    """
    Plot rankings. [Adapted from scanpy.plotting._anndata.ranking]
    See, for example, how this is used in pl.pca_ranking.

    Parameters:
        adata : AnnData
            The data.
        attr : {'var', 'obs', 'uns', 'varm', 'obsm'}
            The attribute of AnnData that contains the score.
        keys : str or list of str
            The scores to look up an array from the attribute of adata.
        indices : list of int
            The column indices of keys for which to plot (e.g. [0,1,2] for first three keys)
    Returns:
        matplotlib gridspec with access to the axes.
    """
    # default to all usages
    if indices is None:
        indices = range(getattr(adata, attr)[keys].shape[1])
    # get scores for each usage
    if isinstance(keys, str) and indices is not None:
        scores = np.array(getattr(adata, attr)[keys])[:, indices]
        keys = ["{}{}".format(keys[:-1], i + 1) for i in indices]
    n_panels = len(keys) if isinstance(keys, list) else 1
    if n_panels == 1:
        scores, keys = scores[:, None], [keys]
    if log:
        scores = np.log(scores)
    if labels is None:
        labels = (
            adata.var_names
            if attr in {"var", "varm"}
            else np.arange(scores.shape[0]).astype(str)
        )
    if isinstance(labels, str):
        labels = [labels + str(i + 1) for i in range(scores.shape[0])]
    if n_panels <= 5:
        n_rows, n_cols = 1, n_panels
    else:
        n_rows, n_cols = int(n_panels / 4 + 0.5), 4
    fig = plt.figure(figsize=(n_cols * figsize[0], n_rows * figsize[1]))
    left, bottom = 0.2 / n_cols, 0.13 / n_rows
    gs = gridspec.GridSpec(
        nrows=n_rows,
        ncols=n_cols,
        wspace=0.2,
        left=left,
        bottom=bottom,
        right=1 - (n_cols - 1) * left - 0.01 / n_cols,
        top=1 - (n_rows - 1) * bottom - 0.1 / n_rows,
    )
    for iscore, score in enumerate(scores.T):
        plt.subplot(gs[iscore])
        indices = np.argsort(score)[::-1][: n_points + 1]
        for ig, g in enumerate(indices):
            plt.text(
                x=ig,
                y=score[g],
                s=labels[g],
                color=color,
                rotation="vertical",
                verticalalignment="bottom",
                horizontalalignment="center",
                fontsize="large",
            )
        plt.title(keys[iscore].replace("_", " "), fontsize="x-large")
        plt.xlim(-0.9, ig + 0.9)
        score_min, score_max = np.min(score[indices]), np.max(score[indices])
        plt.ylim(
            (0.95 if score_min > 0 else 1.05) * score_min,
            (1.05 if score_max > 0 else 0.95) * score_max,
        )
        plt.tick_params(labelsize="x-large")
        gs.tight_layout(fig)
    if show == False:
        return gs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-c",
        "--counts",
        type=str,
        help="Input (cell x gene) counts matrix as .h5ad file",
        required=True,
    )
    parser.add_argument(
        "-l",
        "--labels",
        type=str,
        help="Labels defining cell sets to compare",
        nargs="+",
        default=["manual_label", "dropkick_label"],
    )
    parser.add_argument(
        "-m",
        "--metrics",
        type=str,
        help="Heuristics for comparing. Several can be specified with e.g. '--metrics arcsinh_total_counts arcsinh_n_genes_by_counts pct_counts_mito'",
        nargs="+",
        default=[
            "arcsinh_total_counts",
            "arcsinh_n_genes_by_counts",
            "pct_counts_ambient",
            "pct_counts_mito",
        ],
    )
    parser.add_argument(
        "--mito-names",
        type=str,
        help="Substring or regex defining mitochondrial genes",
        default="^mt-|^MT-",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory. All output will be placed in [output-dir]/[name]...",
        nargs="?",
        default=".",
    )
    parser.add_argument(
        "--cnmf", help="Are cNMF usages available for testing?", action="store_true"
    )

    args = parser.parse_args()
    # get name for saving outputs
    name = os.path.splitext(os.path.basename(args.counts))[0]
    # read in AnnData object
    print("\nReading in counts data from {}".format(args.counts))
    adata = sc.read(args.counts)
    # print regression model coefficient inventory to console
    coef_inventory(adata, n=10)
    # preprocess data and calculate metrics
    recipe_dropkick(adata, mito_names=args.mito_names, verbose=False)
    # print set differences to console
    set_diff(adata, labels=args.labels, metrics=args.metrics)
    # generate plot of chosen metrics' distribution in two cell label populations
    print(
        "\nSaving distribution plots to {}/{}_metrics.png".format(args.output_dir, name)
    )
    plot_set_obs(adata, labels=args.labels, metrics=args.metrics, bins=40, show=False)
    plt.savefig("{}/{}_metrics.png".format(args.output_dir, name))
    if args.cnmf:
        # print significant GEP usages to console
        fig, sig, insig = cnmf_usage_test(adata, labels=args.labels)
        # generate plot of significant GEP distributions
        print(
            "Saving significant NMF GEP distribution plots to {}/{}_sigGEPs.png".format(
                args.output_dir, name
            )
        )
        fig.savefig("{}/{}_sigGEPs.png".format(args.output_dir, name))
        # generate plot of significant GEP loadings
        print(
            "Saving significant NMF GEP loadings to {}/{}_sigspectra.png".format(
                args.output_dir, name
            )
        )
        rank_genes(
            adata, indices=[int(i.split("_", 1)[1]) - 1 for i in sig], show=False
        )
        plt.savefig("{}/{}_sigspectra.png".format(args.output_dir, name))
