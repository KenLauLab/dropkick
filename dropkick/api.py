# -*- coding: utf-8 -*-
"""
Automated QC classifier pipeline

@author: C Heiser
"""
import argparse
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc
import time
import threading
from skimage.filters import threshold_li, threshold_otsu, threshold_mean

from .logistic import LogitNet


def recipe_dropkick(
    adata,
    filter=True,
    min_genes=50,
    calc_metrics=True,
    mito_names="^mt-|^MT-",
    n_ambient=10,
    target_sum=None,
    n_hvgs=2000,
    X_final="raw_counts",
    verbose=True,
):
    """
    dropkick preprocessing recipe

    Parameters:
        adata (AnnData.AnnData): object with raw counts data in .X
        filter (bool): remove cells with less than min_genes detected
            and genes with zero total counts
        min_genes (int): threshold for minimum genes detected. Default 50.
            Ignored if filter==False.
        calc_metrics (bool): if False, do not calculate metrics in .obs/.var
        mito_names (str): substring encompassing mitochondrial gene names for
            calculation of mito expression. Ignored if calc_metrics==False.
        n_ambient (int): number of ambient genes to call. top genes by cells.
            Ignored if calc_metrics==False.
        target_sum (int): total sum of counts for each cell prior to arcsinh 
            or log1p transformations; default None to use median counts.
        n_hvgs (int or None): number of HVGs to calculate using Seurat method.
            if None, do not calculate HVGs.
        X_final (str): which normalization should be left in .X slot?
            ("raw_counts","arcsinh_norm","norm_counts")
        verbose (bool): print updates to the console?

    Returns:
        adata (AnnData.AnnData): updated object includes:
            - useful .obs and .var columns
                ("total_counts", "pct_counts_mito", "n_genes_by_counts", etc.)
            - raw counts (adata.layers["raw_counts"])
            - normalized counts (adata.layers["norm_counts"])
            - arcsinh transformation of normalized counts (adata.X)
            - highly variable genes if desired (adata.var["highly_variable"])
    """
    if filter:
        # remove cells and genes with zero total counts
        orig_shape = adata.shape
        sc.pp.filter_cells(adata, min_genes=min_genes)
        sc.pp.filter_genes(adata, min_counts=1)
        if adata.shape[0] != orig_shape[0]:
            print(
                "Ignoring {} cells with less than {} genes detected".format(
                    orig_shape[0] - adata.shape[0], min_genes
                )
            )
        if adata.shape[1] != orig_shape[1]:
            print(
                "Ignoring {} genes with zero total counts".format(
                    orig_shape[1] - adata.shape[1]
                )
            )

    # store raw counts before manipulation
    adata.layers["raw_counts"] = adata.X.copy()

    if calc_metrics:
        if verbose:
            print("Calculating metrics:")
        # identify mitochondrial genes
        adata.var["mito"] = adata.var_names.str.contains(mito_names)
        # identify putative ambient genes by lowest dropout pct (top n_ambient)
        adata.var["dropout_rate"] = np.array(
            1 - (adata.X.astype(bool).sum(axis=0) / adata.n_obs)
        ).squeeze()
        lowest_dropout = round(
            adata.var.dropout_rate.nsmallest(n=n_ambient).min() * 100, 3
        )
        highest_dropout = round(
            adata.var.dropout_rate.nsmallest(n=n_ambient).max() * 100, 3
        )
        adata.var["ambient"] = (
            adata.var.dropout_rate
            <= adata.var.dropout_rate.nsmallest(n=n_ambient).max()
        )
        # reorder genes by dropout rate
        adata = adata[:, np.argsort(adata.var.dropout_rate)].copy()
        if verbose:
            print(
                "Top {} ambient genes have dropout rates between {} and {} percent:\n\t{}".format(
                    len(adata.var_names[adata.var.ambient]),
                    lowest_dropout,
                    highest_dropout,
                    adata.var_names[adata.var.ambient].tolist(),
                )
            )
        # calculate standard qc .obs and .var
        sc.pp.calculate_qc_metrics(
            adata, qc_vars=["mito", "ambient"], inplace=True, percent_top=None
        )
        # other arcsinh-transformed metrics
        adata.obs["arcsinh_total_counts"] = np.arcsinh(adata.obs["total_counts"])
        adata.obs["arcsinh_n_genes_by_counts"] = np.arcsinh(
            adata.obs["n_genes_by_counts"]
        )

    # normalize counts (adata.layers["norm_counts"])
    sc.pp.normalize_total(adata, target_sum=target_sum, layers=None, layer_norm=None)
    adata.layers["norm_counts"] = adata.X.copy()  # save to .layers

    # HVGs
    if n_hvgs is not None:
        if verbose:
            print("Determining {} highly variable genes".format(n_hvgs))
        # log1p transform for HVGs (adata.layers["log1p_norm"])
        sc.pp.log1p(adata)
        adata.layers["log1p_norm"] = adata.X.copy()  # save to .layers
        sc.pp.highly_variable_genes(
            adata, n_top_genes=n_hvgs, n_bins=20, flavor="seurat"
        )

    # arcsinh-transform normalized counts
    adata.X = np.arcsinh(adata.layers["norm_counts"])
    sc.pp.scale(adata)  # scale genes for feeding into model
    adata.layers[
        "arcsinh_norm"
    ] = adata.X.copy()  # save arcsinh scaled counts in .layers

    # set .X as desired for downstream processing; default raw_counts
    adata.X = adata.layers[X_final].copy()

    return adata


def auto_thresh_obs(
    adata, obs_cols=["arcsinh_n_genes_by_counts", "pct_counts_ambient"], method="otsu",
):
    """
    automated thresholding on metrics in adata.obs

    Parameters:
        adata (anndata.AnnData): object containing unfiltered scRNA-seq data
        obs_cols (list of str): name of column(s) to threshold from adata.obs
        method (str): one of 'otsu' (default), 'li', or 'mean'

    Returns:
        thresholds (dict): keys are obs_cols and values are threshold results
    """
    thresholds = dict.fromkeys(obs_cols)  # initiate output dictionary
    for col in obs_cols:
        tmp = np.array(adata.obs[col])
        if method == "otsu":
            thresholds[col] = threshold_otsu(tmp)
        elif method == "li":
            thresholds[col] = threshold_li(tmp)
        elif method == "mean":
            thresholds[col] = threshold_mean(tmp)
        else:
            raise ValueError(
                "Please provide a valid threshold method ('otsu', 'li', 'mean')."
            )

    return thresholds


def plot_thresh_obs(adata, thresholds, bins=40, show=True):
    """
    plot automated thresholding on metrics in adata.obs as output by auto_thresh_obs()

    Parameters:
        adata (anndata.AnnData): object containing unfiltered scRNA-seq data
        thresholds (dict): output of auto_thresh_obs() function
        bins (int): number of bins for histogram
        show (bool): show plot or return object

    Returns:
        plot of distributions of obs_cols in thresholds dictionary with corresponding threshold values
    """
    fig, axes = plt.subplots(
        ncols=len(thresholds), nrows=1, figsize=(len(thresholds) * 4, 4), sharey=True
    )
    axes[0].set_ylabel("cells")
    for i in range(len(thresholds)):
        axes[i].hist(adata.obs[list(thresholds.keys())[i]], bins=bins)
        axes[i].axvline(list(thresholds.values())[i], color="r")
        axes[i].set_title(list(thresholds.keys())[i])
    fig.tight_layout()
    if show:
        plt.show()
    else:
        return fig


def filter_thresh_obs(
    adata,
    thresholds,
    obs_cols=["arcsinh_n_genes_by_counts", "pct_counts_ambient"],
    directions=["above", "below"],
    inclusive=True,
    name="thresh_filter",
):
    """
    filter cells by thresholding on metrics in adata.obs as output by auto_thresh_obs()

    Parameters:
        adata (anndata.AnnData): object containing unfiltered scRNA-seq data
        thresholds (dict): output of auto_thresh_obs() function
        obs_cols (list of str): name of column(s) to threshold from adata.obs
        directions (list of str): 'below' or 'above', indicating which direction to keep (label=1)
        inclusive (bool): include cells at the thresholds? default True.
        name (str): name of .obs col containing final labels

    Returns:
        updated adata with filter labels in adata.obs[name]
    """
    # initialize .obs column as all "good" cells
    adata.obs[name] = 1
    # if any criteria are NOT met, label cells "bad"
    for i in range(len(obs_cols)):
        if directions[i] == "above":
            if inclusive:
                adata.obs.loc[
                    (adata.obs[name] == 1)
                    & (adata.obs[obs_cols[i]] <= thresholds[obs_cols[i]]),
                    name,
                ] = 0
            else:
                adata.obs.loc[
                    (adata.obs[name] == 1)
                    & (adata.obs[obs_cols[i]] < thresholds[obs_cols[i]]),
                    name,
                ] = 0
        elif directions[i] == "below":
            if inclusive:
                adata.obs.loc[
                    (adata.obs[name] == 1)
                    & (adata.obs[obs_cols[i]] >= thresholds[obs_cols[i]]),
                    name,
                ] = 0
            else:
                adata.obs.loc[
                    (adata.obs[name] == 1)
                    & (adata.obs[obs_cols[i]] > thresholds[obs_cols[i]]),
                    name,
                ] = 0


def dropkick(
    adata,
    min_genes=50,
    mito_names="^mt-|^MT-",
    n_ambient=10,
    n_hvgs=2000,
    thresh_method="otsu",
    metrics=["arcsinh_n_genes_by_counts", "pct_counts_ambient",],
    directions=["above", "below"],
    alphas=[0.1],
    max_iter=1000,
    n_jobs=-1,
    seed=18,
    verbose=False,
):
    """
    generate logistic regression model of cell quality

    Parameters:
        adata (anndata.AnnData): object containing unfiltered, raw scRNA-seq
            counts in .X layer
        min_genes (int): threshold for minimum genes detected. Default 50.
            Ignores all cells with less than min_genes (dropkick label = 0).
        mito_names (str): substring encompassing mitochondrial gene names for
            calculation of mito expression
        n_ambient (int): number of ambient genes to call. top genes by cells.
        n_hvgs (int or None): number of HVGs to calculate using Seurat method
            if None, do not calculate HVGs
        thresh_method (str): one of 'otsu' (default), 'li', or 'mean'
        metrics (list of str): name of column(s) to threshold from adata.obs
        directions (list of str): 'below' or 'above', indicating which
            direction to keep (label=1)
        alphas (tuple of int): alpha values to test using glmnet with n-fold
            cross validation
        max_iter (int): number of iterations for glmnet optimization
        n_jobs (int): number of threads for cross validation by glmnet
        seed (int): random state for cross validation by glmnet
        verbose (bool): verbosity for glmnet training and warnings

    Returns:
        adata_thresh (dict): dictionary of automated thresholds on heuristics
        rc (LogisticRegression): trained logistic regression classifier

        updates adata inplace to include 'train', 'dropkick_score', and
            'dropkick_label' columns in .obs
    """
    # 0) preprocess counts and calculate required QC metrics
    a = adata.copy()  # make copy of anndata before manipulating
    a = recipe_dropkick(
        a,
        filter=True,
        min_genes=min_genes,
        calc_metrics=True,
        mito_names=mito_names,
        n_ambient=n_ambient,
        target_sum=None,
        n_hvgs=n_hvgs,
        X_final="arcsinh_norm",
        verbose=verbose,
    )

    # 1) threshold chosen heuristics using automated method
    print("Thresholding on heuristics for training labels:\n\t{}".format(metrics))
    adata_thresh = auto_thresh_obs(a, method=thresh_method, obs_cols=metrics)

    # 2) create labels from combination of thresholds
    filter_thresh_obs(
        a,
        adata_thresh,
        obs_cols=metrics,
        directions=directions,
        inclusive=True,
        name="train",
    )

    X = a.X[:, a.var.highly_variable].copy()  # final X is HVGs
    y = a.obs["train"].copy(deep=True)  # final y is "train" labels from step 2
    print("Training LogitNet with alphas: {}".format(alphas))

    if len(alphas) > 1:
        # 3.1) cross-validation to choose alpha and lambda values
        cv_scores = {"rc": [], "lambda": [], "alpha": [], "score": []}  # dictionary o/p
        for alpha in alphas:
            # print("Training LogitNet with alpha: {}".format(alpha), end="  ")
            rc = LogitNet(
                alpha=alpha,
                n_lambda=100,
                cut_point=1,
                n_splits=5,
                max_iter=max_iter,
                n_jobs=n_jobs,
                random_state=seed,
                verbose=verbose,
            )
            rc.fit(adata=a, y=y, n_hvgs=n_hvgs)
            cv_scores["rc"].append(rc)
            cv_scores["alpha"].append(alpha)
            cv_scores["lambda"].append(rc.lambda_best_)
            cv_scores["score"].append(rc.score(X, y, lamb=rc.lambda_best_))
        # determine optimal lambda and alpha values by accuracy score
        lambda_ = cv_scores["lambda"][
            cv_scores["score"].index(max(cv_scores["score"]))
        ]  # choose alpha value
        alpha_ = cv_scores["alpha"][
            cv_scores["score"].index(max(cv_scores["score"]))
        ]  # choose l1 ratio
        rc_ = cv_scores["rc"][
            cv_scores["score"].index(max(cv_scores["score"]))
        ]  # choose classifier
        print("Chosen lambda value: {}; Chosen alpha value: {}".format(lambda_, alpha_))
    else:
        # 3.2) train model with single alpha value
        rc_ = LogitNet(
            alpha=alphas[0],
            n_lambda=100,
            cut_point=1,
            n_splits=5,
            max_iter=max_iter,
            n_jobs=n_jobs,
            random_state=seed,
            verbose=verbose,
        )
        rc_.fit(adata=a, y=y, n_hvgs=n_hvgs)
        print("Chosen lambda value: {}".format(rc_.lambda_best_))
        lambda_, alpha_ = rc_.lambda_best_, alphas[0]

    # 4) use model to assign scores and labels to original adata
    print("Assigning scores and labels")
    adata.obs.loc[a.obs_names, "dropkick_score"] = rc_.predict_proba(X)[:, 1]
    adata.obs.dropkick_score.fillna(0, inplace=True)  # fill ignored cells with zeros
    adata.obs.loc[a.obs_names, "dropkick_label"] = rc_.predict(X)
    adata.obs.dropkick_label.fillna(0, inplace=True)  # fill ignored cells with zeros
    for metric in metrics:
        adata.obs.loc[a.obs_names, metric] = a.obs[metric]
        adata.obs[metric].fillna(0, inplace=True)  # fill ignored cells with zeros
    adata.var.loc[
        a.var_names[a.var.highly_variable], "dropkick_coef"
    ] = rc_.coef_.squeeze()

    # 5) save model hyperparameters in .uns
    adata.uns["dropkick_thresholds"] = adata_thresh
    adata.uns["dropkick_args"] = {
        "n_ambient": n_ambient,
        "n_hvgs": n_hvgs,
        "thresh_method": thresh_method,
        "metrics": metrics,
        "directions": directions,
        "alphas": alphas,
        "chosen_alpha": alpha_,
        "lambda_path": rc_.lambda_path_,
        "chosen_lambda": lambda_,
        "coef_path": rc_.coef_path_.squeeze().T,
        "cv_mean_score": rc_.cv_mean_score_,
        "cv_standard_error": rc_.cv_standard_error_,
        "max_iter": max_iter,
        "seed": seed,
    }  # save command-line arguments to .uns for reference

    print("Done!\n")
    return rc_


def coef_inventory(adata, n=10):
    """
    return highest and lowest coefficient values from logistic regression model,
    along with sparsity

    Parameters:
        adata (anndata.AnnData): object generated from dropkick
        n (int): number of genes to show at top and bottom of coefficient list

    Returns:
        prints top and bottom n genes by their coefficient values
    """
    print("\nTop HVGs by coefficient value (good cells):")
    print(adata.var.loc[-adata.var.dropkick_coef.isna(), "dropkick_coef"].nlargest(n))
    print("\nBottom HVGs by coefficient value (bad droplets):")
    print(adata.var.loc[-adata.var.dropkick_coef.isna(), "dropkick_coef"].nsmallest(n))
    n_zero = (adata.var.dropkick_coef == 0).sum()
    n_coef = (-adata.var.dropkick_coef.isna()).sum()
    sparsity = round((n_zero / n_coef) * 100, 3)
    print(
        "\n{} coefficients equal to zero. Model sparsity: {} %\n".format(
            n_zero, sparsity
        )
    )


def coef_plot(adata, show=True):
    """
    plot dropkick coefficient values and cross validation (CV) scores for tested values
    of lambda (lambda_path)

    Parameters:
        adata (anndata.AnnData): object generated from dropkick
        show (bool): show plot or return object

    Returns:
        plot of CV scores (mean +/- SEM) and coefficient values (coef_path) versus
        log(lambda_path). includes indicator of chosen lambda value.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    # plot coefficient values versus log(lambda) on left y-axis
    ax.set_title("Dropkick Coefficients")
    ax.set_xlabel("Log (lambda)")
    ax.set_ylabel("Coefficient Value")
    ax.plot(
        np.log(adata.uns["dropkick_args"]["lambda_path"]),
        adata.uns["dropkick_args"]["coef_path"],
        alpha=0.5,
    )
    # plot top three genes by coefficient value
    [
        ax.text(
            x=np.log(adata.uns["dropkick_args"]["chosen_lambda"]),
            y=adata.var.dropkick_coef.max() + 0.15 - 0.05 * x,
            s=" "
            + adata.var.loc[-adata.var.dropkick_coef.isna(), "dropkick_coef"]
            .nlargest(3)
            .index[x],
            fontsize=9,
            color="g",
        )
        for x in range(3)
    ]
    # plot bottom three genes by coefficient value
    [
        ax.text(
            x=np.log(adata.uns["dropkick_args"]["chosen_lambda"]),
            y=adata.var.dropkick_coef.min() - 0.20 + 0.05 * x,
            s=" "
            + adata.var.loc[-adata.var.dropkick_coef.isna(), "dropkick_coef"]
            .nsmallest(3)
            .index[x],
            fontsize=9,
            color="r",
        )
        for x in range(3)
    ]

    # plot CV scores versus log(lambda) on right y-axis
    ax2 = ax.twinx()
    ax2.set_ylabel("CV Mean Score", color="b")
    ax2.plot(
        np.log(adata.uns["dropkick_args"]["lambda_path"]),
        adata.uns["dropkick_args"]["cv_mean_score"],
        label="CV Score Mean",
        color="b",
    )
    ax2.fill_between(
        np.log(adata.uns["dropkick_args"]["lambda_path"]),
        y1=adata.uns["dropkick_args"]["cv_mean_score"]
        - adata.uns["dropkick_args"]["cv_standard_error"],
        y2=adata.uns["dropkick_args"]["cv_mean_score"]
        + adata.uns["dropkick_args"]["cv_standard_error"],
        color="b",
        alpha=0.2,
        label="CV Score SEM",
    )
    ax2.tick_params(axis="y", labelcolor="b")
    # plot vertical line at chosen lambda value and add to legend
    plt.axvline(
        np.log(adata.uns["dropkick_args"]["chosen_lambda"]),
        label="Chosen lambda: {:.2e}".format(
            adata.uns["dropkick_args"]["chosen_lambda"][0]
        ),
        color="k",
        ls="--",
    )
    plt.legend()
    fig.tight_layout()
    if show:
        plt.show()
    else:
        return fig
