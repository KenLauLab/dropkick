# -*- coding: utf-8 -*-
"""
Automated QC classifier pipeline

@author: C Heiser
"""
import argparse
import sys
import os, errno
import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc
import time
import threading
from skimage.filters import threshold_li, threshold_otsu, threshold_mean

from logistic import LogitNet


class Spinner:
    busy = False
    delay = 0.1

    @staticmethod
    def spinning_cursor():
        while 1:
            for cursor in "|/-\\":
                yield cursor

    def __init__(self, delay=None):
        self.spinner_generator = self.spinning_cursor()
        if delay and float(delay):
            self.delay = delay

    def spinner_task(self):
        while self.busy:
            sys.stdout.write(next(self.spinner_generator))
            sys.stdout.flush()
            time.sleep(self.delay)
            sys.stdout.write("\b")
            sys.stdout.flush()

    def __enter__(self):
        self.busy = True
        threading.Thread(target=self.spinner_task).start()

    def __exit__(self, exception, value, tb):
        self.busy = False
        time.sleep(self.delay)
        if exception is not None:
            return False


def check_dir_exists(path):
    """
    Checks if directory already exists or not and creates it if it doesn't
    """
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def recipe_dropkick(
    adata,
    X_final="raw_counts",
    filter=True,
    calc_metrics=True,
    mito_names="^mt-|^MT-",
    n_ambient=10,
    target_sum=None,
    n_hvgs=2000,
    verbose=True,
):
    """
    scanpy preprocessing recipe

    Parameters:
        adata (AnnData.AnnData): object with raw counts data in .X
        X_final (str): which normalization should be left in .X slot?
            ("raw_counts","arcsinh_norm","norm_counts")
        filter (bool): remove cells and genes with zero total counts
        calc_metrics (bool): if False, do not calculate metrics in .obs/.var
        mito_names (str): substring encompassing mitochondrial gene names for
            calculation of mito expression
        n_ambient (int): number of ambient genes to call. top genes by cells.
        target_sum (int): total sum of counts for each cell prior to arcsinh 
            and log1p transformations; default None to use median counts.
        n_hvgs (int or None): number of HVGs to calculate using Seurat method
            if None, do not calculate HVGs
        verbose (bool): print updates to the console?

    Returns:
        AnnData.AnnData: adata is edited in place to include:
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
        sc.pp.filter_cells(adata, min_genes=10)
        sc.pp.filter_genes(adata, min_counts=1)
        if adata.shape[0] != orig_shape[0]:
            print(
                "Ignoring {} cells with zero total counts".format(
                    orig_shape[0] - adata.shape[0]
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
        # identify putative ambient genes by lowest dropout pct (top 10)
        adata.var["ambient"] = np.array(
            adata.X.astype(bool).sum(axis=0) / adata.n_obs
        ).squeeze()
        if verbose:
            print(
                "Top {} ambient genes have dropout rates between {} and {} percent:\n\t{}".format(
                    n_ambient,
                    round((1 - adata.var.ambient.nlargest(n=n_ambient).max()) * 100, 2),
                    round((1 - adata.var.ambient.nlargest(n=n_ambient).min()) * 100, 2),
                    adata.var.ambient.nlargest(n=n_ambient).index.tolist(),
                )
            )
        adata.var["ambient"] = (
            adata.var.ambient >= adata.var.ambient.nlargest(n=n_ambient).min()
        )
        # calculate standard qc .obs and .var
        sc.pp.calculate_qc_metrics(
            adata, qc_vars=["mito", "ambient"], inplace=True, percent_top=[10, 50, 100]
        )
        # other arcsinh-transformed metrics
        adata.obs["arcsinh_total_counts"] = np.arcsinh(adata.obs["total_counts"])
        adata.obs["arcsinh_n_genes_by_counts"] = np.arcsinh(
            adata.obs["n_genes_by_counts"]
        )

    # log1p transform (adata.layers["log1p_norm"])
    sc.pp.normalize_total(adata, target_sum=target_sum, layers=None, layer_norm=None)
    adata.layers["norm_counts"] = adata.X.copy()  # save to .layers
    sc.pp.log1p(adata)
    adata.layers["log1p_norm"] = adata.X.copy()  # save to .layers

    # HVGs
    if n_hvgs is not None:
        if verbose:
            print("Determining {} highly variable genes".format(n_hvgs))
        sc.pp.highly_variable_genes(
            adata, n_top_genes=n_hvgs, n_bins=20, flavor="seurat"
        )

    # arcsinh-transform normalized counts to leave in .X
    adata.X = np.arcsinh(adata.layers["norm_counts"])
    sc.pp.scale(adata)  # scale genes for feeding into model
    adata.layers[
        "arcsinh_norm"
    ] = adata.X.copy()  # save arcsinh scaled counts in .layers

    # set .X as desired for downstream processing; default raw_counts
    adata.X = adata.layers[X_final].copy()


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
    mito_names="^mt-|^MT-",
    n_hvgs=2000,
    thresh_method="otsu",
    metrics=["arcsinh_n_genes_by_counts", "pct_counts_ambient",],
    directions=["above", "below"],
    alphas=[0.1],
    n_lambda=10,
    cut_point=1,
    n_splits=5,
    max_iter=100000,
    n_jobs=-1,
    seed=18,
):
    """
    generate logistic regression model of cell quality

    Parameters:
        adata (anndata.AnnData): object containing unfiltered, raw scRNA-seq
            counts in .X layer
        mito_names (str): substring encompassing mitochondrial gene names for
            calculation of mito expression
        n_hvgs (int or None): number of HVGs to calculate using Seurat method
            if None, do not calculate HVGs
        thresh_method (str): one of 'otsu' (default), 'li', or 'mean'
        metrics (list of str): name of column(s) to threshold from adata.obs
        directions (list of str): 'below' or 'above', indicating which
            direction to keep (label=1)
        alphas (tuple of int): alpha values to test using glmnet with n-fold
            cross validation
        n_lambda (int): number of lambda values to test in glmnet
        cut_point (float): The cut point to use for selecting lambda_best.
            arg_max lambda
            cv_score(lambda)>=cv_score(lambda_max)-cut_point*standard_error(lambda_max)
        n_splits (int): number of splits for n-fold cross validation
        max_iter (int): number of iterations for glmnet optimization
        n_jobs (int): number of threads for cross validation by glmnet
        seed (int): random state for cross validation by glmnet

    Returns:
        adata_thresh (dict): dictionary of automated thresholds on heuristics
        rc (LogisticRegression): trained logistic regression classifier

        updated adata inplace to include 'train', 'dropkick_score', and
            'dropkick_label' columns in .obs
    """
    # 0) preprocess counts and calculate required QC metrics
    a = adata.copy()  # make copy of anndata before manipulating
    recipe_dropkick(
        a,
        X_final="arcsinh_norm",
        filter=True,
        calc_metrics=True,
        mito_names=mito_names,
        n_hvgs=n_hvgs,
        target_sum=None,
        verbose=True,
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

    if len(alphas) > 1:
        # 3.1) cross-validation to choose alpha and lambda values
        cv_scores = {"rc": [], "lambda": [], "alpha": [], "score": []}  # dictionary o/p
        for alpha in alphas:
            print("Training LogitNet with alpha: {}".format(alpha), end="  ")
            rc = LogitNet(
                alpha=alpha,
                n_lambda=n_lambda,
                cut_point=cut_point,
                n_splits=n_splits,
                max_iter=max_iter,
                n_jobs=n_jobs,
                random_state=seed,
            )
            with Spinner():
                rc.fit(adata=a, y=y, n_hvgs=n_hvgs)
            print("\n", end="")
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
        print("Training LogitNet with alpha: {}".format(alphas[0]), end="  ")
        rc_ = LogitNet(
            alpha=alphas[0],
            n_lambda=n_lambda,
            cut_point=cut_point,
            n_splits=n_splits,
            max_iter=max_iter,
            n_jobs=n_jobs,
            random_state=seed,
        )
        with Spinner():
            rc_.fit(adata=a, y=y, n_hvgs=n_hvgs)
        print("\n", end="")
        lambda_, alpha_ = rc_.lambda_best_, alphas[0]

    # 4) use ridge model to assign scores and labels to original adata
    print("Assigning scores and labels")
    adata.obs.loc[a.obs_names, "dropkick_score"] = rc_.predict_proba(X)[:, 1]
    adata.obs.dropkick_score.fillna(0, inplace=True)  # fill ignored cells with zeros
    adata.obs.loc[a.obs_names, "dropkick_label"] = rc_.predict(X)
    adata.obs.dropkick_label.fillna(0, inplace=True)  # fill ignored cells with zeros
    for metric in metrics:
        adata.obs.loc[a.obs_names, metric] = a.obs[metric]
        adata.obs[metric].fillna(0, inplace=True)  # fill ignored cells with zeros
    adata.var.loc[a.var_names[a.var.highly_variable], "dropkick_coef"] = rc_.coef_.squeeze()

    # 5) save model hyperparameters in .uns
    adata.uns["dropkick_thresholds"] = adata_thresh
    adata.uns["dropkick_args"] = {
        "n_hvgs": n_hvgs,
        "thresh_method": thresh_method,
        "metrics": metrics,
        "directions": directions,
        "alphas": alphas,
        "chosen_alpha": alpha_,
        "chosen_lambda": lambda_,
        "n_lambda": n_lambda,
        "cut_point": cut_point,
        "n_splits": n_splits,
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
        adata (anndata.AnnData): object generated from dropkick.py ("regression")
        n (int): number of genes to show at top and bottom of coefficient list

    Returns:
        prints top and bottom n genes by their coefficient values
    """
    print("\nTop HVGs by coefficient value (good cells):")
    print(adata.var.loc[-adata.var.dropkick_coef.isna(), "dropkick_coef"].nlargest(n))
    print("\nBottom HVGs by coefficient value (bad droplets):")
    print(adata.var.loc[-adata.var.dropkick_coef.isna(), "dropkick_coef"].nsmallest(n))
    n_zero = (adata.var.dropkick_coef==0).sum()
    n_coef = (-adata.var.dropkick_coef.isna()).sum()
    sparsity = round((n_zero/n_coef)*100, 3)
    print("\n{} coefficients equal to zero. Model sparsity: {} %\n".format(n_zero, sparsity))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "counts",
        type=str,
        help="Input (cell x gene) counts matrix as .h5ad or tab delimited text file",
    )
    parser.add_argument(
        "--obs-cols",
        type=str,
        help="Heuristics for thresholding. Several can be specified with '--obs-cols arcsinh_n_genes_by_counts pct_counts_ambient'",
        nargs="+",
        default=["arcsinh_n_genes_by_counts", "pct_counts_ambient"],
    )
    parser.add_argument(
        "--directions",
        type=str,
        help="Direction of thresholding for each heuristic. Several can be specified with '--obs-cols above below'",
        nargs="+",
        default=["above", "below"],
    )
    parser.add_argument(
        "--thresh-method",
        type=str,
        help="Method used for automatic thresholding on heuristics. One of ['otsu','li','mean']. Default 'Otsu'",
        default="otsu",
    )
    parser.add_argument(
        "--mito-names",
        type=str,
        help="Substring or regex defining mitochondrial genes. Default '^mt-|^MT-'",
        default="^mt-|^MT-",
    )
    parser.add_argument(
        "--n-hvgs",
        type=int,
        help="Number of highly variable genes for training model. Default 2000",
        default=2000,
    )
    parser.add_argument(
        "--seed", type=int, help="Random state for cross validation", default=18,
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory. Output will be placed in [output-dir]/[name]...",
        nargs="?",
        default=".",
    )
    parser.add_argument(
        "--alphas",
        type=float,
        help="Ratios between l1 and l2 regularization for regression model",
        nargs="*",
        default=[0.1],
    )
    parser.add_argument(
        "--n-lambda",
        type=int,
        help="Number of lambda (regularization strength) values to test. Default 10",
        default=10,
    )
    parser.add_argument(
        "--cut-point",
        type=float,
        help="The cut point to use for selecting lambda_best. Default 1.0",
        default=1.0,
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        help="Number of splits for cross validation. Default 5",
        default=5,
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        help="Maximum number of iterations for optimization. Default 100000",
        default=100000,
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        help="Maximum number of threads for cross validation. Default -1",
        default=-1,
    )

    args = parser.parse_args()

    # read in counts data
    print("\nReading in unfiltered counts from {}".format(args.counts), end="")
    adata = sc.read(args.counts)
    print(" - {} barcodes and {} genes".format(adata.shape[0], adata.shape[1]))

    # check that output directory exists, create it if needed.
    check_dir_exists(args.output_dir)
    # get basename of file for writing outputs
    name = os.path.splitext(os.path.basename(args.counts))[0]

    regression_model = dropkick(
        adata,
        mito_names=args.mito_names,
        n_hvgs=args.n_hvgs,
        thresh_method=args.thresh_method,
        metrics=args.obs_cols,
        directions=args.directions,
        alphas=args.alphas,
        n_lambda=args.n_lambda,
        cut_point=args.cut_point,
        n_splits=args.n_splits,
        max_iter=args.n_iter,
        n_jobs=args.n_jobs,
        seed=args.seed,
    )
    # generate plot of chosen training thresholds on heuristics
    print(
        "Saving threshold plots to {}/{}_{}_thresholds.png".format(
            args.output_dir, name, args.thresh_method
        )
    )
    thresh_plt = plot_thresh_obs(
        adata, adata.uns["dropkick_thresholds"], bins=40, show=False
    )
    plt.savefig(
        "{}/{}_{}_thresholds.png".format(args.output_dir, name, args.thresh_method)
    )
    # save new labels
    print(
        "Writing updated counts to {}/{}_{}.h5ad".format(
            args.output_dir, name, args.command
        )
    )
    adata.write(
        "{}/{}_dropkick.h5ad".format(args.output_dir, name), compression="gzip",
    )
