# -*- coding: utf-8 -*-
"""
Automated QC classifier command line interface

@author: C Heiser
"""
import os, errno, argparse
import scanpy as sc
import matplotlib.pyplot as plt

from .api import dropkick, plot_thresh_obs, coef_plot


def check_dir_exists(path):
    """
    Checks if directory already exists or not and creates it if it doesn't
    """
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def main():
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
        help="Direction of thresholding for each heuristic. Several can be specified with '--directions above below'",
        nargs="+",
        default=["above", "below"],
    )
    parser.add_argument(
        "--thresh-method",
        type=str,
        help="Method used for automatic thresholding on heuristics. One of ['otsu','li','mean']. Default 'otsu'",
        default="otsu",
    )
    parser.add_argument(
        "--n-hvgs",
        type=int,
        help="Number of highly variable genes for training model. Default 2000",
        default=2000,
    )
    parser.add_argument(
        "--min-genes",
        type=int,
        help="Minimum number of genes detected to keep cell. Default 50",
        default=50,
    )
    parser.add_argument(
        "--seed", type=int, help="Random state for cross validation", default=18,
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory. Output will be placed in [output-dir]/[name]_dropkick.h5ad. Default './'",
        nargs="?",
        default=".",
    )
    parser.add_argument(
        "--alphas",
        type=float,
        help="Ratios between l1 and l2 regularization for regression model. Default [0.1]",
        nargs="*",
        default=[0.1],
    )
    parser.add_argument(
        "--n-lambda",
        type=int,
        help="Number of lambda (regularization strength) values to test. Default 100",
        default=100,
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
        help="Maximum number of iterations for optimization. Default 1000",
        default=1000,
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        help="Maximum number of threads for cross validation. Default -1",
        default=-1,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        help="Verbosity of glmnet module. Default False",
        action="store_true",
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

    _ = dropkick(
        adata,
        min_genes=args.min_genes,
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
        verbose=args.verbose,
    )
    # generate plot of chosen training thresholds on heuristics
    print(
        "Saving threshold plots to {}/{}_{}_thresholds.png".format(
            args.output_dir, name, args.thresh_method
        )
    )
    _ = plot_thresh_obs(adata, adata.uns["dropkick_thresholds"], bins=40, show=False)
    plt.savefig(
        "{}/{}_{}_thresholds.png".format(args.output_dir, name, args.thresh_method)
    )
    # generate plot of dropkick coefficient values and CV scores vs tested lambda_path
    print("Saving coefficient plot to {}/{}_coef.png".format(args.output_dir, name))
    _ = coef_plot(adata, show=False)
    plt.savefig("{}/{}_coef.png".format(args.output_dir, name))
    # save new labels
    print("Writing updated counts to {}/{}_dropkick.h5ad".format(args.output_dir, name))
    adata.write(
        "{}/{}_dropkick.h5ad".format(args.output_dir, name), compression="gzip",
    )


main()
