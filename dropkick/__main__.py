# -*- coding: utf-8 -*-
"""
Automated QC classifier command line interface

@author: C Heiser
"""
import os, errno, argparse
import scanpy as sc
import matplotlib.pyplot as plt

from .api import dropkick, recipe_dropkick, plot_thresh_obs, coef_plot
from .qc import summary_plot


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
        "--n-ambient",
        type=int,
        help="Number of top genes by dropout rate to use for ambient profile. Default 10",
        default=10,
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
    parser.add_argument(
        "--qc",
        help="Perform analysis of ambient expression content and overall QC.",
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

    # if --qc flag, perform ambient analysis and QC
    if args.qc:
        # preprocess and calculate metrics
        adata = recipe_dropkick(
            adata,
            filter=False,
            n_hvgs=None,
            X_final="raw_counts",
            n_ambient=args.n_ambient,
            verbose=args.verbose,
        )
        # plot total counts distribution, gene dropout rates, and highest expressed genes
        print(
            "Saving QC summary plot to {}/{}_dropkickqc.png".format(
                args.output_dir, name
            )
        )
        _ = summary_plot(adata, show=False)
        plt.savefig("{}/{}_dropkickqc.png".format(args.output_dir, name))

    # otherwise, run main dropkick module
    else:
        _ = dropkick(
            adata,
            min_genes=args.min_genes,
            n_ambient=args.n_ambient,
            n_hvgs=args.n_hvgs,
            thresh_method=args.thresh_method,
            metrics=args.obs_cols,
            directions=args.directions,
            alphas=args.alphas,
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
        a = (
            adata.copy()
        )  # copy anndata to re-calculate metrics for plotting using all barcodes
        recipe_dropkick(a, filter=False)
        _ = plot_thresh_obs(a, adata.uns["dropkick_thresholds"], bins=40, show=False)
        plt.savefig(
            "{}/{}_{}_thresholds.png".format(args.output_dir, name, args.thresh_method)
        )
        # generate plot of dropkick coefficient values and CV scores vs tested lambda_path
        print("Saving coefficient plot to {}/{}_coef.png".format(args.output_dir, name))
        _ = coef_plot(adata, show=False)
        plt.savefig("{}/{}_coef.png".format(args.output_dir, name))
        # save new labels
        print(
            "Writing updated counts to {}/{}_dropkick.h5ad".format(
                args.output_dir, name
            )
        )
        adata.write(
            "{}/{}_dropkick.h5ad".format(args.output_dir, name), compression="gzip",
        )


main()
