# -*- coding: utf-8 -*-
"""
Automated QC classifier command line interface

@author: C Heiser
"""
import os, errno, argparse
import scanpy as sc
import matplotlib.pyplot as plt

from .api import dropkick, recipe_dropkick, coef_plot, score_plot
from .qc import qc_summary
from ._version import get_versions


def check_dir_exists(path):
    """
    Checks if directory already exists or not and creates it if it doesn't
    """
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def prepare(args):
    """
    read counts into AnnData object and get directory and name for outputs
    """
    # read in counts data
    print("\nReading in unfiltered counts from {}".format(args.counts), end="")
    adata = sc.read(args.counts)
    print(" - {} barcodes and {} genes".format(adata.shape[0], adata.shape[1]))

    # check that output directory exists, create it if needed.
    check_dir_exists(args.output_dir)
    # get basename of file for writing outputs
    name = os.path.splitext(os.path.basename(args.counts))[0]

    return adata, name


def run(args):
    """
    run dropkick filtering pipeline and save results
    """
    # read in counts and prepare output directory
    adata, name = prepare(args)
    # run main dropkick module
    _ = dropkick(
        adata,
        min_genes=args.min_genes,
        n_ambient=args.n_ambient,
        n_hvgs=args.n_hvgs,
        metrics=args.metrics,
        thresh_methods=args.thresh_methods,
        directions=args.directions,
        alphas=args.alphas,
        max_iter=args.n_iter,
        n_jobs=args.n_jobs,
        seed=args.seed,
        verbose=args.verbose,
    )
    # save new labels in .h5ad
    print("Writing updated counts to {}/{}_dropkick.h5ad".format(args.output_dir, name))
    adata.write(
        "{}/{}_dropkick.h5ad".format(args.output_dir, name), compression="gzip",
    )
    # generate plot of dropkick coefficient values and CV scores vs tested lambda_path
    coef_plot(
        adata,
        save_to="{}/{}_coef.png".format(args.output_dir, name),
        verbose=args.verbose,
    )
    # generate plot of chosen training thresholds on heuristics
    adata = recipe_dropkick(
        adata, filter=True, min_genes=args.min_genes, n_hvgs=None, verbose=False
    )
    score_plot(
        adata,
        ["arcsinh_n_genes_by_counts", "pct_counts_ambient"],
        save_to="{}/{}_score.png".format(args.output_dir, name),
        verbose=args.verbose,
    )


def qc(args):
    """
    generate dropkick qc report
    """
    # read in counts and prepare output directory
    adata, name = prepare(args)

    # perform ambient analysis and QC
    # preprocess and calculate metrics
    adata = recipe_dropkick(
        adata,
        filter=True,
        min_genes=args.min_genes,
        n_hvgs=None,
        X_final="raw_counts",
        n_ambient=args.n_ambient,
        verbose=args.verbose,
    )
    # plot total counts distribution, gene dropout rates, and highest expressed genes
    qc_summary(
        adata,
        save_to="{}/{}_qc.png".format(args.output_dir, name),
        verbose=args.verbose,
    )


def main():
    parser = argparse.ArgumentParser(prog="dropkick")
    parser.add_argument(
        "-V", "--version", action="version", version=get_versions()["version"],
    )

    subparsers = parser.add_subparsers()

    run_parser = subparsers.add_parser(
        "run", help="Automated filtering of scRNA-seq data.",
    )
    run_parser.add_argument(
        "counts",
        type=str,
        help="Input (cell x gene) counts matrix as .h5ad or tab delimited text file",
    )
    run_parser.add_argument(
        "--output-dir",
        required=False,
        type=str,
        help="Output directory. Output will be placed in [output-dir]/[name]_dropkick.h5ad. Default './'.",
        nargs="?",
        default=".",
    )
    run_parser.add_argument(
        "-v",
        "--verbose",
        required=False,
        help="Print processing updates to console.",
        action="store_true",
    )
    run_parser.add_argument(
        "--min-genes",
        required=False,
        type=int,
        help="Minimum number of genes detected to keep cell. Default 50.",
        default=50,
    )
    run_parser.add_argument(
        "--n-ambient",
        required=False,
        type=int,
        help="Number of top genes by dropout rate to use for ambient profile. Default 10.",
        default=10,
    )
    run_parser.add_argument(
        "-m",
        "--metrics",
        required=False,
        type=str,
        help="Heuristics for thresholding.",
        nargs="+",
        default=["arcsinh_n_genes_by_counts", "pct_counts_ambient"],
    )
    run_parser.add_argument(
        "--thresh-methods",
        required=False,
        type=str,
        help="Method used for automatic thresholding on each heuristic in '--metrics'.",
        nargs="+",
        default=["multiotsu", "otsu"],
    )
    run_parser.add_argument(
        "--directions",
        required=False,
        type=str,
        help="Direction of thresholding for each heuristic in '--metrics'.",
        nargs="+",
        default=["above", "below"],
    )
    run_parser.add_argument(
        "--n-hvgs",
        required=False,
        type=int,
        help="Number of highly variable genes for training model. Default 2000.",
        default=2000,
    )
    run_parser.add_argument(
        "--alphas",
        required=False,
        type=float,
        help="Ratio(s) between l1 and l2 regularization for regression model. Default 0.1.",
        nargs="*",
        default=[0.1],
    )
    run_parser.add_argument(
        "--n-iter",
        required=False,
        type=int,
        help="Maximum number of iterations for optimization. Default 1000.",
        default=1000,
    )
    run_parser.add_argument(
        "--n-jobs",
        required=False,
        type=int,
        help="Maximum number of threads for cross validation. Default -1.",
        default=-1,
    )
    run_parser.add_argument(
        "--seed",
        required=False,
        type=int,
        help="Random state for cross validation.",
        default=18,
    )
    run_parser.set_defaults(func=run)

    qc_parser = subparsers.add_parser("qc", help="scRNA-seq quality control report.",)
    qc_parser.add_argument(
        "counts",
        type=str,
        help="Input (cell x gene) counts matrix as .h5ad or tab delimited text file",
    )
    qc_parser.add_argument(
        "--output-dir",
        required=False,
        type=str,
        help="Output directory. Output will be placed in [output-dir]/[name]_dropkick.h5ad. Default './'.",
        nargs="?",
        default=".",
    )
    qc_parser.add_argument(
        "-v",
        "--verbose",
        required=False,
        help="Print processing updates to console.",
        action="store_true",
    )
    qc_parser.add_argument(
        "--min-genes",
        required=False,
        type=int,
        help="Minimum number of genes detected to keep cell. Default 50.",
        default=50,
    )
    qc_parser.add_argument(
        "--n-ambient",
        required=False,
        type=int,
        help="Number of top genes by dropout rate to use for ambient profile. Default 10.",
        default=10,
    )
    qc_parser.set_defaults(func=qc)

    args = parser.parse_args()
    args.func(args)
