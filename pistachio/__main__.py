#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd
import os
from .nmf import nmf_main

def main():
    parser = argparse.ArgumentParser(description="Run Spatial-NMF on a dataset.")
    parser.add_argument("--input", type=str, required=True, help="Path to the input matrix (CSV).")
    parser.add_argument("--mask", type=str, required=True, help="Path to the mask file (CSV).")
    parser.add_argument("--components", type=int, default=5, help="Number of components.")
    parser.add_argument("--method", type=str, default="coordinate_descent",
                        choices=["coordinate_descent", "multiplicative", "negative_binomial"],
                        help="NMF optimization method.")
    parser.add_argument("--init", type=str, default="random", help="Initialization method.")
    parser.add_argument("--max_iter", type=int, default=200, help="Maximum number of iterations.")
    parser.add_argument("--tol", type=float, default=1e-4, help="Convergence tolerance.")
    parser.add_argument("--alpha", type=float, default=10.0, help="Alpha parameter for NB-NMF.")
    parser.add_argument("--outdir", type=str, default="results", help="Output directory for results.")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    
    # Load data
    Y_df = pd.read_csv(args.input, index_col=0)   # keep as DataFrame to get names
    mask = pd.read_csv(args.mask, index_col=0)
    gene_names = Y_df.columns                     # capture gene names
    Y = Y_df.values                               # use numeric array for NMF
    
    # Run NMF
    W, H, errors = nmf_main(
        Y,
        Z_df_final=mask,
        n_components=args.components,
        max_iter=args.max_iter,
        tol=args.tol,
        method=args.method,
        init_method=args.init,
        alpha=args.alpha,
        outdir=args.outdir,
        gene_names=gene_names                     
    )

if __name__ == "__main__":
    main()