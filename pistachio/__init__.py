"""
spatial_nmf package
-------------------
A modular framework for Non-Negative Matrix Factorization (NMF)
with support for coordinate descent, multiplicative updates,
and negative binomial-based loss.

Usage (Python):
    import spatial_nmf
    W, H, errors = spatial_nmf.run(Y, mask)

Usage (CLI):
    spatial_nmf --input input.csv --mask mask.csv --method coordinate_descent
"""

from .nmf import nmf_main as run