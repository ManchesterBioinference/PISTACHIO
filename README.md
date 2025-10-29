# PISTACHIO
---

##  Overview

**PISTACHIO** is a modular Python framework for spatial omics deconvolution, integrating spatial transcriptomics and spatial proteomics data through constrained non-negative matrix factorization (NMF).  

It provides a flexible implementation of several NMF-based optimization strategies (Coordinate Descent, Multiplicative Update, and Negative Binomial NMF), supports spatial masking to constrain cell-type distributions, and outputs interpretable W and H matrices with detailed evaluation metrics.

<img width="2513" height="1425" alt="Methods" src="https://github.com/user-attachments/assets/cd81f8a9-aea6-4815-80d1-1db048a76d0b" />

---

## Installation

Clone the repository and install locally:

```bash
git clone https://github.com/ManchesterBioinference/PISTACHIO.git
cd PISTACHIO
pip install .
```

Or install directly from GitHub.

## Arguments

| Argument | Type | Required | Description | Default |
|----------|------:|:--------:|-------------|:--------|
| `--input` | `str` | **Yes** | Path to spatial transcriptomics CSV (rows = spots, columns = genes). First column may be spot IDs. | — |
| `--mask` | `str` | **Yes** | Path to spatial proteomics mask CSV (rows = spots, columns = cell types). Binary (0/1) or soft (0..1). Must have same number of rows as `--input`. | — |
| `--components` | `int` | No | Number of latent components (interpreted as cell types). | `5` |
| `--method` | `str` | No | NMF optimization method. Options: `coordinate_descent`, `multiplicative`, `negative_binomial`. | `coordinate_descent` |
| `--init` | `str` | No | Initialization method. Options: `random`, `real`, `nndsvd`, `initialize_nmf`. | `random` |
| `--max_iter` | `int` | No | Maximum number of iterations for the selected algorithm. | `200` |
| `--tol` | `float` | No | Convergence tolerance (stop if improvement < tol). | `1e-4` |
| `--alpha` | `float` | No | Overdispersion parameter for Negative Binomial NMF (only used if `--method negative_binomial`). | `10.0` |
| `--shuffle` | `flag` | No | If set, shuffle update order in coordinate descent. Use as `--shuffle`. | *not set* |
| `--seed` | `int` | No | Random seed for reproducible initialization and shuffling. | `42` |
| `--outdir` | `str` | No | Output directory where all results and figures will be saved. Created if it doesn't exist. | `results/` |


## Output Files

All results are automatically saved in the directory specified by `--outdir` (default: `results/`).  
If the directory does not exist, PISTACHIO will create it.

| File Name | Format | Description |
|------------|:-------:|-------------|
| `W_output.csv` | CSV | Spot × Component matrix representing the cell-type proportions per spatial location. Rows correspond to spot IDs, columns correspond to inferred components or cell types. |
| `H_output.csv` | CSV | Component × Gene matrix containing the gene weights or expression signatures for each inferred component. Rows correspond to components, columns to genes. |
| `Reconstruction_error.png` | PNG | Line plot showing reconstruction loss or error versus iteration number, indicating convergence behavior. |
| `Y.png` | PNG | Scatter plot comparing original vs reconstructed data (`Y` vs `Ŷ`), annotated with R² and Pearson correlation metrics. |


---

## Example

Example files are provided in the examples/ directory.

For realistic testing, you can use spatial transcriptomics (ST) and spatial proteomics (SP) matrices:
- glioblastoma_synthetic_ST.csv (gene expression counts)
- glioblastoma_synthetic_SP.csv (cell-type distribution mask)

```
  pistachio \
  --input glioblastoma_synthetic_ST.csv \
  --mask glioblastoma_synthetic_SP.csv \
  --components 9 \
  --method negative_binomial \
  --alpha 20 \
  --outdir results_gbm/
```
