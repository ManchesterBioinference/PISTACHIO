import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for CLI use
import matplotlib.pyplot as plt
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils import check_random_state
from sklearn.decomposition import NMF
from sklearn.metrics import r2_score, mean_absolute_error


# ---------------------------------------------------------
# Initialization methods
# ---------------------------------------------------------
def _initialize_nmf(X, n_components, eps=1e-6, init='random'):
    """Algorithms for NMF initialization.
    Computes an initial guess for the non-negative rank-k matrix approximation for X: X = WH.
    Based on sklearn NMF initialization logic.
    """
    random_state = 42
    rng = check_random_state(random_state)
    n_samples, n_features = X.shape

    if init == 'random':
        X_mean = X.mean()
        avg = np.sqrt(X_mean / n_components)

        W = avg * rng.standard_normal(size=(n_samples, n_components)).astype(X.dtype, copy=False)
        H = avg * rng.standard_normal(size=(n_components, n_features)).astype(X.dtype, copy=False)

        np.abs(H, out=H)
        np.abs(W, out=W)
        return W, H
    else:
        raise ValueError(f"Unsupported init method: {init}")


def initialize_nmf(Y, n_components, init_method='random', random_state=42):
    """Initialize W and H based on the selected method."""
    n_samples, n_features = Y.shape
    if init_method == 'random':
        W_init = np.random.random((n_samples, n_components))
        H_init = np.random.random((n_components, n_features))
    elif init_method == 'real':
        W_init = Y[:, :n_components]
        H_init = np.random.random((n_components, n_features))
    elif init_method == 'nndsvd':
        nmf = NMF(n_components=n_components, init='nndsvd', random_state=random_state)
        W_init = nmf.fit_transform(Y)
        H_init = nmf.components_
    elif init_method == "initialize_nmf":
        W_init, H_init = _initialize_nmf(Y, n_components, eps=1e-6, init='random')
    else:
        raise ValueError("Unknown initialization method.")
    return W_init, H_init


# ---------------------------------------------------------
# Coordinate Descent
# ---------------------------------------------------------
def _update_cdnmf_fast(W, HHt, XHt, permutation, mask=None):
    """Fast coordinate descent update for W or H."""
    violation = 0.0
    for s in range(W.shape[1]):
        t = permutation[s]
        for i in range(W.shape[0]):
            if W[i, t] == 0:
                continue
            grad = -XHt[i, t] + np.dot(W[i, :], HHt[:, t])
            pg = min(0., grad)
            violation += abs(pg)
            hess = HHt[t, t]
            if hess != 0:
                W[i, t] = max(W[i, t] - grad / hess, 0.)
    return violation


def nmf_coordinate_descent(X, W, H, max_iter, tol, shuffle, random_state, mask=None):
    """Perform NMF with Coordinate Descent."""
    Ht = H.T.copy()
    rng = check_random_state(random_state)
    reconstruction_errors = []
    for n_iter in range(1, max_iter + 1):
        permutation = rng.permutation(W.shape[1]) if shuffle else np.arange(W.shape[1])
        violation = _update_cdnmf_fast(W, np.dot(Ht.T, Ht), safe_sparse_dot(X, Ht), permutation, mask)
        violation += _update_cdnmf_fast(Ht, np.dot(W.T, W), safe_sparse_dot(X.T, W), permutation, mask)
        WH = np.dot(W, Ht.T)
        error = np.linalg.norm(X - WH, "fro")
        reconstruction_errors.append(error)
        if len(reconstruction_errors) > 1 and abs(reconstruction_errors[-2] - error) < tol:
            break
    return W, Ht.T, reconstruction_errors


# ---------------------------------------------------------
# Multiplicative Update
# ---------------------------------------------------------
def _update_multiplicative(W, H, X, l1_reg_W, l2_reg_W, l1_reg_H, l2_reg_H, epsilon=1e-8):
    """Multiplicative update rule for W and H."""
    W *= safe_sparse_dot(X, H.T) / (np.dot(W, np.dot(H, H.T)) + l1_reg_W + l2_reg_W * W + epsilon)
    H *= np.dot(W.T, X) / (np.dot(W.T, W).dot(H) + l1_reg_H + l2_reg_H * H + epsilon)
    return W, H


def nmf_multiplicative_update(X, W, H, max_iter, tol, l1_reg_W, l2_reg_W, l1_reg_H, l2_reg_H):
    """Perform NMF with Multiplicative Updates."""
    reconstruction_errors = []
    for n_iter in range(1, max_iter + 1):
        W, H = _update_multiplicative(W, H, X, l1_reg_W, l2_reg_W, l1_reg_H, l2_reg_H)
        WH = np.dot(W, H)
        error = np.linalg.norm(X - WH, "fro")
        reconstruction_errors.append(error)
        if len(reconstruction_errors) > 1 and abs(reconstruction_errors[-2] - error) < tol:
            break
    return W, H, reconstruction_errors


# ---------------------------------------------------------
# Post-hoc normalization
# ---------------------------------------------------------
def posthoc_normalize_WH(W, H, eps=1e-12):
    """Rescale columns of W and rows of H for better interpretability."""
    alphas = np.linalg.norm(H, axis=1) + eps
    W *= alphas[None, :]
    H = H / alphas[:, None]
    return W, H


# ---------------------------------------------------------
# Plotting and evaluation
# ---------------------------------------------------------
def plot_results(X, X_reconstructed, reconstruction_errors, outdir="."):
    """Plots reconstruction error and comparison of original vs reconstructed X."""
    os.makedirs(outdir, exist_ok=True)

    r2 = r2_score(X.flatten(), X_reconstructed.flatten())
    mae = mean_absolute_error(X, X_reconstructed)
    corr = np.corrcoef(X.flatten(), X_reconstructed.flatten())[0, 1]

    print(f"R² Score: {r2:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"Pearson Correlation: {corr:.4f}")

    # Reconstruction error plot
    plt.figure(figsize=(8, 6))
    plt.plot(range(len(reconstruction_errors)), reconstruction_errors, label="Reconstruction Error")
    plt.xlabel("Iteration")
    plt.ylabel("Reconstruction Error")
    plt.title("Reconstruction Error Over Iterations")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(outdir, "Reconstruction_error.png"), dpi=300, bbox_inches='tight', transparent=True)
    plt.close()

    # Scatter comparison plot
    plt.figure(figsize=(7, 6))
    plt.scatter(X.flatten(), X_reconstructed.flatten(), alpha=0.5, s=10, edgecolor='k', linewidth=0.1)
    plt.xlabel("Original $Y$", fontsize=12)
    plt.ylabel(r"Reconstructed $\hat{Y}$", fontsize=12)
    plt.title(f"Original vs Reconstructed Gene Expression\nR² = {r2:.3f}, Corr = {corr:.3f}", fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "Y.png"), dpi=300, bbox_inches='tight', transparent=True)
    plt.close()


# ---------------------------------------------------------
# Negative Binomial NMF
# ---------------------------------------------------------
def negative_binomial_loss(X, X_pred, alpha, epsilon=1e-8):
    """Negative Binomial loss."""
    return np.sum((X + alpha) * np.log(X_pred + alpha + epsilon) - X * np.log(X_pred + epsilon))


def update_nb_nmf(W, H, X, alpha, epsilon=1e-8):
    """Multiplicative update rule for NB-NMF."""
    X_hat = np.dot(W, H) + epsilon
    numerator_W = np.dot((X / X_hat), H.T)
    denominator_W = np.dot(((X + alpha) / (X_hat + alpha)), H.T)
    W *= numerator_W / (denominator_W + epsilon)
    X_hat = np.dot(W, H) + epsilon
    numerator_H = np.dot(W.T, (X / X_hat))
    denominator_H = np.dot(W.T, ((X + alpha) / (X_hat + alpha)))
    H *= numerator_H / (denominator_H + epsilon)
    return W, H


def nmf_negative_binomial(X, W, H, alpha=10.0, max_iter=200, tol=1e-4, mask=None):
    """NB-NMF training loop."""
    reconstruction_errors = []
    for n_iter in range(1, max_iter + 1):
        W, H = update_nb_nmf(W, H, X, alpha)
        if mask is not None:
            W[mask == 0] = 0
        X_hat = np.dot(W, H)
        loss = negative_binomial_loss(X, X_hat, alpha)
        reconstruction_errors.append(loss)
        if len(reconstruction_errors) > 1 and abs(reconstruction_errors[-2] - loss) < tol:
            break
    return W, H, reconstruction_errors


# ---------------------------------------------------------
# Main entry point
# ---------------------------------------------------------
def nmf_main(Y, Z_df_final, n_components=5, max_iter=200, tol=1e-4,
             method='coordinate_descent', init_method='random', shuffle=True,
             alpha=10.0, outdir="results", gene_names=None):
    """Main function to run NMF with user-selected options."""
    os.makedirs(outdir, exist_ok=True)

    W_init, H_init = initialize_nmf(Y, n_components, init_method)
    mask_array = Z_df_final.values if hasattr(Z_df_final, "values") else Z_df_final
    W_init[mask_array == 0] = 0

    if method == 'coordinate_descent':
        W_final, H_final, reconstruction_errors = nmf_coordinate_descent(
            Y, W_init, H_init, max_iter, tol, shuffle, 42, mask_array)
    elif method == 'multiplicative':
        W_final, H_final, reconstruction_errors = nmf_multiplicative_update(
            Y, W_init, H_init, max_iter, tol, 0, 0, 0, 0)
    elif method == 'negative_binomial':
        W_final, H_final, reconstruction_errors = nmf_negative_binomial(
            Y, W_init, H_init, alpha=alpha, max_iter=max_iter, tol=tol, mask=mask_array)
    else:
        raise ValueError("Unknown method. Choose from ['coordinate_descent', 'multiplicative', 'negative_binomial'].")

    W_final, H_final = posthoc_normalize_WH(W_final, H_final)
    X_reconstructed = np.dot(W_final, H_final)
    plot_results(Y, X_reconstructed, reconstruction_errors, outdir=outdir)

    # Preserve row/column labels
    spot_ids = getattr(Z_df_final, "index", [f"spot_{i}" for i in range(W_final.shape[0])])
    cell_types = getattr(Z_df_final, "columns", [f"celltype_{i}" for i in range(W_final.shape[1])])
    if gene_names is None:
        gene_names = [f"gene_{i}" for i in range(H_final.shape[1])]

    W_final = pd.DataFrame(W_final, index=spot_ids, columns=cell_types)
    H_final = pd.DataFrame(H_final, index=cell_types, columns=gene_names)

    # Save outputs
    W_final.to_csv(os.path.join(outdir, "W_output.csv"))
    H_final.to_csv(os.path.join(outdir, "H_output.csv"))
    print(f"✅ Done. Results saved in {outdir}/")

    return W_final, H_final, reconstruction_errors