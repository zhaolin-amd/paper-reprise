"""Optimal scalar-quantizer codebook for TurboQuant (arXiv 2504.19874, sec:mse_turbo_alg).

After the random rotation, each coordinate of a unit vector is distributed as
    f_X(x) = Gamma(d/2) / (sqrt(pi) Gamma((d-1)/2)) * (1 - x^2)^((d-3)/2),  x in [-1,1]
which converges to N(0, 1/d) in high dimension (the paper states this explicitly and
quotes the resulting centroids). The optimal b-bit scalar quantizer solves the continuous
1-D k-means / Lloyd-Max problem (eq:continuous_k_means):
    C(f_X, b) = min over c_1<=...<=c_{2^b} of  sum_i  int_{cell_i} |x - c_i|^2 f_X(x) dx
with cell boundaries at the midpoints of consecutive centroids.

We solve it once on the standardized limit N(0,1) (giving the codebook in units of the
coordinate std) and scale by sigma = 1/sqrt(d). This reproduces the paper's quoted
centroids exactly, e.g. b=1 -> {+/- sqrt(2/pi)/sqrt(d)} and the normalized MSE values
0.3634, 0.1175, 0.0345, 0.0095 for b=1..4 (== the classic Lloyd-Max Gaussian distortions,
matching the paper's reported 0.36, 0.117, 0.03, 0.009).

Nothing here reads the paper's target numbers; the codebook and its cost are COMPUTED.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

# +/-CLIP stands in for +/-inf for the outer cell boundaries: for N(0,1) the density and
# x*density are 0 well before 40 sigma, so this is exact to machine precision but keeps all
# arithmetic finite (no inf*0 = nan).
_CLIP = 40.0


def _cell_moments(lo: float, hi: float) -> tuple[float, float, float]:
    """(P, M1, M2) = integrals of (1, x, x^2) * phi(x) over [lo, hi] for the standard
    normal, in closed form: P = Phi(hi)-Phi(lo); M1 = phi(lo)-phi(hi);
    M2 = P + (lo*phi(lo) - hi*phi(hi))."""
    P = norm.cdf(hi) - norm.cdf(lo)
    M1 = norm.pdf(lo) - norm.pdf(hi)
    M2 = P + (lo * norm.pdf(lo) - hi * norm.pdf(hi))
    return P, M1, M2


def lloyd_max_gaussian(K: int, iters: int = 500, tol: float = 1e-13
                       ) -> tuple[np.ndarray, float]:
    """Lloyd-Max optimal K-level scalar quantizer for N(0,1).

    Returns (centroids ascending, normalized MSE). For K==1 the optimal centroid is the
    mean (0) and the cost is the variance (1.0) -- this is the b=0 case TurboQuant_prod
    uses internally (no MSE stage, residual == input)."""
    if K < 1:
        raise ValueError("K must be >= 1")
    if K == 1:
        return np.array([0.0]), 1.0

    # Initialize centroids at the (i/(K+1)) quantiles of N(0,1); Lloyd iterations follow.
    c = norm.ppf(np.arange(1, K + 1) / (K + 1))
    for _ in range(iters):
        bnd = np.empty(K + 1)
        bnd[0], bnd[K] = -_CLIP, _CLIP
        bnd[1:K] = (c[:-1] + c[1:]) / 2.0
        new_c = c.copy()
        for i in range(K):
            P, M1, _ = _cell_moments(bnd[i], bnd[i + 1])
            if P > 0:
                new_c[i] = M1 / P  # conditional mean = centroid of the Voronoi cell
        if np.max(np.abs(new_c - c)) < tol:
            c = new_c
            break
        c = new_c

    # Final cost with the converged centroids.
    bnd = np.empty(K + 1)
    bnd[0], bnd[K] = -_CLIP, _CLIP
    bnd[1:K] = (c[:-1] + c[1:]) / 2.0
    cost = 0.0
    for i in range(K):
        P, M1, M2 = _cell_moments(bnd[i], bnd[i + 1])
        cost += M2 - 2.0 * c[i] * M1 + c[i] ** 2 * P
    return c, float(cost)


def codebook(d: int, b: int) -> np.ndarray:
    """The b-bit (2^b level) TurboQuant_mse codebook for dimension d: the standardized
    Lloyd-Max centroids scaled by the coordinate std 1/sqrt(d). b==0 -> single centroid 0."""
    K = 2 ** b
    c_std, _ = lloyd_max_gaussian(K)
    return c_std / np.sqrt(d)


def normalized_mse(b: int) -> float:
    """C(f_X,b) * d == the standardized Lloyd-Max MSE == the paper's D_mse for b bits.
    (D_mse = d * C(f_X,b) and C ~ (Lloyd-Max MSE)/d, so the d's cancel.)"""
    _, cost = lloyd_max_gaussian(2 ** b)
    return cost
