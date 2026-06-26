"""TurboQuant (arXiv 2504.19874): the two data-oblivious vector quantizers.

  TurboQuant_mse  (Algorithm 1): random rotation Pi -> per-coordinate nearest-centroid
                  quantization against the Lloyd-Max codebook -> DeQuant retrieves
                  centroids and rotates back with Pi^T. Optimized for MSE.

  TurboQuant_prod (Algorithm 2): TurboQuant_mse at (b-1) bits, then a 1-bit QJL on the
                  residual r = x - dequant_mse(x). Inner-product estimator
                      <y, x_mse> + ||r|| * <y, QJL^{-1}(QJL(r))>
                  is UNBIASED (Theorem 2).

Row-vector convention: a batch X is (N, d). The paper writes y = Pi.x for column vectors;
for rows that is Y = X @ Pi^T, and DeQuant x = Pi^T.y_tilde is X_tilde = Y_tilde @ Pi.
Likewise QJL quant sign(S.r) is sign(R @ S^T) and QJL^{-1} (sqrt(pi/2)/d) S^T z is
(sqrt(pi/2)/d) * (Z @ S).

Everything is COMPUTED from the rotation + codebook; no paper target value is referenced.
"""
from __future__ import annotations

import numpy as np

from codebook import codebook


def random_rotation(d: int, rng: np.random.Generator) -> np.ndarray:
    """Haar-uniform d x d rotation via QR of an i.i.d. Gaussian matrix (paper: 'QR
    decomposition on a random matrix with i.i.d Normal entries'), with the standard
    sign correction so the distribution is exactly Haar."""
    a = rng.standard_normal((d, d))
    q, r = np.linalg.qr(a)
    q *= np.sign(np.diag(r))            # remove QR sign ambiguity -> Haar measure
    return q


class TurboQuantMSE:
    """Algorithm 1. Holds the rotation Pi and the b-bit codebook for dimension d."""

    def __init__(self, d: int, b: int, rng: np.random.Generator):
        self.d, self.b = d, b
        self.pi = random_rotation(d, rng)
        self.c = codebook(d, b)                       # ascending centroids (1-D)
        # Voronoi boundaries = midpoints of consecutive centroids (for searchsorted).
        self.bnd = (self.c[:-1] + self.c[1:]) / 2.0 if len(self.c) > 1 else np.array([])

    def quant(self, X: np.ndarray) -> np.ndarray:
        """X (N,d) -> integer indices (N,d) of the nearest centroid per rotated coord."""
        Y = X @ self.pi.T
        if self.bnd.size == 0:                        # b==0: single centroid
            return np.zeros_like(Y, dtype=np.int64)
        return np.searchsorted(self.bnd, Y)           # nearest centroid (sorted boundaries)

    def dequant(self, idx: np.ndarray) -> np.ndarray:
        """indices (N,d) -> reconstructed vectors (N,d) (retrieve centroids, rotate back)."""
        Y_tilde = self.c[idx]
        return Y_tilde @ self.pi


class TurboQuantProd:
    """Algorithm 2. TurboQuant_mse at (b-1) bits + 1-bit QJL on the residual."""

    def __init__(self, d: int, b: int, rng: np.random.Generator):
        self.d, self.b = d, b
        self.mse = TurboQuantMSE(d, b - 1, rng)       # b-1 bits (b=1 -> 0-bit MSE stage)
        self.S = rng.standard_normal((d, d))          # QJL projection ~ N(0,1)
        self._scale = np.sqrt(np.pi / 2.0) / d

    def quant(self, X: np.ndarray):
        idx = self.mse.quant(X)
        r = X - self.mse.dequant(idx)                 # residual
        qjl = np.sign(r @ self.S.T)                   # 1-bit QJL of the residual
        qjl[qjl == 0] = 1.0                           # sign(0) -> +1 (measure-zero tie)
        gamma = np.linalg.norm(r, axis=1)             # ||r||_2 per row
        return idx, qjl, gamma

    def dequant(self, idx: np.ndarray, qjl: np.ndarray, gamma: np.ndarray) -> np.ndarray:
        x_mse = self.mse.dequant(idx)
        x_qjl = self._scale * gamma[:, None] * (qjl @ self.S)
        return x_mse + x_qjl
