"""Per-claim entrypoint for the TurboQuant §4.1 reproduction.

Usage:  python eval.py <claim_id>        # computes & prints `<metric>: <value>`
        python eval.py --smoke            # tiny synthetic self-test (proves the code runs)

Claim id -> what is computed (all metrics are COMPUTED from the algorithm; no paper target
value is read here):
  mse-distortion-b{1..4}  -> mse_distortion : D_mse = E||x - x_hat||^2          (Theorem 1)
  prod-distortion-b{1..4} -> ip_distortion  : D_prod = E|<y,x> - <y,x_hat>|^2   (Theorem 2)
  mse-bias-b1             -> ip_ratio        : slope of <y,x_hat_mse> vs <y,x>  (~2/pi bias)
  prod-unbiased-b2        -> ip_ratio        : slope of <y,x_hat_prod> vs <y,x> (~1, unbiased)

Env overrides (each falls back to the paper's setup):
  SEED (default 0); PAPER_REPRISE_N_TRAIN (rows for D_mse, default 100000);
  PAPER_REPRISE_N_DB / PAPER_REPRISE_N_QUERY (database/query rows for the IP metrics).
"""
from __future__ import annotations

import os
import sys

import numpy as np

from data import DIM, load
from turboquant import TurboQuantMSE, TurboQuantProd

SEED = int(os.environ.get("SEED", "0"))
N_TRAIN = int(os.environ.get("PAPER_REPRISE_N_TRAIN", "100000"))
N_DB = int(os.environ.get("PAPER_REPRISE_N_DB", "5000"))
N_QUERY = int(os.environ.get("PAPER_REPRISE_N_QUERY", "1000"))


def _emit(metric: str, value: float) -> None:
    # Standalone `metric: number` line the grade stage parses (parsers.parse_metric).
    print(f"{metric}: {value:.8g}")


def mse_distortion(b: int, X: np.ndarray) -> float:
    """D_mse = mean_n ||x_n - dequant(quant(x_n))||^2, in row-chunks to bound memory."""
    rng = np.random.default_rng(SEED)
    q = TurboQuantMSE(DIM, b, rng)
    total, n = 0.0, X.shape[0]
    for s in range(0, n, 10000):
        xb = X[s:s + 10000]
        xt = q.dequant(q.quant(xb))
        total += float(np.sum((xb - xt) ** 2))
    return total / n


def _ip_pairs(quantizer, X_db: np.ndarray, Y_q: np.ndarray):
    """Return (true, est) inner-product matrices (n_query, n_db) for a quantizer exposing
    quant/dequant. Handles both the MSE quantizer (idx only) and the prod quantizer."""
    out = quantizer.quant(X_db)
    Xt = quantizer.dequant(*out) if isinstance(out, tuple) else quantizer.dequant(out)
    return Y_q @ X_db.T, Y_q @ Xt.T


def ip_distortion(b: int, X_db: np.ndarray, Y_q: np.ndarray) -> float:
    rng = np.random.default_rng(SEED)
    q = TurboQuantProd(DIM, b, rng)
    true, est = _ip_pairs(q, X_db, Y_q)
    return float(np.mean((true - est) ** 2))


def ip_ratio(method: str, b: int, X_db: np.ndarray, Y_q: np.ndarray) -> float:
    """Least-squares slope (through the origin) of estimated vs true inner product:
    E[<y,x_hat>] = slope * <y,x>. ~2/pi for MSE b=1, ~1 for the unbiased prod estimator."""
    rng = np.random.default_rng(SEED)
    q = TurboQuantMSE(DIM, b, rng) if method == "mse" else TurboQuantProd(DIM, b, rng)
    true, est = _ip_pairs(q, X_db, Y_q)
    return float(np.sum(true * est) / np.sum(true * true))


def _smoke() -> None:
    """Tiny synthetic self-test: no dataset, a few unit vectors, prints a metric line."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((64, DIM)); X /= np.linalg.norm(X, axis=1, keepdims=True)
    q = TurboQuantMSE(DIM, 2, rng)
    xt = q.dequant(q.quant(X))
    _emit("smoke_mse_distortion", float(np.mean(np.sum((X - xt) ** 2, axis=1))))


def main(argv: list[str]) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    if argv[0] == "--smoke":
        _smoke()
        return 0

    claim = argv[0]
    if claim.startswith("mse-distortion-b"):
        b = int(claim.rsplit("b", 1)[1])
        X, _ = load(n_train=N_TRAIN)
        _emit("mse_distortion", mse_distortion(b, X))
    elif claim.startswith("prod-distortion-b"):
        b = int(claim.rsplit("b", 1)[1])
        Xtr, Yq = load(n_query=N_QUERY, n_train=N_DB)
        _emit("ip_distortion", ip_distortion(b, Xtr, Yq))
    elif claim == "mse-bias-b1":
        Xtr, Yq = load(n_query=N_QUERY, n_train=N_DB)
        _emit("ip_ratio", ip_ratio("mse", 1, Xtr, Yq))
    elif claim == "prod-unbiased-b2":
        Xtr, Yq = load(n_query=N_QUERY, n_train=N_DB)
        _emit("ip_ratio", ip_ratio("prod", 2, Xtr, Yq))
    else:
        print(f"unknown claim id: {claim}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
