"""Correctness tests for the TurboQuant implementation, checked against the paper's
closed-form values (independent of the dataset). These are the honesty backstop: a wrong
codebook or quantizer cannot silently "match" the graded distortion numbers.

Run: python -m pytest impl/test_algo.py -q   (from the run root, inside the env)
"""
import numpy as np
import pytest

from codebook import codebook, lloyd_max_gaussian, normalized_mse
from turboquant import TurboQuantMSE, TurboQuantProd, random_rotation


# --- codebook: Lloyd-Max Gaussian centroids / costs == paper's quoted values ----------
def test_lloyd_max_costs_match_paper():
    # Theorem 1: D_mse ~ 0.36, 0.117, 0.03, 0.009 for b=1..4 (classic Lloyd-Max Gaussian).
    assert normalized_mse(1) == pytest.approx(0.3634, abs=2e-3)
    assert normalized_mse(2) == pytest.approx(0.1175, abs=2e-3)
    assert normalized_mse(3) == pytest.approx(0.0345, abs=2e-3)
    assert normalized_mse(4) == pytest.approx(0.0095, abs=5e-4)


def test_lloyd_max_centroids_b1_b2():
    c1, _ = lloyd_max_gaussian(2)
    assert np.allclose(np.sort(c1), [-np.sqrt(2 / np.pi), np.sqrt(2 / np.pi)], atol=2e-3)
    c2, _ = lloyd_max_gaussian(4)
    assert np.allclose(np.sort(np.abs(c2)), [0.4528, 0.4528, 1.510, 1.510], atol=3e-3)


def test_codebook_scaled_by_coord_std():
    d = 1024
    c = codebook(d, 1)
    assert np.allclose(np.sort(c), np.array([-1, 1]) * np.sqrt(2 / np.pi) / np.sqrt(d), atol=1e-4)


def test_b0_codebook_single_zero_centroid():
    assert np.allclose(codebook(64, 0), [0.0])
    _, cost = lloyd_max_gaussian(1)
    assert cost == pytest.approx(1.0)  # variance of N(0,1)


# --- TurboQuant_mse: measured distortion on unit vectors matches the codebook cost -----
def test_random_rotation_is_orthogonal():
    rng = np.random.default_rng(0)
    q = random_rotation(128, rng)
    assert np.allclose(q @ q.T, np.eye(128), atol=1e-10)


@pytest.mark.parametrize("b,expected", [(1, 0.3634), (2, 0.1175), (3, 0.0345)])
def test_mse_distortion_matches_theory(b, expected):
    d, n = 512, 8000
    rng = np.random.default_rng(1)
    X = rng.standard_normal((n, d))
    X /= np.linalg.norm(X, axis=1, keepdims=True)        # unit sphere
    q = TurboQuantMSE(d, b, rng)
    Xt = q.dequant(q.quant(X))
    d_mse = np.mean(np.sum((X - Xt) ** 2, axis=1))
    assert d_mse == pytest.approx(expected, abs=0.02)


# --- TurboQuant_prod: unbiased; TurboQuant_mse at b=1 has the 2/pi multiplicative bias --
def test_prod_inner_product_unbiased():
    d, nx, nq = 256, 1500, 300
    rng = np.random.default_rng(2)
    X = rng.standard_normal((nx, d)); X /= np.linalg.norm(X, axis=1, keepdims=True)
    Y = rng.standard_normal((nq, d)); Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    q = TurboQuantProd(d, 2, rng)
    idx, qjl, gamma = q.quant(X)
    Xt = q.dequant(idx, qjl, gamma)
    true = Y @ X.T
    est = Y @ Xt.T
    slope = np.sum(true * est) / np.sum(true * true)
    assert slope == pytest.approx(1.0, abs=0.05)         # unbiased


def test_mse_b1_has_two_over_pi_bias():
    d, nx, nq = 256, 1500, 300
    rng = np.random.default_rng(3)
    X = rng.standard_normal((nx, d)); X /= np.linalg.norm(X, axis=1, keepdims=True)
    Y = rng.standard_normal((nq, d)); Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    q = TurboQuantMSE(d, 1, rng)
    Xt = q.dequant(q.quant(X))
    true = Y @ X.T
    est = Y @ Xt.T
    slope = np.sum(true * est) / np.sum(true * true)
    assert slope == pytest.approx(2 / np.pi, abs=0.03)   # the bias prod is designed to fix
