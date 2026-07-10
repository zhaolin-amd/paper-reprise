"""Fake-quant core for the enhanced MXFP4 formats of arXiv 2603.08713v1.

Implements, as numerically-faithful quantize->dequantize (no custom kernel needed;
the paper's CUTLASS/SM100 kernels only affect SPEED, not the numbers):

  * FP4 (E2M1) round-to-nearest with clamp at Fmax = 6.0
  * MXFP4-OCP (baseline): block_size=32, E8M0 OCP scale (maps amax to (4,8])
  * MXFP4 at block size 16 with an E8M0 (power-of-two) per-block scale
    -- realized via the NVFP4 pipeline but constraining scales to powers of two
       (paper 4.1 "Quantization Block Granularity")
  * Overflow-Aware Scaling (OAS): map block absmax to (3.5, 7] instead of (3, 6]
    (paper 4.2)
  * Macro Block Scaling (MBS), 1x128 macro block, 8-bit mantissa refinement factor
    (paper 4.3):
      - Static  (MBS-S): factor from top-8 mantissa bits of 6/absmax128 (eq. 1)
      - Dynamic (MBS-D): LUT/search over 16 mantissa slots minimizing macro-block SSE

Everything operates on the LAST dim (the contraction / K dim). For a weight (N, K)
that is the input-feature dim; for an activation (T, K) it is the hidden dim. All
Qwen3-8B linear in-features are multiples of 128, so blocking is exact (no padding).

The numbers in the paper (Table 2) are NOT read here; correctness is pinned by
independent closed-form checks in test_mxquant.py.
"""
from __future__ import annotations

import torch

FP4_MAX = 6.0
# Positive FP4 (E2M1) representable magnitudes.
_FP4_LEVELS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)
# Midpoints between consecutive levels -> round-to-nearest via bucketize.
_FP4_MIDS = ((_FP4_LEVELS[1:] + _FP4_LEVELS[:-1]) / 2.0)  # 7 boundaries

_TINY = 1e-30


def quant_fp4(v: torch.Tensor) -> torch.Tensor:
    """Round each element to the nearest FP4 (E2M1) value, clamped to [-6, 6].

    Nearest with ties rounding up (bucketize on midpoints); ties are vanishingly
    rare for real activations/weights and have no measurable downstream effect.
    """
    mids = _FP4_MIDS.to(v.device, v.dtype)
    levels = _FP4_LEVELS.to(v.device, v.dtype)
    sign = torch.sign(v)
    a = v.abs()
    idx = torch.bucketize(a, mids, right=True)  # 0..7
    return sign * levels[idx]


def _pow2_scale_oas(amax: torch.Tensor, oas: bool) -> torch.Tensor:
    """E8M0 power-of-two scale SF for a block with the given absmax.

    Standard: SF = 2^floor(log2(6/amax)) so amax*SF in (3, 6] (masking the mantissa
    bits of 6/amax == truncating to its power of two). OAS: if amax*SF <= 3.5, double
    SF (one more power of two), mapping amax to (6, 7] -> overall (3.5, 7].

    Computed via frexp so the exponent is exact (no log2 rounding hazard at powers
    of two). Returns SF with the same shape as amax; zero-absmax blocks get SF = 1.
    """
    amax = amax.clamp_min(_TINY)
    target = FP4_MAX / amax
    # frexp: target = mant * 2^exp, mant in [0.5, 1)  ->  floor(log2(target)) = exp - 1
    _mant, exp = torch.frexp(target)
    e = (exp - 1).to(torch.float32)
    sf = torch.exp2(e)
    if oas:
        scaled_amax = amax * sf
        e = torch.where(scaled_amax <= 3.5, e + 1.0, e)
        sf = torch.exp2(e)
    return sf


_OCP_REF = 8.0  # OCP maps amax to (4, 8] using 8 as scale reference


def _pow2_scale_ocp(amax: torch.Tensor) -> torch.Tensor:
    """E8M0 power-of-two scale per the OCP MXFP4 spec: SF = 2^floor(log2(8/amax)).

    Maps block absmax to (4, 8]: when amax > 6, overflow occurs (values clamped to ±6).
    The paper notes this is the key difference from the enhanced methods — MXFP4-OCP
    allows ~15% of blocks to saturate (paper §4.2).
    """
    amax = amax.clamp_min(_TINY)
    target = _OCP_REF / amax
    _mant, exp = torch.frexp(target)
    e = (exp - 1).to(torch.float32)
    return torch.exp2(e)


def _quant_blocks(x: torch.Tensor, block_size: int, sf: torch.Tensor) -> torch.Tensor:
    """Generic MXFP4 quantize->dequantize given a pre-computed per-block SF tensor.

    sf shape: (*lead, k//block_size, 1). Returns dequantized x, same shape as x.
    """
    *lead, k = x.shape
    xb = x.reshape(*lead, k // block_size, block_size)
    amax = xb.abs().amax(dim=-1, keepdim=True)
    q = quant_fp4(xb * sf)
    deq = torch.where(amax > 0, q / sf, torch.zeros_like(q))
    return deq.reshape(*lead, k)


def _quant_blocks16(x: torch.Tensor, oas: bool) -> torch.Tensor:
    """MXFP4 quantize->dequantize with a 1x16 E8M0(+OAS) scale, along the last dim."""
    *lead, k = x.shape
    assert k % 16 == 0, f"last dim {k} not divisible by 16"
    xb = x.reshape(*lead, k // 16, 16)
    amax = xb.abs().amax(dim=-1, keepdim=True)
    sf = _pow2_scale_oas(amax, oas)
    q = quant_fp4(xb * sf)
    deq = torch.where(amax > 0, q / sf, torch.zeros_like(q))
    return deq.reshape(*lead, k)


def _quant_blocks_ocp(x: torch.Tensor) -> torch.Tensor:
    """MXFP4-OCP: block_size=32, OCP scale (maps amax to (4,8], allows overflow)."""
    *lead, k = x.shape
    assert k % 32 == 0, f"last dim {k} not divisible by 32"
    xb = x.reshape(*lead, k // 32, 32)
    amax = xb.abs().amax(dim=-1, keepdim=True)
    sf = _pow2_scale_ocp(amax)
    q = quant_fp4(xb * sf)
    deq = torch.where(amax > 0, q / sf, torch.zeros_like(q))
    return deq.reshape(*lead, k)


def _float32_bits(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.float32).contiguous().view(torch.int32)


def _mbs_factor_static(xr: torch.Tensor) -> torch.Tensor:
    """MBS-Static factor (1 + m8/256) per 1x128 macro block (paper eq. 1).

    xr: (..., n_macro, 128). Returns (..., n_macro, 1) in [1, 2).
    m8 = (bits(6/amax128) & 0x007F8000) >> 15  -- the top 8 mantissa bits of the
    full-precision scale, i.e. the fractional part of the ideal scale's significand.
    """
    amax = xr.abs().amax(dim=-1, keepdim=True)  # (..., n_macro, 1)
    sf_full = FP4_MAX / amax.clamp_min(_TINY)
    bits = _float32_bits(sf_full)
    m8 = (bits & 0x007F8000) >> 15  # 0..255
    factor = 1.0 + m8.to(torch.float32) / 256.0
    return torch.where(amax > 0, factor, torch.ones_like(factor))


def _mbs_factor_dynamic(xr: torch.Tensor, oas: bool, n_slots: int = 16) -> torch.Tensor:
    """MBS-Dynamic factor per 1x128 macro block: pick the mantissa slot minimizing
    the macro-block sum of squared quantization error (paper 4.3.3).

    xr: (..., n_macro, 128). Candidate factors (1 + j/n_slots), j in [0, n_slots).
    j=0 -> factor 1.0 (== no MBS), so the result never increases error vs OAS-only.
    """
    lead = xr.shape[:-1]
    best_sse = xr.new_full((*lead, 1), float("inf"))
    best_factor = xr.new_ones((*lead, 1))
    for j in range(n_slots):
        c = 1.0 + j / n_slots
        xs = (xr * c).reshape(*xr.shape[:-2], -1)          # apply factor, flatten K
        q = _quant_blocks16(xs, oas).reshape(*lead, 128)   # local 1x16 OAS quant
        deq = q / c
        sse = ((deq - xr) ** 2).sum(dim=-1, keepdim=True)  # (..., n_macro, 1)
        better = sse < best_sse
        best_sse = torch.where(better, sse, best_sse)
        best_factor = torch.where(better, xr.new_full((*lead, 1), c), best_factor)
    return best_factor


def fake_quant(x: torch.Tensor, mbs: str = "none", oas: bool = True,
               ocp: bool = False) -> torch.Tensor:
    """Direct-cast MXFP4 fake-quant of `x` along its last dim.

    ocp=True -> MXFP4-OCP: block_size=32, OCP scale (maps amax to (4,8]); mbs/oas ignored.
    mbs (only when ocp=False):
         "none"   -> MXFP4-16 (+OAS if oas)
         "static" -> + Macro Block Scaling (static)
         "dynamic"-> + Macro Block Scaling (dynamic search)
    Returns a tensor of the same shape/dtype as x.
    """
    orig_dtype = x.dtype
    xf = x.to(torch.float32)

    if ocp:
        return _quant_blocks_ocp(xf).to(orig_dtype)

    if mbs == "none":
        return _quant_blocks16(xf, oas).to(orig_dtype)

    *lead, k = xf.shape
    assert k % 128 == 0, f"last dim {k} not divisible by 128 (MBS macro block)"
    xr = xf.reshape(*lead, k // 128, 128)
    if mbs == "static":
        factor = _mbs_factor_static(xr)
    elif mbs == "dynamic":
        factor = _mbs_factor_dynamic(xr, oas)
    else:
        raise ValueError(f"unknown mbs mode: {mbs!r}")

    xs = (xr * factor).reshape(*lead, k)
    q = _quant_blocks16(xs, oas).reshape(*lead, k // 128, 128)
    deq = q / factor
    return deq.reshape(*lead, k).to(orig_dtype)


# Method registry: how each Table-2 row maps to fake_quant kwargs (weight & activation).
# All rows quantize BOTH weights and activations (paper Setups).
# ocp=True -> OCP path (block_size=32, OCP scale); ocp=False -> enhanced path (block_size=16).
METHODS = {
    "MXFP4-OCP":    {"weight_mbs": "none", "act_mbs": "none", "oas": False, "ocp": True},
    "MXFP4-16":     {"weight_mbs": "none", "act_mbs": "none", "oas": False, "ocp": False},
    "MXFP4-16-OAS": {"weight_mbs": "none", "act_mbs": "none", "oas": True,  "ocp": False},
    "MXFP4-MBS-S":  {"weight_mbs": "static",  "act_mbs": "static",  "oas": True, "ocp": False},
    # MBS-Hybrid: Dynamic for weights, Static for activations (paper default).
    "MXFP4-MBS-H":  {"weight_mbs": "dynamic", "act_mbs": "static",  "oas": True, "ocp": False},
}


def qsnr_db(x: torch.Tensor, xq: torch.Tensor) -> float:
    """Quantization SNR in dB (signal power over error power) -- diagnostic only."""
    x = x.to(torch.float32)
    xq = xq.to(torch.float32)
    sig = (x ** 2).mean().clamp_min(_TINY)
    err = ((x - xq) ** 2).mean().clamp_min(_TINY)
    return float(10.0 * torch.log10(sig / err))
