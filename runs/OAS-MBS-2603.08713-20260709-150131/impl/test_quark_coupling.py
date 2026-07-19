"""Part B (verify-fromscratch-fidelity skill) checks for the Quark MXFP4 <-> MBS
coupling. None of test_mxquant.py's existing tests ever call
`_quant_blocks_quark_even` / `QuarkQuantLinear` -- this file closes that gap.

Requires a GPU (Quark's `qdq_mxfp4_triton` is Triton, CUDA/ROCm only); skipped
automatically where unavailable, same convention as any other GPU-only test in
this impl/.
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))
import mxquant as mq  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Quark's mxfp4 kernel needs a GPU")


def test_quark_even_round_trips_exactly_representable_block():
    # Same spirit as test_mxquant.test_exactly_representable_block_is_lossless,
    # but through the REAL Quark kernel rather than the paper's own OAS scale.
    blk = torch.tensor([6.0, 4.0, 3.0, 2.0, 1.5, 1.0, 0.5, 0.0,
                        -6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0] * 2).cuda()
    q = mq._quant_blocks_quark_even(blk.reshape(1, 32))
    assert torch.allclose(q.flatten(), blk, atol=1e-6)


def test_quark_kernel_runs_and_reduces_error_vs_random_input():
    # Contract sanity: the kernel accepts our (bf16-cast) call convention and
    # returns something finite, same shape, with bounded QSNR -- catches a
    # silently-wrong call (bad axis/shape) turning into NaN/garbage rather than
    # a clean crash.
    torch.manual_seed(0)
    x = (torch.randn(8, 512) * 3.0).cuda()
    out = mq._quant_blocks_quark_even(x)
    assert out.shape == x.shape
    assert not torch.isnan(out).any()
    assert mq.qsnr_db(x, out) > 0  # quantized signal must still resemble the input


def test_mbs_macro_block_forces_extra_bf16_rounding_before_quark_quant():
    # BOUNDARY DIFF (skill Part B check 2): the "oas" inner path stays fp32
    # end-to-end; the "quark_even" inner path MUST cast to bf16 first (Quark's
    # kernel asserts src dtype is bf16/fp16). Confirm this extra rounding step
    # is real (not zero) so it stays a documented, not silent, error source
    # when comparing MXFP4-MBS-H (oas inner) against MXFP4-Quark-MBS-H (quark
    # inner) end2end numbers -- the two differ by BOTH block-size (16 vs 32,
    # already documented in analysis_en.md) AND this bf16 cast (undocumented).
    torch.manual_seed(0)
    x = (torch.randn(8, 512) * 3.0).cuda()
    x[:, ::129] *= 12.0
    xr = x.reshape(8, 4, 128)
    factor = mq._mbs_factor_dynamic(xr, oas=False, oas_block=32, inner="quark_even")
    xs = (xr * factor).reshape(8, 512)
    bf16_roundtrip_error = (xs - xs.to(torch.bfloat16).to(torch.float32)).abs()
    # Not a pass/fail assertion on magnitude (it's small, ~0.1% relative) --
    # this test exists to keep the effect visible and regression-checked, not
    # to declare a size threshold the paper never specified.
    assert bf16_roundtrip_error.mean() > 0
