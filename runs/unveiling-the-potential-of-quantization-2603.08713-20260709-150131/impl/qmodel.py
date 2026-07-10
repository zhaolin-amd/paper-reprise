"""Wrap a HuggingFace causal-LM's linear layers with MXFP4 fake-quant.

Direct-cast, calibration-free (paper Setups): every linear layer's WEIGHT is
quantized once at load time, and its input ACTIVATION is quantized on every forward
pass. Quantizes all QKVO + FFN projections (the transformer's Linear layers);
leaves the embedding and the lm_head in full precision, matching the paper's
"we quantize all linear layers" for the compute-bound GEMMs.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from mxquant import METHODS, fake_quant


class QuantLinear(nn.Module):
    """A drop-in for nn.Linear: pre-quantized weight + per-forward activation quant."""

    def __init__(self, lin: nn.Linear, weight_mbs: str, act_mbs: str,
                 oas: bool, ocp: bool = False):
        super().__init__()
        self.in_features = lin.in_features
        self.out_features = lin.out_features
        self.act_mbs = act_mbs
        self.oas = oas
        self.ocp = ocp
        with torch.no_grad():
            wq = fake_quant(lin.weight.data.to(torch.float32),
                            mbs=weight_mbs, oas=oas, ocp=ocp).to(lin.weight.dtype)
        self.weight = nn.Parameter(wq, requires_grad=False)
        if lin.bias is not None:
            self.bias = nn.Parameter(lin.bias.data.clone(), requires_grad=False)
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xq = fake_quant(x, mbs=self.act_mbs, oas=self.oas, ocp=self.ocp)
        return F.linear(xq, self.weight, self.bias)


def _quantizable(name: str, module: nn.Module) -> bool:
    if not isinstance(module, nn.Linear):
        return False
    # Skip the output head (not one of the paper's compute-bound body GEMMs).
    if name.split(".")[-1] == "lm_head" or name == "lm_head":
        return False
    # Only quantize when both dims allow exact 1x128 / 1x16 blocking.
    return module.in_features % 128 == 0


def quantize_model_(model: nn.Module, method: str) -> int:
    """In-place: replace eligible nn.Linear with QuantLinear per `method`. Returns count."""
    cfg = METHODS[method]
    targets = [(n, m) for n, m in model.named_modules() if _quantizable(n, m)]
    for name, lin in targets:
        parent = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
        child = name.rsplit(".", 1)[-1]
        ql = QuantLinear(lin, weight_mbs=cfg["weight_mbs"],
                         act_mbs=cfg["act_mbs"], oas=cfg["oas"],
                         ocp=cfg.get("ocp", False))
        ql = ql.to(next(lin.parameters()).device)
        setattr(parent, child, ql)
    return len(targets)
