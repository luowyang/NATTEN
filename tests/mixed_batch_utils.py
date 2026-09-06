"""Capability fixtures and a CPU reference independent of NATTEN kernels."""

import itertools
import math
from dataclasses import asdict, dataclass
from typing import Optional, Tuple

import torch

from . import varlen_numerics
from .varlen_numerics import (
    exact_interval,
    grouped_head_prediction,
    product_interval,
    rank_one_match,
    rank_one_scalar,
    reference_interval,
    single_key_scalar_bound,
)


@dataclass(frozen=True)
class MixedCase:
    name: str
    ability: str
    shapes: Tuple[Tuple[int, ...], ...]
    kernel: Tuple[int, ...]
    causal: Tuple[bool, ...]
    dtype: str = "float32"
    stride: Tuple[int, ...] = ()
    dilation: Tuple[int, ...] = ()
    heads: int = 2
    kv_heads: int = 2
    dim: int = 16
    vdim: int = 16
    scale: Optional[float] = None
    seed: int = 314159
    focus: Optional[Tuple[int, int]] = None
    pattern: str = "random"
    noncontiguous: bool = False
    deterministic: bool = True

    @property
    def rank(self):
        return len(self.kernel)

    @property
    def strides(self):
        return self.stride or (1,) * self.rank

    @property
    def dilations(self):
        return self.dilation or (1,) * self.rank

    @property
    def lengths(self):
        return tuple(math.prod(s) for s in self.shapes)

    @property
    def offsets(self):
        return tuple(itertools.accumulate((0,) + self.lengths))

    def descriptor(self):
        return asdict(self)


def capability_cases():
    C = MixedCase
    cases = [
        C(
            "uniform-videos",
            "uniform dispatch",
            ((7, 3, 4),) * 2,
            (3, 3, 3),
            (True, False, False),
        ),
        C(
            "image-video-3d",
            "heterogeneous image/video",
            ((1, 3, 4), (7, 3, 4), (3, 4, 3)),
            (5, 3, 3),
            (True, False, False),
            dtype="bfloat16",
        ),
        C(
            "image-video-spatial",
            "spatial attention",
            ((1, 3, 4), (7, 4, 3)),
            (1, 3, 3),
            (False, False, False),
        ),
        C(
            "image-video-temporal",
            "temporal attention",
            ((1, 3, 4), (7, 4, 3)),
            (5, 1, 1),
            (True, False, False),
            dtype="bfloat16",
            focus=(0, 5),
        ),
        C(
            "thin-height",
            "implicit H=1",
            ((3, 1, 4), (5, 3, 4)),
            (3, 3, 3),
            (True, False, False),
        ),
        C(
            "thin-width",
            "implicit W=1",
            ((3, 4, 1), (5, 4, 3)),
            (3, 3, 3),
            (True, False, False),
            dtype="float16",
        ),
        C(
            "all-degenerate-axis-patterns",
            "mixed rank reductions",
            (
                (1, 1, 1),
                (1, 1, 5),
                (1, 4, 1),
                (3, 1, 1),
                (1, 4, 5),
                (3, 1, 5),
                (3, 4, 1),
                (3, 4, 5),
            ),
            (3, 3, 3),
            (True, False, False),
            dtype="bfloat16",
        ),
        C(
            "explicit-middle-axis",
            "permutation lowering",
            ((3, 4, 5), (4, 3, 5)),
            (3, 1, 3),
            (False, False, False),
        ),
        C(
            "explicit-last-axis",
            "permutation lowering",
            ((3, 4, 5), (4, 3, 5)),
            (3, 3, 1),
            (False, False, False),
        ),
        C(
            "all-kernel-one-gqa",
            "identity with grouped heads",
            ((1, 2, 3), (3, 2, 1)),
            (1, 1, 1),
            (True, False, True),
            dtype="bfloat16",
            heads=4,
            kv_heads=1,
            focus=(1, 2),
        ),
        C(
            "clamp-below-at-above",
            "effective kernel",
            ((1,), (3,), (5,), (9,)),
            (5,),
            (False,),
        ),
        C(
            "clamp-even-2d",
            "effective even kernel",
            ((2, 3), (4, 4), (5, 7)),
            (4, 4),
            (False, False),
            dtype="float16",
        ),
        C(
            "stride-causal-tail",
            "strided causal boundary",
            ((8,), (13,)),
            (4,),
            (True,),
            stride=(2,),
            focus=(1, 4),
        ),
        C(
            "stride-even-boundary",
            "noncausal shifted windows",
            ((7, 9), (9, 7)),
            (4, 4),
            (False, False),
            stride=(3, 2),
            focus=(0, 0),
        ),
        C(
            "dilation-residue",
            "dilated reachable keys",
            ((7,), (11,)),
            (3,),
            (True,),
            dilation=(2,),
            focus=(0, 4),
        ),
        C(
            "dilation-2d",
            "independent residue classes",
            ((7, 9), (9, 7)),
            (3, 3),
            (False, False),
            dilation=(2, 2),
            dtype="bfloat16",
            focus=(1, 30),
        ),
        C(
            "causal-spatial-axis",
            "axis-specific causal mask",
            ((4, 5), (5, 4)),
            (3, 3),
            (False, True),
            focus=(0, 7),
        ),
        C(
            "causal-multiple-axes",
            "multi-axis reachable region",
            ((3, 4, 5), (4, 3, 5)),
            (3, 3, 3),
            (True, True, False),
            focus=(0, 26),
        ),
        C(
            "causal-middle-video",
            "future frame isolation",
            ((7, 3, 4), (9, 4, 3)),
            (5, 3, 3),
            (True, False, False),
            dtype="bfloat16",
            focus=(0, 30),
        ),
        C(
            "causal-final-query",
            "past local-window boundary",
            ((9,), (13,)),
            (5,),
            (True,),
            focus=(0, 8),
        ),
        C(
            "tile-seams",
            "short and partial tiles",
            ((31,), (32,), (33,), (65,)),
            (5,),
            (True,),
            dtype="float16",
            focus=(2, 1),
        ),
        C(
            "empty-each-axis",
            "zero-token no-op",
            ((0, 3, 4), (1, 3, 4), (3, 0, 4), (3, 3, 4), (3, 4, 0)),
            (3, 3, 3),
            (True, False, False),
        ),
        C(
            "all-empty",
            "empty forward and backward",
            ((0, 3, 4), (3, 0, 4), (3, 4, 0)),
            (3, 3, 3),
            (True, False, False),
        ),
        C(
            "gqa-vdim",
            "KV head gradient accumulation",
            ((7,), (11,)),
            (5,),
            (True,),
            dtype="bfloat16",
            heads=4,
            kv_heads=2,
            dim=32,
            vdim=16,
        ),
        C(
            "mqa-noncontiguous",
            "strides and shared KV heads",
            ((4, 5), (5, 4)),
            (3, 3),
            (False, False),
            dtype="float16",
            heads=4,
            kv_heads=1,
            vdim=32,
            noncontiguous=True,
        ),
        C(
            "custom-scale",
            "explicit attention scale",
            ((7,), (11,)),
            (5,),
            (False,),
            scale=0.125,
        ),
        C(
            "zero-logits",
            "known uniform weights",
            ((7,), (11,)),
            (4,),
            (False,),
            pattern="zero_qk",
        ),
        C(
            "constant-values",
            "output independent of scores",
            ((7,), (11,)),
            (5,),
            (True,),
            dtype="bfloat16",
            pattern="constant_v",
        ),
        C(
            "saturated-logits",
            "stable softmax",
            ((9,), (13,)),
            (5,),
            (True,),
            dtype="float16",
            pattern="saturated",
            focus=(0, 3),
        ),
        C(
            "fp16-product-overflow",
            "finite-input gradient overflow",
            ((9,), (13,)),
            (5,),
            (True,),
            dtype="float16",
            pattern="overflow",
            focus=(0, 0),
        ),
        C(
            "fp16-product-underflow",
            "tiny representable dot product",
            ((9,), (13,)),
            (5,),
            (True,),
            dtype="float16",
            pattern="underflow",
            focus=(0, 0),
        ),
        C(
            "signed-cancellation",
            "delta cancellation",
            ((9,), (13,)),
            (5,),
            (True,),
            dtype="bfloat16",
            pattern="cancellation",
            focus=(0, 0),
        ),
    ]
    cases.extend(
        (
            C(
                "large-qk-head",
                "multiple QK reduction iterations",
                ((33,), (65,)),
                (5,),
                (True,),
                dtype="bfloat16",
                dim=256,
                vdim=128,
            ),
            C(
                "large-value-head",
                "multiple value output iterations",
                ((33,), (65,)),
                (5,),
                (True,),
                dtype="float16",
                dim=64,
                vdim=256,
            ),
            C(
                "head-vector-tail",
                "non-power-of-two head dimensions",
                ((7,), (11,)),
                (5,),
                (False,),
                dtype="bfloat16",
                dim=80,
                vdim=48,
            ),
        )
    )
    for dtype in ("float32", "float16", "bfloat16"):
        cases.append(
            C(
                "single-key-video-" + dtype,
                "singleton inside video",
                ((9,), (13,)),
                (5,),
                (True,),
                dtype=dtype,
                heads=8,
                kv_heads=8,
                dim=64,
                vdim=64,
                seed=271828,
                focus=(0, 0),
            )
        )
        cases.append(
            C(
                "single-token-mixed-" + dtype,
                "singleton document",
                ((1,), (9,)),
                (5,),
                (True,),
                dtype=dtype,
                focus=(0, 0),
            )
        )
    return tuple(cases)


def coordinate_mask(case):
    """The pack's neighborhood mask, block diagonal by construction: attention
    never crosses a document, so each block is that document's own
    `varlen_numerics.document_mask`.
    """
    mask = torch.zeros((sum(case.lengths),) * 2, dtype=torch.bool)
    for shape, length, offset in zip(case.shapes, case.lengths, case.offsets):
        block = slice(offset, offset + length)
        mask[block, block] = varlen_numerics.document_mask(
            shape, case.kernel, case.strides, case.dilations, case.causal
        )
    return mask


def make_inputs(case, device="cpu"):
    gen = torch.Generator().manual_seed(case.seed)
    n = sum(case.lengths)
    dtype = getattr(torch, case.dtype)
    q = torch.randn(n, case.heads, case.dim, generator=gen)
    k = torch.randn(n, case.kv_heads, case.dim, generator=gen)
    v = torch.randn(n, case.kv_heads, case.vdim, generator=gen)
    grad = torch.randn(n, case.heads, case.vdim, generator=gen)
    if case.pattern == "zero_qk":
        q.zero_()
        k.zero_()
    elif case.pattern == "constant_v":
        v.fill_(1)
    elif case.pattern == "saturated":
        q.mul_(16)
        k.mul_(16)
    elif case.pattern == "overflow":
        v.fill_(256)
        grad.fill_(256)
    elif case.pattern == "underflow":
        v.fill_(2**-14)
        grad.fill_(2**-14)
    elif case.pattern == "cancellation":
        v.fill_(1)
        v[..., 1::2] = -1
        grad.fill_(1)
    if case.focus is not None:
        row = case.offsets[case.focus[0]] + case.focus[1]
        selected = grad[row].clone()
        grad.zero_()
        grad[row] = selected
    data = [x.to(device=device, dtype=dtype) for x in (q, k, v)]
    if case.noncontiguous:
        data = [torch.stack((x, x), dim=-1)[..., 0] for x in data]
    return data, grad.to(device=device, dtype=dtype)


def case_scale(case):
    return case.scale if case.scale is not None else case.dim**-0.5


def dense_attention(data, case, mask=None):
    """`varlen_numerics.dense_attention` over a whole pack."""
    if mask is None:
        mask = coordinate_mask(case)
    return varlen_numerics.dense_attention(*data, mask, case_scale(case))


def reference(data, grad, case, dtype=torch.float64, mask=None):
    """The pack's float64 CPU reference: `varlen_numerics.dense_reference` on the
    block-diagonal mask."""
    if mask is None:
        mask = coordinate_mask(case)
    return varlen_numerics.dense_reference(
        *data, grad, mask, scale=case_scale(case), dtype=dtype, device="cpu"
    )


def error_metrics(actual, expected):
    a, b = actual.detach().cpu().double(), expected.detach().cpu().double()
    nonfinite = int((~torch.isfinite(a)).sum())
    if a.shape != b.shape:
        return {"shape_mismatch": [list(a.shape), list(b.shape)], "exact": False}
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    diff = a - b
    return {
        "exact": torch.equal(a, b),
        "nonfinite": nonfinite,
        "max_abs": float(diff.abs().max())
        if a.numel() and finite
        else (0.0 if not a.numel() else None),
        "rms": float(diff.square().mean().sqrt())
        if a.numel() and finite
        else (0.0 if not a.numel() else None),
        "reference_rms": (
            float(b.square().mean().sqrt())
            if b.numel() and bool(torch.isfinite(b).all())
            else (0.0 if not b.numel() else None)
        ),
    }


def zero_stats(value):
    return {
        "nonzero": int(torch.count_nonzero(value)),
        "nonfinite": int((~torch.isfinite(value)).sum()),
    }


def effective_kernel(case, shape):
    """`varlen_numerics.effective_kernel` for one document of this case."""
    return varlen_numerics.effective_kernel(case.kernel, case.dilations, shape)


def pack_takes_identity_path(case):
    """True when the whole call is answered by the lowering's Python identity path.

    ``maybe_lower_degenerate_axes`` clamps kernel_size against the extent the
    token-carrying documents share (``varlen_numerics.pack_uniform_shape``); when
    they disagree it judges the pack on the caller's own kernel_size and leaves each
    document's narrowing to the CUDA kernel. Either way the identity path is taken
    exactly when every axis of the resulting effective kernel is 1, and that is the
    only case with no attention kernel launch at all. A pack whose documents are all
    empty never reaches it: the lowering hands that one to the generic all-empty fast
    path first.
    """
    if not any(math.prod(shape) for shape in case.shapes):
        return False
    uniform_shape = varlen_numerics.pack_uniform_shape(case.shapes)
    resulting = (
        varlen_numerics.effective_kernel(case.kernel, case.dilations, uniform_shape)
        if uniform_shape is not None
        else tuple(case.kernel)
    )
    return all(k == 1 for k in resulting)


def lowering_differs_from_pack(case, shape):
    """True when this document lowers differently inside this pack than it does
    isolated."""
    return varlen_numerics.lowering_differs_from_pack(
        case.kernel, case.dilations, case.shapes, shape
    )


def document_comparison_rule(case, shape):
    """How one document's packed result is compared with its isolated call."""
    return varlen_numerics.document_comparison_rule(
        case.kernel, case.dilations, case.shapes, shape
    )


def tf32_toward_zero(value):
    """CUTLASS NumericConverter<tfloat32_t, float, round_toward_zero>."""
    bits = value.to(torch.float32).contiguous().view(torch.int32)
    return (bits & -8192).view(torch.float32)


def tf32_half_ulp_truncate(value):
    """tfloat32_t::round_half_ulp_truncate plus the mask applied on read-back."""
    bits = value.to(torch.float32).contiguous().view(torch.int32)
    return ((bits + 4096) & -8192).view(torch.float32)


def tf32_split(value):
    """NumericConverterFastF32's (big, small) pair from mma_tensor_op_fast_f32.h."""
    value = value.to(torch.float32)
    big = tf32_toward_zero(value)
    return big, tf32_half_ulp_truncate(value - big)


def matmul_3xtf32(left, right):
    """One OpMultiplyAddFastF32 GEMM: three TF32 products summed exactly, then fp32."""
    left_big, left_small = tf32_split(left)
    right_big, right_small = tf32_split(right)
    return (
        left_small.double() @ right_big.double()
        + left_big.double() @ right_small.double()
        + left_big.double() @ right_big.double()
    ).float()


def dense_backward_3xtf32(data, grad, case, mask=None):
    """Dense reference backward with every GEMM routed through 3xTF32 arithmetic.

    Reproduces what the FP32 kernels do internally, so the directional-derivative
    interval can carry the backward's own arithmetic error as a measured term.
    """
    repeats = case.heads // case.kv_heads
    q, k, v = (x.detach().to(device="cpu", dtype=torch.float32) for x in data)
    k = k.repeat_interleave(repeats, dim=1)
    v = v.repeat_interleave(repeats, dim=1)
    upstream = grad.detach().to(device="cpu", dtype=torch.float32)
    grads = {
        "dq": torch.zeros_like(q),
        "dk": torch.zeros_like(k),
        "dv": torch.zeros_like(v),
    }
    if q.shape[0]:
        mask = coordinate_mask(case) if mask is None else mask.cpu()
        scale = float(case.scale if case.scale is not None else case.dim**-0.5)
        blocked = torch.zeros(mask.shape, dtype=torch.float32)
        for head in range(case.heads):
            scores = matmul_3xtf32(q[:, head], k[:, head].t()) * scale
            scores = torch.where(mask, scores, torch.full_like(scores, -torch.inf))
            weights = torch.where(
                mask, (scores - scores.amax(1, keepdim=True)).exp(), blocked
            )
            weights = weights / weights.sum(1, keepdim=True)
            out = matmul_3xtf32(weights, v[:, head])
            delta = (upstream[:, head] * out).sum(1, keepdim=True)
            dp = matmul_3xtf32(upstream[:, head], v[:, head].t())
            ds = torch.where(mask, (dp - delta) * weights * scale, blocked)
            grads["dv"][:, head] = matmul_3xtf32(weights.t(), upstream[:, head])
            grads["dq"][:, head] = matmul_3xtf32(ds, k[:, head])
            grads["dk"][:, head] = matmul_3xtf32(ds.t(), q[:, head])
    for name in ("dk", "dv"):
        value = grads[name]
        grads[name] = value.reshape(value.shape[0], case.kv_heads, repeats, -1).sum(2)
    return grads


def singleton_kernel_checks(result, data, grad, case, row, key):
    """Single-key row computed inside the kernel, i.e. not a fully degenerate document.

    O = V and dV = dO hold bitwise only for fp16/bf16; the FP32 GEMMs run 3xTF32, so
    1 * V costs up to 2^-21 relative. dQ/dK are the rank-1 image of one scalar per
    head -- the residual of the same dot product taken over two different summation
    paths -- bounded by `single_key_scalar_bound`.
    """
    dtype = getattr(torch, case.dtype)
    repeats = case.heads // case.kv_heads
    q, k, v = (x.detach().cpu() for x in data)
    upstream = grad.detach().cpu()
    scale = float(case.scale if case.scale is not None else case.dim**-0.5)
    interval = product_interval if dtype == torch.float32 else exact_interval
    expected_out = v[key].repeat_interleave(repeats, dim=0)
    expected_dv = torch.zeros_like(v, dtype=result["dv"].dtype)
    expected_dv[key] = (
        upstream[row]
        .to(result["dv"].dtype)
        .reshape(case.kv_heads, repeats, case.vdim)
        .sum(1)
    )
    dv_magnitude = torch.zeros_like(expected_dv, dtype=torch.float64)
    dv_magnitude[key] = (
        upstream[row].double().abs().reshape(case.kv_heads, repeats, case.vdim).sum(1)
    )
    other_queries = torch.ones(result["dq"].shape[0], dtype=torch.bool)
    other_queries[row] = False
    other_keys = torch.ones(result["dk"].shape[0], dtype=torch.bool)
    other_keys[key] = False

    heads = []
    for head in range(case.heads):
        kv_head = head // repeats
        bound = single_key_scalar_bound(upstream[row, head], v[key, kv_head], dtype)
        fitted, stored = rank_one_scalar(result["dq"][row, head], k[key, kv_head], dtype)
        heads.append(
            {
                "head": head,
                "least_squares": fitted,
                "coefficient": stored,
                "coefficient_over_scale": stored / scale,
                "bound_over_scale": bound,
                "within_bound": abs(stored / scale) <= bound,
                "reconstruction": rank_one_match(
                    result["dq"][row, head], k[key, kv_head], stored, dtype
                ),
            }
        )
    dq_check = {
        "pass": bool(
            not int(torch.count_nonzero(result["dq"][other_queries]))
            and all(h["within_bound"] and h["reconstruction"]["pass"] for h in heads)
        ),
        "other_rows_nonzero": int(torch.count_nonzero(result["dq"][other_queries])),
        "nonfinite": int((~torch.isfinite(result["dq"])).sum()),
        "focus_row": row,
        "heads": heads,
    }

    if case.heads == case.kv_heads:
        key_heads = []
        for head in range(case.heads):
            bound = heads[head]["bound_over_scale"]
            fitted, stored = rank_one_scalar(
                result["dk"][key, head], q[row, head], dtype
            )
            key_heads.append(
                {
                    "head": head,
                    "least_squares": fitted,
                    "coefficient": stored,
                    "coefficient_over_scale": stored / scale,
                    "bound_over_scale": bound,
                    "within_bound": abs(stored / scale) <= bound,
                    "agrees_with_dq": stored == heads[head]["coefficient"],
                    "reconstruction": rank_one_match(
                        result["dk"][key, head], q[row, head], stored, dtype
                    ),
                }
            )
        dk_reconstruction = {
            "mode": "per-head",
            "pass": all(
                h["within_bound"] and h["reconstruction"]["pass"] for h in key_heads
            ),
            "heads": key_heads,
        }
    else:
        # Python repeats K/V to the query head count, so dK arrives back as a sum of
        # per-Q-head rounded products; only that sum is observable here.
        predicted, radius = grouped_head_prediction(
            [h["coefficient"] for h in heads], q[row], repeats, dtype
        )
        difference = (result["dk"][key].double() - predicted).abs()
        dk_reconstruction = {
            "mode": "grouped-heads",
            "pass": bool((difference <= radius).all()),
            "outside": int((difference > radius).sum()),
            "max_abs": float(difference.max()) if difference.numel() else 0.0,
            "within_bound": all(h["within_bound"] for h in heads),
        }
        dk_reconstruction["pass"] = (
            dk_reconstruction["pass"] and dk_reconstruction["within_bound"]
        )
    dk_check = {
        "pass": bool(
            not int(torch.count_nonzero(result["dk"][other_keys]))
            and dk_reconstruction["pass"]
        ),
        "other_rows_nonzero": int(torch.count_nonzero(result["dk"][other_keys])),
        "nonfinite": int((~torch.isfinite(result["dk"])).sum()),
        "key_row": key,
        "reconstruction": dk_reconstruction,
    }
    return {
        "singleton_out_is_v": interval(result["out"][row], expected_out),
        "singleton_dq_zero": dq_check,
        "singleton_dk_zero": dk_check,
        "singleton_dv_route": (
            product_interval(result["dv"], expected_dv, magnitude=dv_magnitude)
            if dtype == torch.float32
            else exact_interval(result["dv"], expected_dv)
        ),
        "singleton_row": {"row": row, "key": key, "path": "cuda-kernel"},
    }


def singleton_identity_checks(result, data, grad, case, row, key):
    """Fully degenerate document: the Python identity path, bitwise on every dtype."""
    repeats = case.heads // case.kv_heads
    v = data[2].detach().cpu()
    expected_out = v[key].repeat_interleave(repeats, dim=0)
    expected_dv = torch.zeros_like(v, dtype=result["dv"].dtype)
    expected_dv[key] = (
        grad[row]
        .detach()
        .to(device="cpu", dtype=result["dv"].dtype)
        .reshape(case.kv_heads, repeats, case.vdim)
        .sum(1)
    )
    return {
        "singleton_out_is_v": {"exact": torch.equal(result["out"][row], expected_out)},
        "singleton_dq_zero": zero_stats(result["dq"]),
        "singleton_dk_zero": zero_stats(result["dk"]),
        "singleton_dv_route": {"exact": torch.equal(result["dv"], expected_dv)},
        "singleton_row": {"row": row, "key": key, "path": "python-identity"},
    }


def single_key_document_checks(packed, isolated, data, grad, case, lo, hi):
    """A fully degenerate document that a heterogeneous pack computes in the kernel.

    Every row is its own single visible key, so the isolated call takes the Python
    identity path and is exact -- which makes `reference_interval` collapse to one
    ulp and demand an accuracy the kernel's 1 * V product does not have. The rows
    are held to the same gate `singleton_kernel_checks` applies to a focus row
    instead: O and dV bitwise for fp16/bf16 and inside the 3xTF32 product interval
    for float32, dQ and dK the rank-1 image of one scalar per row and head under
    `single_key_scalar_bound`.

    Such a row's logsumexp is `scale * (q . k)` on its one key, which both sides
    accumulate in float32 over different summation orders (and, for float32 inputs,
    with 3xTF32 rather than IEEE products), so it takes `single_key_scalar_bound` on
    that dot product instead, scaled.
    """
    dtype = getattr(torch, case.dtype)
    repeats = case.heads // case.kv_heads
    q, k, v = (x.detach().cpu()[lo:hi] for x in data)
    upstream = grad.detach().cpu()[lo:hi]
    scale = float(case.scale if case.scale is not None else case.dim**-0.5)
    tokens = hi - lo
    slices = {name: value[lo:hi] for name, value in packed.items()}
    isolated_slices = {name: value[lo:hi] for name, value in isolated.items()}

    if dtype == torch.float32:
        magnitude = (
            upstream.double()
            .abs()
            .reshape(tokens, case.kv_heads, repeats, case.vdim)
            .sum(2)
        )
        out_check = product_interval(slices["out"], isolated_slices["out"])
        dv_check = product_interval(
            slices["dv"], isolated_slices["dv"], magnitude=magnitude
        )
    else:
        out_check = exact_interval(slices["out"], isolated_slices["out"])
        dv_check = exact_interval(slices["dv"], isolated_slices["dv"])

    fits = {"dq": [], "dk": []}
    score_bound = torch.zeros(tokens, case.heads, dtype=torch.float64)
    for row in range(tokens):
        for head in range(case.heads):
            kv_head = head // repeats
            bound = single_key_scalar_bound(upstream[row, head], v[row, kv_head], dtype)
            score_bound[row, head] = single_key_scalar_bound(
                q[row, head], k[row, kv_head], dtype
            ) * abs(scale)
            fitted, stored = rank_one_scalar(
                slices["dq"][row, head], k[row, kv_head], dtype
            )
            fits["dq"].append(
                {
                    "row": row,
                    "head": head,
                    "coefficient_over_scale": stored / scale,
                    "bound_over_scale": bound,
                    "within_bound": abs(stored / scale) <= bound,
                    "reconstruction": rank_one_match(
                        slices["dq"][row, head], k[row, kv_head], stored, dtype
                    ),
                }
            )
            if case.heads == case.kv_heads:
                fitted, stored = rank_one_scalar(
                    slices["dk"][row, head], q[row, head], dtype
                )
                fits["dk"].append(
                    {
                        "row": row,
                        "head": head,
                        "coefficient_over_scale": stored / scale,
                        "bound_over_scale": bound,
                        "within_bound": abs(stored / scale) <= bound,
                        "reconstruction": rank_one_match(
                            slices["dk"][row, head], q[row, head], stored, dtype
                        ),
                    }
                )

    gradient_checks = {}
    for name, entries in fits.items():
        if not entries:
            continue
        outside = [
            {
                key: e[key]
                for key in ("row", "head", "coefficient_over_scale", "bound_over_scale")
            }
            for e in entries
            if not (e["within_bound"] and e["reconstruction"]["pass"])
        ]
        ratios = [
            abs(e["coefficient_over_scale"]) / e["bound_over_scale"]
            for e in entries
            if e["bound_over_scale"]
        ]
        gradient_checks[name] = {
            "pass": not outside,
            "mode": "per-head-rank-one",
            "fits": len(entries),
            "outside": len(outside),
            "first_outside": outside[:8],
            "max_bound_occupancy": max(ratios) if ratios else 0.0,
            "reconstruction_bitwise": sum(
                bool(e["reconstruction"].get("bitwise")) for e in entries
            ),
        }
    if "dk" not in gradient_checks:
        # GQA: Python sums the kernel's per-Q-head dK, so only that sum is visible.
        coefficients = [[] for _ in range(tokens)]
        for entry in fits["dq"]:
            coefficients[entry["row"]].append(entry["coefficient_over_scale"] * scale)
        rows = [
            grouped_head_prediction(coefficients[row], q[row], repeats, dtype)
            for row in range(tokens)
        ]
        predicted = torch.stack([row[0] for row in rows])
        radius = torch.stack([row[1] for row in rows])
        difference = (slices["dk"].double() - predicted).abs()
        occupied = torch.where(
            radius > 0, difference / radius, torch.zeros_like(radius)
        )
        gradient_checks["dk"] = {
            "pass": bool((difference <= radius).all()),
            "mode": "grouped-heads",
            "outside": int((difference > radius).sum()),
            "max_abs": float(difference.max()) if difference.numel() else 0.0,
            "max_bound_occupancy": float(occupied.max()) if occupied.numel() else 0.0,
        }
    lse_difference = (slices["lse"].double() - isolated_slices["lse"].double()).abs()
    occupancy = torch.where(
        score_bound > 0, lse_difference / score_bound, torch.zeros_like(score_bound)
    )
    return {
        "out": out_check,
        "lse": {
            "pass": bool((lse_difference <= score_bound).all()),
            "outside": int((lse_difference > score_bound).sum()),
            "max_abs": float(lse_difference.max()) if lse_difference.numel() else 0.0,
            "max_bound_occupancy": float(occupancy.max()) if occupancy.numel() else 0.0,
        },
        "dv": dv_check,
        "dq": gradient_checks["dq"],
        "dk": gradient_checks["dk"],
    }


def document_adjudication(packed, isolated, ref, data, grad, case):
    """Packed-vs-isolated verdict per document under the single-launch dispatch.

    A pack runs in one kernel launch, so a document whose isolated call lowers to a
    different kernel family is not required to agree bitwise with it; every other
    document still is.
    """
    rows = []
    for index, shape in enumerate(case.shapes):
        lo, hi = case.offsets[index], case.offsets[index + 1]
        rule = document_comparison_rule(case, shape)
        row = {
            "document": index,
            "shape": list(shape),
            "effective_kernel": list(effective_kernel(case, shape)),
            "rule": rule,
        }
        if rule == "bitwise":
            row["tensors"] = {
                name: exact_interval(packed[name][lo:hi], isolated[name][lo:hi])
                for name in ("out", "dq", "dk", "dv")
            }
        elif rule == "reference-interval":
            row["tensors"] = {
                name: reference_interval(
                    packed[name][lo:hi], isolated[name][lo:hi], ref[name][lo:hi]
                )
                for name in ("out", "lse", "dq", "dk", "dv")
            }
        else:
            row["tensors"] = single_key_document_checks(
                packed, isolated, data, grad, case, lo, hi
            )
        row["pass"] = all(check["pass"] for check in row["tensors"].values())
        rows.append(row)
    return rows


def structural_checks(result, data, grad, case, mask=None):
    """Checks dependencies using the mathematical mask, not the tested kernel."""
    mask = coordinate_mask(case) if mask is None else mask
    active = grad.detach().cpu().ne(0).any(dim=(1, 2))
    reachable = mask[active].any(dim=0)
    checks = {
        "inactive_query_dq_zero": zero_stats(result["dq"][~active]),
        "unreachable_key_dk_zero": zero_stats(result["dk"][~reachable]),
        "unreachable_key_dv_zero": zero_stats(result["dv"][~reachable]),
    }
    if case.focus is not None:
        row = case.offsets[case.focus[0]] + case.focus[1]
        keys = mask[row].nonzero().flatten()
        if keys.numel() == 1:
            key = int(keys[0])
            singleton = (
                singleton_identity_checks
                if pack_takes_identity_path(case)
                else singleton_kernel_checks
            )
            checks.update(singleton(result, data, grad, case, row, key))
    if case.pattern in ("zero_qk", "constant_v", "cancellation"):
        checks["known_zero_dq"] = zero_stats(result["dq"])
        checks["known_zero_dk"] = zero_stats(result["dk"])
    return checks


def checks_pass(checks):
    return all(
        v.get("pass", True)
        and v.get("exact", True)
        and v.get("nonzero", 0) == 0
        and v.get("nonfinite", 0) == 0
        for v in checks.values()
    )
