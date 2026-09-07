"""The numerical gates a varlen kernel result is judged by.

Plain functions over tensors and geometry tuples. Nothing here imports natten or
knows about a test fixture, so the unittest modules and the mixed-batch bench read
the same arithmetic instead of each carrying their own reading of it.

Comparisons return a record -- ``pass`` plus the counts and the measured worst case
that explain the verdict -- rather than raising, so a unittest can assert on one
field and the bench can write the whole record into its JSONL.

The FP32 kernels' GEMMs are CUTLASS OpMultiplyAddFastF32 (3xTF32), not IEEE FP32,
so a single product costs up to ``TF32_RELATIVE`` of its own magnitude plus
``TF32_ABSOLUTE``; FP16 and BF16 products enter the FP32 accumulator exactly and
cost nothing. That asymmetry is why almost every gate below branches on dtype.
"""

import itertools
import math
from typing import Any, Dict, Tuple

import torch

TF32_RELATIVE = 2.0**-21
TF32_ABSOLUTE = 2.0**-30

_DOCUMENT_MASK_CACHE: Dict[Any, torch.Tensor] = {}


# -- spacing and rounding ----------------------------------------------------


def dtype_spacing(values: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Spacing of ``dtype`` above each element's own magnitude, its smallest
    subnormal at zero. Float64 on the CPU, whatever the input carries.

    The one place a float format's spacing is computed, for the gates below and
    for the bench that reads them. ``torch.nextafter`` gives it directly, and
    stays right where reconstructing the spacing from an exponent is not: a
    subnormal's spacing is the subnormal floor rather than
    ``2 ** (exponent - significand_bits)``.
    """
    magnitude = values.detach().cpu().abs().to(dtype)
    above = torch.nextafter(magnitude, torch.full_like(magnitude, torch.inf))
    return above.double() - magnitude.double()


def dtype_ulp(value, dtype: torch.dtype) -> float:
    """Spacing of ``dtype`` above ``|value|``, its smallest subnormal at zero."""
    scalar = torch.tensor([float(value)], dtype=torch.float64)
    return float(dtype_spacing(scalar, dtype)[0])


def ulp32(value) -> float:
    """Spacing of float32 above ``|value|`` -- the accumulator's own step, which
    is what a reassociation bound is counted in whatever the operands are stored
    as."""
    return dtype_ulp(value, torch.float32)


def dtype_rounding_radius(values: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Half an ulp of ``dtype`` at each element's own magnitude."""
    return dtype_spacing(values, dtype) * 0.5


def head_sum_term_radius(term: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Allowance for one query head's kernel product before Python sums the heads.

    Each term carries the dtype's storage rounding and, for float32, the 3xTF32
    GEMM's own relative error. Building the allowance per term is what keeps a
    cancelling head sum from shrinking it below the error it has to cover.
    """
    radius = dtype_rounding_radius(term, dtype)
    if dtype == torch.float32:
        radius = radius + TF32_RELATIVE * term.abs()
    return radius


# -- bounds ------------------------------------------------------------------


def prefix_sum_bound(left: torch.Tensor, right: torch.Tensor) -> float:
    """Largest sequential prefix sum of ``left * right``, the full dot product
    included."""
    products = left.detach().cpu().double() * right.detach().cpu().double()
    return float(products.cumsum(0).abs().max()) if products.numel() else 0.0


def single_key_scalar_bound(
    upstream_row: torch.Tensor, value_row: torch.Tensor, dtype: torch.dtype
) -> float:
    """Bound on ``|dS / scale|`` for a row whose only visible key is one token.

    Such a row's exact dS is zero, and what the kernel returns instead is the
    residual of a single dot product taken over two paths -- the backward's
    ``dP = dO . V`` GEMM against ``delta = rowsum(dO * O)`` -- so it carries their
    reassociation error: 8 float32 ulps of the largest sequential partial sum. The
    FP32 kernels run both GEMMs in 3xTF32, whose per-product error is proportional
    to that product's own magnitude rather than to the (possibly cancelling) running
    sum, so float32 adds ``TF32_RELATIVE`` of the sum of absolute products.
    """
    bound = 8 * ulp32(prefix_sum_bound(upstream_row, value_row))
    if dtype == torch.float32:
        products = (
            upstream_row.detach().cpu().double() * value_row.detach().cpu().double()
        )
        bound += TF32_RELATIVE * float(products.abs().sum())
    return bound


# -- comparisons -------------------------------------------------------------


def product_interval(
    actual: torch.Tensor,
    expected: torch.Tensor,
    relative: float = TF32_RELATIVE,
    absolute: float = TF32_ABSOLUTE,
    magnitude=None,
) -> Dict[str, Any]:
    """Elementwise 3xTF32 product tolerance, for a quantity the kernel obtains by
    multiplying by one.

    ``magnitude`` is the size the arithmetic error is proportional to, defaulting to
    the expected value itself. Where a grouped-query backward has already summed one
    3xTF32 product per query head, pass the sum of the per-head magnitudes instead:
    each term carries its own error, so a sum that cancels would otherwise buy an
    allowance far below the error it has to cover.
    """
    a = actual.detach().cpu().double()
    b = expected.detach().cpu().double()
    difference = (a - b).abs()
    basis = b.abs() if magnitude is None else magnitude.detach().cpu().double()
    allowed = basis * relative + absolute
    inside = difference <= allowed
    return {
        "pass": bool(torch.isfinite(a).all() and bool(inside.all())),
        "bitwise": torch.equal(a, b),
        "matches": int(inside.sum()),
        "of": int(inside.numel()),
        "outside": int((~inside).sum()),
        "max_abs": float(difference.max()) if difference.numel() else 0.0,
        "relative": relative,
        "absolute": absolute,
        "magnitude_basis": "expected" if magnitude is None else "pre-sum",
    }


def exact_interval(actual: torch.Tensor, expected: torch.Tensor) -> Dict[str, Any]:
    """Bitwise form of ``product_interval``, for the dtypes whose products are
    exact."""
    a = actual.detach().cpu()
    b = expected.detach().cpu()
    inside = a.double().eq(b.double())
    return {
        "pass": bool(torch.isfinite(a.double()).all() and bool(inside.all())),
        "bitwise": bool(inside.all()),
        "matches": int(inside.sum()),
        "of": int(inside.numel()),
        "outside": int((~inside).sum()),
    }


def dtype_interval(
    actual: torch.Tensor, expected: torch.Tensor, dtype: torch.dtype, magnitude=None
) -> Dict[str, Any]:
    """``product_interval`` on float32, ``exact_interval`` on fp16/bf16 -- the gate
    for a product of one, which only float32 computes inexactly."""
    if dtype == torch.float32:
        return product_interval(actual, expected, magnitude=magnitude)
    return exact_interval(actual, expected)


def rank_one_scalar(
    values: torch.Tensor, basis: torch.Tensor, dtype: torch.dtype
) -> Tuple[float, float]:
    """Least-squares scalar in float64, and that scalar stored back in the kernel's
    dtype -- the form the kernel could have held it in."""
    y = values.detach().cpu().double().flatten()
    x = basis.detach().cpu().double().flatten()
    denominator = float(x @ x)
    fitted = float(x @ y) / denominator if denominator else 0.0
    stored = float(torch.tensor([fitted], dtype=torch.float64).to(dtype).double()[0])
    return fitted, stored


def rank_one_match(
    values: torch.Tensor, basis: torch.Tensor, coefficient: float, dtype: torch.dtype
) -> Dict[str, Any]:
    """Whether ``values`` is ``coefficient * basis``: ``round_dtype(c * basis)``
    bitwise for fp16/bf16, the 3xTF32 interval for float32."""
    scalar = torch.tensor([coefficient], dtype=torch.float64)
    expected = scalar * basis.detach().cpu().double().flatten()
    actual = values.detach().cpu().flatten()
    if dtype == torch.float32:
        return {"mode": "3xtf32-interval", **product_interval(actual, expected)}
    return {"mode": "bitwise", **exact_interval(actual, expected.float().to(dtype))}


def grouped_head_prediction(
    coefficients, basis: torch.Tensor, repeats: int, dtype: torch.dtype
) -> Tuple[torch.Tensor, torch.Tensor]:
    """One row's predicted grouped-query dK, and the radius it is allowed.

    Python repeats key/value to the query head count, so the kernel's per-query-head
    rank-1 products arrive back summed per key/value head and only that sum is
    observable. ``basis`` is that row's ``[heads, dim]`` query; the returned pair is
    ``[kv_heads, dim]``.
    """
    kv_heads = len(coefficients) // repeats
    predicted = torch.zeros(kv_heads, basis.shape[-1], dtype=torch.float64)
    radius = torch.zeros_like(predicted)
    for head, coefficient in enumerate(coefficients):
        term = (
            torch.tensor([coefficient], dtype=torch.float64)
            * basis[head].detach().cpu().double()
        )
        predicted[head // repeats] += term
        radius[head // repeats] += head_sum_term_radius(term, dtype)
    radius += dtype_rounding_radius(predicted, dtype) * repeats
    return predicted, radius


def reference_interval(
    packed: torch.Tensor, isolated: torch.Tensor, ref: torch.Tensor
) -> Dict[str, Any]:
    """``max|packed - ref| <= 2 max|isolated - ref| + 1 ulp``, both sides measured
    against the same float64 dense reference.

    For a document whose isolated call lowers to a different kernel family than the
    pack does, neither side is the other's oracle; the isolated call's own distance
    from the reference is the scale a second implementation of the same mathematics
    is allowed, and the ulp is read at the reference magnitude where the packed
    error peaks.
    """
    a = packed.detach().cpu().double()
    b = isolated.detach().cpu().double()
    r = ref.detach().cpu().double()
    packed_error = (a - r).abs()
    isolated_error = (b - r).abs()
    if not packed_error.numel():
        return {"pass": True, "bitwise": True, "packed_max_abs": 0.0}
    position = int(packed_error.flatten().argmax())
    term = dtype_ulp(float(r.flatten()[position]), packed.dtype)
    allowed = 2 * float(isolated_error.max()) + term
    worst = float(packed_error.max())
    return {
        "pass": bool(torch.isfinite(a).all()) and worst <= allowed,
        "bitwise": torch.equal(a, b),
        "packed_max_abs": worst,
        "isolated_max_abs": float(isolated_error.max()),
        "ulp_term": term,
        "allowed": allowed,
    }


# -- neighborhood geometry and the dense reference ---------------------------


def axis_neighbors(index, extent, kernel, stride=1, dilation=1, causal=False):
    """Keys one query at ``index`` sees along one axis of one document.

    The kernel is clamped to the document's own extent on an undilated axis -- the
    clamp the CUDA kernel applies per document, and the one
    ``effective_kernel_for_uniform_shape`` applies host-side for a uniform pack. A
    dilated axis has no such clamp in the public contract, so an extent too short
    for it is a caller error rather than a narrower window.
    """
    if kernel == 1:
        return (index,)
    if dilation == 1:
        kernel = min(kernel, extent)
    elif extent < kernel * dilation:
        raise ValueError("Dilated extent does not fit the public contract")
    residue = index % dilation
    pos = index // dilation
    size = len(range(residue, extent, dilation))
    group = pos // stride
    if causal:
        anchor = min(group * stride + stride - 1, size - 1)
        begin, end = max(0, anchor - kernel + 1), pos + 1
    else:
        anchor = min(group * stride + stride // 2, size - 1)
        begin = min(max(anchor - kernel // 2, 0), size - kernel)
        end = begin + kernel
    return tuple(residue + dilation * k for k in range(begin, end))


def document_mask(shape, kernel, stride, dilation, causal) -> torch.Tensor:
    """``[tokens, tokens]`` bool for one document: is key n in query m's
    neighborhood, from ``axis_neighbors`` on every axis.

    Cached on the geometry, which is all it depends on, and returned to be read
    rather than written.
    """
    memo_key = (
        tuple(shape),
        tuple(kernel),
        tuple(stride),
        tuple(dilation),
        tuple(bool(c) for c in causal),
    )
    mask = _DOCUMENT_MASK_CACHE.get(memo_key)
    if mask is None:
        total = math.prod(shape)
        mask = torch.zeros(total, total, dtype=torch.bool)
        for row, coord in enumerate(itertools.product(*(range(n) for n in shape))):
            axes = [
                axis_neighbors(i, n, k, s, d, c)
                for i, n, k, s, d, c in zip(
                    coord, shape, kernel, stride, dilation, causal
                )
            ]
            for key_coord in itertools.product(*axes):
                column = 0
                for i, n in zip(key_coord, shape):
                    column = column * n + i
                mask[row, column] = True
        _DOCUMENT_MASK_CACHE[memo_key] = mask
    return mask


def dense_attention(query, key, value, mask, scale=None):
    """Dense neighborhood attention over one token block, in whatever dtype the
    inputs carry. Grouped-query aware: key/value heads are repeated up to the query
    head count first, so gradients flow back to the unrepeated tensors.
    """
    repeats = query.shape[-2] // key.shape[-2]
    key = key.repeat_interleave(repeats, dim=-2)
    value = value.repeat_interleave(repeats, dim=-2)
    if not query.shape[0]:
        return (
            value + (query.sum() + key.sum()) * 0,
            query.new_empty((0, query.shape[-2])),
        )
    if scale is None:
        scale = query.shape[-1] ** -0.5
    scores = torch.einsum("ihd,jhd->hij", query, key) * scale
    scores = scores.masked_fill(~mask.to(query.device).unsqueeze(0), -torch.inf)
    out = torch.einsum("hij,jhd->ihd", scores.softmax(-1), value)
    lse = scores.logsumexp(-1).transpose(0, 1)
    return out, lse


def dense_reference(
    query, key, value, gradient, mask, scale=None, dtype=torch.float64, device=None
) -> Dict[str, torch.Tensor]:
    """The answer a kernel result approximates: ``dense_attention`` and its
    gradients, computed in float64 from the same quantized inputs without calling
    NATTEN. ``device`` defaults to the inputs' own.
    """
    leaves = [
        x.detach().to(device=device or x.device, dtype=dtype).requires_grad_(True)
        for x in (query, key, value)
    ]
    out, lse = dense_attention(*leaves, mask, scale)
    upstream = gradient.detach().to(device=out.device, dtype=dtype)
    grads = torch.autograd.grad(out, leaves, upstream)
    return dict(
        zip(
            ("out", "lse", "dq", "dk", "dv"),
            (out.detach(), lse.detach(), *(x.detach() for x in grads)),
        )
    )


# -- how one document's packed result is compared with its isolated call -----


def effective_kernel(kernel, dilation, shape):
    """``min(kernel_size, extent)`` on every dilation-1 axis, kernel_size
    elsewhere."""
    return tuple(
        k if d > 1 else min(k, extent) for k, extent, d in zip(kernel, shape, dilation)
    )


def pack_uniform_shape(shapes):
    """The shape shared by the documents that carry at least one token, or ``None``
    when there is no such document or they disagree.

    ``VarlenLayout.uniform_shape``, the uniformity ``maybe_lower_degenerate_axes``
    reads: a zero-token document is never scheduled, so it neither supplies an extent
    to clamp against nor stops the documents that do carry tokens from agreeing on
    one. ``VarlenLayout.is_uniform`` -- every document, empty ones included -- is the
    stricter property the fixed-shape batched-view dispatch needs instead.
    """
    carrying = [tuple(shape) for shape in shapes if math.prod(shape) > 0]
    if not carrying or any(shape != carrying[0] for shape in carrying):
        return None
    return carrying[0]


def isolated_takes_identity_path(kernel, dilation, shape) -> bool:
    """True when this document called on its own is fully degenerate, so its
    isolated result is the identity path's exact answer rather than a
    measurement."""
    return math.prod(shape) > 0 and all(
        k == 1 for k in effective_kernel(kernel, dilation, shape)
    )


def lowering_differs_from_pack(kernel, dilation, shapes, shape) -> bool:
    """True when this document lowers differently inside this pack than it does
    isolated.

    An extent-1 axis under a kernel_size > 1 is a Python-side fold or permute for
    the isolated call, whose pack of one is uniform by construction. Inside the
    given pack it is the same fold or permute whenever the token-carrying
    documents share a shape (``pack_uniform_shape``): ``maybe_lower_degenerate_axes``
    then applies that shared shape's host-side clamp to the whole pack, and it is
    this document's own clamp. Only where they disagree is no host-side clamp
    defined, leaving the axis to the kernel's own window clamp -- the two runs then
    reach different kernel families on the same mathematics.
    """
    if pack_uniform_shape(shapes) is not None:
        return False
    return any(
        extent == 1 and k > 1 and d == 1
        for extent, k, d in zip(shape, kernel, dilation)
    )


def document_comparison_rule(kernel, dilation, shapes, shape) -> str:
    """``bitwise``, ``reference-interval`` or ``single-key-rows``: how one
    document's packed result is compared with its isolated call.

    ``bitwise`` for an empty document, and for a document whose lowering inside
    this pack does not differ from its isolated call's
    (``lowering_differs_from_pack``) -- which is every document of a pack whose
    token-carrying documents share one shape, whatever the shared clamp reduces
    the kernel to, and not only the packs that reach the identity path. A pack
    on the identity path is one of those: it is either
    uniform, or heterogeneous under a kernel that is already 1 on every axis,
    and neither differs. Otherwise ``single-key-rows`` when the document's
    isolated call is itself fully degenerate
    (``isolated_takes_identity_path``), ``reference-interval`` when it is not:
    a heterogeneous pack leaves this document's narrowing to the CUDA kernel's
    own per-document clamp, which the isolated call does not share.
    """
    if math.prod(shape) == 0:
        return "bitwise"
    if not lowering_differs_from_pack(kernel, dilation, shapes, shape):
        return "bitwise"
    if isolated_takes_identity_path(kernel, dilation, shape):
        return "single-key-rows"
    return "reference-interval"
