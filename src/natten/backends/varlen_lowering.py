#################################################################################################
# Copyright (c) 2022 - 2026 Ali Hassani.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
#################################################################################################
"""Lowers ``kernel_size == 1`` axes of a varlen call before it ever reaches a
CUDA kernel: such an axis mixes nothing (each query attends only to tokens
sharing its coordinate on that axis), so it is either folded away (a pure
view, for a leading run of such axes) or permuted away (a gather/scatter,
for the rest) in Python, never lowered as a real kernel dimension. Called
from ``natten.backends.varlen_fna._neighborhood_attention_varlen_generic``
right after that function's own argument normalization; see
``maybe_lower_degenerate_axes``'s docstring for the exact contract.

Only this module and ``natten.backends.varlen_fna`` know kernel_size = 1 is
possible for varlen calls -- the CUDA kernels and the fixed (non-varlen)
family are unchanged and still reject it (``check_kernel_size_arg``'s
``allow_ones`` stays ``False`` for every other caller).
"""

from typing import Callable, cast, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.autograd import Function

from natten.backends.configs.checks import check_cutlass_fna_device_compatibility
from natten.types import CausalArgType, DimensionType, NoneType
from natten.utils.device import get_device_cc
from natten.varlen import VarlenLayout

# A varlen call's own entry-point signature (na_dim, query, key, value,
# layout, kernel_size, stride, dilation, is_causal, scale, backend,
# q_tile_shape, kv_tile_shape, backward_q_tile_shape, backward_kv_tile_shape,
# backward_kv_splits, backward_use_pt_reduction, return_lse), all by
# keyword. The lowering functions below call this once the residual call
# has no degenerate axes left, so it may re-enter either the fixed-shape
# uniform-dispatch branch or the ordinary varlen kernel path -- both live
# in the caller (_neighborhood_attention_varlen_generic itself), not here;
# this module never imports it (dependency injection instead), which is
# what keeps this a "small pure module" free of a circular import back to
# varlen_fna.py.
DispatchFn = Callable[..., Union[Tensor, Tuple[Tensor, Tensor]]]

_TILE_KNOB_MESSAGE = (
    "Explicit tile shapes (q_tile_shape, kv_tile_shape, backward_q_tile_shape, "
    "backward_kv_tile_shape) and backward_kv_splits are not supported together "
    "with a kernel_size = 1 axis: lowering folds/permutes the token layout to a "
    "lower rank, and an explicit tile shape's rank would not match it. "
    "Omit them (the resolved rank picks its own defaults), or drop the "
    "kernel_size = 1 axes."
)


def effective_kernel_for_uniform_shape(
    kernel_size: DimensionType, dilation: DimensionType, shape: DimensionType
) -> DimensionType:
    """Host-side form of the per-document effective-kernel clamp
    (``min(kernel_size, extent)`` on every ``dilation == 1`` axis), valid
    when every document shares ``shape`` -- the same rule
    ``_neighborhood_attention_varlen_generic``'s uniform-dispatch branch
    and the CUDA kernel's own per-document clamp both apply, so calling it
    here first (to detect axes that clamp down to 1) is consistent with
    dispatching the residual call to the fixed-shape kernels either
    directly (already kernel_size >= 2 everywhere) or after lowering.

    An axis with ``dilation > 1`` is never clamped -- ``shape`` must still
    fit ``kernel_size * dilation`` there, exactly as the varlen fit check
    (``_build_varlen_fna_state``) and the fixed-shape uniform-dispatch
    branch (``_neighborhood_attention_varlen_generic``) require; this
    raises the same ``ValueError`` otherwise.
    """
    effective = []
    for kernel_axis, extent, dilation_axis in zip(kernel_size, shape, dilation):
        if dilation_axis > 1:
            if extent < kernel_axis * dilation_axis:
                raise ValueError(
                    "kernel_size * dilation must fit every token layout on any "
                    f"axis with dilation > 1; token_layouts[*]={shape}, "
                    f"dilation={dilation}."
                )
            effective.append(kernel_axis)
        else:
            effective.append(min(kernel_axis, extent))
    return cast(DimensionType, tuple(effective))


class _PermuteTokens(Function):
    """Permutes the token (row) dimension of a ``[tokens, heads, *]`` (or
    ``[tokens, heads]``, for logsumexp) tensor by a precomputed int64 index,
    with a hand-written backward rather than relying on autograd to
    differentiate through ``index_select`` generically: a permutation's
    transpose is its own inverse permutation (no ``index_add``, so no
    nondeterministic atomic-add reduction order), so backward is a second
    ``index_select`` by ``inverse_index`` -- precomputed once alongside
    ``index`` by :meth:`VarlenLayout._permuted`, not rebuilt here.

    The same Function permutes q/k/v into the lowered order (``index=perm``,
    ``inverse_index=inv``) and un-permutes the lowered call's output/lse
    back to the caller's order (``index=inv``, ``inverse_index=perm``) --
    just swapping which of the two mutually-inverse indices plays which
    role.
    """

    @staticmethod
    def forward(ctx, tokens: Tensor, index: Tensor, inverse_index: Tensor) -> Tensor:
        ctx.save_for_backward(inverse_index)
        return tokens.index_select(0, index)

    @staticmethod
    def backward(  # type: ignore[override]
        ctx, grad_output: Tensor
    ) -> Tuple[Tensor, NoneType, NoneType]:
        (inverse_index,) = ctx.saved_tensors
        return grad_output.index_select(0, inverse_index), None, None


def _identity_output(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    scale: Optional[float],
    return_lse: bool,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Every axis degenerate (after folding/permuting away every OTHER
    degenerate axis, whatever remains clamps or starts at kernel_size == 1
    on every axis): each query's only valid window position is itself, so
    output is exactly ``value`` and logsumexp is exactly
    ``scale * (query * key).sum(-1)`` -- no kernel launch, no memo entry.

    GQA (key/value with fewer heads than query) repeats key/value to
    query's head count first, exactly as the kernel's own GQA contract
    does (see ``varlen_cutlass_fna_generic``/``cutlass_fna_generic``).
    query/key/value are tied into BOTH outputs with a zero-weighted
    contribution (``* 0``, the same "differentiable but ignored" idiom
    ``_neighborhood_attention_varlen_generic``'s all-empty fast path uses),
    so requires_grad/.grad parity holds for every tensor regardless of
    which output ``backward()`` is called through -- output alone still
    gives exactly ``dv = grad_out``, ``dq = dk = 0`` (the zero-weighted
    link's local gradient is exactly zero, not merely small).

    Same CUDA/dtype/compute-capability contract as the rest of the varlen
    family, checked explicitly here for the same reason
    ``_neighborhood_attention_varlen_generic``'s all-empty fast path checks
    it: this is a pure tensor-arithmetic terminal (no kernel launch), so
    nothing downstream would otherwise enforce it, and it must not silently
    succeed on an unsupported device/dtype just because kernel_size
    happened to be fully degenerate.
    """
    if query.device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("Variable-length CUTLASS FNA requires a CUDA device.")
    check_cutlass_fna_device_compatibility(query.dtype, get_device_cc(query.device))

    heads = query.shape[-2]
    heads_kv = key.shape[-2]
    if heads != heads_kv:
        repeats = heads // heads_kv
        key_g = torch.repeat_interleave(key, repeats=repeats, dim=-2, output_size=heads)
        value_g = torch.repeat_interleave(
            value, repeats=repeats, dim=-2, output_size=heads
        )
    else:
        key_g = key
        value_g = value

    resolved_scale = scale if scale is not None else query.shape[-1] ** -0.5
    zero_link = (query.sum() + key.sum() + value.sum()) * 0

    output = value_g.clone() + zero_link.to(value_g.dtype)
    if not return_lse:
        return output
    lse = (query.float() * key_g.float()).sum(-1) * resolved_scale + zero_link.to(
        torch.float32
    )
    return output, lse


def maybe_lower_degenerate_axes(
    na_dim: int,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    layout: VarlenLayout,
    kernel_size: DimensionType,
    stride: DimensionType,
    dilation: DimensionType,
    is_causal: CausalArgType,
    scale: Optional[float],
    backend: str,
    q_tile_shape: Optional[DimensionType],
    kv_tile_shape: Optional[DimensionType],
    backward_q_tile_shape: Optional[DimensionType],
    backward_kv_tile_shape: Optional[DimensionType],
    backward_kv_splits: Optional[DimensionType],
    backward_use_pt_reduction: bool,
    return_lse: bool,
    dispatch: DispatchFn,
) -> Optional[Union[Tensor, Tuple[Tensor, Tensor]]]:
    """Returns the lowered call's result, or ``None`` if there is nothing to
    lower (no axis is degenerate) -- the caller proceeds with its own
    (unmodified) kernel_size in that case.

    An axis is degenerate when its kernel_size is 1 -- either the caller's
    own ``kernel_size`` entry, or (only for a uniform layout, where every
    document's extent on an axis is a single known value) the per-axis
    clamp ``effective_kernel_for_uniform_shape`` computes turning a >= 2
    entry into 1. A non-uniform (heterogeneous) layout is never
    axis-clamped here: an individual document narrower than kernel_size on
    some axis remains a *device-side* concern (the CUDA kernel's own
    per-document clamp, unrelated to this Python-level lowering), since a
    single scalar clamp cannot represent per-document extents that differ.

    Explicit tile shapes / backward_kv_splits raise immediately once any
    axis is found degenerate (checked before any lowering happens): folding
    or permuting changes the call's rank, so a caller-supplied tile shape's
    rank would not match.

    An all-empty layout (``total_tokens == 0``) always returns ``None``
    after that tile-shape check: its output/lse never depend on
    kernel_size, so it is left entirely to
    ``_neighborhood_attention_varlen_generic``'s existing all-empty fast
    path (kernel_size-independent, and the only path that still works
    correctly with zero real tokens to view/permute) instead of being
    special-cased again here.

    Lowering itself is at most a two-step pipeline, not a general loop:
    1. Leading fold (view-only, see ``VarlenLayout._folded``): consumes the
       *entire* leading run of degenerate axes in one step, since a
       document's own axes are already row-major (no partial-run
       recursion needed -- the next axis after the run is never
       degenerate, by construction of "run").
    2. Permute (gather/scatter, see ``VarlenLayout._permuted``): consumes
       every axis still degenerate after the fold (necessarily non-leading,
       for the same reason) in one step, moving them to the front of each
       document's own token block and repartitioning into more, smaller
       documents.

    Either step's residual call (rank reduced by the fold, or further
    reduced and reordered by the permute) is guaranteed to have every
    remaining kernel_size >= 2, so the one ``dispatch`` call this function
    makes at the end of whichever step actually ran needs no further
    degenerate-axis handling itself -- it goes straight through
    ``dispatch``'s own normal validation and (for a uniform derived layout)
    fixed-shape dispatch or (otherwise) varlen kernel path.
    """
    effective_kernel = kernel_size
    if layout.is_uniform and layout.total_tokens > 0:
        effective_kernel = effective_kernel_for_uniform_shape(
            kernel_size, dilation, layout.shapes[0]
        )
    degenerate = tuple(index for index, k in enumerate(effective_kernel) if k == 1)
    if not degenerate:
        return None

    # Checked once, unconditionally, right here: fold and permute each
    # build a derived VarlenLayout from this one, and identity computes
    # straight from query/key/value without ever touching layout, so this
    # is the only point common to every lowering path below.
    layout._check_device_pin(query.device)

    if (
        q_tile_shape is not None
        or kv_tile_shape is not None
        or backward_q_tile_shape is not None
        or backward_kv_tile_shape is not None
        or backward_kv_splits is not None
    ):
        raise ValueError(_TILE_KNOB_MESSAGE)

    if layout.total_tokens == 0:
        return None

    kernel_size = effective_kernel
    if len(degenerate) == len(kernel_size):
        return _identity_output(query, key, value, scale, return_lse)

    # -- leading fold: consumes the entire leading degenerate run --
    fold_length = 0
    while fold_length < len(kernel_size) and kernel_size[fold_length] == 1:
        fold_length += 1
    if fold_length > 0:
        layout = layout._folded(fold_length)
        kernel_size = cast(DimensionType, kernel_size[fold_length:])
        stride = cast(DimensionType, stride[fold_length:])
        dilation = cast(DimensionType, dilation[fold_length:])
        is_causal = cast(CausalArgType, is_causal[fold_length:])
        na_dim -= fold_length

    # -- permute: consumes every axis still degenerate (non-leading now) --
    group_axes = tuple(index for index, k in enumerate(kernel_size) if k == 1)
    if not group_axes:
        return dispatch(
            na_dim=na_dim,
            query=query,
            key=key,
            value=value,
            layout=layout,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            is_causal=is_causal,
            scale=scale,
            backend=backend,
            q_tile_shape=None,
            kv_tile_shape=None,
            backward_q_tile_shape=None,
            backward_kv_tile_shape=None,
            backward_kv_splits=None,
            backward_use_pt_reduction=backward_use_pt_reduction,
            return_lse=return_lse,
        )

    keep_axes = tuple(
        index for index in range(len(kernel_size)) if index not in group_axes
    )
    derived_layout, perm, inv = layout._permuted(group_axes, query.device)
    permuted_query = _PermuteTokens.apply(query, perm, inv)
    permuted_key = _PermuteTokens.apply(key, perm, inv)
    permuted_value = _PermuteTokens.apply(value, perm, inv)

    result = dispatch(
        na_dim=len(keep_axes),
        query=permuted_query,
        key=permuted_key,
        value=permuted_value,
        layout=derived_layout,
        kernel_size=tuple(kernel_size[index] for index in keep_axes),
        stride=tuple(stride[index] for index in keep_axes),
        dilation=tuple(dilation[index] for index in keep_axes),
        is_causal=tuple(is_causal[index] for index in keep_axes),
        scale=scale,
        backend=backend,
        q_tile_shape=None,
        kv_tile_shape=None,
        backward_q_tile_shape=None,
        backward_kv_tile_shape=None,
        backward_kv_splits=None,
        backward_use_pt_reduction=backward_use_pt_reduction,
        return_lse=return_lse,
    )
    if return_lse:
        output, lse = result
        return (
            _PermuteTokens.apply(output, inv, perm),
            _PermuteTokens.apply(lse, inv, perm),
        )
    return _PermuteTokens.apply(result, inv, perm)
