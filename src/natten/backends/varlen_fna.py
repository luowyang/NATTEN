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

import functools
import math
from dataclasses import dataclass
from typing import cast, List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor
from torch.amp import custom_bwd, custom_fwd
from torch.autograd import Function

from natten._libnatten import (
    varlen_na1d_backward,
    varlen_na1d_forward,
    varlen_na2d_backward,
    varlen_na2d_forward,
    varlen_na3d_backward,
    varlen_na3d_forward,
)
from natten.backends.configs.checks import check_cutlass_fna_device_compatibility
from natten.backends.configs.cutlass import (
    check_cutlass_fna_backward_config,
    check_cutlass_fna_forward_config,
)
from natten.backends.configs.cutlass.backward_knobs import (
    _get_max_grid_size_allowed,
    get_default_varlen_fna_kv_splits,
    get_max_splits,
)
from natten.context import is_kv_parallelism_in_fused_na_enabled
from natten.types import (
    CausalArgType,
    CausalArgTypeOrDed,
    CutlassFnaBackwardConfigType,
    CutlassFnaForwardConfigType,
    DimensionType,
    DimensionTypeOrDed,
    NoneType,
)
from natten.utils.checks import check_all_args, fmha_tensor_checks
from natten.utils.device import get_device_cc
from natten.varlen import _prefix_offsets, _VARLEN_INT32_MAX, VarlenLayout

amp_fwd = functools.partial(custom_fwd, device_type="cuda")
amp_bwd = functools.partial(custom_bwd, device_type="cuda")


def _tiny_rank_shaped_view(tensor: Tensor, na_dim: int) -> Tensor:
    shape = (1,) + (1,) * na_dim + tuple(tensor.shape[-2:])
    return tensor.narrow(0, 0, 1).view(shape)


def make_varlen_cutlass_fna_autograd_fn(na_dim: int):
    forward_ops = {
        1: varlen_na1d_forward,
        2: varlen_na2d_forward,
        3: varlen_na3d_forward,
    }
    backward_ops = {
        1: varlen_na1d_backward,
        2: varlen_na2d_backward,
        3: varlen_na3d_backward,
    }

    class VarlenCutlassFNAAutogradFn(Function):
        @staticmethod
        @amp_fwd
        def forward(
            ctx,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            cumulative_seqlens: Tensor,
            token_layouts: Tensor,
            backward_kv_splits: Tensor,
            forward_worklist: Tensor,
            backward_worklist: Tensor,
            backward_q_tile_offsets: Tensor,
            backward_kv_split_offsets: Tensor,
            kernel_size: DimensionType,
            stride: DimensionType,
            dilation: DimensionType,
            is_causal: CausalArgType,
            scale: float,
            forward_config: CutlassFnaForwardConfigType,
            backward_config: CutlassFnaBackwardConfigType,
            forward_work_count: int,
            backward_work_count: int,
            total_backward_q_tiles: int,
            backward_use_pt_reduction: bool,
            deterministic: bool,
        ) -> Tuple[Tensor, Tensor]:
            query = query.contiguous()
            key = key.contiguous()
            value = value.contiguous()

            q_tile_shape, kv_tile_shape = forward_config
            output, logsumexp = forward_ops[na_dim](
                query,
                key,
                value,
                cumulative_seqlens,
                token_layouts,
                forward_worklist,
                kernel_size,
                stride,
                dilation,
                is_causal,
                scale,
                q_tile_shape,
                kv_tile_shape,
                forward_work_count,
            )

            ctx.save_for_backward(
                query,
                key,
                value,
                output,
                logsumexp,
                cumulative_seqlens,
                token_layouts,
                backward_kv_splits,
                backward_worklist,
                backward_q_tile_offsets,
                backward_kv_split_offsets,
            )
            ctx.kernel_size = kernel_size
            ctx.stride = stride
            ctx.dilation = dilation
            ctx.is_causal = is_causal
            ctx.scale = scale
            ctx.backward_config = backward_config
            ctx.backward_work_count = backward_work_count
            ctx.total_backward_q_tiles = total_backward_q_tiles
            ctx.backward_use_pt_reduction = backward_use_pt_reduction
            ctx.deterministic = deterministic
            return output, logsumexp

        @staticmethod
        @amp_bwd
        def backward(ctx, grad_out: Tensor, grad_lse: Tensor) -> Tuple[
            Tensor,
            Tensor,
            Tensor,
            NoneType,
            NoneType,
            NoneType,
            NoneType,
            NoneType,
            NoneType,
            NoneType,
            NoneType,
            NoneType,
            NoneType,
            NoneType,
            NoneType,
            NoneType,
            NoneType,
            NoneType,
            NoneType,
            NoneType,
            NoneType,
            NoneType,
        ]:
            (
                query,
                key,
                value,
                output,
                logsumexp,
                cumulative_seqlens,
                token_layouts,
                backward_kv_splits,
                backward_worklist,
                backward_q_tile_offsets,
                backward_kv_split_offsets,
            ) = ctx.saved_tensors
            q_tile_shape, kv_tile_shape = ctx.backward_config
            compute_delta_with_torch = (
                ctx.backward_use_pt_reduction and not ctx.deterministic
            )
            d_query, d_key, d_value = backward_ops[na_dim](
                query,
                key,
                value,
                output,
                grad_out.contiguous(),
                logsumexp,
                cumulative_seqlens,
                token_layouts,
                backward_kv_splits,
                backward_worklist,
                backward_q_tile_offsets,
                backward_kv_split_offsets,
                ctx.kernel_size,
                ctx.stride,
                ctx.dilation,
                ctx.is_causal,
                ctx.scale,
                q_tile_shape,
                kv_tile_shape,
                ctx.backward_work_count,
                ctx.total_backward_q_tiles,
                compute_delta_with_torch,
                ctx.deterministic,
            )
            return (
                d_query,
                d_key,
                d_value,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )

    return VarlenCutlassFNAAutogradFn


VarlenCutlassFNAAutogradFns = {
    na_dim: make_varlen_cutlass_fna_autograd_fn(na_dim) for na_dim in (1, 2, 3)
}


def varlen_cutlass_fna_generic(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    cumulative_seqlens: Tensor,
    token_layouts: Tensor,
    na_dim: int,
    kernel_size: DimensionTypeOrDed,
    forward_worklist: Tensor,
    backward_worklist: Tensor,
    backward_q_tile_offsets: Tensor,
    backward_kv_split_offsets: Tensor,
    backward_kv_splits: Tensor,
    forward_work_count: int,
    backward_work_count: int,
    total_backward_q_tiles: int,
    deterministic: bool,
    stride: DimensionTypeOrDed = 1,
    dilation: DimensionTypeOrDed = 1,
    is_causal: Optional[CausalArgTypeOrDed] = False,
    scale: Optional[float] = None,
    q_tile_shape: Optional[DimensionType] = None,
    kv_tile_shape: Optional[DimensionType] = None,
    backward_q_tile_shape: Optional[DimensionType] = None,
    backward_kv_tile_shape: Optional[DimensionType] = None,
    backward_use_pt_reduction: bool = False,
    backend: str = "cutlass-fna",
    return_lse: bool = False,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Dispatches varlen FNA with schedule metadata resolved by the caller.

    The public entry points (``natten.na{1,2,3}d_varlen``) resolve that
    metadata from the caller's ``VarlenLayout`` before calling this.
    """
    if backend != "cutlass-fna":
        raise NotImplementedError("Varlen FNA only supports backend='cutlass-fna'.")

    kernel_size, stride, dilation, is_causal = check_all_args(
        na_dim, kernel_size, stride, dilation, is_causal
    )

    if q_tile_shape is not None and kv_tile_shape is not None:
        forward_config = (q_tile_shape, kv_tile_shape)
    else:
        forward_config_input = query if value.shape[-1] <= query.shape[-1] else value
        forward_config = check_cutlass_fna_forward_config(
            input_tensor=_tiny_rank_shaped_view(forward_config_input, na_dim),
            dilation=dilation,
            q_tile_shape=q_tile_shape,
            kv_tile_shape=kv_tile_shape,
        )

    if backward_q_tile_shape is not None and backward_kv_tile_shape is not None:
        backward_config = (backward_q_tile_shape, backward_kv_tile_shape)
    else:
        backward_config_input = key if value.shape[-1] <= key.shape[-1] else value
        backward_config = check_cutlass_fna_backward_config(
            input_tensor=_tiny_rank_shaped_view(backward_config_input, na_dim),
            q_tile_shape=backward_q_tile_shape,
            kv_tile_shape=backward_kv_tile_shape,
        )

    scale = scale or query.shape[-1] ** -0.5

    if query.shape[-2] != key.shape[-2]:
        repeats = query.shape[-2] // key.shape[-2]
        key = torch.repeat_interleave(
            key, repeats=repeats, dim=-2, output_size=query.shape[-2]
        )
        value = torch.repeat_interleave(
            value, repeats=repeats, dim=-2, output_size=query.shape[-2]
        )

    output, logsumexp = VarlenCutlassFNAAutogradFns[na_dim].apply(
        query,
        key,
        value,
        cumulative_seqlens,
        token_layouts,
        backward_kv_splits,
        forward_worklist,
        backward_worklist,
        backward_q_tile_offsets,
        backward_kv_split_offsets,
        kernel_size,
        stride,
        dilation,
        is_causal,
        scale,
        forward_config,
        backward_config,
        forward_work_count,
        backward_work_count,
        total_backward_q_tiles,
        backward_use_pt_reduction,
        deterministic,
    )

    if return_lse:
        return output, logsumexp
    return output


def _normalize_split_cap(
    backward_kv_splits: Optional[DimensionType], na_dim: int
) -> Optional[DimensionType]:
    if backward_kv_splits is None:
        return None
    # Strict fixed-family mirror (check_fna_kv_splits in
    # configs/cutlass/backward_knobs.py): a tuple, not an integer -- see
    # that family's kv_splits docstrings ("Like tile shapes, this is a
    # tuple and not an integer...").
    if not isinstance(backward_kv_splits, tuple):
        raise ValueError(
            f"Invalid type {type(backward_kv_splits)} for backward_kv_splits."
        )
    split_cap = backward_kv_splits
    if len(split_cap) != na_dim:
        raise ValueError(
            f"backward_kv_splits must have {na_dim} values, got {len(split_cap)=}."
        )
    if any(
        isinstance(split, bool) or not isinstance(split, int) for split in split_cap
    ):
        raise TypeError("backward_kv_splits must contain only integers.")
    if any(split <= 0 for split in split_cap):
        raise ValueError("backward_kv_splits must contain only positive values.")
    return split_cap


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _tile_count(
    token_layout: DimensionType,
    dilation: DimensionType,
    tile_shape: DimensionType,
) -> int:
    return math.prod(
        [
            _ceil_div(_ceil_div(extent, dilation_axis), tile)
            for extent, dilation_axis, tile in zip(token_layout, dilation, tile_shape)
        ]
    )


def _make_worklist_tensor(counts: Sequence[int], device: torch.device) -> Tensor:
    # Explicit (document, local_tile) worklists rather than an
    # (n_docs x max_tiles) grid with device-side early exit: packed batches
    # are skew-heavy by design (one long document beside many tiny ones),
    # where a max-sized grid launches almost entirely empty blocks. The
    # worklist launches exactly the live tiles regardless of skew.
    work_count = sum(counts)
    counts_tensor = torch.tensor(counts, dtype=torch.int64, device=device)
    document_ids = torch.repeat_interleave(
        torch.arange(len(counts), dtype=torch.int32, device=device),
        counts_tensor,
        output_size=work_count,
    )
    document_offsets = torch.repeat_interleave(
        torch.tensor(_prefix_offsets(counts)[:-1], dtype=torch.int64, device=device),
        counts_tensor,
        output_size=work_count,
    )
    local_ids = torch.arange(work_count, dtype=torch.int64, device=device)
    local_ids = (local_ids - document_offsets).to(torch.int32)
    return torch.stack((document_ids, local_ids), dim=1)


@dataclass(frozen=True)
class _VarlenFnaResolvedState:
    """One VarlenLayout memo entry: build artifacts for one geometry key.

    Holds everything a memo miss derives from (kernel_size, dilation,
    num_heads, backward_kv_splits cap, deterministic, the KV-parallelism
    switch, the resolved max-grid bound, resolved forward and backward tile
    configs, dtype, device -- see the memo key built in
    _neighborhood_attention_varlen_generic) that a memo hit must not
    recompute: resolved tile configs, KV-split selection, and the
    worklist/offset tensors the raw ops consume. Values only -- never layout
    composition (that lives on the owning VarlenLayout, not per-entry).
    """

    forward_config: CutlassFnaForwardConfigType
    backward_config: CutlassFnaBackwardConfigType
    uses_kv_parallelism: bool
    selected_splits: Tuple[DimensionType, ...]
    backward_kv_splits: Tensor
    forward_worklist: Tensor
    backward_worklist: Tensor
    backward_q_tile_offsets: Tensor
    backward_kv_split_offsets: Tensor
    forward_work_count: int
    backward_work_count: int
    total_backward_q_tiles: int


@dataclass(frozen=True)
class _VarlenFnaPlanKey:
    """Memoization key for one VarlenLayout schedule-resolution call (see the
    key built in _neighborhood_attention_varlen_generic).

    Every field the memo-miss build (_build_varlen_fna_state) actually
    consumes, and nothing else: kernel_size is included because it drives
    the build-time kernel_size/dilation fit check, so an invalid
    (kernel_size, layout) combination always misses and always raises in
    build, instead of forcing the O(num_docs) fit loop onto every call.
    stride and is_causal are deliberately absent -- neither is consumed by
    the build (tile counts depend on dilation/tile shapes, the KV-split
    heuristic on shapes/num_heads/dilation/kv_tile_shape); both act per-call
    inside the kernel from the call's own arguments. dtype/device/
    kv_parallel_enabled/max_grid_size are snapshots the caller takes once
    per call (mutable global policy state, in the latter two cases), so a
    memo hit is always consistent with the key it was built under.

    frozen=True with the default eq=True derives __hash__ from every field,
    so instances are hashable dict keys, like the positional tuple this
    replaces.
    """

    kernel_size: DimensionType
    dilation: DimensionType
    num_heads: int
    split_cap: Optional[DimensionType]
    deterministic: bool
    forward_config: CutlassFnaForwardConfigType
    backward_config: CutlassFnaBackwardConfigType
    dtype: torch.dtype
    device: torch.device
    kv_parallel_enabled: bool
    max_grid_size: int


@torch.library.custom_op("natten::varlen_build_schedule_tensors", mutates_args=())
def _varlen_build_schedule_tensors(
    forward_q_tiles: List[int],
    backward_split_counts: List[int],
    selected_splits_flat: List[int],
    na_dim: int,
    backward_q_tile_offsets: List[int],
    backward_kv_split_offsets: List[int],
    dtype: torch.dtype,
    device: torch.device,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """VarlenLayout memo-miss device work, behind an opaque custom-op
    boundary: device compatibility check plus construction of the 5
    schedule tensors (backward_kv_splits, forward/backward worklists,
    backward offsets) from already-host-resolved shapes/splits/offsets.

    The boundary matters only under torch.compile: dynamo traces a custom op
    as a single opaque call rather than inlining the ATen ops inside it. A
    geometry miss taken at trace time is a one-shot graph -- after it runs,
    the memo has the entry, so the next call's ``self._memo.get(key)``
    returns non-None, dynamo guards on that dict-membership flip, and
    recompiles into a hit-path graph that never calls this op again. Without
    the op boundary, inlined tensor-construction ops would instead be baked
    into that first graph and silently replayed on every subsequent call
    even after the memo is warm. The project's test suite covers both
    halves of this memoization behavior: a cold miss under compile builds
    exactly once, and a prewarmed geometry never re-enters the build path.

    Keeping this behind one opaque op instead of tracing it inline follows
    the same direction taken elsewhere in NATTEN: moving KV-split decisions
    out of traced Python and into ops to avoid recompilations.
    """
    major, minor = torch.cuda.get_device_capability(device)
    check_cutlass_fna_device_compatibility(dtype, major * 10 + minor)

    num_docs = len(forward_q_tiles)
    selected_splits = [
        tuple(selected_splits_flat[index * na_dim : (index + 1) * na_dim])
        for index in range(num_docs)
    ]
    with torch.inference_mode(False):
        backward_kv_splits_tensor = torch.tensor(
            selected_splits, dtype=torch.int32, device=device
        )
        forward_worklist_tensor = _make_worklist_tensor(forward_q_tiles, device=device)
        backward_worklist_tensor = _make_worklist_tensor(
            backward_split_counts, device=device
        )
        backward_q_tile_offsets_tensor = torch.tensor(
            backward_q_tile_offsets, dtype=torch.int64, device=device
        )
        backward_kv_split_offsets_tensor = torch.tensor(
            backward_kv_split_offsets, dtype=torch.int64, device=device
        )
    return (
        backward_kv_splits_tensor,
        forward_worklist_tensor,
        backward_worklist_tensor,
        backward_q_tile_offsets_tensor,
        backward_kv_split_offsets_tensor,
    )


@_varlen_build_schedule_tensors.register_fake
def _(
    forward_q_tiles: List[int],
    backward_split_counts: List[int],
    selected_splits_flat: List[int],
    na_dim: int,
    backward_q_tile_offsets: List[int],
    backward_kv_split_offsets: List[int],
    dtype: torch.dtype,
    device: torch.device,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    num_docs = len(forward_q_tiles)
    return (
        torch.empty((num_docs, na_dim), dtype=torch.int32, device=device),
        torch.empty((sum(forward_q_tiles), 2), dtype=torch.int32, device=device),
        torch.empty((sum(backward_split_counts), 2), dtype=torch.int32, device=device),
        torch.empty((len(backward_q_tile_offsets),), dtype=torch.int64, device=device),
        torch.empty(
            (len(backward_kv_split_offsets),), dtype=torch.int64, device=device
        ),
    )


def _build_varlen_fna_state(
    shapes: Tuple[DimensionType, ...],
    kernel_size: DimensionType,
    dilation: DimensionType,
    num_heads: int,
    dtype: torch.dtype,
    device: torch.device,
    forward_config: CutlassFnaForwardConfigType,
    backward_config: CutlassFnaBackwardConfigType,
    split_cap: Optional[DimensionType],
    deterministic: bool,
    kv_parallelism_enabled: bool,
    max_grid_size: int,
) -> _VarlenFnaResolvedState:
    """Memo-miss build: resolves KV-split selection and constructs the
    worklist/offset tensors for one geometry. Invoked by VarlenLayout's
    resolve path on a memo miss. Host-only shape/split arithmetic stays
    inline below; device tensor construction is delegated to
    _varlen_build_schedule_tensors (see its docstring for why).

    ``kv_parallelism_enabled`` and ``max_grid_size`` are snapshots the
    caller took once and folded into the memo key; this function reads
    only those snapshots (never the live global policy state) so a memo
    hit is always consistent with the key it was built under.
    """
    na_dim = len(shapes[0])
    is_empty_doc = tuple(math.prod(layout) == 0 for layout in shapes)
    for index, layout in enumerate(shapes):
        # A zero-token document is never scheduled (see the backward_split_
        # counts and forward_q_tiles/backward_q_tiles derivations below, both
        # 0 for it), so it is exempt from the kernel_size * dilation fit
        # check: an empty document is a no-op, not a document a kernel needs
        # to fit into.
        if is_empty_doc[index]:
            continue
        if any(
            extent < kernel_axis * dilation_axis
            for extent, kernel_axis, dilation_axis in zip(layout, kernel_size, dilation)
        ):
            raise ValueError(
                "kernel_size * dilation must fit every token layout; "
                f"token_layouts[{index}]={layout}."
            )

    q_tile_shape, _ = forward_config
    backward_q_tile_shape, backward_kv_tile_shape = backward_config
    forward_q_tiles = tuple(
        _tile_count(layout, dilation, q_tile_shape) for layout in shapes
    )
    backward_q_tiles = tuple(
        _tile_count(layout, dilation, backward_q_tile_shape) for layout in shapes
    )

    if deterministic:
        selected_splits = cast(
            Tuple[DimensionType, ...], tuple((1,) * na_dim for _ in shapes)
        )
    elif split_cap is None:
        selected_splits = get_default_varlen_fna_kv_splits(
            input_shapes=shapes,
            num_heads=num_heads,
            dilation=dilation,
            kv_tile_shape=backward_kv_tile_shape,
            deterministic=deterministic,
            kv_parallelism_enabled=kv_parallelism_enabled,
            max_grid_size=max_grid_size,
        )
    else:
        selected_splits = cast(
            Tuple[DimensionType, ...],
            tuple(
                tuple(
                    min(available, cap)
                    for available, cap in zip(
                        get_max_splits(
                            layout,
                            dilation=dilation,
                            kv_tile_shape=backward_kv_tile_shape,
                        ),
                        split_cap,
                    )
                )
                for layout in shapes
            ),
        )

    # A zero-token document's split row is never read by any work item (it
    # schedules none -- backward_split_counts below is always exactly 0 for
    # it), so it is normalized to a (1,) * na_dim placeholder here regardless
    # of which branch above produced it. The count itself is read straight
    # off is_empty_doc rather than math.prod(selected_splits[i]): the
    # deterministic branch's uniform (1,) * na_dim placeholder would
    # otherwise contribute one spurious backward work item per empty
    # document.
    selected_splits = cast(
        Tuple[DimensionType, ...],
        tuple(
            (1,) * na_dim if empty else splits
            for empty, splits in zip(is_empty_doc, selected_splits)
        ),
    )
    backward_split_counts = tuple(
        0 if empty else math.prod(splits)
        for empty, splits in zip(is_empty_doc, selected_splits)
    )
    forward_work_count = sum(forward_q_tiles)
    backward_work_count = sum(backward_split_counts)
    backward_q_tile_offsets = _prefix_offsets(backward_q_tiles)
    backward_kv_split_offsets = _prefix_offsets(backward_split_counts)

    (
        backward_kv_splits_tensor,
        forward_worklist_tensor,
        backward_worklist_tensor,
        backward_q_tile_offsets_tensor,
        backward_kv_split_offsets_tensor,
    ) = _varlen_build_schedule_tensors(
        list(forward_q_tiles),
        list(backward_split_counts),
        [component for splits in selected_splits for component in splits],
        na_dim,
        list(backward_q_tile_offsets),
        list(backward_kv_split_offsets),
        dtype,
        device,
    )

    return _VarlenFnaResolvedState(
        forward_config=forward_config,
        backward_config=backward_config,
        uses_kv_parallelism=any(count > 1 for count in backward_split_counts),
        selected_splits=selected_splits,
        backward_kv_splits=backward_kv_splits_tensor,
        forward_worklist=forward_worklist_tensor,
        backward_worklist=backward_worklist_tensor,
        backward_q_tile_offsets=backward_q_tile_offsets_tensor,
        backward_kv_split_offsets=backward_kv_split_offsets_tensor,
        forward_work_count=forward_work_count,
        backward_work_count=backward_work_count,
        total_backward_q_tiles=backward_q_tile_offsets[-1],
    )


def _neighborhood_attention_varlen_generic(
    na_dim: int,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    layout: VarlenLayout,
    kernel_size: DimensionTypeOrDed,
    stride: DimensionTypeOrDed = 1,
    dilation: DimensionTypeOrDed = 1,
    is_causal: Optional[CausalArgTypeOrDed] = False,
    scale: Optional[float] = None,
    backend: Optional[str] = None,
    q_tile_shape: Optional[DimensionType] = None,
    kv_tile_shape: Optional[DimensionType] = None,
    backward_q_tile_shape: Optional[DimensionType] = None,
    backward_kv_tile_shape: Optional[DimensionType] = None,
    backward_kv_splits: Optional[DimensionType] = None,
    backward_use_pt_reduction: bool = False,
    return_lse: bool = False,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    if not isinstance(layout, VarlenLayout):
        raise TypeError(f"layout must be a natten.VarlenLayout, got {type(layout)=}.")
    # None picks the implementation, same convention as the fixed family's
    # backend=None -- there is only one choice today, so this is a fixed
    # point rather than a search, but the parameter still reads the same way
    # at every call site.
    backend = backend or "cutlass-fna"
    if backend != "cutlass-fna":
        raise NotImplementedError(
            "Variable-length FNA currently only supports backend='cutlass-fna'."
        )
    if not isinstance(backward_use_pt_reduction, bool):
        raise TypeError("backward_use_pt_reduction must be a bool.")
    if query.dim() != 3 or key.dim() != 3 or value.dim() != 3:
        raise ValueError(
            "Variable-length FNA expects flat [total_tokens, heads, head_dim] "
            f"tensors (no batch dim), got {query.dim()=}, {key.dim()=}, "
            f"{value.dim()=}."
        )
    if layout.rank != na_dim:
        raise ValueError(
            f"layout.rank={layout.rank} does not match this entry point's "
            f"rank ({na_dim})."
        )

    # Device/dtype/rank/head_dim/GQA consistency across Q/K/V, via the shared
    # fmha_tensor_checks so the messages match the rest of the library. That
    # helper expects 4-D [batch, ...] tensors, so the packed 3-D tensors are
    # checked through a temporary batch-dim view; nothing downstream keeps it.
    fmha_tensor_checks(
        query.unsqueeze(0),
        key.unsqueeze(0),
        value.unsqueeze(0),
        must_match_head_dims=False,
        supports_gqa_mqa=True,
        backend_name="Variable-length CUTLASS FNA",
    )
    if query.shape[0] != key.shape[0] or query.shape[0] != value.shape[0]:
        raise ValueError("Variable-length FNA requires equal Q/K/V token counts.")
    if query.shape[0] != layout.total_tokens:
        raise ValueError(
            f"query.shape[0]={query.shape[0]} does not match "
            f"layout.total_tokens={layout.total_tokens}."
        )
    if query.shape[-1] % 8 != 0 or value.shape[-1] % 8 != 0:
        raise ValueError("head_dim and head_dim_v must be multiples of 8.")
    if max(query.shape[-1], value.shape[-1]) > 2**16:
        raise ValueError("head_dim and head_dim_v must not exceed 65536.")

    for arg_name, arg_value in (
        ("kernel_size", kernel_size),
        ("stride", stride),
        ("dilation", dilation),
    ):
        if isinstance(arg_value, bool) or (
            isinstance(arg_value, Sequence)
            and any(isinstance(item, bool) for item in arg_value)
        ):
            raise TypeError(f"{arg_name} must contain integers, not booleans.")
    kernel_size, stride, dilation, is_causal = check_all_args(
        na_dim, kernel_size, stride, dilation, is_causal
    )
    if any(
        stride_axis > kernel_axis
        for kernel_axis, stride_axis in zip(kernel_size, stride)
    ):
        raise ValueError("stride cannot be larger than kernel_size along any axis.")
    split_cap = _normalize_split_cap(backward_kv_splits, na_dim)

    num_heads = query.shape[-2]
    if (
        num_heads * query.shape[-1] > _VARLEN_INT32_MAX
        or num_heads * value.shape[-1] > _VARLEN_INT32_MAX
    ):
        raise ValueError("heads * head dimension must fit in int32.")

    # A layout already pinned to a device fails fast here against a
    # mismatched one (host-only compare), before any schedule state is
    # built. A no-op when unmaterialized -- there is no "wrong" device yet,
    # so a CPU/meta query can still reach the memo-miss build's dtype/extent
    # checks below without a CUDA device.
    layout._check_device_pin(query.device)

    # Empty-layout contract, matching NATTEN's varlen FMHA path (where an
    # all-empty batch launches no attention kernel either): an all-empty
    # layout is an exact no-op -- no kernel launch, no tile-config
    # resolution (which needs a real token to probe below), no memo entry.
    # Autograd stays connected by deriving the (zero-element) output AND
    # logsumexp from query/key/value through one shared zero-valued link, so
    # backward() still produces correctly-shaped zero grads and
    # logsumexp.requires_grad matches the non-empty path (an
    # autograd.Function output that is differentiable but, like the rest of
    # the FNA family, ignored) instead of silently detaching.
    if layout.total_tokens == 0:
        # The non-empty path's CUDA requirement is enforced further down
        # this function (device-tensor construction in the memo-miss build,
        # or layout materialization); this fast path bypasses all of that,
        # so it must check for itself instead of silently succeeding on CPU.
        if query.device.type != "cuda" or not torch.cuda.is_available():
            raise ValueError("Variable-length CUTLASS FNA requires a CUDA device.")
        # Same reason: the dtype/compute-capability gate normally runs inside
        # the memo-miss build (_varlen_build_schedule_tensors), which this
        # fast path never reaches -- so an unsupported dtype or device must
        # be checked here too, instead of silently succeeding.
        check_cutlass_fna_device_compatibility(query.dtype, get_device_cc(query.device))
        head_dim_v = value.shape[-1]
        zero_link = (query.sum() + key.sum() + value.sum()) * 0
        output = query.new_zeros((0, num_heads, head_dim_v)) + zero_link
        if return_lse:
            logsumexp = query.new_zeros(
                (0, num_heads), dtype=torch.float32
            ) + zero_link.to(torch.float32)
            return output, logsumexp
        return output

    config_input = query if value.shape[-1] <= query.shape[-1] else value
    forward_config = check_cutlass_fna_forward_config(
        input_tensor=_tiny_rank_shaped_view(config_input, na_dim),
        dilation=dilation,
        q_tile_shape=q_tile_shape,
        kv_tile_shape=kv_tile_shape,
    )
    backward_config_input = key if value.shape[-1] <= key.shape[-1] else value
    backward_config = check_cutlass_fna_backward_config(
        input_tensor=_tiny_rank_shaped_view(backward_config_input, na_dim),
        q_tile_shape=backward_q_tile_shape,
        kv_tile_shape=backward_kv_tile_shape,
    )

    # Single read for the whole call: threaded through varlen_cutlass_fna_generic
    # into the autograd Function's ctx (see its forward()) instead of being
    # re-read there, so the memo key/schedule build and the backward policy
    # always agree, even if the global flips between this read and backward().
    deterministic = torch.are_deterministic_algorithms_enabled()
    # Snapshotted once per call, not read inside the build: both are mutable
    # global policy state (the KV-parallelism switch and the memory-usage
    # preference the grid-size bound derives from), so they must be part of
    # the memo key -- otherwise flipping either after warming a layout would
    # reuse a schedule resolved under the old value. Mirrors how
    # `deterministic` is already handled above.
    kv_parallel_enabled = is_kv_parallelism_in_fused_na_enabled()
    max_grid_size = _get_max_grid_size_allowed()
    plan_key = _VarlenFnaPlanKey(
        kernel_size=kernel_size,
        dilation=dilation,
        num_heads=num_heads,
        split_cap=split_cap,
        deterministic=deterministic,
        forward_config=forward_config,
        backward_config=backward_config,
        dtype=query.dtype,
        device=query.device,
        kv_parallel_enabled=kv_parallel_enabled,
        max_grid_size=max_grid_size,
    )
    resolved = layout._resolve(
        plan_key,
        lambda: _build_varlen_fna_state(
            shapes=layout._shapes,
            kernel_size=kernel_size,
            dilation=dilation,
            num_heads=num_heads,
            dtype=query.dtype,
            device=query.device,
            forward_config=forward_config,
            backward_config=backward_config,
            split_cap=split_cap,
            deterministic=deterministic,
            kv_parallelism_enabled=kv_parallel_enabled,
            max_grid_size=max_grid_size,
        ),
    )

    # The device-pin check ran before any schedule state was built
    # (host-only compare), so a call against the wrong device fails before
    # doing any device work; materialization itself (first-call device
    # tensor construction) is deferred to here, keeping the reuse path free
    # of device interaction until the raw-op dispatch below.
    layout._ensure_materialized(query.device)

    return varlen_cutlass_fna_generic(
        query=query,
        key=key,
        value=value,
        cumulative_seqlens=layout.cu_seqlens,
        token_layouts=layout.token_layouts,
        na_dim=na_dim,
        kernel_size=kernel_size,
        forward_worklist=resolved.forward_worklist,
        backward_worklist=resolved.backward_worklist,
        backward_q_tile_offsets=resolved.backward_q_tile_offsets,
        backward_kv_split_offsets=resolved.backward_kv_split_offsets,
        backward_kv_splits=resolved.backward_kv_splits,
        forward_work_count=resolved.forward_work_count,
        backward_work_count=resolved.backward_work_count,
        total_backward_q_tiles=resolved.total_backward_q_tiles,
        deterministic=deterministic,
        stride=stride,
        dilation=dilation,
        is_causal=is_causal,
        scale=scale,
        q_tile_shape=resolved.forward_config[0],
        kv_tile_shape=resolved.forward_config[1],
        backward_q_tile_shape=resolved.backward_config[0],
        backward_kv_tile_shape=resolved.backward_config[1],
        backward_use_pt_reduction=backward_use_pt_reduction,
        backend="cutlass-fna",
        return_lse=return_lse,
    )
