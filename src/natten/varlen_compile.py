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
"""Variable-length FNA entry points whose layout travels as an opaque handle.

``na{1,2,3}d_varlen`` resolves its schedule from a ``VarlenLayout`` in Python,
so ``torch.compile`` traces that resolution and guards on the per-document
extents it reads along the way (``VarlenLayout._shapes`` and everything derived
from it in ``natten.backends.varlen_lowering``). A caller whose document shapes
change from call to call -- sequence packing's whole point -- therefore gets one
specialization per packing, and a graph whose document geometry is baked in.

The entry points here take the same layout behind a CPU 0-dim int64 handle
tensor (``VarlenLayoutHandle.tensor``) and reach the kernels through a
``torch.library.custom_op``. Dynamo records the operator as one opaque call and
never traces into it, so the schedule resolution runs at execution time and the
compiled graph carries no document geometry at all: one graph serves every
packing.

This is a different boundary from the two operators already in the varlen path
(``natten::varlen_build_permutation_tensors`` in ``natten.varlen``,
``natten::varlen_build_schedule_tensors`` in ``natten.backends.varlen_fna``).
Those are inner boundaries: they hide a memo miss's *device tensor
construction*, not the host-side Python that decides what to build, which is
where the per-document guards come from. This module's operator wraps the whole
entry point, so none of that Python is traced.

Numerics are the stock path's, literally: the operator body calls the public
``na{1,2,3}d_varlen``, and backward re-runs it under ``enable_grad`` and returns
``torch.autograd.grad``. That re-run costs one extra forward per backward; a
backward that drives the inner kernels from a saved output/logsumexp is the
planned follow-up, not this.
"""

import threading
import weakref
from itertools import count
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor

from natten.functional import na1d_varlen, na2d_varlen, na3d_varlen
from natten.types import (
    CausalArg1DTypeOrDed,
    CausalArg2DTypeOrDed,
    CausalArg3DTypeOrDed,
    Dimension1DTypeOrDed,
    Dimension2DTypeOrDed,
    Dimension3DTypeOrDed,
)
from natten.varlen import VarlenLayout

_VARLEN_ENTRY_POINTS: Dict[int, Callable[..., Any]] = {
    1: na1d_varlen,
    2: na2d_varlen,
    3: na3d_varlen,
}

# Handle id -> layout. Entries are removed by VarlenLayoutHandle's finalizer,
# so the registry is bounded by the handles the caller keeps alive rather than
# by a second eviction policy of its own: whoever caches handles (per geometry,
# with whatever bound it wants) also bounds this.
_REGISTRY: Dict[int, VarlenLayout] = {}
_REGISTRY_LOCK = threading.Lock()
_REGISTRY_IDS = count()


def _register(layout: VarlenLayout) -> int:
    with _REGISTRY_LOCK:
        key = next(_REGISTRY_IDS)
        _REGISTRY[key] = layout
    return key


def _unregister(key: int) -> None:
    with _REGISTRY_LOCK:
        _REGISTRY.pop(key, None)


def _lookup(layout_handle: Tensor) -> VarlenLayout:
    if layout_handle.dim() != 0 or layout_handle.dtype != torch.int64:
        raise ValueError(
            "layout_handle must be the 0-dim int64 tensor a VarlenLayoutHandle "
            f"exposes as .tensor; got a {layout_handle.dim()}-dim "
            f"{layout_handle.dtype} tensor."
        )
    key = int(layout_handle.item())
    with _REGISTRY_LOCK:
        layout = _REGISTRY.get(key)
    if layout is None:
        raise RuntimeError(
            f"Layout handle {key} is not registered. The handle tensor is a "
            "reference, not a value: keep the VarlenLayoutHandle object alive "
            "for as long as any call -- including any compiled graph that "
            "captured its tensor -- may still use it."
        )
    return layout


class VarlenLayoutHandle:
    """Opaque, compile-safe reference to a [VarlenLayout][natten.VarlenLayout].

    Wraps one layout and exposes it as a CPU 0-dim int64 tensor
    ([tensor][natten.VarlenLayoutHandle.tensor]) that
    ``na{1,2,3}d_varlen_handle`` accepts in place of the layout object. Under
    ``torch.compile`` that tensor is an ordinary graph input: dynamo guards on
    its dtype/device/rank and never on its value, so two packings with different
    document shapes share one graph::

        handle = natten.VarlenLayoutHandle(natten.VarlenLayout(shapes))
        out = natten.na3d_varlen_handle(q, k, v, handle.tensor, kernel_size=...)

    The tensor is a reference into a process-local registry, not a value. It is
    meaningful only while this object is alive and only in the process that
    created it, so it is not picklable and must not be sent to a DataLoader
    worker or another rank. Pickle the ``VarlenLayout`` instead (which carries
    only its shapes) and build a handle on the other side.

    A handle owns nothing else: it does not deduplicate geometries, cache, or
    expire. Callers that want one handle per geometry keep their own map from
    geometry to handle, and the registry entry disappears when their entry does.

    Parameters:
        layout (VarlenLayout): The layout to reference. Held by strong
            reference for this object's lifetime.
    """

    __slots__ = ("_layout", "_tensor", "_finalizer", "__weakref__")

    def __init__(self, layout: VarlenLayout) -> None:
        if not isinstance(layout, VarlenLayout):
            raise TypeError(
                f"layout must be a natten.VarlenLayout, got {type(layout)=}."
            )
        key = _register(layout)
        self._layout = layout
        # inference_mode(False) for the same reason the rest of the varlen path
        # takes it (see natten.varlen): a handle built inside inference_mode
        # would otherwise carry an inference tensor into later normal-mode
        # calls.
        with torch.inference_mode(False):
            self._tensor = torch.tensor(key, dtype=torch.int64)
        self._finalizer = weakref.finalize(self, _unregister, key)

    def __repr__(self) -> str:
        return f"VarlenLayoutHandle(id={int(self._tensor)}, layout={self._layout!r})"

    def __reduce__(self) -> Any:
        raise TypeError(
            "VarlenLayoutHandle is not picklable: the handle id is only "
            "meaningful in the process that registered it. Pickle the "
            "VarlenLayout instead and build a handle in the target process."
        )

    @property
    def tensor(self) -> Tensor:
        """CPU 0-dim int64 tensor to pass as ``layout_handle``."""
        return self._tensor

    @property
    def layout(self) -> VarlenLayout:
        """The referenced layout."""
        return self._layout


@torch.library.custom_op("natten::varlen_attention_fwd", mutates_args=())
def _varlen_attention_fwd(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    layout_handle: Tensor,
    na_dim: int,
    kernel_size: List[int],
    stride: List[int],
    dilation: List[int],
    is_causal: List[bool],
    scale: Optional[float],
    backward_use_pt_reduction: bool,
) -> Tuple[Tensor, Tensor]:
    """Opaque forward: resolves the layout from the handle and calls the public
    ``na{na_dim}d_varlen``.

    Everything the stock entry point does -- argument checks, degenerate-axis
    lowering, the uniform and all-empty fast paths, schedule memoization --
    happens here, at execution time, where dynamo cannot see it.

    ``logsumexp`` is returned unconditionally because an operator's schema
    cannot vary its return type with a bool argument; the Python wrappers drop
    it when ``return_lse=False``. The only place that costs anything is the
    fully-degenerate path, which computes lse from q/k rather than getting it
    from a kernel for free.
    """
    layout = _lookup(layout_handle)
    # Grad is already off inside the generated autograd Function's forward;
    # stated explicitly so a direct call below the autograd key behaves the
    # same and never records the inner autograd.Function's saved tensors.
    with torch.no_grad():
        output, logsumexp = _VARLEN_ENTRY_POINTS[na_dim](
            query,
            key,
            value,
            layout,
            kernel_size=tuple(kernel_size),
            stride=tuple(stride),
            dilation=tuple(dilation),
            is_causal=tuple(is_causal),
            scale=scale,
            backend="cutlass-fna",
            backward_use_pt_reduction=backward_use_pt_reduction,
            return_lse=True,
        )
    return output, logsumexp


@_varlen_attention_fwd.register_fake
def _(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    layout_handle: Tensor,
    na_dim: int,
    kernel_size: List[int],
    stride: List[int],
    dilation: List[int],
    is_causal: List[bool],
    scale: Optional[float],
    backward_use_pt_reduction: bool,
) -> Tuple[Tensor, Tensor]:
    # Reads tensor metadata only -- no layout, no document geometry. This is
    # what keeps a compiled graph geometry-free: the packed token count stays
    # whatever symbol the caller had, and the layout/token-count agreement is
    # checked at execution time by the entry point itself instead.
    total_tokens, heads = query.shape[0], query.shape[1]
    return (
        query.new_empty((total_tokens, heads, value.shape[-1])),
        query.new_empty((total_tokens, heads), dtype=torch.float32),
    )


def _autograd_keyset() -> Any:
    keyset = torch._C.DispatchKeySet(torch._C.DispatchKey.AutogradFunctionality)
    for key in (
        torch._C.DispatchKey.AutogradOther,
        torch._C.DispatchKey.AutogradNestedTensor,
        torch._C.DispatchKey.AutogradCPU,
        torch._C.DispatchKey.AutogradCUDA,
    ):
        keyset = keyset | torch._C.DispatchKeySet(key)
    return keyset


_AUTOGRAD_KEYSET = _autograd_keyset()


def _recording_autograd() -> Any:
    """Re-enables autograd recording inside an operator's body.

    An operator's implementation runs under ``_C._AutoDispatchBelowAutograd``
    (torch/_library/autograd.py), which excludes the autograd dispatch keys:
    ATen calls inside the body are not recorded no matter what grad mode says.
    ``autograd.Function.apply`` is a Python-level construct and still records,
    so a varlen call whose last step is such a Function comes out
    differentiable while one that ends in a plain tensor op (the uniform
    path's reshape, the all-empty path's arithmetic, GQA's repeat_interleave)
    silently does not. Clearing those keys from the thread-local excluded set
    gives the replay below a complete graph to differentiate.
    """
    included = torch._C._dispatch_tls_local_include_set()
    excluded = torch._C._dispatch_tls_local_exclude_set()
    return torch._C._ForceDispatchKeyGuard(included, excluded - _AUTOGRAD_KEYSET)


def _copy_aliases_of(
    tensors: Tuple[Tensor, ...], inputs: Tuple[Tensor, ...]
) -> Tuple[Tensor, ...]:
    """Copies whichever of ``tensors`` share storage with an input or with an
    earlier one of themselves.

    An operator's outputs may alias neither its inputs nor each other, and the
    stock paths produce both: the fully-degenerate lowering's dV is the
    incoming ``grad_out`` itself
    (``natten.backends.varlen_lowering._IdentityValue``), and the all-empty
    path's three gradients are all expansions of one scalar, hence one storage.
    The kernel paths allocate their own gradients and copy nothing here.
    """
    storages = {tensor.untyped_storage().data_ptr() for tensor in inputs}
    copied = []
    for tensor in tensors:
        if tensor.untyped_storage().data_ptr() in storages:
            tensor = tensor.clone()
        storages.add(tensor.untyped_storage().data_ptr())
        copied.append(tensor)
    return tuple(copied)


@torch.library.custom_op("natten::varlen_attention_bwd", mutates_args=())
def _varlen_attention_bwd(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    grad_output: Tensor,
    layout_handle: Tensor,
    na_dim: int,
    kernel_size: List[int],
    stride: List[int],
    dilation: List[int],
    is_causal: List[bool],
    scale: Optional[float],
    backward_use_pt_reduction: bool,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Opaque backward: replays the stock forward on detached inputs and
    returns its gradients.

    An operator, rather than plain Python inside the ``register_autograd``
    callback, because that callback *is* traced by AOTAutograd: dereferencing
    the handle there raises ``GuardOnDataDependentSymNode`` on ``.item()``. The
    dereference has to sit one level further down, in an operator body, same as
    the forward's -- which is also why the replay needs
    ``_recording_autograd`` to get a graph at all.

    Replaying instead of driving the kernels from a saved output/logsumexp
    keeps the numerics identical to ``na{1,2,3}d_varlen``'s own backward by
    construction, degenerate-axis lowering included (a hand-written backward
    would have to reproduce the fold, the permute and its inverse, and the
    uniform path's separate fixed-shape configuration). It costs one extra
    forward per backward.
    """
    layout = _lookup(layout_handle)
    with _recording_autograd(), torch.enable_grad():
        query_ = query.detach().requires_grad_(True)
        key_ = key.detach().requires_grad_(True)
        value_ = value.detach().requires_grad_(True)
        output = _VARLEN_ENTRY_POINTS[na_dim](
            query_,
            key_,
            value_,
            layout,
            kernel_size=tuple(kernel_size),
            stride=tuple(stride),
            dilation=tuple(dilation),
            is_causal=tuple(is_causal),
            scale=scale,
            backend="cutlass-fna",
            backward_use_pt_reduction=backward_use_pt_reduction,
            return_lse=False,
        )
        gradients = torch.autograd.grad(
            output, (query_, key_, value_), grad_output.contiguous()
        )
    d_query, d_key, d_value = _copy_aliases_of(
        gradients, (query, key, value, grad_output)
    )
    return d_query, d_key, d_value


@_varlen_attention_bwd.register_fake
def _(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    grad_output: Tensor,
    layout_handle: Tensor,
    na_dim: int,
    kernel_size: List[int],
    stride: List[int],
    dilation: List[int],
    is_causal: List[bool],
    scale: Optional[float],
    backward_use_pt_reduction: bool,
) -> Tuple[Tensor, Tensor, Tensor]:
    return (
        torch.empty_like(query),
        torch.empty_like(key),
        torch.empty_like(value),
    )


def _varlen_attention_setup_context(ctx: Any, inputs: Any, output: Any) -> None:
    query, key, value, layout_handle = inputs[:4]
    ctx.save_for_backward(query, key, value, layout_handle)
    ctx.varlen_args = tuple(inputs[4:])


def _varlen_attention_backward(
    ctx: Any, grad_output: Optional[Tensor], grad_logsumexp: Optional[Tensor]
) -> Tuple[Optional[Tensor], ...]:
    # grad_logsumexp is ignored, exactly as the stock autograd Function ignores
    # it (natten.backends.varlen_fna.VarlenCutlassFNAAutogradFn.backward): FNA's
    # logsumexp is a differentiable output whose incoming gradient the family
    # does not propagate. With no gradient on the output either, q/k/v get none
    # from this call, which is what the stock path's all-zero gradients would
    # accumulate to anyway.
    if grad_output is None:
        return (None,) * 11
    query, key, value, layout_handle = ctx.saved_tensors
    d_query, d_key, d_value = _varlen_attention_bwd(
        query, key, value, grad_output, layout_handle, *ctx.varlen_args
    )
    return (d_query, d_key, d_value) + (None,) * 8


torch.library.register_autograd(
    "natten::varlen_attention_fwd",
    _varlen_attention_backward,
    setup_context=_varlen_attention_setup_context,
)


def _as_int_list(value: Any, na_dim: int, name: str) -> List[int]:
    # Booleans are rejected here rather than downstream: the operator schema
    # takes int[], which would silently coerce True into 1 before
    # check_all_args ever sees it.
    if isinstance(value, bool) or (
        isinstance(value, Sequence) and any(isinstance(item, bool) for item in value)
    ):
        raise TypeError(f"{name} must contain integers, not booleans.")
    if isinstance(value, int):
        return [value] * na_dim
    return [int(item) for item in value]


def _as_bool_list(value: Any, na_dim: int) -> List[bool]:
    if value is None:
        return [False] * na_dim
    if isinstance(value, bool):
        return [value] * na_dim
    return [bool(item) for item in value]


def _varlen_handle_call(
    na_dim: int,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    layout_handle: Tensor,
    kernel_size: Any,
    stride: Any,
    dilation: Any,
    is_causal: Any,
    scale: Optional[float],
    backend: Optional[str],
    backward_use_pt_reduction: bool,
    return_lse: bool,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    if backend is not None and backend != "cutlass-fna":
        raise NotImplementedError(
            "Variable-length FNA currently only supports backend='cutlass-fna'."
        )
    if not isinstance(layout_handle, Tensor):
        raise TypeError(
            "layout_handle must be a VarlenLayoutHandle's .tensor, got "
            f"{type(layout_handle)=}. Pass handle.tensor, not the handle."
        )
    output, logsumexp = _varlen_attention_fwd(
        query,
        key,
        value,
        layout_handle,
        na_dim,
        _as_int_list(kernel_size, na_dim, "kernel_size"),
        _as_int_list(stride, na_dim, "stride"),
        _as_int_list(dilation, na_dim, "dilation"),
        _as_bool_list(is_causal, na_dim),
        None if scale is None else float(scale),
        bool(backward_use_pt_reduction),
    )
    if return_lse:
        return output, logsumexp
    return output


def na1d_varlen_handle(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    layout_handle: Tensor,
    kernel_size: Dimension1DTypeOrDed,
    stride: Dimension1DTypeOrDed = 1,
    dilation: Dimension1DTypeOrDed = 1,
    is_causal: Optional[CausalArg1DTypeOrDed] = False,
    scale: Optional[float] = None,
    backend: Optional[str] = None,
    backward_use_pt_reduction: bool = False,
    return_lse: bool = False,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """1-D variable-length neighborhood attention, layout passed as a handle.

    Same computation as [na1d_varlen][natten.na1d_varlen] -- it is what this
    calls -- with the layout arriving as
    [VarlenLayoutHandle.tensor][natten.VarlenLayoutHandle.tensor] instead of a
    ``VarlenLayout``. Use it when the call sits inside ``torch.compile`` and the
    document shapes change between calls: the layout is then invisible to
    dynamo, so one graph covers every packing instead of one per packing. In
    eager there is no reason to prefer it.

    Explicit tile shapes (``q_tile_shape``, ``kv_tile_shape``,
    ``backward_q_tile_shape``, ``backward_kv_tile_shape``) and
    ``backward_kv_splits`` are not accepted: a packed call with a
    ``kernel_size = 1`` axis lowers to a different rank, where the stock entry
    point rejects them anyway, and leaving them out of the operator's schema
    keeps the traced call free of avoidable constants. Use
    [na1d_varlen][natten.na1d_varlen] when you need them.

    Parameters:
        query (Tensor): 3-D query tensor, with the heads last layout
            (`[total_tokens, heads, head_dim]`).

        key (Tensor): 3-D key tensor, with the heads last layout
            (`[total_tokens, heads_kv, head_dim]`).

        value (Tensor): 3-D value tensor, with the heads last layout
            (`[total_tokens, heads_kv, head_dim_v]`).

        layout_handle (Tensor): CPU 0-dim int64 tensor from
            [VarlenLayoutHandle][natten.VarlenLayoutHandle]; the referenced
            layout must have rank 1, and its `total_tokens` must equal
            `query.shape[0]`. Both are checked when the call executes, not when
            it is traced.

        kernel_size (Tuple[int] | int): Neighborhood window (kernel) size.

        stride (Tuple[int] | int): Sliding window step size. Defaults to `1`.

        dilation (Tuple[int] | int): Dilation step size. Defaults to `1`.

        is_causal (Tuple[bool] | bool): Toggle causal masking. Defaults to
            `False`.

        scale (float): Attention scale. `None` selects `head_dim ** -0.5`.

    Other Parameters:
        backend (Optional[str]): Backend implementation to run with. Choices
            are: `None` (defaults to `"cutlass-fna"`, the only backend
            currently supported), `"cutlass-fna"`.

        backward_use_pt_reduction (bool): Whether to use PyTorch eager for
            computing the `dO * O` product required by the backward pass.

        return_lse (bool): Whether or not to return the `logsumexp` tensor.

    Returns:
        output (Tensor): `[total_tokens, heads, head_dim_v]` packed output.

        logsumexp (Tensor): only returned when `return_lse=True`.
            `[total_tokens, heads]` packed logsumexp.
    """
    return _varlen_handle_call(
        1,
        query,
        key,
        value,
        layout_handle,
        kernel_size,
        stride,
        dilation,
        is_causal,
        scale,
        backend,
        backward_use_pt_reduction,
        return_lse,
    )


def na2d_varlen_handle(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    layout_handle: Tensor,
    kernel_size: Dimension2DTypeOrDed,
    stride: Dimension2DTypeOrDed = 1,
    dilation: Dimension2DTypeOrDed = 1,
    is_causal: Optional[CausalArg2DTypeOrDed] = False,
    scale: Optional[float] = None,
    backend: Optional[str] = None,
    backward_use_pt_reduction: bool = False,
    return_lse: bool = False,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """2-D variable-length neighborhood attention, layout passed as a handle.

    See [na1d_varlen_handle][natten.na1d_varlen_handle] for the full
    description (identical, modulo rank); the referenced layout must have rank
    2 here, and `kernel_size`/`stride`/`dilation`/`is_causal` are 2-tuples.
    """
    return _varlen_handle_call(
        2,
        query,
        key,
        value,
        layout_handle,
        kernel_size,
        stride,
        dilation,
        is_causal,
        scale,
        backend,
        backward_use_pt_reduction,
        return_lse,
    )


def na3d_varlen_handle(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    layout_handle: Tensor,
    kernel_size: Dimension3DTypeOrDed,
    stride: Dimension3DTypeOrDed = 1,
    dilation: Dimension3DTypeOrDed = 1,
    is_causal: Optional[CausalArg3DTypeOrDed] = False,
    scale: Optional[float] = None,
    backend: Optional[str] = None,
    backward_use_pt_reduction: bool = False,
    return_lse: bool = False,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """3-D variable-length neighborhood attention, layout passed as a handle.

    See [na1d_varlen_handle][natten.na1d_varlen_handle] for the full
    description (identical, modulo rank); the referenced layout must have rank
    3 here, and `kernel_size`/`stride`/`dilation`/`is_causal` are 3-tuples.
    """
    return _varlen_handle_call(
        3,
        query,
        key,
        value,
        layout_handle,
        kernel_size,
        stride,
        dilation,
        is_causal,
        scale,
        backend,
        backward_use_pt_reduction,
        return_lse,
    )


__all__ = [
    "VarlenLayoutHandle",
    "na1d_varlen_handle",
    "na2d_varlen_handle",
    "na3d_varlen_handle",
]
