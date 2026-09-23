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
"""What ``na{1,2,3}d_varlen`` does instead of tracing, under torch.compile.

Private: nothing here is part of the public API. The entry points in
``natten.functional`` call :func:`_varlen_compiled_call` as their first
statement when ``torch.compiler.is_compiling()``, and are otherwise unchanged.

Why they must not be traced. Those entry points resolve a schedule from a
``VarlenLayout`` in Python, and dynamo guards on every per-document extent that
resolution reads (``VarlenLayout._shapes``, and everything derived from it in
``natten.backends.varlen_lowering``). A caller whose document shapes change
from call to call -- sequence packing's whole point -- would get one
specialization per packing, and a graph with the geometry baked into it.

So under compile the call goes through a ``torch.library.custom_op``, which
dynamo records as one opaque node and never enters, and the layout is that
operator's argument as itself. ``VarlenLayout`` is a reference-type opaque
object (registered in ``natten.varlen``): dynamo makes it a graph input guarded
by its type alone, and when the graph runs the operator receives the caller's
own layout, memo included. The schedule resolution then happens at execution
time, and the graph carries no document geometry at all: one graph serves every
packing. While the operators are traced they see a ``FakeScriptObject`` in the
layout's place, and their fake kernels read tensor metadata only.

This is a different boundary from the two operators already in the varlen path
(``natten::varlen_build_permutation_tensors`` in ``natten.varlen``,
``natten::varlen_build_schedule_tensors`` in ``natten.backends.varlen_fna``).
Those are inner boundaries: they hide a memo miss's *device tensor
construction*, not the host-side Python that decides what to build, which is
where the per-document guards come from. The operator here wraps the whole
entry point, so none of that Python is traced.

The numerics are the entry point's own: the operator body calls
``na{1,2,3}d_varlen``, which -- not compiling, at that point -- takes its
ordinary path. Backward re-runs that call under ``enable_grad`` and returns
``torch.autograd.grad``, which costs one extra forward per backward; driving
the inner kernels from a saved output/logsumexp is the planned follow-up.

The operators exist only where torch can pass a ``VarlenLayout`` to an operator
(``_LAYOUT_IS_OPAQUE``); elsewhere the entry points trace their own Python under
compile.
"""

from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor

from natten.varlen import _LAYOUT_IS_OPAQUE, VarlenLayout

# natten.functional imports this module, so the entry points are resolved
# inside the operator bodies instead of here. Those bodies run at execution
# time, where the import is a sys.modules lookup.
_ENTRY_POINT_NAMES = {1: "na1d_varlen", 2: "na2d_varlen", 3: "na3d_varlen"}


def _entry_point(na_dim: int) -> Callable[..., Any]:
    from natten import functional

    return getattr(functional, _ENTRY_POINT_NAMES[na_dim])


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


def _opt_tuple(value: Optional[List[int]]) -> Optional[Tuple[int, ...]]:
    return None if value is None else tuple(value)


def _stock_call(
    na_dim: int,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    layout: VarlenLayout,
    kernel_size: List[int],
    stride: List[int],
    dilation: List[int],
    is_causal: List[bool],
    scale: Optional[float],
    backend: Optional[str],
    q_tile_shape: Optional[List[int]],
    kv_tile_shape: Optional[List[int]],
    backward_q_tile_shape: Optional[List[int]],
    backward_kv_tile_shape: Optional[List[int]],
    backward_kv_splits: Optional[List[int]],
    backward_use_pt_reduction: bool,
    return_lse: bool,
) -> Any:
    """The public entry point, called with the caller's own layout.

    ``torch.compiler.is_compiling()`` is False here -- this runs when the
    compiled graph executes, not while it is traced -- so the entry point
    takes its ordinary path and every argument means exactly what it means in
    eager.
    """
    return _entry_point(na_dim)(
        query,
        key,
        value,
        layout,
        kernel_size=tuple(kernel_size),
        stride=tuple(stride),
        dilation=tuple(dilation),
        is_causal=tuple(is_causal),
        scale=scale,
        backend=backend,
        q_tile_shape=_opt_tuple(q_tile_shape),
        kv_tile_shape=_opt_tuple(kv_tile_shape),
        backward_q_tile_shape=_opt_tuple(backward_q_tile_shape),
        backward_kv_tile_shape=_opt_tuple(backward_kv_tile_shape),
        backward_kv_splits=_opt_tuple(backward_kv_splits),
        backward_use_pt_reduction=backward_use_pt_reduction,
        return_lse=return_lse,
    )


if _LAYOUT_IS_OPAQUE:

    @torch.library.custom_op("natten::varlen_attention_fwd", mutates_args=())
    def _varlen_attention_fwd(
        query: Tensor,
        key: Tensor,
        value: Tensor,
        layout: VarlenLayout,
        na_dim: int,
        kernel_size: List[int],
        stride: List[int],
        dilation: List[int],
        is_causal: List[bool],
        scale: Optional[float],
        backend: Optional[str],
        q_tile_shape: Optional[List[int]],
        kv_tile_shape: Optional[List[int]],
        backward_q_tile_shape: Optional[List[int]],
        backward_kv_tile_shape: Optional[List[int]],
        backward_kv_splits: Optional[List[int]],
        backward_use_pt_reduction: bool,
    ) -> Tuple[Tensor, Tensor]:
        """Opaque forward: the entry point's own work, where dynamo cannot see it.

        Argument checks, degenerate-axis lowering, the uniform and all-empty fast
        paths and schedule memoization all happen here, at execution time.

        ``logsumexp`` comes back unconditionally because an operator's schema
        cannot vary its return type with a bool argument; the caller drops it when
        ``return_lse=False``. The only place that costs anything is the
        fully-degenerate path, which computes lse from q/k rather than getting it
        from a kernel for free.
        """
        # Grad is already off inside the generated autograd Function's forward;
        # stated explicitly so a direct call below the autograd key behaves the
        # same and never records the inner autograd.Function's saved tensors.
        with torch.no_grad():
            output, logsumexp = _stock_call(
                na_dim,
                query,
                key,
                value,
                layout,
                kernel_size,
                stride,
                dilation,
                is_causal,
                scale,
                backend,
                q_tile_shape,
                kv_tile_shape,
                backward_q_tile_shape,
                backward_kv_tile_shape,
                backward_kv_splits,
                backward_use_pt_reduction,
                True,
            )
        return output, logsumexp

    @_varlen_attention_fwd.register_fake
    def _(
        query: Tensor,
        key: Tensor,
        value: Tensor,
        layout: VarlenLayout,
        na_dim: int,
        kernel_size: List[int],
        stride: List[int],
        dilation: List[int],
        is_causal: List[bool],
        scale: Optional[float],
        backend: Optional[str],
        q_tile_shape: Optional[List[int]],
        kv_tile_shape: Optional[List[int]],
        backward_q_tile_shape: Optional[List[int]],
        backward_kv_tile_shape: Optional[List[int]],
        backward_kv_splits: Optional[List[int]],
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

    @torch.library.custom_op("natten::varlen_attention_bwd", mutates_args=())
    def _varlen_attention_bwd(
        query: Tensor,
        key: Tensor,
        value: Tensor,
        grad_output: Tensor,
        layout: VarlenLayout,
        na_dim: int,
        kernel_size: List[int],
        stride: List[int],
        dilation: List[int],
        is_causal: List[bool],
        scale: Optional[float],
        backend: Optional[str],
        q_tile_shape: Optional[List[int]],
        kv_tile_shape: Optional[List[int]],
        backward_q_tile_shape: Optional[List[int]],
        backward_kv_tile_shape: Optional[List[int]],
        backward_kv_splits: Optional[List[int]],
        backward_use_pt_reduction: bool,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Opaque backward: replays the entry point on detached inputs and returns
        its gradients.

        An operator, rather than plain Python inside the ``register_autograd``
        callback, because that callback *is* traced by AOTAutograd, where the layout
        is a ``FakeScriptObject`` whose contents cannot be read and q/k/v are fake.
        The replay needs the real layout and real tensors, which only an operator
        body gets, same as the forward's -- which is also why the replay needs
        ``_recording_autograd`` to get a graph at all.

        Replaying instead of driving the kernels from a saved output/logsumexp
        keeps the numerics identical to the entry point's own backward by
        construction, degenerate-axis lowering included (a hand-written backward
        would have to reproduce the fold, the permute and its inverse, and the
        uniform path's separate fixed-shape configuration). It costs one extra
        forward per backward.

        This operator registers no autograd of its own, so
        ``backward(create_graph=True)`` is unsupported -- same as the stock path,
        whose ``VarlenCutlassFNAAutogradFn.backward`` is not differentiable either.
        """
        with _recording_autograd(), torch.enable_grad():
            query_ = query.detach().requires_grad_(True)
            key_ = key.detach().requires_grad_(True)
            value_ = value.detach().requires_grad_(True)
            output = _stock_call(
                na_dim,
                query_,
                key_,
                value_,
                layout,
                kernel_size,
                stride,
                dilation,
                is_causal,
                scale,
                backend,
                q_tile_shape,
                kv_tile_shape,
                backward_q_tile_shape,
                backward_kv_tile_shape,
                backward_kv_splits,
                backward_use_pt_reduction,
                False,
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
        layout: VarlenLayout,
        na_dim: int,
        kernel_size: List[int],
        stride: List[int],
        dilation: List[int],
        is_causal: List[bool],
        scale: Optional[float],
        backend: Optional[str],
        q_tile_shape: Optional[List[int]],
        kv_tile_shape: Optional[List[int]],
        backward_q_tile_shape: Optional[List[int]],
        backward_kv_tile_shape: Optional[List[int]],
        backward_kv_splits: Optional[List[int]],
        backward_use_pt_reduction: bool,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        return (
            torch.empty_like(query),
            torch.empty_like(key),
            torch.empty_like(value),
        )

    def _varlen_attention_setup_context(ctx: Any, inputs: Any, output: Any) -> None:
        query, key, value, layout = inputs[:4]
        ctx.save_for_backward(query, key, value)
        # save_for_backward takes tensors only; the layout rides on ctx, and
        # AOTAutograd saves it for the backward graph like any non-tensor value.
        ctx.layout = layout
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
            return (None,) * 17
        query, key, value = ctx.saved_tensors
        d_query, d_key, d_value = _varlen_attention_bwd(
            query, key, value, grad_output, ctx.layout, *ctx.varlen_args
        )
        return (d_query, d_key, d_value) + (None,) * 14

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


def _as_opt_int_list(value: Any) -> Optional[List[int]]:
    return None if value is None else [int(item) for item in value]


def _varlen_compiled_call(
    na_dim: int,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    layout: VarlenLayout,
    kernel_size: Any,
    stride: Any,
    dilation: Any,
    is_causal: Any,
    scale: Optional[float],
    backend: Optional[str],
    q_tile_shape: Any,
    kv_tile_shape: Any,
    backward_q_tile_shape: Any,
    backward_kv_tile_shape: Any,
    backward_kv_splits: Any,
    backward_use_pt_reduction: bool,
    return_lse: bool,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """What ``na{1,2,3}d_varlen`` runs while it is being traced.

    The layout object itself goes into the graph, and nothing is read off it:
    dynamo guards it by its type alone, so nothing that varies with the packing
    is ever a Python value in the graph. Everything else is handed through
    unchanged.
    """
    output, logsumexp = _varlen_attention_fwd(
        query,
        key,
        value,
        layout,
        na_dim,
        _as_int_list(kernel_size, na_dim, "kernel_size"),
        _as_int_list(stride, na_dim, "stride"),
        _as_int_list(dilation, na_dim, "dilation"),
        _as_bool_list(is_causal, na_dim),
        None if scale is None else float(scale),
        backend,
        _as_opt_int_list(q_tile_shape),
        _as_opt_int_list(kv_tile_shape),
        _as_opt_int_list(backward_q_tile_shape),
        _as_opt_int_list(backward_kv_tile_shape),
        _as_opt_int_list(backward_kv_splits),
        bool(backward_use_pt_reduction),
    )
    if return_lse:
        return output, logsumexp
    return output
