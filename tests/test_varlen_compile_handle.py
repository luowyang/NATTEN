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
"""natten.varlen_compile: the handle registry, parity with the stock varlen
entry points, and the property the whole module exists for -- a call whose
document geometry changes does not recompile.

Parity is asserted bitwise against na{1,2,3}d_varlen because the operator body
calls exactly those entry points; anything but equality means the wrapper
changed an argument on the way in.
"""

import gc
import math
import pickle
from dataclasses import dataclass
from typing import Tuple

import natten
import pytest
import torch
from natten._environment import _IS_CUDA_AVAILABLE, HAS_LIBNATTEN
from natten.varlen_compile import _REGISTRY

# CompileCounter is a private torch._dynamo.testing utility; the sibling
# test_varlen_layout.py depends on it for the same reason (counting compiled
# frames is not otherwise observable).
from torch._dynamo.testing import CompileCounter

SEED = 20260922

# The sibling suites' skip_if_libnatten_is_not_supported is a unittest method
# decorator (it calls self.skipTest); these are plain pytest functions.
requires_libnatten = pytest.mark.skipif(
    not (_IS_CUDA_AVAILABLE and HAS_LIBNATTEN),
    reason="CUDA and libnatten are required",
)


@dataclass(frozen=True)
class HandleCase:
    name: str
    shapes: Tuple[Tuple[int, ...], ...]
    kernel_size: Tuple[int, ...]
    is_causal: Tuple[bool, ...]
    heads: int = 4
    heads_kv: int = 4
    head_dim: int = 32
    head_dim_v: int = 32

    @property
    def rank(self) -> int:
        return len(self.shapes[0])

    @property
    def total_tokens(self) -> int:
        return sum(math.prod(shape) for shape in self.shapes)


# One case per path _neighborhood_attention_varlen_generic can take, since the
# operator wraps that whole function: the varlen kernel, both degenerate-axis
# lowerings, the fully-degenerate identity, the uniform fixed-shape dispatch,
# the all-empty short circuit, plus GQA and the other two ranks.
CASES = (
    HandleCase("3d-varlen", ((2, 4, 4), (1, 6, 5)), (2, 3, 3), (True, False, False)),
    HandleCase("3d-fold", ((2, 4, 4), (1, 6, 5)), (1, 3, 3), (False, False, False)),
    HandleCase("3d-permute", ((3, 4, 4), (5, 6, 5)), (3, 1, 1), (True, False, False)),
    HandleCase("3d-identity", ((2, 4, 4), (1, 6, 5)), (1, 1, 1), (False, False, False)),
    HandleCase("3d-uniform", ((2, 4, 4), (2, 4, 4)), (2, 3, 3), (False, False, False)),
    HandleCase("3d-empty", ((0, 4, 4), (0, 2, 2)), (2, 3, 3), (False, False, False)),
    HandleCase(
        "3d-gqa",
        ((2, 4, 4), (1, 6, 5)),
        (2, 3, 3),
        (False, False, False),
        heads=4,
        heads_kv=2,
        head_dim_v=64,
    ),
    HandleCase("2d-varlen", ((4, 5), (3, 7)), (3, 3), (False, False)),
    HandleCase("1d-varlen", ((17,), (9,)), (5,), (True,)),
)


def _entry_points(rank: int):
    return (
        getattr(natten, f"na{rank}d_varlen"),
        getattr(natten, f"na{rank}d_varlen_handle"),
    )


def _inputs(case: HandleCase, dtype: torch.dtype, requires_grad: bool = False):
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    total = case.total_tokens
    tensors = []
    for heads, head_dim in (
        (case.heads, case.head_dim),
        (case.heads_kv, case.head_dim),
        (case.heads_kv, case.head_dim_v),
    ):
        raw = torch.randn((total, heads, head_dim), generator=generator)
        tensor = raw.to(device="cuda", dtype=dtype)
        tensors.append(tensor.requires_grad_(requires_grad))
    return tensors


def _call_kwargs(case: HandleCase):
    return {
        "kernel_size": case.kernel_size,
        "is_causal": case.is_causal,
        "scale": 0.125,
    }


# ------------------------------------------------------------------ registry


def test_handle_registers_and_releases_with_the_object():
    layout = natten.VarlenLayout(((2, 3), (1, 4)))
    handle = natten.VarlenLayoutHandle(layout)
    key = int(handle.tensor)

    assert handle.tensor.dtype == torch.int64
    assert handle.tensor.dim() == 0
    assert handle.tensor.device.type == "cpu"
    assert handle.layout is layout
    assert _REGISTRY[key] is layout

    del handle
    gc.collect()
    assert key not in _REGISTRY


def test_two_handles_for_one_layout_get_distinct_ids():
    layout = natten.VarlenLayout(((2, 3),))
    first = natten.VarlenLayoutHandle(layout)
    second = natten.VarlenLayoutHandle(layout)
    assert int(first.tensor) != int(second.tensor)
    assert first.layout is second.layout


def test_handle_rejects_non_layout():
    with pytest.raises(TypeError):
        natten.VarlenLayoutHandle(((2, 3),))  # type: ignore[arg-type]


def test_handle_is_not_picklable():
    handle = natten.VarlenLayoutHandle(natten.VarlenLayout(((2, 3),)))
    with pytest.raises(TypeError, match="not picklable"):
        pickle.dumps(handle)


def test_released_handle_tensor_fails_loudly():
    handle = natten.VarlenLayoutHandle(natten.VarlenLayout(((4,),)))
    tensor = handle.tensor
    del handle
    gc.collect()
    query = torch.zeros(4, 1, 32)
    with pytest.raises(RuntimeError, match="not registered"):
        natten.na1d_varlen_handle(query, query, query, tensor, kernel_size=(2,))


def test_malformed_handle_tensor_is_rejected():
    query = torch.zeros(4, 1, 32)
    with pytest.raises(ValueError, match="0-dim int64"):
        natten.na1d_varlen_handle(
            query, query, query, torch.zeros(1, dtype=torch.int64), kernel_size=(2,)
        )


def test_handle_must_be_a_tensor():
    handle = natten.VarlenLayoutHandle(natten.VarlenLayout(((4,),)))
    query = torch.zeros(4, 1, 32)
    with pytest.raises(TypeError, match="handle.tensor"):
        natten.na1d_varlen_handle(query, query, query, handle, kernel_size=(2,))


def test_boolean_window_arguments_are_rejected_before_the_schema_coerces_them():
    handle = natten.VarlenLayoutHandle(natten.VarlenLayout(((4,),)))
    query = torch.zeros(4, 1, 32)
    with pytest.raises(TypeError, match="not booleans"):
        natten.na1d_varlen_handle(
            query, query, query, handle.tensor, kernel_size=(True,)
        )


def test_unsupported_backend_is_rejected():
    handle = natten.VarlenLayoutHandle(natten.VarlenLayout(((4,),)))
    query = torch.zeros(4, 1, 32)
    with pytest.raises(NotImplementedError):
        natten.na1d_varlen_handle(
            query, query, query, handle.tensor, kernel_size=(2,), backend="flex-fna"
        )


# ------------------------------------------------------------------- parity


@requires_libnatten
@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("return_lse", [False, True])
def test_forward_matches_stock_entry_point_bitwise(case, dtype, return_lse):
    stock, handled = _entry_points(case.rank)
    query, key, value = _inputs(case, dtype)
    layout = natten.VarlenLayout(case.shapes)
    handle = natten.VarlenLayoutHandle(layout)

    # The handle stays on the CPU while q/k/v are on the GPU: the operator has
    # to accept that mix, so every call below exercises it.
    assert handle.tensor.device.type == "cpu"
    assert query.device.type == "cuda"

    expected = stock(
        query, key, value, layout, return_lse=return_lse, **_call_kwargs(case)
    )
    actual = handled(
        query, key, value, handle.tensor, return_lse=return_lse, **_call_kwargs(case)
    )

    if return_lse:
        for name, want, got in zip(("output", "logsumexp"), expected, actual):
            assert torch.equal(want, got), name
    else:
        assert torch.equal(expected, actual)


@requires_libnatten
@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_backward_matches_stock_entry_point_bitwise(case, dtype):
    # Deterministic mode pins the backward's KV-split selection to 1 per axis
    # (backends.varlen_fna._build_varlen_fna_state), without which the split
    # reduction makes "bitwise" undefined run to run, for the stock path as
    # much as for this one.
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        stock, handled = _entry_points(case.rank)
        layout = natten.VarlenLayout(case.shapes)
        handle = natten.VarlenLayoutHandle(layout)
        kwargs = _call_kwargs(case)

        stock_inputs = _inputs(case, dtype, requires_grad=True)
        output = stock(*stock_inputs, layout, **kwargs)
        grad = torch.ones_like(output)
        expected = torch.autograd.grad(output, stock_inputs, grad)

        handle_inputs = _inputs(case, dtype, requires_grad=True)
        output = handled(*handle_inputs, handle.tensor, **kwargs)
        actual = torch.autograd.grad(output, handle_inputs, grad)

        for name, want, got in zip(("dq", "dk", "dv"), expected, actual):
            assert torch.equal(want, got), name
    finally:
        torch.use_deterministic_algorithms(previous)


@requires_libnatten
def test_fully_degenerate_output_does_not_alias_value():
    # The fully-degenerate path's output is exactly `value`, and a custom op
    # may not return a view of an input.
    case = HandleCase("alias", ((2, 4, 4), (1, 6, 5)), (1, 1, 1), (False, False, False))
    query, key, value = _inputs(case, torch.float32)
    handle = natten.VarlenLayoutHandle(natten.VarlenLayout(case.shapes))
    output = natten.na3d_varlen_handle(
        query, key, value, handle.tensor, **_call_kwargs(case)
    )
    assert output.data_ptr() != value.data_ptr()
    assert torch.equal(output, value)


@requires_libnatten
def test_opcheck_accepts_the_operator():
    case = CASES[0]
    query, key, value = _inputs(case, torch.float32, requires_grad=True)
    handle = natten.VarlenLayoutHandle(natten.VarlenLayout(case.shapes))
    torch.library.opcheck(
        torch.ops.natten.varlen_attention_fwd,
        (
            query,
            key,
            value,
            handle.tensor,
            3,
            list(case.kernel_size),
            [1, 1, 1],
            [1, 1, 1],
            list(case.is_causal),
            0.125,
            False,
        ),
    )


# ------------------------------------------------------------------ compile


def _compiled_geometry_stream(attention, layout_argument):
    """Runs two different packings through one compiled callable."""

    def run(query, key, value, layout_like):
        return attention(
            query, key, value, layout_like, kernel_size=(2, 3, 3), scale=0.125
        )

    counter = CompileCounter()
    compiled = torch.compile(run, backend=counter, fullgraph=True, dynamic=True)
    for shapes in (((2, 4, 4), (1, 6, 5)), ((3, 4, 4), (2, 6, 5), (1, 2, 2))):
        case = HandleCase("stream", shapes, (2, 3, 3), (False, False, False))
        query, key, value = _inputs(case, torch.float32)
        compiled(query, key, value, layout_argument(shapes))
    return counter.frame_count


@requires_libnatten
def test_changing_geometry_compiles_once_with_a_handle():
    torch._dynamo.reset()
    handles = []

    def layout_argument(shapes):
        handle = natten.VarlenLayoutHandle(natten.VarlenLayout(shapes))
        handles.append(handle)  # keep alive for the duration of the run
        return handle.tensor

    try:
        frames = _compiled_geometry_stream(natten.na3d_varlen_handle, layout_argument)
    finally:
        torch._dynamo.reset()
    assert frames == 1


@requires_libnatten
def test_changing_geometry_recompiles_without_a_handle():
    # Negative control for the test above: with the layout passed as a Python
    # object, the same stream specializes. Without this, a stream that never
    # reached the compiler at all would pass the positive test.
    torch._dynamo.reset()
    try:
        frames = _compiled_geometry_stream(
            natten.na3d_varlen, lambda shapes: natten.VarlenLayout(shapes)
        )
    finally:
        torch._dynamo.reset()
    assert frames > 1


@requires_libnatten
def test_backward_survives_aot_autograd():
    # The registered backward is traced by AOTAutograd, unlike the forward
    # operator's body: dereferencing the handle there (rather than inside
    # natten::varlen_attention_bwd) raises GuardOnDataDependentSymNode on
    # .item(). This is that regression.
    torch._dynamo.reset()
    case = CASES[0]
    handle = natten.VarlenLayoutHandle(natten.VarlenLayout(case.shapes))
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)

    def run(query, key, value, layout_handle):
        output = natten.na3d_varlen_handle(
            query, key, value, layout_handle, **_call_kwargs(case)
        )
        return output.square().sum()

    try:
        compiled = torch.compile(run, backend="aot_eager", fullgraph=True, dynamic=True)
        inputs = _inputs(case, torch.float32, requires_grad=True)
        compiled(*inputs, handle.tensor).backward()
        for name, tensor in zip(("dq", "dk", "dv"), inputs):
            assert tensor.grad is not None, name

        eager_inputs = _inputs(case, torch.float32, requires_grad=True)
        run(*eager_inputs, handle.tensor).backward()
        for name, compiled_input, eager_input in zip(
            ("dq", "dk", "dv"), inputs, eager_inputs
        ):
            assert torch.equal(compiled_input.grad, eager_input.grad), name
    finally:
        torch.use_deterministic_algorithms(previous)
        torch._dynamo.reset()
