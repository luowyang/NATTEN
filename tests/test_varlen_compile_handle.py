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
"""na{1,2,3}d_varlen under torch.compile: same answers, one graph.

The entry points are the whole public surface; compile-safety is a property
they have, not a second set of names. So everything here is stated against
``na{1,2,3}d_varlen`` itself -- compiled against eager -- and the operator
underneath (``natten.varlen_compile``) is only visible where the test is about
the operator contract (opcheck) or about what the captured graph contains.

Parity is asserted bitwise because the operator body calls the entry point:
anything but equality means an argument changed meaning on the way in.
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
from natten.varlen import _HANDLE_REGISTRY

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
# the all-empty short circuit, plus GQA and the other two ranks. The uniform,
# all-empty and GQA cases are also the sentinels for _recording_autograd: they
# are the paths whose last step is a plain tensor op, which come back with no
# gradient at all if the operator body stops re-entering autograd.
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


def _entry_point(rank: int):
    return getattr(natten, f"na{rank}d_varlen")


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


def _compiled(rank: int, backend="aot_eager"):
    return torch.compile(
        _entry_point(rank), backend=backend, fullgraph=True, dynamic=True
    )


# --------------------------------------------------- layout handle lifetime


def test_layout_registers_a_handle_at_construction():
    layout = natten.VarlenLayout(((2, 3), (1, 4)))
    key = int(layout._handle)

    assert layout._handle.dtype == torch.int64
    assert layout._handle.dim() == 0
    assert layout._handle.device.type == "cpu"
    assert _HANDLE_REGISTRY[key] is layout


def test_registry_entry_dies_with_the_layout():
    layout = natten.VarlenLayout(((2, 3),))
    key = int(layout._handle)
    del layout
    gc.collect()
    assert key not in _HANDLE_REGISTRY


def test_two_layouts_get_distinct_handles():
    first = natten.VarlenLayout(((2, 3),))
    second = natten.VarlenLayout(((2, 3),))
    assert int(first._handle) != int(second._handle)


def test_pickle_round_trip_reregisters():
    # A layout pickled to a DataLoader worker or another rank has to keep
    # working there; the handle id means nothing across processes, so the
    # unpickled layout takes a fresh one.
    layout = natten.VarlenLayout(((2, 3), (1, 4)))
    restored = pickle.loads(pickle.dumps(layout))
    assert restored.shapes == layout.shapes
    assert int(restored._handle) != int(layout._handle)
    assert _HANDLE_REGISTRY[int(restored._handle)] is restored


def test_stale_handle_fails_loudly():
    from natten.varlen import _layout_from_handle

    layout = natten.VarlenLayout(((4,),))
    handle = layout._handle
    del layout
    gc.collect()
    with pytest.raises(RuntimeError, match="not registered"):
        _layout_from_handle(handle)


def test_malformed_handle_is_rejected():
    from natten.varlen import _layout_from_handle

    with pytest.raises(ValueError, match="0-dim int64"):
        _layout_from_handle(torch.zeros(1, dtype=torch.int64))


def test_no_handle_names_are_public():
    # The compile path must not add public API. This is the invariant the
    # rework exists for.
    public = set(natten.__all__)
    assert not [name for name in public if "handle" in name.lower()]
    assert {"VarlenLayout", "na1d_varlen", "na2d_varlen", "na3d_varlen"} <= public
    for name in ("VarlenLayoutHandle", "na3d_varlen_handle"):
        assert not hasattr(natten, name), name


# ------------------------------------------------------------------- parity


@requires_libnatten
@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("return_lse", [False, True])
def test_compiled_forward_matches_eager_bitwise(case, dtype, return_lse):
    torch._dynamo.reset()
    try:
        entry = _entry_point(case.rank)
        query, key, value = _inputs(case, dtype)
        layout = natten.VarlenLayout(case.shapes)

        # The handle is CPU while q/k/v are CUDA: the operator has to take
        # that mix, so every case below exercises it.
        assert layout._handle.device.type == "cpu"
        assert query.device.type == "cuda"

        expected = entry(
            query, key, value, layout, return_lse=return_lse, **_call_kwargs(case)
        )
        actual = _compiled(case.rank)(
            query, key, value, layout, return_lse=return_lse, **_call_kwargs(case)
        )

        if return_lse:
            for name, want, got in zip(("output", "logsumexp"), expected, actual):
                assert torch.equal(want, got), name
        else:
            assert torch.equal(expected, actual)
    finally:
        torch._dynamo.reset()


@requires_libnatten
@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_compiled_backward_matches_eager_bitwise(case, dtype):
    # Deterministic mode pins the backward's KV-split selection to 1 per axis
    # (backends.varlen_fna._build_varlen_fna_state), without which the split
    # reduction makes "bitwise" undefined run to run, for the eager path as
    # much as for this one.
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    torch._dynamo.reset()
    try:
        entry = _entry_point(case.rank)
        layout = natten.VarlenLayout(case.shapes)
        kwargs = _call_kwargs(case)

        eager_inputs = _inputs(case, dtype, requires_grad=True)
        output = entry(*eager_inputs, layout, **kwargs)
        grad = torch.ones_like(output)
        expected = torch.autograd.grad(output, eager_inputs, grad)

        compiled_inputs = _inputs(case, dtype, requires_grad=True)
        output = _compiled(case.rank)(*compiled_inputs, layout, **kwargs)
        actual = torch.autograd.grad(output, compiled_inputs, grad)

        for name, want, got in zip(("dq", "dk", "dv"), expected, actual):
            assert torch.equal(want, got), name
    finally:
        torch.use_deterministic_algorithms(previous)
        torch._dynamo.reset()


@requires_libnatten
def test_eager_is_untouched_by_the_compile_branch():
    # Nothing about the eager path may change: same call, no compile in sight.
    case = CASES[0]
    query, key, value = _inputs(case, torch.float32)
    layout = natten.VarlenLayout(case.shapes)
    first = natten.na3d_varlen(query, key, value, layout, **_call_kwargs(case))
    second = natten.na3d_varlen(query, key, value, layout, **_call_kwargs(case))
    assert torch.equal(first, second)
    assert not torch.compiler.is_compiling()


# ----------------------------------------------------------------- operator


def _op_args(case: HandleCase, query, key, value, layout):
    return (
        query,
        key,
        value,
        layout._handle,
        case.rank,
        list(case.kernel_size),
        [1] * case.rank,
        [1] * case.rank,
        list(case.is_causal),
        0.125,
        None,
        None,
        None,
        None,
        None,
        None,
        False,
    )


@requires_libnatten
@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_opcheck_accepts_the_forward_operator(case):
    # Every case, not just the plain varlen kernel: inductor takes the fake
    # kernel's metadata -- shape, dtype and strides -- at face value for
    # everything downstream, and the lowered paths (fold, permute, identity,
    # uniform, all-empty) each build their output differently.
    query, key, value = _inputs(case, torch.float32, requires_grad=True)
    layout = natten.VarlenLayout(case.shapes)
    torch.library.opcheck(
        torch.ops.natten.varlen_attention_fwd,
        _op_args(case, query, key, value, layout),
    )


@requires_libnatten
@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_opcheck_accepts_the_backward_operator(case):
    query, key, value = _inputs(case, torch.float32)
    grad_output = torch.ones(
        (case.total_tokens, case.heads, case.head_dim_v),
        device="cuda",
        dtype=torch.float32,
    )
    layout = natten.VarlenLayout(case.shapes)
    args = _op_args(case, query, key, value, layout)
    torch.library.opcheck(
        torch.ops.natten.varlen_attention_bwd,
        (*args[:3], grad_output, *args[3:]),
        # This operator *is* a backward and registers no autograd of its own,
        # so the autograd-registration check has nothing to check here.
        test_utils=("test_schema", "test_faketensor", "test_aot_dispatch_dynamic"),
    )


@requires_libnatten
def test_fully_degenerate_output_does_not_alias_value():
    # The fully-degenerate path's output is exactly `value`, and a custom op
    # may not return a view of an input.
    case = HandleCase("alias", ((2, 4, 4), (1, 6, 5)), (1, 1, 1), (False, False, False))
    query, key, value = _inputs(case, torch.float32)
    layout = natten.VarlenLayout(case.shapes)
    output = _compiled(3)(query, key, value, layout, **_call_kwargs(case))
    assert output.data_ptr() != value.data_ptr()
    assert torch.equal(output, value)


@requires_libnatten
def test_explicit_tile_shapes_reach_the_compiled_call():
    # The operator schema carries the tile knobs, so a compiled call accepts
    # exactly what an eager one does -- including the arguments that make the
    # eager path raise, which must raise the same way.
    case = CASES[0]
    query, key, value = _inputs(case, torch.float32)
    layout = natten.VarlenLayout(case.shapes)
    kwargs = dict(_call_kwargs(case), q_tile_shape=(2, 4, 4), kv_tile_shape=(2, 4, 4))
    expected = natten.na3d_varlen(query, key, value, layout, **kwargs)
    actual = _compiled(3)(query, key, value, layout, **kwargs)
    assert torch.equal(expected, actual)


# ------------------------------------------------------------------ compile


@requires_libnatten
def test_changing_geometry_compiles_once():
    torch._dynamo.reset()
    counter = CompileCounter()
    compiled = torch.compile(
        natten.na3d_varlen, backend=counter, fullgraph=True, dynamic=True
    )
    layouts = []
    try:
        for shapes in (((2, 4, 4), (1, 6, 5)), ((3, 4, 4), (2, 6, 5), (1, 2, 2))):
            case = HandleCase("stream", shapes, (2, 3, 3), (False, False, False))
            query, key, value = _inputs(case, torch.float32)
            layout = natten.VarlenLayout(shapes)
            layouts.append(layout)  # keep alive for the duration of the run
            compiled(query, key, value, layout, kernel_size=(2, 3, 3), scale=0.125)
    finally:
        torch._dynamo.reset()
    assert counter.frame_count == 1


@requires_libnatten
def test_captured_graph_holds_exactly_one_natten_op():
    # The positive form of "the layout was not traced": whatever dynamo
    # captured contains the one opaque call and nothing else of natten's --
    # no schedule build, no lowering, no second kernel dispatch.
    captured = []

    def inspecting_backend(gm, example_inputs):
        names = []
        for node in gm.graph.nodes:
            if node.op != "call_function":
                continue
            name = getattr(node.target, "name", None)
            qualified = name() if callable(name) else str(node.target)
            if "natten" in qualified:
                names.append(qualified)
        captured.append(names)
        return gm.forward

    torch._dynamo.reset()
    case = CASES[0]
    layout = natten.VarlenLayout(case.shapes)
    try:
        compiled = torch.compile(
            natten.na3d_varlen,
            backend=inspecting_backend,
            fullgraph=True,
            dynamic=True,
        )
        compiled(*_inputs(case, torch.float32), layout, **_call_kwargs(case))
    finally:
        torch._dynamo.reset()

    assert len(captured) == 1, captured
    assert captured[0] == ["natten::varlen_attention_fwd"], captured[0]


@requires_libnatten
def test_backward_survives_aot_autograd():
    # The registered backward is traced by AOTAutograd, unlike the forward
    # operator's body: dereferencing the handle there (rather than inside
    # natten::varlen_attention_bwd) raises GuardOnDataDependentSymNode on
    # .item(). This is that regression.
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    torch._dynamo.reset()
    case = CASES[0]
    layout = natten.VarlenLayout(case.shapes)

    def run(entry, query, key, value):
        return entry(query, key, value, layout, **_call_kwargs(case)).square().sum()

    try:
        inputs = _inputs(case, torch.float32, requires_grad=True)
        run(_compiled(3), *inputs).backward()
        for name, tensor in zip(("dq", "dk", "dv"), inputs):
            assert tensor.grad is not None, name

        eager_inputs = _inputs(case, torch.float32, requires_grad=True)
        run(natten.na3d_varlen, *eager_inputs).backward()
        for name, compiled_input, eager_input in zip(
            ("dq", "dk", "dv"), inputs, eager_inputs
        ):
            assert torch.equal(compiled_input.grad, eager_input.grad), name
    finally:
        torch.use_deterministic_algorithms(previous)
        torch._dynamo.reset()
