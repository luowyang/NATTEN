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

The layout reaches that operator as itself: ``VarlenLayout`` is a reference-type
opaque object, which dynamo guards by its type alone.

Parity is asserted bitwise because the operator body calls the entry point:
anything but equality means an argument changed meaning on the way in.
"""

import copy
import inspect
import math
import pickle
from dataclasses import dataclass
from typing import Tuple

import natten
import pytest
import torch
from natten._environment import _IS_CUDA_AVAILABLE, HAS_LIBNATTEN
from natten.varlen_compile import _LAYOUT_IS_OPAQUE

# CompileCounter is a private torch._dynamo.testing utility; the sibling
# test_varlen_layout.py depends on it for the same reason (counting compiled
# frames is not otherwise observable).
from torch._dynamo.testing import CompileCounter

SEED = 20260922

# The operator path exists only where torch can pass a VarlenLayout to an
# operator, and everything here is about that path.
pytestmark = pytest.mark.skipif(
    not _LAYOUT_IS_OPAQUE,
    reason="this torch cannot pass a VarlenLayout to a custom op",
)

# The sibling suites' skip_if_libnatten_is_not_supported is a unittest method
# decorator (it calls self.skipTest); these are plain pytest functions.
requires_libnatten = pytest.mark.skipif(
    not (_IS_CUDA_AVAILABLE and HAS_LIBNATTEN),
    reason="CUDA and libnatten are required",
)


@dataclass(frozen=True)
class VarlenCase:
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
    VarlenCase("3d-varlen", ((2, 4, 4), (1, 6, 5)), (2, 3, 3), (True, False, False)),
    VarlenCase("3d-fold", ((2, 4, 4), (1, 6, 5)), (1, 3, 3), (False, False, False)),
    VarlenCase("3d-permute", ((3, 4, 4), (5, 6, 5)), (3, 1, 1), (True, False, False)),
    VarlenCase("3d-identity", ((2, 4, 4), (1, 6, 5)), (1, 1, 1), (False, False, False)),
    VarlenCase("3d-uniform", ((2, 4, 4), (2, 4, 4)), (2, 3, 3), (False, False, False)),
    VarlenCase("3d-empty", ((0, 4, 4), (0, 2, 2)), (2, 3, 3), (False, False, False)),
    VarlenCase(
        "3d-gqa",
        ((2, 4, 4), (1, 6, 5)),
        (2, 3, 3),
        (False, False, False),
        heads=4,
        heads_kv=2,
        head_dim_v=64,
    ),
    VarlenCase("2d-varlen", ((4, 5), (3, 7)), (3, 3), (False, False)),
    VarlenCase("1d-varlen", ((17,), (9,)), (5,), (True,)),
)


def _entry_point(rank: int):
    return getattr(natten, f"na{rank}d_varlen")


def _inputs(case: VarlenCase, dtype: torch.dtype, requires_grad: bool = False):
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


def _call_kwargs(case: VarlenCase):
    return {
        "kernel_size": case.kernel_size,
        "is_causal": case.is_causal,
        "scale": 0.125,
    }


def _compiled(rank: int, backend="aot_eager"):
    return torch.compile(
        _entry_point(rank), backend=backend, fullgraph=True, dynamic=True
    )


# ---------------------------------------------------------- layout object

# natten's public surface. Compile support lives inside na{1,2,3}d_varlen and
# VarlenLayout, so it adds nothing here; a new name has to be a deliberate edit
# of these two sets.
PUBLIC_NAMES = frozenset(
    {
        "__version__",
        "NeighborhoodAttention1D",
        "NeighborhoodAttention2D",
        "NeighborhoodAttention3D",
        "are_deterministic_algorithms_enabled",
        "use_deterministic_algorithms",
        "use_kv_parallelism_in_fused_na",
        "is_kv_parallelism_in_fused_na_enabled",
        "set_memory_usage_preference",
        "get_memory_usage_preference",
        "is_memory_usage_default",
        "is_memory_usage_strict",
        "is_memory_usage_unrestricted",
        "is_flex_compile_allowed",
        "is_flex_compile_backprop_allowed",
        "allow_flex_compile",
        "allow_flex_compile_backprop",
        "disable_flex_compile",
        "disable_flex_compile_backprop",
        "get_bwd_configs_for_cutlass_fmha",
        "get_bwd_configs_for_cutlass_fna",
        "get_configs_for_cutlass_fmha",
        "get_configs_for_cutlass_fna",
        "get_configs_for_cutlass_hopper_fmha",
        "get_bwd_configs_for_cutlass_hopper_fmha",
        "get_configs_for_cutlass_hopper_fna",
        "get_bwd_configs_for_cutlass_hopper_fna",
        "get_bwd_configs_for_cutlass_blackwell_fmha",
        "get_bwd_configs_for_cutlass_blackwell_fna",
        "get_configs_for_cutlass_blackwell_fmha",
        "get_configs_for_cutlass_blackwell_fna",
        "get_configs_for_flex_fmha",
        "get_configs_for_flex_fna",
        "HAS_LIBNATTEN",
        "na1d",
        "na2d",
        "na3d",
        "na1d_varlen",
        "na2d_varlen",
        "na3d_varlen",
        "attention",
        "merge_attentions",
        "VarlenLayout",
    }
)
LAYOUT_PUBLIC_MEMBERS = frozenset(
    {
        "cu_seqlens",
        "device",
        "from_tensor_list",
        "is_uniform",
        "max_seqlen",
        "num_docs",
        "rank",
        "shapes",
        "split",
        "token_layouts",
        "total_tokens",
        "uniform_shape",
    }
)


def test_public_surface_is_unchanged():
    public = set(natten.__all__)
    assert public == PUBLIC_NAMES, (public - PUBLIC_NAMES, PUBLIC_NAMES - public)
    members = {name for name in dir(natten.VarlenLayout) if not name.startswith("_")}
    assert members == LAYOUT_PUBLIC_MEMBERS, members ^ LAYOUT_PUBLIC_MEMBERS
    parameters = inspect.signature(natten.VarlenLayout).parameters
    assert list(parameters) == ["shapes", "device"]


def test_layout_is_a_reference_opaque_type():
    from torch._library.opaque_object import is_opaque_reference_type

    assert is_opaque_reference_type(natten.VarlenLayout)


def _used_layout(case):
    # A layout that has run: materialized on the device, with derived state.
    layout = natten.VarlenLayout(case.shapes)
    entry = _entry_point(case.rank)
    entry(*_inputs(case, torch.float32), layout, **_call_kwargs(case))
    assert layout.device is not None
    assert layout._memo or layout._fold_memo or layout._permute_memo
    return layout


def _assert_host_shapes_only(layout: natten.VarlenLayout) -> None:
    tensors = [
        name for name, value in vars(layout).items() if isinstance(value, torch.Tensor)
    ]
    assert not tensors, tensors
    assert layout.device is None
    assert layout._memo == {}
    assert layout._fold_memo == {}
    assert layout._permute_memo == {}


@requires_libnatten
def test_pickle_round_trip_carries_shapes_only():
    # A layout pickled to a DataLoader worker or another rank takes its host
    # shapes and nothing else; the copy rebuilds derived state on first use.
    layout = _used_layout(CASES[0])
    assert layout.__getstate__() == {"shapes": layout.shapes}
    restored = pickle.loads(pickle.dumps(layout))
    assert restored.shapes == layout.shapes
    _assert_host_shapes_only(restored)


@requires_libnatten
def test_deepcopy_carries_shapes_only():
    # torch deep-copies a layout into the FakeScriptObject a traced operator
    # sees: that copy must not carry device tensors or the memo.
    layout = _used_layout(CASES[0])
    copied = copy.deepcopy(layout)
    assert copied is not layout
    assert copied.shapes == layout.shapes
    _assert_host_shapes_only(copied)


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


def _op_args(case: VarlenCase, query, key, value, layout):
    return (
        query,
        key,
        value,
        layout,
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
    case = VarlenCase("alias", ((2, 4, 4), (1, 6, 5)), (1, 1, 1), (False, False, False))
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
    # Ask the library which tile shapes this device/dtype/head_dim supports
    # rather than hardcoding a pair that may not be in the set.
    probe = query.new_zeros((1, 1, 1, 1, case.heads, case.head_dim))
    configs = natten.get_configs_for_cutlass_fna(probe, probe, probe)
    if not configs:
        pytest.skip("no CUTLASS FNA forward config for this device")
    q_tile, kv_tile = configs[0]
    kwargs = dict(_call_kwargs(case), q_tile_shape=q_tile, kv_tile_shape=kv_tile)
    expected = natten.na3d_varlen(query, key, value, layout, **kwargs)
    actual = _compiled(3)(query, key, value, layout, **kwargs)
    assert torch.equal(expected, actual)


# ------------------------------------------------------------------ compile


# Different document counts and shapes, each carried by its own layout.
GEOMETRIES = (
    ((2, 4, 4), (1, 6, 5)),
    ((3, 4, 4), (2, 6, 5), (1, 2, 2)),
    ((1, 8, 8), (2, 5, 7), (3, 4, 4), (1, 3, 9)),
)


@requires_libnatten
def test_changing_geometry_compiles_once():
    # CompileCounter counts frames; the cache entries' guard trees are the only
    # place the installed guards can be read back (private, like the counter).
    from torch._dynamo.eval_frame import _debug_get_cache_entry_list

    torch._dynamo.reset()
    counter = CompileCounter()
    compiled = torch.compile(
        natten.na3d_varlen, backend=counter, fullgraph=True, dynamic=True
    )
    try:
        for shapes in GEOMETRIES:
            case = VarlenCase("stream", shapes, (2, 3, 3), (False, False, False))
            query, key, value = _inputs(case, torch.float32)
            layout = natten.VarlenLayout(shapes)
            compiled(query, key, value, layout, kernel_size=(2, 3, 3), scale=0.125)
        guards = "\n".join(
            str(entry.guard_manager)
            for entry in _debug_get_cache_entry_list(natten.na3d_varlen)
        )
    finally:
        torch._dynamo.reset()

    assert counter.frame_count == 1
    # Every guard that mentions the layout is TYPE_MATCH on the object itself:
    # no identity match, and nothing read off it.
    lines = [line for line in guards.splitlines() if "L['layout']" in line]
    managers = [line for line in lines if "GuardManager:" in line]
    leaves = [line for line in lines if "GuardManager:" not in line]
    assert managers and all("source=L['layout']," in line for line in managers), lines
    assert leaves and all(
        line.strip(" |+-").startswith("TYPE_MATCH:") for line in leaves
    ), lines


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
    # operator's body: there the layout is a FakeScriptObject whose contents
    # cannot be read, so the replay has to happen inside
    # natten::varlen_attention_bwd, which gets the real layout from ctx.
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
