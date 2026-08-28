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
"""natten.VarlenLayout's own construction-contract tests (bad shapes, the
int32 total_tokens fence, lazy/eager device pinning) live in
test_varlen_layout.py, not here -- this file covers na{1,2,3}d_varlen's
per-call validation (the entry point takes real tensors, not host-side
token-count scalars), the tile/split configuration and
determinism/inference-mode behavior exercised against real layouts and
tensors, and the pure-host-math tests that need neither.
"""

import itertools
import math
import unittest
from dataclasses import fields, replace
from typing import Tuple
from unittest import mock

import natten
import torch
from natten.backends.configs.cutlass import (
    check_cutlass_fna_backward_config,
    check_cutlass_fna_forward_config,
)
from natten.backends.configs.cutlass.backward_knobs import _get_max_grid_size_allowed
from natten.backends.varlen_fna import (
    _build_varlen_fna_state,
    _make_worklist_tensor,
    _normalize_split_cap,
)
from natten.context import (
    get_memory_usage_preference,
    is_kv_parallelism_in_fused_na_enabled,
)
from natten.utils.testing import (
    skip_if_fewer_than_n_gpus,
    skip_if_libnatten_is_not_supported,
)

from .utils import (
    _make_layout,
    _prod,
    _regular_reference,
    _resolved_state,
    DEFAULT_CASES,
    NONDETERMINISTIC_CONTROL_CASE,
    PAIRWISE_CASES,
    VarlenCase,
)

_PAIRWISE_FEATURE_FACTORS = (
    (1, 2, 3),
    (torch.float16, torch.bfloat16, torch.float32),
    ("mha", "mqa", "gqa"),
    ("equal", "v-wide", "qk-wide"),
    ("odd", "even"),
    ("unit", "middle", "kernel"),
    ("unit", "dilated"),
    (False, True),
    (True, False),
    ("compact", "expanded"),
)


def _pairwise_features(case: VarlenCase) -> Tuple[object, ...]:
    if case.heads == case.heads_kv:
        head_mode = "mha"
    elif case.heads_kv == 1:
        head_mode = "mqa"
    else:
        head_mode = "gqa"
    if case.head_dim == case.head_dim_v:
        dim_mode = "equal"
    elif case.head_dim < case.head_dim_v:
        dim_mode = "v-wide"
    else:
        dim_mode = "qk-wide"
    kernel_mode = "odd" if all(kernel % 2 for kernel in case.kernel_size) else "even"
    if all(stride == 1 for stride in case.stride):
        stride_mode = "unit"
    elif case.stride == case.kernel_size:
        stride_mode = "kernel"
    else:
        stride_mode = "middle"
    dilation_mode = (
        "unit" if all(dilation == 1 for dilation in case.dilation) else "dilated"
    )
    expanded_threshold = {1: 513, 2: 65, 3: 33}[case.rank]
    shape_regime = (
        "expanded"
        if max(max(layout) for layout in case.layouts) >= expanded_threshold
        else "compact"
    )
    return (
        case.rank,
        case.dtype,
        head_mode,
        dim_mode,
        kernel_mode,
        stride_mode,
        dilation_mode,
        any(case.is_causal),
        case.deterministic,
        shape_regime,
    )


class VarlenFnaValidationTests(unittest.TestCase):
    def _qkv(
        self, total_tokens=8, heads=1, head_dim=8, dtype=torch.float16, device="cpu"
    ):
        query = torch.zeros(total_tokens, heads, head_dim, dtype=dtype, device=device)
        return query, query.clone(), query.clone()

    def test_scalar_and_configuration_contract_fails_before_cuda_access(self):
        # Everything up to (not including) the memo-miss build is pure
        # host/tensor-metadata validation and runs without touching a real
        # device -- these cases use CPU tensors and no CUDA mocking.
        layout = natten.VarlenLayout(((8,),))
        query, key, value = self._qkv()
        cases = (
            ({"backend": "reference"}, NotImplementedError, "backend"),
            (
                {"backward_use_pt_reduction": 1},
                TypeError,
                "backward_use_pt_reduction",
            ),
            ({"kernel_size": True}, TypeError, "kernel_size"),
            ({"kernel_size": (True,)}, TypeError, "kernel_size"),
            ({"stride": True}, TypeError, "stride"),
            ({"stride": (True,)}, TypeError, "stride"),
            ({"dilation": True}, TypeError, "dilation"),
            ({"dilation": (True,)}, TypeError, "dilation"),
            ({"kernel_size": 1}, ValueError, "kernel_size"),
            ({"stride": 0}, ValueError, "stride"),
            ({"dilation": 0}, ValueError, "dilation"),
            ({"is_causal": (False, False)}, ValueError, "is_causal"),
            ({"stride": 4}, ValueError, "stride cannot be larger"),
        )
        for overrides, exception, message in cases:
            with self.subTest(overrides=overrides):
                call_kwargs = {"kernel_size": 3, **overrides}
                with self.assertRaisesRegex(exception, message):
                    natten.na1d_varlen(query, key, value, layout, **call_kwargs)

    def test_all_empty_cpu_requires_cuda(self):
        # The all-empty fast path (backends/varlen_fna.py) returns before any
        # device work, so it carries its own CUDA check: it must raise on CPU
        # inputs like every other input, before doing any CPU computation.
        layout = natten.VarlenLayout(((0,), (0,)))
        query, key, value = self._qkv(total_tokens=0)
        with self.assertRaisesRegex(ValueError, "requires a CUDA device"):
            natten.na1d_varlen(query, key, value, layout, kernel_size=3)

    def test_non_empty_cpu_requires_cuda_same_error_class(self):
        # Non-empty CPU tensors raise deeper in the function, once device
        # work starts; this only pins that it is the same exception CLASS as
        # the all-empty fast path's check above, not that it is the same
        # message or raised at the same call site.
        layout = natten.VarlenLayout(((8,),))
        query, key, value = self._qkv()
        with self.assertRaises(ValueError):
            natten.na1d_varlen(query, key, value, layout, kernel_size=3)

    def test_backend_none_resolves_to_cutlass_fna(self):
        # An explicit backend=None must resolve to "cutlass-fna", like the
        # fixed family's None-picks-the-default convention, and reach the
        # same CUDA-device check as an unspecified backend -- rather than
        # being rejected as an unsupported backend value.
        layout = natten.VarlenLayout(((0,), (0,)))
        query, key, value = self._qkv(total_tokens=0)
        with self.assertRaisesRegex(ValueError, "requires a CUDA device"):
            natten.na1d_varlen(query, key, value, layout, kernel_size=3, backend=None)

    def test_heads_times_head_dim_int32_fence(self):
        # "meta" device: real shape/dtype metadata, zero memory backing --
        # this check only ever reads .shape, so an un-backed 268M-head tensor
        # costs nothing to construct.
        layout = natten.VarlenLayout(((8,),))
        query, key, value = self._qkv(
            total_tokens=8, heads=268_435_456, head_dim=8, device="meta"
        )
        with self.assertRaisesRegex(ValueError, r"heads \* head dimension"):
            natten.na1d_varlen(query, key, value, layout, kernel_size=3)

    def test_head_dim_bounds_fail_before_cuda_access(self):
        layout = natten.VarlenLayout(((8,),))
        cases = (
            (7, 8, "multiples of 8"),
            (8, 7, "multiples of 8"),
            (65_544, 8, "must not exceed 65536"),
            (8, 65_544, "must not exceed 65536"),
        )
        for head_dim, head_dim_v, message in cases:
            with self.subTest(head_dim=head_dim, head_dim_v=head_dim_v):
                query, key, _ = self._qkv(head_dim=head_dim)
                value = torch.zeros(8, 1, head_dim_v, dtype=torch.float16, device="cpu")
                with self.assertRaisesRegex(ValueError, message):
                    natten.na1d_varlen(query, key, value, layout, kernel_size=3)

    def test_dtype_and_extent_contract_needs_the_build_path(self):
        # These two checks (allowed dtypes; kernel_size * dilation fits every
        # document) live inside the memo-miss build, which needs a device
        # capability query -- mocked here so the test still runs without a
        # real GPU.
        with (
            mock.patch.object(torch.cuda, "get_device_capability", return_value=(8, 0)),
        ):
            layout = natten.VarlenLayout(((8,),))
            query, key, value = self._qkv(total_tokens=8, dtype=torch.int32)
            with self.assertRaisesRegex(ValueError, "FP16, BF16, and FP32"):
                natten.na1d_varlen(query, key, value, layout, kernel_size=3)

            layout = natten.VarlenLayout(((5,),))
            query, key, value = self._qkv(total_tokens=5)
            with self.assertRaisesRegex(ValueError, "must fit every token layout"):
                natten.na1d_varlen(query, key, value, layout, kernel_size=3, dilation=2)

    def test_packed_token_count_at_scale_succeeds(self):
        # Construction succeeds well past the point where the element count
        # (tokens * heads * head_dim) would exceed int32: only the packed-
        # token-count limit and the heads*head_dim limit are enforced, and
        # this stays far clear of both. Construction-only (cheap): no device,
        # no real 268M-token tensor allocated.
        layout = natten.VarlenLayout(((268_435_456,),))
        self.assertEqual(layout.total_tokens, 268_435_456)

    def test_worklist_tensor_matches_document_local_order(self):
        observed = _make_worklist_tensor((2, 1, 3), torch.device("cpu"))
        expected = torch.tensor(
            ((0, 0), (0, 1), (1, 0), (2, 0), (2, 1), (2, 2)),
            dtype=torch.int32,
            device="cpu",
        )
        self.assertTrue(torch.equal(observed, expected))

    @skip_if_libnatten_is_not_supported()
    def test_varlen_capability_dtype_matrix(self):
        # Unlike the other cases in this file, the *success* path here needs
        # layout._ensure_materialized to actually succeed (a real CUDA
        # device), not just the mocked capability query the failure path
        # alone would need -- so this uses a real (tiny) CUDA tensor.
        def configure(dtype, capability):
            with mock.patch.object(
                torch.cuda, "get_device_capability", return_value=capability
            ):
                layout = natten.VarlenLayout(((8,),))
                query, key, value = self._qkv(dtype=dtype, device="cuda")
                natten.na1d_varlen(query, key, value, layout, kernel_size=3)

        # Per-dtype floor mirroring the C++ dispatch guard the kernels launch
        # under (fna_forward.cu/fna_backward.cu): FP16/FP32 need CC >= 50,
        # BF16 needs CC >= 80.
        device_ccs = (49, 50, 59, 60, 75, 79, 80)
        dtypes = (torch.float16, torch.float32, torch.bfloat16)
        for device_cc, dtype in itertools.product(device_ccs, dtypes):
            threshold = 80 if dtype == torch.bfloat16 else 50
            expected = device_cc >= threshold
            capability = divmod(device_cc, 10)
            with self.subTest(device_cc=device_cc, dtype=dtype):
                if expected:
                    configure(dtype, capability)
                    continue
                with self.assertRaisesRegex(
                    ValueError,
                    f"compute capability {threshold} or higher",
                ):
                    configure(dtype, capability)

    def test_dilated_residue_surplus_tiles_are_culled(self):
        cases = (
            ((129,), (2,), (64,), 1),
            ((17, 19), (2, 2), (8, 8), 4),
            ((7, 9, 11), (2, 2, 2), (4, 4, 4), 8),
        )

        def ceil_div(left, right):
            return (left + right - 1) // right

        def tile_grid(extents, tile_shape):
            return tuple(
                ceil_div(extent, tile) for extent, tile in zip(extents, tile_shape)
            )

        def index_to_coord(index, shape):
            coord = []
            for extent in reversed(shape[1:]):
                coord.append(index % extent)
                index //= extent
            return (index, *reversed(coord))

        for layout, dilation, tile_shape, expected_surplus_count in cases:
            max_residue_extents = tuple(
                ceil_div(extent, dilation_axis)
                for extent, dilation_axis in zip(layout, dilation)
            )
            worklist_grid = tile_grid(max_residue_extents, tile_shape)
            worklist_count = math.prod(worklist_grid)
            surplus_count = 0

            for residue in itertools.product(
                *(range(dilation_axis) for dilation_axis in dilation)
            ):
                residue_extents = tuple(
                    ceil_div(extent - residue_axis, dilation_axis)
                    for extent, residue_axis, dilation_axis in zip(
                        layout, residue, dilation
                    )
                )
                residue_grid = tile_grid(residue_extents, tile_shape)
                residue_tile_count = math.prod(residue_grid)
                valid_coords = [
                    index_to_coord(tile_id, residue_grid)
                    for tile_id in range(worklist_count)
                    if tile_id < residue_tile_count
                ]
                expected_coords = set(
                    itertools.product(*(range(x) for x in residue_grid))
                )

                self.assertEqual(len(valid_coords), residue_tile_count)
                self.assertEqual(len(set(valid_coords)), residue_tile_count)
                self.assertEqual(set(valid_coords), expected_coords)
                self.assertEqual(
                    sum(
                        tile_id >= residue_tile_count
                        for tile_id in range(worklist_count)
                    ),
                    worklist_count - residue_tile_count,
                )
                surplus_count += worklist_count - residue_tile_count

            with self.subTest(rank=len(layout)):
                self.assertTrue(all(extent % 2 == 1 for extent in layout))
                self.assertTrue(
                    any(
                        extent % tile != 0
                        for extent, tile in zip(max_residue_extents, tile_shape)
                    )
                )
                self.assertEqual(surplus_count, expected_surplus_count)

    def test_pairwise_matrix_covers_every_factor_pair(self):
        observed = {
            (left, features[left], right, features[right])
            for case in PAIRWISE_CASES
            for features in (_pairwise_features(case),)
            for left in range(len(features))
            for right in range(left + 1, len(features))
        }
        required = {
            (left, left_value, right, right_value)
            for left in range(len(_PAIRWISE_FEATURE_FACTORS))
            for right in range(left + 1, len(_PAIRWISE_FEATURE_FACTORS))
            for left_value in _PAIRWISE_FEATURE_FACTORS[left]
            for right_value in _PAIRWISE_FEATURE_FACTORS[right]
        }
        self.assertEqual(observed, required)

    @skip_if_fewer_than_n_gpus(2)
    def test_layout_and_data_device_mismatch_raises(self):
        # gpu_count>=2 self-skips under `make test_parallel` (each test file is
        # pinned to a single GPU there); runs only under serial `make test`.
        layout = natten.VarlenLayout(((8,),), device="cuda:0")
        query = torch.randn(8, 1, 8, device="cuda:1", dtype=torch.float16)
        with self.assertRaisesRegex(ValueError, "was pinned to device"):
            natten.na1d_varlen(query, query, query, layout, kernel_size=3)


# NOTE: There is intentionally no test asserting that na1d_varlen detects a
# VarlenLayout built for one token layout being used against QKV data packed
# in a different order (same total token count, different per-document row
# order/content). This is undetectable by design: VarlenLayout only carries
# per-document element counts, never row identity or content, so there is no
# signal available to check against. See the docstring warning on
# natten.na1d_varlen (and na2d_varlen/na3d_varlen). Do not add a test that
# mocks/monkeypatches internal state to fake a detection the real API cannot
# perform.


def _build_state_for_split_heuristic(
    layouts, heads, head_dim, dtype, kernel_size, dilation, backward_tile
):
    # Resolves split selection directly against the shared build core, the
    # same way natten.backends.varlen_fna._neighborhood_attention_varlen_generic does
    # internally, but from a synthetic config tensor instead of real QKV data
    # -- this test only cares about the split heuristic's behavior across
    # natten.context memory-usage preferences, not about running attention
    # itself, and some of its layouts are large enough (32768 tokens) that
    # building real QKV tensors would be wasted allocation.
    na_dim = len(layouts[0])
    config_input = torch.empty(
        (1, *((1,) * na_dim), 1, head_dim), device="cuda", dtype=dtype
    )
    forward_config = check_cutlass_fna_forward_config(
        input_tensor=config_input, dilation=dilation
    )
    backward_config = check_cutlass_fna_backward_config(
        input_tensor=config_input,
        q_tile_shape=backward_tile,
        kv_tile_shape=backward_tile,
    )
    return _build_varlen_fna_state(
        shapes=layouts,
        kernel_size=kernel_size,
        dilation=dilation,
        num_heads=heads,
        dtype=dtype,
        device=torch.device("cuda"),
        forward_config=forward_config,
        backward_config=backward_config,
        split_cap=_normalize_split_cap(None, na_dim),
        deterministic=torch.are_deterministic_algorithms_enabled(),
        kv_parallelism_enabled=is_kv_parallelism_in_fused_na_enabled(),
        max_grid_size=_get_max_grid_size_allowed(),
    )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
class VarlenFnaApiTests(unittest.TestCase):
    def tearDown(self):
        torch.use_deterministic_algorithms(False)

    def test_tile_and_split_configuration_fails_fast(self):
        layout = natten.VarlenLayout(((17,),), device="cuda")
        query = torch.randn(17, 1, 64, device="cuda", dtype=torch.float16)
        cases = (
            ({"q_tile_shape": (64,)}, ValueError, "both q_tile_shape"),
            ({"kv_tile_shape": (64,)}, ValueError, "both q_tile_shape"),
            (
                {"backward_q_tile_shape": (64,)},
                ValueError,
                "both q_tile_shape",
            ),
            (
                {"backward_kv_tile_shape": (64,)},
                ValueError,
                "both q_tile_shape",
            ),
            (
                {"q_tile_shape": (1,), "kv_tile_shape": (1,)},
                ValueError,
                "Invalid configuration",
            ),
            (
                {
                    "backward_q_tile_shape": (1,),
                    "backward_kv_tile_shape": (1,),
                },
                ValueError,
                "Invalid configuration",
            ),
            # Strict fixed-family mirror (check_fna_kv_splits): only a tuple
            # is accepted -- not an int, not a bool, not an arbitrary
            # Sequence -- so all of these hit the same "Invalid type" gate.
            ({"backward_kv_splits": True}, ValueError, "Invalid type"),
            ({"backward_kv_splits": object()}, ValueError, "Invalid type"),
            ({"backward_kv_splits": 2}, ValueError, "Invalid type"),  # int now raises
            ({"backward_kv_splits": (1, 1)}, ValueError, "must have 1 values"),
            ({"backward_kv_splits": (1.0,)}, TypeError, "only integers"),
            ({"backward_kv_splits": (True,)}, TypeError, "only integers"),
            ({"backward_kv_splits": (0,)}, ValueError, "positive values"),
        )
        for overrides, exception, message in cases:
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(exception, message):
                    natten.na1d_varlen(
                        query, query, query, layout, kernel_size=3, **overrides
                    )

    @skip_if_libnatten_is_not_supported()
    def test_layout_build_and_call_inside_inference_mode(self):
        case = replace(DEFAULT_CASES[0], layouts=((17,), (23,)))
        total = sum(_prod(doc_layout) for doc_layout in case.layouts)
        query = torch.randn(
            total, case.heads, case.head_dim, device="cuda", dtype=case.dtype
        )
        key = torch.randn(
            total, case.heads_kv, case.head_dim, device="cuda", dtype=case.dtype
        )
        value = torch.randn(
            total, case.heads_kv, case.head_dim_v, device="cuda", dtype=case.dtype
        )

        with torch.inference_mode():
            layout = _make_layout(case)
            output, logsumexp = natten.na1d_varlen(
                query,
                key,
                value,
                layout,
                kernel_size=case.kernel_size,
                stride=case.stride,
                dilation=case.dilation,
                is_causal=case.is_causal,
                return_lse=True,
            )

        self.assertEqual(output.shape, (total, case.heads, case.head_dim_v))
        # _regular_reference only reads the resolved state's non-tensor
        # forward/backward_config fields, so it's safe to call outside
        # inference_mode with a layout resolved inside it; query/key/value
        # are ordinary tensors built before the block.
        output_ref, logsumexp_ref = _regular_reference(case, layout, query, key, value)
        self.assertTrue(torch.equal(output, output_ref))
        self.assertTrue(torch.equal(logsumexp, logsumexp_ref))

    @skip_if_libnatten_is_not_supported()
    def test_deterministic_mode_transparently_serializes_kv_splits(self):
        case = NONDETERMINISTIC_CONTROL_CASE
        torch.use_deterministic_algorithms(False)
        layout = _make_layout(case)
        total = sum(_prod(doc_layout) for doc_layout in case.layouts)
        torch.manual_seed(2026)
        inputs = (
            torch.randn(
                total, case.heads, case.head_dim, device="cuda", dtype=case.dtype
            ),
            torch.randn(
                total, case.heads_kv, case.head_dim, device="cuda", dtype=case.dtype
            ),
            torch.randn(
                total, case.heads_kv, case.head_dim_v, device="cuda", dtype=case.dtype
            ),
        )
        gradient = torch.randn(
            total, case.heads, case.head_dim_v, device="cuda", dtype=case.dtype
        )

        def run_once():
            query, key, value = (
                tensor.detach().clone().requires_grad_(True) for tensor in inputs
            )
            output, logsumexp = natten.na1d_varlen(
                query,
                key,
                value,
                layout,
                kernel_size=case.kernel_size,
                stride=case.stride,
                dilation=case.dilation,
                is_causal=case.is_causal,
                return_lse=True,
            )
            output.backward(gradient)
            return (output, logsumexp, query.grad, key.grad, value.grad)

        # Step 1 (deterministic OFF): warm the memo and confirm this
        # geometry actually selects real KV parallelism, so step 2 below is
        # a genuine serialization test rather than a vacuous one.
        run_once()
        nondeterministic_entry = _resolved_state(layout)
        self.assertTrue(nondeterministic_entry.uses_kv_parallelism)

        # Step 2 (deterministic ON): the same geometry on the same layout
        # must resolve its own memo entry with every KV split serialized,
        # rather than raising, and must reproduce bit-identical
        # outputs/gradients run to run.
        torch.use_deterministic_algorithms(True)
        first = run_once()
        second = run_once()
        self.assertTrue(
            all(torch.equal(left, right) for left, right in zip(first, second))
        )

        self.assertEqual(len(layout._memo), 2)
        deterministic_entry = next(
            entry
            for entry in layout._memo.values()
            if entry is not nondeterministic_entry
        )
        self.assertFalse(deterministic_entry.uses_kv_parallelism)
        self.assertTrue(
            all(
                split == 1
                for splits in deterministic_entry.selected_splits
                for split in splits
            )
        )

        # The two memo entries share every key component except the
        # determinism flag that was folded into the key.
        key_a, key_b = layout._memo.keys()
        diffs = [
            field.name
            for field in fields(key_a)
            if getattr(key_a, field.name) != getattr(key_b, field.name)
        ]
        self.assertEqual(diffs, ["deterministic"])
        self.assertEqual({key_a.deterministic, key_b.deterministic}, {True, False})

    @skip_if_libnatten_is_not_supported()
    def test_kv_parallelism_switch_is_part_of_memo_key(self):
        # is_kv_parallelism_in_fused_na_enabled() is mutable global state read
        # by the default split selector; flipping it after warming a layout
        # must not reuse a schedule resolved under the old value.
        case = NONDETERMINISTIC_CONTROL_CASE
        torch.use_deterministic_algorithms(False)
        previous_kv_parallelism = is_kv_parallelism_in_fused_na_enabled()
        layout = _make_layout(case)
        total = sum(_prod(doc_layout) for doc_layout in case.layouts)
        torch.manual_seed(2026)
        query = torch.randn(
            total, case.heads, case.head_dim, device="cuda", dtype=case.dtype
        )
        key = torch.randn(
            total, case.heads_kv, case.head_dim, device="cuda", dtype=case.dtype
        )
        value = torch.randn(
            total, case.heads_kv, case.head_dim_v, device="cuda", dtype=case.dtype
        )

        def run_once():
            natten.na1d_varlen(
                query,
                key,
                value,
                layout,
                kernel_size=case.kernel_size,
                stride=case.stride,
                dilation=case.dilation,
                is_causal=case.is_causal,
            )

        try:
            natten.use_kv_parallelism_in_fused_na(True)
            run_once()
            enabled_entry = _resolved_state(layout)
            # The assertion below is only meaningful if this geometry
            # actually selects KV parallelism while the switch is enabled.
            self.assertTrue(enabled_entry.uses_kv_parallelism)

            natten.use_kv_parallelism_in_fused_na(False)
            run_once()
            self.assertEqual(len(layout._memo), 2)
            disabled_entry = next(
                entry for entry in layout._memo.values() if entry is not enabled_entry
            )
            self.assertFalse(disabled_entry.uses_kv_parallelism)
            self.assertTrue(
                all(
                    split == 1
                    for splits in disabled_entry.selected_splits
                    for split in splits
                )
            )
        finally:
            natten.use_kv_parallelism_in_fused_na(previous_kv_parallelism)

    @skip_if_libnatten_is_not_supported()
    def test_memory_usage_preference_is_part_of_memo_key(self):
        # _get_max_grid_size_allowed() derives from the memory-usage-
        # preference global; switching preference after warming a layout
        # must not reuse a schedule resolved under the old (looser) bound.
        torch.use_deterministic_algorithms(False)
        previous_kv_parallelism = is_kv_parallelism_in_fused_na_enabled()
        previous_preference = get_memory_usage_preference().name.lower()
        natten.use_kv_parallelism_in_fused_na(True)
        # Same (layouts, dilation, kernel_size, backward tile) combination
        # test_default_split_heuristic_respects_rank_and_memory_limits
        # already proved selects (256,) under "unrestricted"/"default" and
        # (128,) under "strict" -- reused here so the expected split counts
        # below are pinned to an independently-verified selector result,
        # not a value this test invents and could tautologically satisfy.
        layouts = ((32768,), (32768,))
        dilation = (2,)
        kernel_size = (3,)
        backward_tile = (64,)
        total = sum(_prod(s) for s in layouts)
        layout = natten.VarlenLayout(layouts, device="cuda")
        torch.manual_seed(2026)
        query = torch.randn(total, 2, 64, device="cuda", dtype=torch.float16)
        key = torch.randn(total, 2, 64, device="cuda", dtype=torch.float16)
        value = torch.randn(total, 2, 64, device="cuda", dtype=torch.float16)

        def run_once():
            natten.na1d_varlen(
                query,
                key,
                value,
                layout,
                kernel_size=kernel_size,
                dilation=dilation,
                backward_q_tile_shape=backward_tile,
                backward_kv_tile_shape=backward_tile,
            )

        try:
            natten.set_memory_usage_preference("unrestricted")
            run_once()
            unrestricted_entry = _resolved_state(layout)

            natten.set_memory_usage_preference("strict")
            run_once()
            self.assertEqual(len(layout._memo), 2)
            strict_entry = next(
                entry
                for entry in layout._memo.values()
                if entry is not unrestricted_entry
            )
            self.assertEqual(unrestricted_entry.selected_splits, ((256,), (256,)))
            self.assertEqual(strict_entry.selected_splits, ((128,), (128,)))
        finally:
            natten.set_memory_usage_preference(previous_preference)
            natten.use_kv_parallelism_in_fused_na(previous_kv_parallelism)

    @skip_if_libnatten_is_not_supported()
    def test_explicit_backward_kv_splits_cap_ignores_kv_parallelism_switch(self):
        # Family-parity semantics (matches check_fna_kv_splits/
        # check_fmha_kv_splits for the fixed path): an explicit
        # backward_kv_splits cap is respected regardless of the KV-
        # parallelism switch -- it is NOT forced to 1 just because the
        # switch is disabled. Only the *default* (uncapped) selection reads
        # the switch.
        case = NONDETERMINISTIC_CONTROL_CASE
        torch.use_deterministic_algorithms(False)
        previous_kv_parallelism = is_kv_parallelism_in_fused_na_enabled()
        layout = _make_layout(case)
        total = sum(_prod(doc_layout) for doc_layout in case.layouts)
        torch.manual_seed(2026)
        query = torch.randn(
            total, case.heads, case.head_dim, device="cuda", dtype=case.dtype
        )
        key = torch.randn(
            total, case.heads_kv, case.head_dim, device="cuda", dtype=case.dtype
        )
        value = torch.randn(
            total, case.heads_kv, case.head_dim_v, device="cuda", dtype=case.dtype
        )
        try:
            natten.use_kv_parallelism_in_fused_na(False)
            natten.na1d_varlen(
                query,
                key,
                value,
                layout,
                kernel_size=case.kernel_size,
                stride=case.stride,
                dilation=case.dilation,
                is_causal=case.is_causal,
                backward_kv_splits=(2,),
            )
            entry = _resolved_state(layout)
            self.assertEqual(entry.selected_splits, ((2,), (2,)))
        finally:
            natten.use_kv_parallelism_in_fused_na(previous_kv_parallelism)

    def test_default_split_heuristic_respects_rank_and_memory_limits(self):
        previous_preference = get_memory_usage_preference().name.lower()
        previous_kv_parallelism = is_kv_parallelism_in_fused_na_enabled()
        torch.use_deterministic_algorithms(False)
        natten.use_kv_parallelism_in_fused_na(True)
        problems = (
            (((32768,), (32768,)), (2,), (64,), ((128,), (256,))),
            (
                ((256, 256), (256, 256)),
                (2, 1),
                (8, 8),
                ((16, 8), (16, 32)),
            ),
            (
                ((8, 64, 64), (8, 64, 64)),
                (2, 1, 1),
                (4, 4, 4),
                ((1, 16, 8), (1, 16, 16)),
            ),
        )
        try:
            for layouts, dilation, backward_tile, expected in problems:
                observed = []
                for preference in ("strict", "default", "unrestricted"):
                    natten.set_memory_usage_preference(preference)
                    state = _build_state_for_split_heuristic(
                        layouts,
                        2,
                        64,
                        torch.float16,
                        (3,) * len(dilation),
                        dilation,
                        backward_tile,
                    )
                    observed.append(state.selected_splits[0])
                    self.assertEqual(
                        state.selected_splits,
                        (state.selected_splits[0],) * len(layouts),
                    )
                self.assertEqual(observed[0], expected[0])
                self.assertEqual(observed[1], expected[1])
                self.assertEqual(observed[2], expected[1])

            natten.use_kv_parallelism_in_fused_na(False)
            disabled = _build_state_for_split_heuristic(
                ((256, 256),), 2, 64, torch.float16, (3, 3), (1, 1), (8, 8)
            )
            self.assertEqual(disabled.selected_splits, ((1, 1),))
        finally:
            natten.set_memory_usage_preference(previous_preference)
            natten.use_kv_parallelism_in_fused_na(previous_kv_parallelism)

    @skip_if_libnatten_is_not_supported()
    def test_nondeterministic_layout_is_allowed_for_deterministic_inference(self):
        # NOTE: despite the name, this test does not use torch.inference_mode();
        # "inference" here means forward-only/no-requires_grad.
        case = NONDETERMINISTIC_CONTROL_CASE
        torch.use_deterministic_algorithms(False)
        layout = _make_layout(case)
        total = sum(_prod(doc_layout) for doc_layout in case.layouts)
        query = torch.randn(
            total, case.heads, case.head_dim, device="cuda", dtype=case.dtype
        )
        key = torch.randn(
            total, case.heads_kv, case.head_dim, device="cuda", dtype=case.dtype
        )
        value = torch.randn(
            total, case.heads_kv, case.head_dim_v, device="cuda", dtype=case.dtype
        )
        output_nondeterministic, logsumexp_nondeterministic = natten.na1d_varlen(
            query,
            key,
            value,
            layout,
            kernel_size=case.kernel_size,
            stride=case.stride,
            dilation=case.dilation,
            is_causal=case.is_causal,
            return_lse=True,
        )  # warm the memo under deterministic=False
        self.assertTrue(_resolved_state(layout).uses_kv_parallelism)
        output_ref, logsumexp_ref = _regular_reference(case, layout, query, key, value)
        self.assertTrue(torch.equal(output_nondeterministic, output_ref))
        self.assertTrue(torch.equal(logsumexp_nondeterministic, logsumexp_ref))

        # A geometry previously resolved non-deterministically must still
        # succeed under deterministic mode: it now resolves its own memo
        # entry rather than reusing (or being rejected against) the earlier
        # one. The forward schedule is split-independent and the forward
        # kernel has no atomics, so the output must match bit-for-bit.
        torch.use_deterministic_algorithms(True)
        output_deterministic, logsumexp_deterministic = natten.na1d_varlen(
            query,
            key,
            value,
            layout,
            kernel_size=case.kernel_size,
            stride=case.stride,
            dilation=case.dilation,
            is_causal=case.is_causal,
            return_lse=True,
        )
        self.assertEqual(
            output_deterministic.shape, (total, case.heads, case.head_dim_v)
        )
        self.assertTrue(torch.equal(output_deterministic, output_nondeterministic))
        self.assertTrue(
            torch.equal(logsumexp_deterministic, logsumexp_nondeterministic)
        )

    @skip_if_libnatten_is_not_supported()
    def test_empty_documents_are_exempt_from_the_fit_check(self):
        # A zero-token document skips "extent >= kernel_size * dilation"
        # (an empty document is a no-op, never scheduled); a non-empty
        # document under the same call is still checked -- the exemption is
        # per-document, not a blanket skip of the whole check.
        layout = natten.VarlenLayout(((0,), (8,)), device="cuda")
        query = torch.zeros(8, 1, 8, device="cuda", dtype=torch.float16)
        natten.na1d_varlen(query, query, query, layout, kernel_size=3, dilation=2)

        tight_layout = natten.VarlenLayout(((0,), (5,)), device="cuda")
        query5 = torch.zeros(5, 1, 8, device="cuda", dtype=torch.float16)
        with self.assertRaisesRegex(ValueError, "must fit every token layout"):
            natten.na1d_varlen(
                query5, query5, query5, tight_layout, kernel_size=3, dilation=2
            )

    def test_zero_token_documents_get_zero_backward_split_count(self):
        # Direct build-core check (no kernel launch): a zero-token document
        # must contribute exactly 0 backward work items, across all three
        # split-selection branches (deterministic; the split_cap=None
        # heuristic; an explicit split_cap) -- not the (1,) * na_dim
        # placeholder's math.prod, which the deterministic branch's uniform
        # placeholder would otherwise turn into 1 spurious work item.
        na_dim = 1
        shapes = ((0,), (24,), (0,), (16,))
        empty_indices = (0, 2)
        non_empty_indices = (1, 3)
        backward_tile = (64,)

        previous_kv_parallelism = is_kv_parallelism_in_fused_na_enabled()
        natten.use_kv_parallelism_in_fused_na(True)
        try:
            for label, deterministic, split_cap in (
                ("deterministic", True, None),
                ("heuristic", False, None),
                ("explicit-cap", False, _normalize_split_cap((2,), na_dim)),
            ):
                with self.subTest(variant=label):
                    config_input = torch.empty(
                        (1, 1, 1, 64), device="cuda", dtype=torch.float16
                    )
                    forward_config = check_cutlass_fna_forward_config(
                        input_tensor=config_input, dilation=(1,)
                    )
                    backward_config = check_cutlass_fna_backward_config(
                        input_tensor=config_input,
                        q_tile_shape=backward_tile,
                        kv_tile_shape=backward_tile,
                    )
                    state = _build_varlen_fna_state(
                        shapes=shapes,
                        kernel_size=(3,),
                        dilation=(1,),
                        num_heads=2,
                        dtype=torch.float16,
                        device=torch.device("cuda"),
                        forward_config=forward_config,
                        backward_config=backward_config,
                        split_cap=split_cap,
                        deterministic=deterministic,
                        kv_parallelism_enabled=is_kv_parallelism_in_fused_na_enabled(),
                        max_grid_size=_get_max_grid_size_allowed(),
                    )

                    # Placeholder row is canonical for every empty document...
                    for index in empty_indices:
                        self.assertEqual(state.selected_splits[index], (1,) * na_dim)

                    # ...but the offsets are flat across an empty document's
                    # slot (zero backward work items), and it never appears
                    # in the backward worklist's document-id column.
                    offsets = state.backward_kv_split_offsets.tolist()
                    counts = [offsets[i + 1] - offsets[i] for i in range(len(shapes))]
                    for index in empty_indices:
                        self.assertEqual(counts[index], 0)
                    for index in non_empty_indices:
                        self.assertGreater(counts[index], 0)

                    referenced_docs = set(state.backward_worklist[:, 0].tolist())
                    for index in empty_indices:
                        self.assertNotIn(index, referenced_docs)
                    for index in non_empty_indices:
                        self.assertIn(index, referenced_docs)

                    # uses_kv_parallelism derives from the actual (post-
                    # zeroing) counts, not the placeholder rows.
                    self.assertEqual(
                        state.uses_kv_parallelism, any(count > 1 for count in counts)
                    )
        finally:
            natten.use_kv_parallelism_in_fused_na(previous_kv_parallelism)


if __name__ == "__main__":
    unittest.main()
