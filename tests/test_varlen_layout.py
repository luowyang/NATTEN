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
"""Mechanism tests for VarlenLayout: construction, the per-object memo (hit/miss/
compile boundary), pickling, and interop helpers.
"""

import pickle
import unittest
from unittest import mock

import torch
from natten import na1d_varlen, na2d_varlen, na3d_varlen, VarlenLayout
from natten.backends.varlen_fna import _tile_count
from natten.utils.testing import (
    skip_if_fewer_than_n_gpus,
    skip_if_libnatten_is_not_supported,
)

# CompileCounter is a private torch._dynamo.testing utility, validated
# against torch 2.11; the frame-count expectations below may shift across
# PyTorch versions and are project-internal assertions, not public API
# contract.
from torch._dynamo.testing import CompileCounter


def _fake_build(counter):
    def build():
        counter[0] += 1
        return counter[0]

    return build


class VarlenLayoutConstructionTests(unittest.TestCase):
    """Host-only validation; no device materialization involved."""

    def test_valid_shapes_populate_host_properties(self):
        layout = VarlenLayout([(8, 32, 32), (4, 16, 16)])
        self.assertEqual(layout.rank, 3)
        self.assertEqual(layout.num_docs, 2)
        self.assertEqual(layout.total_tokens, 8 * 32 * 32 + 4 * 16 * 16)
        self.assertEqual(layout.max_seqlen, 8 * 32 * 32)

    def test_rank1_shapes(self):
        layout = VarlenLayout([(17,), (23,), (5,)])
        self.assertEqual(layout.rank, 1)
        self.assertEqual(layout.num_docs, 3)
        self.assertEqual(layout.total_tokens, 45)
        self.assertEqual(layout.max_seqlen, 23)

    def test_invalid_shapes_raise(self):
        cases = (
            (None, TypeError, "sequence"),
            ("8", TypeError, "sequence"),
            ((), ValueError, "at least one"),
            ((8,), TypeError, r"token_layouts\[0\]"),
            (((True,),), TypeError, "only integers"),
            (((8.0,),), TypeError, "only integers"),
            (((-1,),), ValueError, "non-negative extents"),
            (((),), ValueError, "Only 1-D, 2-D, and 3-D"),
            (((8, 8, 8, 8),), ValueError, "Only 1-D, 2-D, and 3-D"),
            (((8,), (8, 8)), ValueError, "same dimensionality"),
        )
        for shapes, exception, message in cases:
            with self.subTest(shapes=shapes):
                with self.assertRaisesRegex(exception, message):
                    VarlenLayout(shapes)

    def test_zero_extent_shapes_are_accepted(self):
        # A document is empty iff its extent product is 0, on any axis --
        # not just a plain (0,) document, and not just axis 0.
        cases = (
            ([(0,), (8,)], 1, 2, 8, 8),
            ([(0,), (0,)], 1, 2, 0, 0),
            ([(0, 32), (4, 4)], 2, 2, 16, 16),
            ([(4, 0, 16), (2, 2, 2)], 3, 2, 8, 8),
            ([(2, 3, 0)], 3, 1, 0, 0),
        )
        for shapes, rank, num_docs, total_tokens, max_seqlen in cases:
            with self.subTest(shapes=shapes):
                layout = VarlenLayout(shapes)
                self.assertEqual(layout.rank, rank)
                self.assertEqual(layout.num_docs, num_docs)
                self.assertEqual(layout.total_tokens, total_tokens)
                self.assertEqual(layout.max_seqlen, max_seqlen)

    def test_packed_token_count_int32_fence(self):
        with self.assertRaisesRegex(ValueError, "Packed token count"):
            VarlenLayout(((46_341, 46_341),))

    def test_large_single_extent_int32_fence(self):
        # A document with one axis alone over int32 range but another axis
        # zero zero-products its way past the total-token fence above
        # (2**31 * 0 == 0) -- the per-axis extent needs its own int32 fence
        # independent of the total-token one, or this would only fail much
        # later, uncontrolled, when the extent is materialized into an
        # int32 token_layouts tensor.
        with self.assertRaisesRegex(ValueError, r"token_layouts\[0\].*exceed"):
            VarlenLayout(((2**31, 0),))

    def test_large_but_valid_single_extent_is_accepted(self):
        # Right at the int32 boundary, not one past it: the fence must be
        # exclusive (extent == limit is valid), not off-by-one.
        layout = VarlenLayout(((2**31 - 1, 1),))
        self.assertEqual(layout.total_tokens, 2**31 - 1)

    def test_unmaterialized_properties_raise(self):
        layout = VarlenLayout([(8,), (4,)])
        with self.assertRaisesRegex(RuntimeError, "not been materialized"):
            layout.cu_seqlens
        with self.assertRaisesRegex(RuntimeError, "not been materialized"):
            layout.token_layouts

    def test_non_cuda_device_raises(self):
        with self.assertRaisesRegex(ValueError, "requires a CUDA device"):
            VarlenLayout([(8,), (4,)], device="cpu")

    def test_shapes_property_returns_normalized_tuple(self):
        # Constructed from lists; the property must return the normalized
        # tuple-of-tuples form, not the original list-of-lists.
        layout = VarlenLayout([[8, 32, 32], [4, 16, 16]])
        self.assertEqual(layout.shapes, ((8, 32, 32), (4, 16, 16)))
        self.assertIsInstance(layout.shapes, tuple)
        self.assertIsInstance(layout.shapes[0], tuple)

    def test_device_property_none_before_materialization(self):
        layout = VarlenLayout([(8,), (4,)])
        self.assertIsNone(layout.device)

    def test_repr_before_materialization(self):
        layout = VarlenLayout([(8,), (4,), (5,)])
        self.assertEqual(
            repr(layout),
            "VarlenLayout(num_docs=3, rank=1, total_tokens=17, max_seqlen=8, "
            "device=None)",
        )

    def test_from_tensor_list_cpu_packing_stays_unmaterialized(self):
        # Packing is a pure tensor op (cat/reshape): building a layout this
        # way from CPU tensors must not require a device, and the returned
        # layout must be unmaterialized -- same as direct construction with
        # no device=.
        torch.manual_seed(5)
        x1 = torch.randn(4, 2, 8, device="cpu")
        x2 = torch.randn(2, 2, 8, device="cpu")
        layout, packed = VarlenLayout.from_tensor_list([x1, x2])
        self.assertIsNone(layout.device)
        self.assertEqual(packed.shape, (6, 2, 8))
        self.assertEqual(packed.device.type, "cpu")


class TileCountMechanismTests(unittest.TestCase):
    """_tile_count (forward AND backward Q-tile counting share this one
    function) must yield 0 for a zero-token document -- ceil_div(0, d) == 0
    on every axis already gives this without any special-casing; this pins
    that down explicitly, including the any-axis-zero case (empty iff the
    extent PRODUCT is 0, not iff axis 0 is 0)."""

    def test_zero_for_empty_document_any_axis(self):
        cases = (
            ((0,), (1,), (32,)),
            ((0, 32), (1, 1), (8, 8)),
            ((32, 0), (1, 1), (8, 8)),
            ((4, 0, 16), (1, 1, 1), (2, 2, 2)),
            ((4, 16, 0), (2, 1, 1), (2, 2, 2)),
        )
        for token_layout, dilation, tile_shape in cases:
            with self.subTest(token_layout=token_layout):
                self.assertEqual(_tile_count(token_layout, dilation, tile_shape), 0)

    def test_nonzero_for_non_empty_document(self):
        self.assertEqual(_tile_count((32,), (1,), (8,)), 4)
        self.assertEqual(_tile_count((32, 16), (1, 1), (8, 8)), 8)


class VarlenLayoutMemoMechanismTests(unittest.TestCase):
    """Generic memo mechanics (hit/miss/compile guard), decoupled from any
    real geometry-build logic or device materialization.
    """

    def test_miss_then_hit_calls_build_once(self):
        layout = VarlenLayout([(8,), (4,)])
        counter = [0]
        build = _fake_build(counter)
        key = (1,)
        first = layout._resolve(key, build)
        second = layout._resolve(key, build)
        self.assertEqual(first, 1)
        self.assertEqual(second, 1)
        self.assertEqual(counter[0], 1)

    def test_distinct_keys_build_independently(self):
        layout = VarlenLayout([(8,), (4,)])
        counter = [0]
        build = _fake_build(counter)
        self.assertEqual(layout._resolve((1,), build), 1)
        self.assertEqual(layout._resolve((2,), build), 2)
        self.assertEqual(layout._resolve((1,), build), 1)
        self.assertEqual(counter[0], 2)

    def test_distinct_objects_do_not_share_memo(self):
        layout_a = VarlenLayout([(8,), (4,)])
        layout_b = VarlenLayout([(8,), (4,)])
        counter = [0]
        build = _fake_build(counter)
        key = (1,)
        self.assertEqual(layout_a._resolve(key, build), 1)
        self.assertEqual(layout_b._resolve(key, build), 2)
        self.assertEqual(counter[0], 2)

    def test_compile_cold_miss_default_budget_builds_and_hits(self):
        # A cold miss traced inside torch.compile takes the build branch
        # (one compile), and the next call's now-warm memo flips the
        # dict-membership guard dynamo took on the miss branch, forcing
        # exactly one recompile onto the hit branch. Under a DEFAULT
        # (non-strict) budget that's benign, unlike the tight
        # recompile_limit=1 budget the sibling prewarmed test below
        # exercises.
        #
        # The budget is pinned explicitly to torch._dynamo.config's
        # compiled-in defaults (recompile_limit=8, accumulated_recompile_
        # limit=256, cache_size_limit=8, fail_on_recompile_limit_hit=False,
        # as of torch 2.11) rather than left ambient: tests/utils.py's
        # reset_torch_compile helper, used by sibling test modules such as
        # test_torch_compile.py, mutates accumulated_recompile_limit/
        # fail_on_recompile_limit_hit process-globally without restoring
        # them, so an un-pinned "default" budget in a full-suite run can
        # silently inherit an earlier test's tight one.
        default_budget = dict(
            recompile_limit=8,
            accumulated_recompile_limit=256,
            cache_size_limit=8,
            fail_on_recompile_limit_hit=False,
        )
        torch._dynamo.reset()
        layout = VarlenLayout([(8,), (4,)])
        counter = [0]
        build = _fake_build(counter)

        def fn(q):
            entry = layout._resolve((3,), build)
            return q.sum() + entry

        compile_counter = CompileCounter()
        compiled = torch.compile(fn, backend=compile_counter, fullgraph=True)
        q = torch.randn(4)
        try:
            with torch._dynamo.config.patch(**default_budget):
                first = compiled(q)
                second = compiled(q)
                third = compiled(q)
        finally:
            torch._dynamo.reset()

        self.assertEqual(counter[0], 1)  # build ran exactly once
        for result in (first, second, third):
            self.assertTrue(torch.equal(result, q.sum() + 1))
        cold_frame_count = compile_counter.frame_count

        # Bounded cost: at most one extra recompile relative to an identical
        # case that was pre-warmed eagerly and so never takes the miss
        # branch under compile at all.
        torch._dynamo.reset()
        layout_warm = VarlenLayout([(8,), (4,)])
        counter_warm = [0]
        build_warm = _fake_build(counter_warm)
        layout_warm._resolve((3,), build_warm)  # eager prewarm

        def fn_warm(q):
            entry = layout_warm._resolve((3,), build_warm)
            return q.sum() + entry

        warm_compile_counter = CompileCounter()
        compiled_warm = torch.compile(
            fn_warm, backend=warm_compile_counter, fullgraph=True
        )
        try:
            with torch._dynamo.config.patch(**default_budget):
                compiled_warm(q)
                compiled_warm(q)
        finally:
            torch._dynamo.reset()

        self.assertLessEqual(cold_frame_count - warm_compile_counter.frame_count, 1)

    def test_compile_positive_prewarmed_hit_passes_tight_budget(self):
        torch._dynamo.reset()
        layout = VarlenLayout([(8,), (4,)])
        counter = [0]
        build = _fake_build(counter)
        layout._resolve((3,), build)  # eager prewarm

        def fn(q):
            entry = layout._resolve((3,), build)
            return q.sum() + entry

        compiled = torch.compile(fn, fullgraph=True, backend="eager")
        q = torch.randn(4)
        try:
            with torch._dynamo.config.patch(
                recompile_limit=1,
                accumulated_recompile_limit=1,
                fail_on_recompile_limit_hit=True,
            ):
                first = compiled(q)
                second = compiled(q)
            self.assertTrue(torch.equal(first, second))
            # The prewarmed hit must never re-invoke build.
            self.assertEqual(counter[0], 1)
        finally:
            torch._dynamo.reset()


class VarlenLayoutIdentityAndPicklingTests(unittest.TestCase):
    def test_identity_eq_and_hash(self):
        layout_a = VarlenLayout([(8,), (4,)])
        layout_b = VarlenLayout([(8,), (4,)])
        self.assertEqual(layout_a, layout_a)
        self.assertNotEqual(layout_a, layout_b)  # value-equal, but not the same object
        hash(layout_a)  # must not raise
        table = {layout_a: "a", layout_b: "b"}
        self.assertEqual(table[layout_a], "a")
        self.assertEqual(table[layout_b], "b")

    def test_pickle_round_trip_unmaterialized(self):
        layout = VarlenLayout([(8, 4), (2, 2)])
        restored = pickle.loads(pickle.dumps(layout))
        self.assertEqual(restored.rank, layout.rank)
        self.assertEqual(restored.num_docs, layout.num_docs)
        self.assertEqual(restored.total_tokens, layout.total_tokens)
        self.assertEqual(restored.max_seqlen, layout.max_seqlen)
        self.assertIsNot(restored, layout)
        with self.assertRaisesRegex(RuntimeError, "not been materialized"):
            restored.cu_seqlens


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
class VarlenLayoutGpuMechanismTests(unittest.TestCase):
    def test_eager_materialization_at_construction(self):
        layout = VarlenLayout([(8,), (4,)], device="cuda")
        self.assertTrue(
            torch.equal(
                layout.cu_seqlens,
                torch.tensor([0, 8, 12], device="cuda", dtype=torch.int32),
            )
        )
        self.assertTrue(
            torch.equal(
                layout.token_layouts,
                torch.tensor([[8], [4]], device="cuda", dtype=torch.int32),
            )
        )

    def test_eager_materialization_of_all_empty_layout(self):
        # max_seqlen == 0, and cu_seqlens is a flat, all-zero run (a
        # repeated offset per document, since every document is empty).
        layout = VarlenLayout([(0,), (0,), (0,)], device="cuda")
        self.assertEqual(layout.max_seqlen, 0)
        self.assertTrue(
            torch.equal(
                layout.cu_seqlens,
                torch.tensor([0, 0, 0, 0], device="cuda", dtype=torch.int32),
            )
        )
        self.assertTrue(
            torch.equal(
                layout.token_layouts,
                torch.tensor([[0], [0], [0]], device="cuda", dtype=torch.int32),
            )
        )

    def test_lazy_materialization_pins_device_on_first_use(self):
        layout = VarlenLayout([(8,), (4,)])
        layout._ensure_materialized(torch.device("cuda"))
        self.assertEqual(layout.cu_seqlens.device.type, "cuda")

    def test_device_property_reflects_eager_materialization(self):
        layout = VarlenLayout([(8,), (4,)], device="cuda")
        self.assertIsInstance(layout.device, torch.device)
        self.assertEqual(layout.device.type, "cuda")

    def test_device_property_reflects_lazy_materialization(self):
        layout = VarlenLayout([(8,), (4,)])
        self.assertIsNone(layout.device)
        layout._ensure_materialized(torch.device("cuda"))
        self.assertEqual(layout.device.type, "cuda")

    def test_repr_after_materialization(self):
        layout = VarlenLayout([(8,), (4,)], device="cuda")
        text = repr(layout)
        self.assertIn("num_docs=2", text)
        self.assertIn("rank=1", text)
        self.assertIn("total_tokens=12", text)
        self.assertIn("max_seqlen=8", text)
        self.assertIn("device=cuda", text)

    def test_repr_excludes_memo_occupancy(self):
        layout = VarlenLayout([(8,), (4,)], device="cuda")
        before = repr(layout)
        counter = [0]
        layout._resolve((1,), _fake_build(counter))
        layout._resolve((2,), _fake_build(counter))
        self.assertEqual(len(layout._memo), 2)
        after = repr(layout)
        self.assertEqual(before, after)
        self.assertNotIn("memo", after.lower())

    @skip_if_fewer_than_n_gpus(2)
    def test_wrong_device_after_pinning_raises(self):
        layout = VarlenLayout([(8,), (4,)], device="cuda:0")
        counter = [0]
        layout._resolve((1,), _fake_build(counter))
        self.assertEqual(len(layout._memo), 1)
        with self.assertRaisesRegex(ValueError, "was pinned to device"):
            layout._ensure_materialized(torch.device("cuda:1"))
        # The device-pin check runs before any schedule state is touched:
        # a rejected wrong-device call must not disturb the existing memo.
        self.assertEqual(len(layout._memo), 1)

    def test_from_tensor_list_and_split_round_trip_values(self):
        torch.manual_seed(0)
        x1 = torch.randn(4, 4, 2, 8, device="cuda")
        x2 = torch.randn(2, 2, 2, 8, device="cuda")
        layout, packed = VarlenLayout.from_tensor_list([x1, x2])
        self.assertEqual(layout.rank, 2)
        self.assertEqual(packed.shape, (16 + 4, 2, 8))
        docs = layout.split(packed)
        self.assertEqual(len(docs), 2)
        self.assertTrue(torch.equal(docs[0], x1))
        self.assertTrue(torch.equal(docs[1], x2))

    def test_from_tensor_list_and_split_round_trip_zero_token_documents(self):
        torch.manual_seed(2)
        x1 = torch.zeros(0, 4, 2, 8, device="cuda")  # empty: zero spatial extent
        x2 = torch.randn(4, 4, 2, 8, device="cuda")
        x3 = torch.zeros(4, 0, 2, 8, device="cuda")  # empty on the OTHER axis
        layout, packed = VarlenLayout.from_tensor_list([x1, x2, x3])
        self.assertEqual(layout.max_seqlen, 16)
        self.assertEqual(packed.shape, (0 + 16 + 0, 2, 8))
        # from_tensor_list returns an unmaterialized layout regardless of the
        # inputs' device, so pin explicitly before reading the derived device
        # tensor below.
        layout._ensure_materialized(torch.device("cuda"))
        self.assertTrue(
            torch.equal(
                layout.cu_seqlens,
                torch.tensor([0, 0, 16, 16], device="cuda", dtype=torch.int32),
            )
        )
        docs = layout.split(packed)
        self.assertEqual(len(docs), 3)
        self.assertTrue(torch.equal(docs[0], x1))
        self.assertTrue(torch.equal(docs[1], x2))
        self.assertTrue(torch.equal(docs[2], x3))

    def test_from_tensor_list_noncontiguous_input_value_equal_stride_may_differ(self):
        torch.manual_seed(1)
        big = torch.randn(8, 6, 2, 8, device="cuda")
        noncontig_doc = big.narrow(1, 0, 4)  # [4, 4, 2, 8], non-standard strides
        self.assertFalse(noncontig_doc.is_contiguous())
        layout, packed = VarlenLayout.from_tensor_list([noncontig_doc])
        docs = layout.split(packed)
        self.assertTrue(torch.equal(docs[0], noncontig_doc))  # value equality holds
        # ...but the round trip is not required to reproduce identical strides.

    def test_from_tensor_list_cross_doc_mismatch_raises(self):
        x1 = torch.randn(4, 2, 8, device="cuda")
        x2 = torch.randn(4, 4, 8, device="cuda")  # different heads
        with self.assertRaisesRegex(ValueError, "heads, head_dim"):
            VarlenLayout.from_tensor_list([x1, x2])
        x3 = torch.randn(4, 2, 8, device="cuda", dtype=torch.float16)
        with self.assertRaisesRegex(ValueError, "dtype"):
            VarlenLayout.from_tensor_list([x1, x3])

    @skip_if_libnatten_is_not_supported()
    def test_from_tensor_list_cpu_pack_then_cuda_call(self):
        # Packing on CPU (e.g. inside a DataLoader worker) and only moving
        # the packed tensor to CUDA before the first na{1,2,3}d_varlen call
        # must work, with the layout pinning lazily on that call as usual --
        # from_tensor_list itself must not force a device.
        torch.manual_seed(4)
        x1 = torch.randn(4, 2, 8, dtype=torch.float16, device="cpu")
        x2 = torch.randn(3, 2, 8, dtype=torch.float16, device="cpu")
        layout, packed = VarlenLayout.from_tensor_list([x1, x2])
        self.assertIsNone(layout.device)

        query = packed.to("cuda")
        key = query.clone()
        value = query.clone()
        output = na1d_varlen(query, key, value, layout, kernel_size=3)
        self.assertEqual(output.shape, query.shape)
        self.assertEqual(layout.device.type, "cuda")

    def test_split_wrong_total_tokens_raises(self):
        layout = VarlenLayout([(8,), (4,)], device="cuda")
        bad = torch.randn(11, 2, 8, device="cuda")
        with self.assertRaisesRegex(ValueError, "total_tokens"):
            layout.split(bad)

    def test_pickle_round_trip_worker_simulation(self):
        # Simulates a DataLoader worker: build on the main process, pickle,
        # send across a process boundary (here: just unpickle), then use.
        layout = VarlenLayout([(8,), (4,)], device="cuda")
        counter = [0]
        layout._resolve((1,), _fake_build(counter))
        self.assertEqual(len(layout._memo), 1)

        restored = pickle.loads(pickle.dumps(layout))
        self.assertEqual(len(restored._memo), 0)  # memo is not carried over
        with self.assertRaisesRegex(RuntimeError, "not been materialized"):
            restored.cu_seqlens

        restored._ensure_materialized(torch.device("cuda"))
        self.assertTrue(torch.equal(restored.cu_seqlens, layout.cu_seqlens))
        restored_counter = [0]
        self.assertEqual(restored._resolve((1,), _fake_build(restored_counter)), 1)

    def _run_all_empty_case(self, rank, layouts):
        heads, head_dim = 2, 32
        dtype = torch.float16
        varlen_fn = {1: na1d_varlen, 2: na2d_varlen, 3: na3d_varlen}[rank]

        # Deliberately unmaterialized (no device= at construction): the
        # fast-return path must not force materialization.
        layout = VarlenLayout(layouts)
        self.assertIsNone(layout._device)

        query = torch.zeros(
            0, heads, head_dim, device="cuda", dtype=dtype, requires_grad=True
        )
        key = torch.zeros(
            0, heads, head_dim, device="cuda", dtype=dtype, requires_grad=True
        )
        value = torch.zeros(
            0, heads, head_dim, device="cuda", dtype=dtype, requires_grad=True
        )
        output, logsumexp = varlen_fn(
            query, key, value, layout, kernel_size=(3,) * rank, return_lse=True
        )
        self.assertEqual(output.shape, (0, heads, head_dim))
        self.assertEqual(output.dtype, dtype)
        self.assertEqual(logsumexp.shape, (0, heads))
        self.assertEqual(logsumexp.dtype, torch.float32)

        output.sum().backward()
        for tensor in (query, key, value):
            self.assertIsNotNone(tensor.grad)
            self.assertEqual(tensor.grad.shape, tensor.shape)

        # The fast-return path runs entirely before _resolve/materialize:
        # no memo entry, and the layout is still unmaterialized.
        self.assertEqual(len(layout._memo), 0)
        self.assertIsNone(layout._device)

    def test_all_empty_layout_returns_zero_token_outputs_1d(self):
        self._run_all_empty_case(1, ((0,), (0,)))

    def test_all_empty_layout_returns_zero_token_outputs_2d(self):
        self._run_all_empty_case(2, ((0, 8), (4, 0)))

    def test_all_empty_layout_returns_zero_token_outputs_3d(self):
        self._run_all_empty_case(3, ((0, 4, 4), (4, 0, 4)))

    def test_all_empty_unsupported_dtype_raises(self):
        # The all-empty fast path bypasses the memo-miss build (where this
        # check normally lives), so it must run its own dtype/CC check
        # rather than silently succeeding on an unsupported dtype.
        layout = VarlenLayout(((0,), (0,)))
        query = torch.zeros(0, 2, 32, device="cuda", dtype=torch.float64)
        key, value = query.clone(), query.clone()
        with self.assertRaisesRegex(ValueError, "FP16, BF16, and FP32"):
            na1d_varlen(query, key, value, layout, kernel_size=3)

    def test_all_empty_bf16_below_compute_capability_raises(self):
        layout = VarlenLayout(((0,), (0,)))
        query = torch.zeros(0, 2, 32, device="cuda", dtype=torch.bfloat16)
        key, value = query.clone(), query.clone()
        with mock.patch.object(
            torch.cuda, "get_device_capability", return_value=(6, 0)
        ):
            with self.assertRaisesRegex(ValueError, "compute capability 80 or higher"):
                na1d_varlen(query, key, value, layout, kernel_size=3)

    def _run_lse_autograd_parity_case(self, empty):
        # logsumexp must be differentiable on BOTH paths (requires_grad
        # follows the inputs), and its gradient must be ignored -- like the
        # rest of the FNA family -- rather than raising or perturbing
        # dq/dk/dv. Covers three shapes of backward: output-only, lse-only,
        # and both summed together.
        heads, head_dim = 2, 8
        dtype = torch.float16
        if empty:
            layout = VarlenLayout(((0,), (0,)))
            total_tokens = 0
        else:
            layout = VarlenLayout(((8,),), device="cuda")
            total_tokens = 8

        def fresh_inputs():
            torch.manual_seed(7)
            if total_tokens == 0:
                query = torch.zeros(
                    0, heads, head_dim, device="cuda", dtype=dtype, requires_grad=True
                )
            else:
                query = torch.randn(
                    total_tokens,
                    heads,
                    head_dim,
                    device="cuda",
                    dtype=dtype,
                    requires_grad=True,
                )
            key = query.detach().clone().requires_grad_(True)
            value = query.detach().clone().requires_grad_(True)
            return query, key, value

        def run():
            query, key, value = fresh_inputs()
            output, logsumexp = na1d_varlen(
                query, key, value, layout, kernel_size=3, return_lse=True
            )
            self.assertEqual(logsumexp.requires_grad, True)
            return query, key, value, output, logsumexp

        def assert_grads_ok(query, key, value):
            for tensor in (query, key, value):
                self.assertIsNotNone(tensor.grad)
                self.assertEqual(tensor.grad.shape, tensor.shape)

        # (a) output-only backward.
        query, key, value, output, logsumexp = run()
        output.sum().backward()
        assert_grads_ok(query, key, value)

        # (b) lse-only backward.
        query, key, value, output, logsumexp = run()
        logsumexp.sum().backward()
        assert_grads_ok(query, key, value)
        if not empty:
            # Pins the family "differentiable but ignored" semantics: lse
            # is connected to autograd, but its gradient carries no signal.
            for tensor in (query, key, value):
                self.assertTrue(torch.all(tensor.grad == 0))

        # (c) combined backward.
        query, key, value, output, logsumexp = run()
        (output.sum() + logsumexp.sum()).backward()
        assert_grads_ok(query, key, value)

    def test_lse_autograd_parity_all_empty(self):
        self._run_lse_autograd_parity_case(empty=True)

    def test_lse_autograd_parity_non_empty(self):
        self._run_lse_autograd_parity_case(empty=False)

    @skip_if_libnatten_is_not_supported()
    def test_memo_key_ignores_stride_and_is_causal(self):
        layout = VarlenLayout(((32,), (48,)), device="cuda")
        torch.manual_seed(3)
        query = torch.randn(80, 2, 64, device="cuda", dtype=torch.float16)
        key = torch.randn(80, 2, 64, device="cuda", dtype=torch.float16)
        value = torch.randn(80, 2, 64, device="cuda", dtype=torch.float16)

        out_a = na1d_varlen(
            query, key, value, layout, kernel_size=5, stride=1, is_causal=False
        )
        self.assertEqual(len(layout._memo), 1)
        entry_a = next(iter(layout._memo.values()))

        # Different stride AND different is_causal: neither is part of the
        # geometry key -- this must hit the SAME entry, not build a second
        # one.
        out_b = na1d_varlen(
            query, key, value, layout, kernel_size=5, stride=3, is_causal=True
        )
        self.assertEqual(len(layout._memo), 1)
        entry_b = next(iter(layout._memo.values()))

        self.assertIs(entry_a, entry_b)
        # Stride/is_causal still act per-call inside the kernel (from the
        # call's own arguments, not from the shared memo entry) -- outputs
        # correctly differ. Correctness of that masking itself is the main
        # suite's job (DEFAULT_CASES/PAIRWISE_CASES in test_fna_varlen.py);
        # this assertion only guards against the memo sharing silently
        # making the kernel ignore stride/causal too.
        self.assertFalse(torch.equal(out_a, out_b))


if __name__ == "__main__":
    unittest.main()
