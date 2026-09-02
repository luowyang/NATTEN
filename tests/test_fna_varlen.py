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

import contextlib
import math
import unittest
from dataclasses import replace
from typing import Any, Callable, cast, Dict

import natten
import torch
from natten._environment import _NUM_RAND_SWEEP_TESTS as RAND_SWEEP_TESTS
from natten.utils.testing import (
    skip_if_fewer_than_n_gpus,
    skip_if_libnatten_is_not_supported,
    skip_if_not_running_extended_tests,
    supports_bfloat16,
    supports_float16,
)
from torch import Tensor

# CompileCounter is a private torch._dynamo.testing utility, validated
# against torch 2.11; the frame-count expectations below may shift across
# PyTorch versions and are project-internal assertions, not public API
# contract.
from torch._dynamo.testing import CompileCounter

from .utils import (
    _dq_tolerances,
    _dtype_is_supported,
    _explicit_oracle,
    _independent_reference,
    _independent_tolerances,
    _make_layout,
    _prod,
    _regular_reference,
    _resolved_state,
    _set_deterministic,
    _tolerances,
    DEFAULT_CASES,
    NONDETERMINISTIC_CONTROL_CASE,
    PAIRWISE_CASES,
    VarlenCase,
)

# Seeds the RNG for the extended tests exercising the int32 element-count
# boundary.
_INT32_BOUNDARY_SEED = 20260825

_VARLEN_FN_BY_RANK: Dict[int, Callable[..., Any]] = {
    1: natten.na1d_varlen,
    2: natten.na2d_varlen,
    3: natten.na3d_varlen,
}


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
class VarlenFnaGpuTests(unittest.TestCase):
    def tearDown(self):
        torch.use_deterministic_algorithms(False)

    def _run_case(
        self,
        case: VarlenCase,
        sampled: bool = False,
    ) -> None:
        if not _dtype_is_supported(case.dtype):
            self.skipTest(f"{case.dtype} is unavailable on this device")
        previous = _set_deterministic(case.deterministic)
        try:
            torch.manual_seed(117 + case.rank)
            active_tokens = sum(_prod(layout) for layout in case.layouts)
            query = torch.randn(
                active_tokens,
                case.heads,
                case.head_dim,
                device="cuda",
                dtype=case.dtype,
            )
            key = torch.randn(
                active_tokens,
                case.heads_kv,
                case.head_dim,
                device="cuda",
                dtype=case.dtype,
            )
            value = torch.randn(
                active_tokens,
                case.heads_kv,
                case.head_dim_v,
                device="cuda",
                dtype=case.dtype,
            )
            query.requires_grad_(True)
            key.requires_grad_(True)
            value.requires_grad_(True)
            query_ref = query.detach().clone().requires_grad_(True)
            key_ref = key.detach().clone().requires_grad_(True)
            value_ref = value.detach().clone().requires_grad_(True)
            layout = _make_layout(case)
            varlen_fn = _VARLEN_FN_BY_RANK[case.rank]

            output, logsumexp = varlen_fn(
                query,
                key,
                value,
                layout,
                kernel_size=case.kernel_size,
                stride=case.stride,
                dilation=case.dilation,
                is_causal=case.is_causal,
                scale=case.scale,
                backward_kv_splits=case.split_cap,
                q_tile_shape=case.q_tile_shape,
                kv_tile_shape=case.kv_tile_shape,
                backward_q_tile_shape=case.backward_q_tile_shape,
                backward_kv_tile_shape=case.backward_kv_tile_shape,
                backward_use_pt_reduction=case.backward_use_pt_reduction,
                return_lse=True,
            )
            state = _resolved_state(layout)
            if case.deterministic:
                self.assertFalse(state.uses_kv_parallelism)
                self.assertTrue(
                    all(_prod(splits) == 1 for splits in state.selected_splits)
                )
            else:
                self.assertTrue(state.uses_kv_parallelism)

            output_ref, logsumexp_ref = _regular_reference(
                case, layout, query_ref, key_ref, value_ref
            )
            output_ref2, logsumexp_ref2, covered = _independent_reference(
                case, query_ref, key_ref, value_ref
            )
            independent_atol = _independent_tolerances(case.dtype)
            torch.testing.assert_close(
                output[covered], output_ref2[covered], atol=independent_atol, rtol=0
            )
            torch.testing.assert_close(
                logsumexp[covered],
                logsumexp_ref2[covered],
                atol=independent_atol,
                rtol=0,
            )
            gradient = torch.randn_like(output)
            output.backward(gradient)
            output_ref.backward(gradient)
            query_grad = cast(Tensor, query.grad)
            key_grad = cast(Tensor, key.grad)
            value_grad = cast(Tensor, value.grad)
            query_ref_grad = cast(Tensor, query_ref.grad)
            key_ref_grad = cast(Tensor, key_ref.grad)
            value_ref_grad = cast(Tensor, value_ref.grad)

            if case.deterministic:
                self.assertTrue(torch.equal(output, output_ref))
                self.assertTrue(torch.equal(logsumexp, logsumexp_ref))
                self.assertTrue(torch.equal(query_grad, query_ref_grad))
                self.assertTrue(torch.equal(key_grad, key_ref_grad))
                self.assertTrue(torch.equal(value_grad, value_ref_grad))
            else:
                self.assertTrue(torch.equal(output, output_ref))
                self.assertTrue(torch.equal(logsumexp, logsumexp_ref))
                atol, rtol = _tolerances(case.dtype)
                if sampled:
                    # Randomly sampled non-deterministic cases can land on KV-split
                    # configurations with legitimate reduction-order noise larger
                    # than _dq_tolerances' bit-identical-reference-grade bound (raw
                    # dtype eps, no rtol slack) -- the same gap documented for the
                    # CUDA-delta vs. PyTorch-reduction comparison in
                    # test_extended_pt_reduction_matches_cuda. _tolerances
                    # comfortably covers that noise while still catching real
                    # corruption.
                    dq_atol, dq_rtol = atol, rtol
                else:
                    dq_atol, dq_rtol = _dq_tolerances(case.dtype)
                torch.testing.assert_close(
                    query_grad,
                    query_ref_grad,
                    atol=dq_atol,
                    rtol=dq_rtol,
                )
                torch.testing.assert_close(key_grad, key_ref_grad, atol=atol, rtol=rtol)
                torch.testing.assert_close(
                    value_grad, value_ref_grad, atol=atol, rtol=rtol
                )
        finally:
            torch.use_deterministic_algorithms(previous)

    @skip_if_libnatten_is_not_supported()
    def test_default_broad_spectrum_against_per_document_fna(self):
        for case in DEFAULT_CASES + PAIRWISE_CASES:
            with self.subTest(case=case.name):
                self._run_case(case)

    @skip_if_libnatten_is_not_supported()
    def test_zero_scale_matches_fixed_fna_default_scale_semantics(self):
        self._run_case(replace(DEFAULT_CASES[0], scale=0.0))

    @skip_if_libnatten_is_not_supported()
    def test_cross_stream_schedule_build_and_consumption(self):
        # A VarlenLayout's schedule (worklists, KV-split selection, offset
        # tensors) is built once per geometry (memo miss, via the
        # _varlen_build_schedule_tensors custom op) and reused by every
        # later call with that geometry (memo hit) -- including calls that
        # land on a different CUDA stream than the one the schedule was
        # built on. Builds entirely on stream A, then consumes the same
        # (memo-hit, unchanged) schedule on stream B and on the default
        # stream, checking forward+backward correctness against the usual
        # oracle each time.
        case = DEFAULT_CASES[0]
        self.assertTrue(case.deterministic, "needs bit-exact equality below")
        if not _dtype_is_supported(case.dtype):
            self.skipTest(f"{case.dtype} is unavailable on this device")
        previous = _set_deterministic(case.deterministic)
        try:
            active_tokens = sum(_prod(layout) for layout in case.layouts)

            def fresh_inputs():
                torch.manual_seed(20260827)
                query = torch.randn(
                    active_tokens,
                    case.heads,
                    case.head_dim,
                    device="cuda",
                    dtype=case.dtype,
                ).requires_grad_(True)
                key = torch.randn(
                    active_tokens,
                    case.heads_kv,
                    case.head_dim,
                    device="cuda",
                    dtype=case.dtype,
                ).requires_grad_(True)
                value = torch.randn(
                    active_tokens,
                    case.heads_kv,
                    case.head_dim_v,
                    device="cuda",
                    dtype=case.dtype,
                ).requires_grad_(True)
                return query, key, value

            layout = _make_layout(case)
            varlen_fn = _VARLEN_FN_BY_RANK[case.rank]

            def run_and_check(stream):
                query, key, value = fresh_inputs()
                context = (
                    torch.cuda.stream(stream)
                    if stream is not None
                    else contextlib.nullcontext()
                )
                with context:
                    output, logsumexp = varlen_fn(
                        query,
                        key,
                        value,
                        layout,
                        kernel_size=case.kernel_size,
                        stride=case.stride,
                        dilation=case.dilation,
                        is_causal=case.is_causal,
                        scale=case.scale,
                        backward_kv_splits=case.split_cap,
                        q_tile_shape=case.q_tile_shape,
                        kv_tile_shape=case.kv_tile_shape,
                        backward_q_tile_shape=case.backward_q_tile_shape,
                        backward_kv_tile_shape=case.backward_kv_tile_shape,
                        backward_use_pt_reduction=case.backward_use_pt_reduction,
                        return_lse=True,
                    )
                    gradient = torch.randn_like(output)
                    output.backward(gradient)
                if stream is not None:
                    # Validates the documented pattern (VarlenLayout
                    # docstring / docs/backends.md): the producer stream is
                    # synchronized before the schedule built on it is
                    # consumed from a different stream below. This is the
                    # caller's obligation, not something the layout enforces
                    # itself.
                    stream.synchronize()
                else:
                    torch.cuda.synchronize()

                output_ref, logsumexp_ref = _regular_reference(
                    case,
                    layout,
                    *[
                        t.detach().clone().requires_grad_(True)
                        for t in (query, key, value)
                    ],
                )
                output_ref.backward(gradient)
                self.assertTrue(torch.equal(output, output_ref))
                self.assertTrue(torch.equal(logsumexp, logsumexp_ref))
                return query.grad, key.grad, value.grad

            stream_a = torch.cuda.Stream()
            stream_b = torch.cuda.Stream()

            # Step 1: build/warm the schedule (memo miss) entirely on
            # stream A.
            grads_a = run_and_check(stream_a)
            self.assertEqual(len(layout._memo), 1)

            # Step 2: same geometry (memo hit, tensors built on stream A)
            # run on a DIFFERENT stream.
            grads_b = run_and_check(stream_b)
            self.assertEqual(len(layout._memo), 1)

            # Step 3: same geometry again, on the default stream.
            grads_default = run_and_check(None)
            self.assertEqual(len(layout._memo), 1)

            # Deterministic mode plus identical (reseeded) inputs each
            # step: gradients must be bit-identical across builder/
            # consumer stream combinations.
            for left, right in ((grads_a, grads_b), (grads_a, grads_default)):
                for left_grad, right_grad in zip(left, right):
                    self.assertTrue(torch.equal(left_grad, right_grad))
        finally:
            torch.use_deterministic_algorithms(previous)

    # NOTE: there is deliberately no test for out-of-batch/tail-region
    # poisoning. Packed QKV must have total_tokens == layout.total_tokens
    # exactly, so there is no capacity region beyond the packed tokens to
    # poison in the first place. The adjacent concern -- worklist
    # surplus-tile culling at dilation residues -- is covered by
    # test_dilated_residue_surplus_tiles_are_culled (test_varlen_api.py),
    # which is pure host math and needs no capacity region either.

    def _test_varlen_randsweep_against_reference(self, na_dim, max_tests, quick):
        # Mirrors test_fna.py's _test_randsweep_against_reference. Doc-count SCALE
        # is covered by the 4096/84064-document extended tests below, so this
        # fuzzer's job is combinatorial geometry diversity: the caps below
        # (docs<=4, per-axis extents<=test_fna.py's max_size, heads<=4,
        # head_dim<=192) keep every sampled case well clear of the int32
        # fences on token count and heads*head_dim, so there's no
        # retry-on-int32-violation path here, only the dilation self-check
        # below.
        axis_cap = (
            {1: 64, 2: 32, 3: 16}[na_dim]
            if quick
            else {1: 2**15, 2: 128, 3: 64}[na_dim]
        )
        dtype_candidates = [torch.float32]
        if supports_float16(torch.device("cuda")):
            dtype_candidates.append(torch.float16)
        if supports_bfloat16(torch.device("cuda")):
            dtype_candidates.append(torch.bfloat16)

        for i in range(max_tests):
            torch.manual_seed(4051 + i)

            # >= 2 documents: this sampler's own reference machinery
            # (_regular_reference/_resolved_state below) introspects a
            # resolved varlen schedule, which a uniform layout (every
            # document sharing one shape -- guaranteed for a single
            # document) never builds, dispatching straight to the
            # fixed-shape kernels instead (natten.backends.varlen_fna's
            # uniform-dispatch branch). That path is correct -- covered
            # separately and directly by test_varlen_uniform_dispatch.py --
            # just not introspectable this way; single-document coverage
            # already exists there (including via EFFECTIVE_KERNEL_CASES'
            # n=1 case).
            num_docs = int(torch.randint(2, 5, (1,)).item())
            layouts = tuple(
                tuple(
                    int(torch.randint(4, axis_cap + 1, (1,)).item())
                    for _ in range(na_dim)
                )
                for _ in range(num_docs)
            )
            if all(layout == layouts[0] for layout in layouts):
                # Coincidentally-identical document shapes: still a uniform
                # layout by chance. Nudge the last document's first axis to
                # break the tie without consuming the seeded RNG stream (so
                # every other sampled parameter below is unaffected).
                perturbed = list(layouts[-1])
                perturbed[0] = (
                    perturbed[0] - 1 if perturbed[0] > 4 else perturbed[0] + 1
                )
                layouts = layouts[:-1] + (tuple(perturbed),)
            min_extent = tuple(
                min(layout[axis] for layout in layouts) for axis in range(na_dim)
            )
            # One kernel_size for the whole call (not per-document): bounded
            # by the smallest document along each axis.
            kernel_size = tuple(
                int(torch.randint(2, extent + 1, (1,)).item()) for extent in min_extent
            )
            stride = tuple(
                int(torch.randint(1, k + 1, (1,)).item()) for k in kernel_size
            )
            # Dilation off the MINIMUM per-axis extent across documents, then
            # re-verified per document below. This is where a packed batch
            # departs from test_fna.py's fixed-shape sampler, which has only
            # one extent per axis to check against.
            dilation = tuple(
                int(torch.randint(1, max(1, extent // k) + 1, (1,)).item())
                for extent, k in zip(min_extent, kernel_size)
            )

            def dilation_is_valid(dilation):
                return all(
                    extent >= k * d
                    for layout in layouts
                    for extent, k, d in zip(layout, kernel_size, dilation)
                )

            # Sampler self-check: shrink toward dilation=1 (always valid, since
            # kernel_size <= min_extent by construction) on any violation, so a
            # sampler bug fails loudly and locally instead of as a
            # "must fit every token layout" ValueError out of the schedule
            # build.
            while not dilation_is_valid(dilation) and max(dilation) > 1:
                dilation = tuple(max(1, d - 1) for d in dilation)
            assert dilation_is_valid(dilation), (
                f"sampler produced an invalid dilation: {layouts=} {kernel_size=} "
                f"{dilation=}"
            )

            is_causal = tuple(
                bool(torch.randint(0, 2, (1,)).item()) for _ in range(na_dim)
            )
            heads = int(torch.randint(1, 5, (1,)).item())
            heads_kv_choices = [h for h in range(1, heads + 1) if heads % h == 0]
            heads_kv = heads_kv_choices[
                int(torch.randint(0, len(heads_kv_choices), (1,)).item())
            ]
            head_dim = 8 * int(torch.randint(1, 25, (1,)).item())
            head_dim_v = 8 * int(torch.randint(1, 25, (1,)).item())
            dtype = dtype_candidates[
                int(torch.randint(0, len(dtype_candidates), (1,)).item())
            ]
            deterministic = bool(torch.randint(0, 2, (1,)).item())

            case = VarlenCase(
                name=f"randsweep-{na_dim}d-{i}",
                layouts=layouts,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                is_causal=is_causal,
                dtype=dtype,
                heads=heads,
                heads_kv=heads_kv,
                head_dim=head_dim,
                head_dim_v=head_dim_v,
                deterministic=deterministic,
            )

            if not case.deterministic:
                # A sampled small-enough layout can leave every document's
                # backward KV split at 1 even with deterministic=False (the
                # default split heuristic has nothing to parallelize); relabel
                # to match what the resolved state actually does, since
                # _run_case asserts uses_kv_parallelism against
                # case.deterministic. Needs one real (small, forward-only)
                # trial call, because a geometry only resolves when the entry
                # point is called with real QKV tensors.
                previous = _set_deterministic(False)
                try:
                    trial_layout = _make_layout(case)
                    trial_total = sum(_prod(layout) for layout in case.layouts)
                    trial_q = torch.randn(
                        trial_total, heads, head_dim, device="cuda", dtype=dtype
                    )
                    trial_k = torch.randn(
                        trial_total, heads_kv, head_dim, device="cuda", dtype=dtype
                    )
                    trial_v = torch.randn(
                        trial_total, heads_kv, head_dim_v, device="cuda", dtype=dtype
                    )
                    _VARLEN_FN_BY_RANK[na_dim](
                        trial_q,
                        trial_k,
                        trial_v,
                        trial_layout,
                        kernel_size=kernel_size,
                        stride=stride,
                        dilation=dilation,
                        is_causal=is_causal,
                    )
                    if not _resolved_state(trial_layout).uses_kv_parallelism:
                        case = replace(case, deterministic=True)
                finally:
                    torch.use_deterministic_algorithms(previous)

            with self.subTest(na_dim=na_dim, index=i, case=case.name):
                self._run_case(case, sampled=True)

    @skip_if_libnatten_is_not_supported()
    def test_randsweep_1d_against_reference_quick(self):
        self._test_varlen_randsweep_against_reference(1, max_tests=10, quick=True)

    @skip_if_libnatten_is_not_supported()
    def test_randsweep_2d_against_reference_quick(self):
        self._test_varlen_randsweep_against_reference(2, max_tests=10, quick=True)

    @skip_if_libnatten_is_not_supported()
    def test_randsweep_3d_against_reference_quick(self):
        self._test_varlen_randsweep_against_reference(3, max_tests=10, quick=True)

    @skip_if_not_running_extended_tests()
    @skip_if_libnatten_is_not_supported()
    def test_extended_randsweep_1d(self):
        self._test_varlen_randsweep_against_reference(
            1, max_tests=RAND_SWEEP_TESTS, quick=False
        )

    @skip_if_not_running_extended_tests()
    @skip_if_libnatten_is_not_supported()
    def test_extended_randsweep_2d(self):
        self._test_varlen_randsweep_against_reference(
            2, max_tests=RAND_SWEEP_TESTS, quick=False
        )

    @skip_if_not_running_extended_tests()
    @skip_if_libnatten_is_not_supported()
    def test_extended_randsweep_3d(self):
        self._test_varlen_randsweep_against_reference(
            3, max_tests=RAND_SWEEP_TESTS, quick=False
        )

    @skip_if_fewer_than_n_gpus(2)
    @skip_if_libnatten_is_not_supported()
    def test_correctness_on_non_default_cuda_device(self):
        # gpu_count>=2 self-skips under `make test_parallel` (each test file is
        # pinned to a single GPU there); runs only under serial `make test`.
        #
        # Two distinct VarlenLayout objects, one per device: each _run_case
        # call builds (_make_layout) a fresh layout under "cuda" (the current
        # device at construction time), so calling once on the default
        # device and once inside torch.cuda.device(1) exercises two
        # separately-pinned layouts and checks neither's device state leaks
        # into the other's kernel launch.
        self._run_case(DEFAULT_CASES[0])
        with torch.cuda.device(1):
            self._run_case(DEFAULT_CASES[0])

    @skip_if_libnatten_is_not_supported()
    @skip_if_not_running_extended_tests()
    def test_dilated_residue_surplus_against_per_document_fna(self):
        cases = (
            VarlenCase(
                "r2-dilated-residue-surplus",
                ((17, 18), (19, 22)),
                (3, 3),
                (1, 1),
                (2, 1),
                (False, False),
                torch.float32,
                2,
                1,
                64,
                64,
                False,
            ),
            VarlenCase(
                "r3-dilated-residue-surplus",
                ((7, 9, 11), (9, 9, 13)),
                (3, 4, 5),
                (1, 1, 1),
                (2, 2, 2),
                (False, False, False),
                torch.float32,
                2,
                1,
                64,
                64,
                False,
            ),
        )
        for case in cases:
            with self.subTest(case=case.name):
                self._run_case(case)

    @skip_if_libnatten_is_not_supported()
    def test_input_contract_and_noncontiguous_inputs(self):
        # NOTE on scope: a VarlenLayout stores no num_heads/head_dim/dtype
        # (by design -- the same layout is meant to be reused across heads,
        # head dims, and AMP dtypes), so there is nothing for Q/K/V to
        # "mismatch" the layout on for those three, and no such cases below.
        # What is checked below comes from the shared fmha_tensor_checks
        # (Q/K/V consistency) or na1d_varlen's own per-call checks.
        case = replace(DEFAULT_CASES[0], layouts=((17,), (23,)))
        active_tokens = sum(_prod(layout) for layout in case.layouts)
        call_kwargs = dict(
            kernel_size=case.kernel_size,
            stride=case.stride,
            dilation=case.dilation,
            is_causal=case.is_causal,
            q_tile_shape=case.q_tile_shape,
            kv_tile_shape=case.kv_tile_shape,
            backward_q_tile_shape=case.backward_q_tile_shape,
            backward_kv_tile_shape=case.backward_kv_tile_shape,
        )
        layout = _make_layout(case)
        query = torch.randn(
            active_tokens,
            case.heads,
            case.head_dim,
            device="cuda",
            dtype=case.dtype,
        )
        key = torch.randn_like(query)
        value = torch.randn_like(query)

        with self.assertRaisesRegex(TypeError, "VarlenLayout"):
            natten.na1d_varlen(
                query, key, value, object(), kernel_size=case.kernel_size
            )

        noncontiguous_inputs = (
            (
                "query",
                torch.randn(
                    active_tokens,
                    case.head_dim,
                    case.heads,
                    device="cuda",
                    dtype=case.dtype,
                ).transpose(-1, -2),
                key,
                value,
            ),
            (
                "key",
                query,
                torch.randn(
                    active_tokens,
                    case.head_dim,
                    case.heads_kv,
                    device="cuda",
                    dtype=case.dtype,
                ).transpose(-1, -2),
                value,
            ),
            (
                "value",
                query,
                key,
                torch.randn(
                    active_tokens,
                    case.head_dim_v,
                    case.heads_kv,
                    device="cuda",
                    dtype=case.dtype,
                ).transpose(-1, -2),
            ),
        )
        for (
            name,
            candidate_query,
            candidate_key,
            candidate_value,
        ) in noncontiguous_inputs:
            with self.subTest(noncontiguous=name):
                candidate = {
                    "query": candidate_query,
                    "key": candidate_key,
                    "value": candidate_value,
                }[name]
                self.assertFalse(candidate.is_contiguous())
                output = natten.na1d_varlen(
                    candidate_query,
                    candidate_key,
                    candidate_value,
                    layout,
                    **call_kwargs,
                )
                output_ref = natten.na1d_varlen(
                    candidate_query.contiguous(),
                    candidate_key.contiguous(),
                    candidate_value.contiguous(),
                    layout,
                    **call_kwargs,
                )
                self.assertTrue(torch.equal(output, output_ref))

        for name, candidate_key, candidate_value in (
            ("key", key.repeat(2, 1, 1), value),
            ("value", key, value.repeat(2, 1, 1)),
        ):
            with self.subTest(token_count_mismatch=name):
                with self.assertRaisesRegex(ValueError, "match in sequence length"):
                    natten.na1d_varlen(
                        query, candidate_key, candidate_value, layout, **call_kwargs
                    )

        with self.assertRaisesRegex(ValueError, "total_tokens"):
            natten.na1d_varlen(
                query[:-1].contiguous(),
                key[:-1].contiguous(),
                value[:-1].contiguous(),
                layout,
                **call_kwargs,
            )
        query_long = torch.cat((query, query[:1]), dim=0)
        key_long = torch.cat((key, key[:1]), dim=0)
        value_long = torch.cat((value, value[:1]), dim=0)
        token_count_cases = (
            ("query", query_long, key, value, "equal Q/K/V token counts"),
            ("key", query, key_long, value, "match in sequence length"),
            ("value", query, key, value_long, "match in sequence length"),
            ("key-value", query, key_long, value_long, "equal Q/K/V token counts"),
        )
        for (
            name,
            candidate_query,
            candidate_key,
            candidate_value,
            message,
        ) in token_count_cases:
            with self.subTest(token_count_mismatch=name):
                with self.assertRaisesRegex(ValueError, message):
                    natten.na1d_varlen(
                        candidate_query,
                        candidate_key,
                        candidate_value,
                        layout,
                        **call_kwargs,
                    )

        with self.assertRaisesRegex(ValueError, "same data type"):
            natten.na1d_varlen(query.float(), key, value, layout, **call_kwargs)
        with self.assertRaisesRegex(ValueError, "evenly divide query heads"):
            natten.na1d_varlen(
                query,
                key[:, :2].contiguous(),
                value[:, :2].contiguous(),
                layout,
                **call_kwargs,
            )
        with self.assertRaisesRegex(ValueError, "same number of heads"):
            natten.na1d_varlen(
                query, key, value[:, :2].contiguous(), layout, **call_kwargs
            )
        with self.assertRaisesRegex(ValueError, "Q and K head dims must match"):
            natten.na1d_varlen(
                query[..., :56].contiguous(), key, value, layout, **call_kwargs
            )

    @skip_if_libnatten_is_not_supported()
    def test_noncontiguous_input_backward_with_gqa_and_vdim(self):
        torch.use_deterministic_algorithms(True)
        case = replace(
            DEFAULT_CASES[1],
            layouts=((17,), (23,)),
            deterministic=True,
        )
        layout = _make_layout(case)
        varlen_fn = _VARLEN_FN_BY_RANK[case.rank]
        call_kwargs = dict(
            kernel_size=case.kernel_size,
            stride=case.stride,
            dilation=case.dilation,
            is_causal=case.is_causal,
        )
        active_tokens = sum(_prod(layout_shape) for layout_shape in case.layouts)
        torch.manual_seed(1907)
        base_inputs = (
            torch.randn(
                active_tokens,
                case.heads,
                case.head_dim,
                device="cuda",
                dtype=case.dtype,
            ),
            torch.randn(
                active_tokens,
                case.heads_kv,
                case.head_dim,
                device="cuda",
                dtype=case.dtype,
            ),
            torch.randn(
                active_tokens,
                case.heads_kv,
                case.head_dim_v,
                device="cuda",
                dtype=case.dtype,
            ),
        )
        gradient = torch.randn(
            active_tokens,
            case.heads,
            case.head_dim_v,
            device="cuda",
            dtype=case.dtype,
        )

        def as_noncontiguous(tensor: Tensor) -> Tensor:
            storage = torch.empty(
                *tensor.shape[:-1],
                tensor.shape[-1] * 2,
                device=tensor.device,
                dtype=tensor.dtype,
            )
            result = storage[..., ::2]
            result.copy_(tensor)
            result.requires_grad_(True)
            self.assertFalse(result.is_contiguous())
            return result

        for noncontiguous_index, name in enumerate(("query", "key", "value")):
            with self.subTest(noncontiguous=name):
                inputs = tuple(
                    (
                        as_noncontiguous(tensor)
                        if index == noncontiguous_index
                        else tensor.detach().clone().requires_grad_(True)
                    )
                    for index, tensor in enumerate(base_inputs)
                )
                reference_inputs = tuple(
                    tensor.detach().contiguous().requires_grad_(True)
                    for tensor in inputs
                )
                output = varlen_fn(*inputs, layout, **call_kwargs)
                output_ref = varlen_fn(*reference_inputs, layout, **call_kwargs)
                self.assertTrue(torch.equal(output, output_ref))
                output.backward(gradient)
                output_ref.backward(gradient)
                for tensor, tensor_ref in zip(inputs, reference_inputs):
                    self.assertTrue(torch.equal(tensor.grad, tensor_ref.grad))

    @skip_if_libnatten_is_not_supported()
    def test_partial_gradient_and_output_only_paths(self):
        case = replace(DEFAULT_CASES[0], layouts=((17,), (23,)))
        torch.use_deterministic_algorithms(True)
        layout = _make_layout(case)
        total = sum(_prod(layout_shape) for layout_shape in case.layouts)
        torch.manual_seed(412)
        base_inputs = (
            torch.randn(
                total,
                case.heads,
                case.head_dim,
                device="cuda",
                dtype=case.dtype,
            )
            for _ in range(3)
        )
        base_inputs = tuple(base_inputs)
        gradient = torch.randn_like(base_inputs[0])

        gradient_paths = (
            (True, False, False),
            (False, True, False),
            (False, False, True),
        )
        for requires_grad in gradient_paths:
            with self.subTest(requires_grad=requires_grad):
                inputs = tuple(
                    tensor.detach().clone().requires_grad_(enabled)
                    for tensor, enabled in zip(base_inputs, requires_grad)
                )
                reference_inputs = tuple(
                    tensor.detach().clone().requires_grad_(enabled)
                    for tensor, enabled in zip(base_inputs, requires_grad)
                )
                output = natten.na1d_varlen(
                    *inputs,
                    layout,
                    kernel_size=case.kernel_size,
                    stride=case.stride,
                    dilation=case.dilation,
                    is_causal=case.is_causal,
                    return_lse=False,
                )
                output_ref, _ = _regular_reference(case, layout, *reference_inputs)
                self.assertTrue(torch.equal(output, output_ref))
                output.backward(gradient)
                output_ref.backward(gradient)
                for enabled, tensor, tensor_ref in zip(
                    requires_grad, inputs, reference_inputs
                ):
                    if enabled:
                        self.assertTrue(torch.equal(tensor.grad, tensor_ref.grad))
                    else:
                        self.assertIsNone(tensor.grad)
                        self.assertIsNone(tensor_ref.grad)

        output_only = natten.na1d_varlen(
            *base_inputs,
            layout,
            kernel_size=case.kernel_size,
            stride=case.stride,
            dilation=case.dilation,
            is_causal=case.is_causal,
            return_lse=False,
        )
        output_with_lse, _ = natten.na1d_varlen(
            *base_inputs,
            layout,
            kernel_size=case.kernel_size,
            stride=case.stride,
            dilation=case.dilation,
            is_causal=case.is_causal,
            return_lse=True,
        )
        self.assertTrue(torch.equal(output_only, output_with_lse))

    @skip_if_libnatten_is_not_supported()
    def test_document_seams_have_independent_forward_and_backward_oracle(self):
        torch.use_deterministic_algorithms(True)
        for rank in (1, 2, 3):
            with self.subTest(rank=rank):
                doc_shape = (3,) * rank
                document_tokens = _prod(doc_shape)
                active_tokens = 2 * document_tokens
                layout = natten.VarlenLayout((doc_shape, doc_shape), device="cuda")
                query = torch.zeros(
                    active_tokens, 1, 8, device="cuda", requires_grad=True
                )
                key = query.detach().clone().requires_grad_(True)
                value = torch.empty_like(query, requires_grad=False)
                value[:document_tokens] = -3
                value[document_tokens:active_tokens] = 5
                value.requires_grad_(True)
                varlen_fn = _VARLEN_FN_BY_RANK[rank]
                output, logsumexp = varlen_fn(
                    query, key, value, layout, kernel_size=doc_shape, return_lse=True
                )
                gradient = torch.empty_like(output)
                gradient[:document_tokens] = 2
                gradient[document_tokens:active_tokens] = -4
                output.backward(gradient)

                self.assertTrue(torch.equal(output, value))
                torch.testing.assert_close(
                    logsumexp,
                    torch.full_like(logsumexp, math.log(document_tokens)),
                    atol=1e-6,
                    rtol=0,
                )
                self.assertTrue(torch.count_nonzero(query.grad) == 0)
                self.assertTrue(torch.count_nonzero(key.grad) == 0)
                torch.testing.assert_close(value.grad, gradient, atol=1e-6, rtol=1e-6)

    @skip_if_libnatten_is_not_supported()
    def test_deterministic_repeated_runs_are_bitwise_equal(self):
        torch.use_deterministic_algorithms(True)
        for case in (DEFAULT_CASES[0], DEFAULT_CASES[6], DEFAULT_CASES[13]):
            with self.subTest(rank=case.rank):
                layout = _make_layout(case)
                varlen_fn = _VARLEN_FN_BY_RANK[case.rank]
                total = sum(_prod(doc_layout) for doc_layout in case.layouts)
                torch.manual_seed(733 + case.rank)
                inputs = (
                    torch.randn(
                        total,
                        case.heads,
                        case.head_dim,
                        device="cuda",
                        dtype=case.dtype,
                    ),
                    torch.randn(
                        total,
                        case.heads_kv,
                        case.head_dim,
                        device="cuda",
                        dtype=case.dtype,
                    ),
                    torch.randn(
                        total,
                        case.heads_kv,
                        case.head_dim_v,
                        device="cuda",
                        dtype=case.dtype,
                    ),
                )
                gradient = torch.randn(
                    total,
                    case.heads,
                    case.head_dim_v,
                    device="cuda",
                    dtype=case.dtype,
                )

                def run_once():
                    query, key, value = (
                        tensor.detach().clone().requires_grad_(True)
                        for tensor in inputs
                    )
                    output, logsumexp = varlen_fn(
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

                first = run_once()
                second = run_once()
                self.assertTrue(
                    all(torch.equal(left, right) for left, right in zip(first, second))
                )

    def _run_empty_document_case(
        self,
        rank,
        layouts_with_empty,
        kernel_size,
        stride,
        dilation,
        is_causal,
    ) -> None:
        # Oracle: a mixed (empty + non-empty) layout vs. the SAME layout
        # with its empty documents removed, run on IDENTICAL packed
        # query/key/value rows -- an empty document contributes zero rows
        # either way, so the packed tensors are byte-for-byte the same
        # regardless of whether the empty entries are present in the
        # layout. Any deviation between the two runs would mean an "empty"
        # document was not actually a no-op.
        dtype = torch.float16
        heads, head_dim = 2, 32
        reduced_layouts = tuple(doc for doc in layouts_with_empty if _prod(doc) > 0)
        total = sum(_prod(doc) for doc in layouts_with_empty)
        self.assertEqual(total, sum(_prod(doc) for doc in reduced_layouts))
        varlen_fn = _VARLEN_FN_BY_RANK[rank]

        torch.manual_seed(2600 + rank)
        query = torch.randn(total, heads, head_dim, device="cuda", dtype=dtype)
        key = torch.randn(total, heads, head_dim, device="cuda", dtype=dtype)
        value = torch.randn(total, heads, head_dim, device="cuda", dtype=dtype)
        gradient = torch.randn(total, heads, head_dim, device="cuda", dtype=dtype)

        def run(layouts, deterministic, split_cap):
            previous = _set_deterministic(deterministic)
            try:
                layout = natten.VarlenLayout(layouts, device="cuda")
                q = query.detach().clone().requires_grad_(True)
                k = key.detach().clone().requires_grad_(True)
                v = value.detach().clone().requires_grad_(True)
                output, logsumexp = varlen_fn(
                    q,
                    k,
                    v,
                    layout,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=dilation,
                    is_causal=is_causal,
                    backward_kv_splits=split_cap,
                    return_lse=True,
                )
                return output, logsumexp, q, k, v
            finally:
                torch.use_deterministic_algorithms(previous)

        # Forward (+lse): the forward kernel has no atomics (split-
        # independent), so this must be bit-identical in BOTH determinism
        # regimes.
        for deterministic in (True, False):
            with self.subTest(rank=rank, deterministic=deterministic, part="forward"):
                out_mixed, lse_mixed, *_ = run(layouts_with_empty, deterministic, None)
                out_reduced, lse_reduced, *_ = run(reduced_layouts, deterministic, None)
                self.assertTrue(torch.equal(out_mixed, out_reduced))
                self.assertTrue(torch.equal(lse_mixed, lse_reduced))

        # Backward, deterministic=False: split_cap=(1,)*rank keeps every
        # non-empty document single-split (no atomics), isolating the
        # empty-document no-op claim from KV-parallelism's own reduction-
        # order behavior (already covered by DEFAULT_CASES/PAIRWISE_CASES).
        with self.subTest(rank=rank, deterministic=False, part="backward"):
            split_cap = (1,) * rank
            out_mixed, lse_mixed, qm, km, vm = run(layouts_with_empty, False, split_cap)
            out_mixed.backward(gradient)
            out_reduced, lse_reduced, qr, kr, vr = run(
                reduced_layouts, False, split_cap
            )
            out_reduced.backward(gradient)
            for observed, expected in (
                (out_mixed, out_reduced),
                (lse_mixed, lse_reduced),
                (qm.grad, qr.grad),
                (km.grad, kr.grad),
                (vm.grad, vr.grad),
            ):
                self.assertTrue(torch.equal(observed, expected))

        # Backward, deterministic=True: each document contributes at most
        # one backward work item under determinism (fna_backward.cu); an
        # empty document correctly contributes zero (a zero-token document
        # must not schedule spurious work), so a mixed empty/non-empty
        # layout legitimately has backward_work_count < batch_size_64 --
        # not "KV parallelism in use" (uses_kv_parallelism is False and no
        # atomics are involved anywhere in this call). Forward is
        # unaffected (no such check in fna_forward.cu); an all-empty layout
        # is unaffected (the fast-return path never reaches this kernel); a
        # layout with no empty documents is unaffected (backward_work_count
        # == batch_size_64 there already).
        with self.subTest(rank=rank, deterministic=True, part="backward"):
            out_mixed, lse_mixed, qm, km, vm = run(layouts_with_empty, True, None)
            out_mixed.backward(gradient)
            out_reduced, lse_reduced, qr, kr, vr = run(reduced_layouts, True, None)
            out_reduced.backward(gradient)
            for observed, expected in (
                (out_mixed, out_reduced),
                (lse_mixed, lse_reduced),
                (qm.grad, qr.grad),
                (km.grad, kr.grad),
                (vm.grad, vr.grad),
            ):
                self.assertTrue(torch.equal(observed, expected))

    @skip_if_libnatten_is_not_supported()
    def test_mixed_empty_documents_match_reduced_layout_1d(self):
        self._run_empty_document_case(
            1,
            ((0,), (37,), (0,), (24,), (0,)),
            (5,),
            (2,),
            (1,),
            (False,),
        )

    @skip_if_libnatten_is_not_supported()
    def test_mixed_empty_documents_match_reduced_layout_2d(self):
        self._run_empty_document_case(
            2,
            ((0, 16), (12, 16), (9, 0), (8, 10), (0, 9)),
            (3, 3),
            (1, 1),
            (1, 1),
            (False, False),
        )

    @skip_if_libnatten_is_not_supported()
    def test_mixed_empty_documents_match_reduced_layout_3d(self):
        self._run_empty_document_case(
            3,
            ((0, 8, 8), (6, 8, 8), (5, 0, 6), (6, 6, 6), (5, 5, 0)),
            (3, 3, 3),
            (1, 1, 1),
            (1, 1, 1),
            (False, False, False),
        )

    @skip_if_libnatten_is_not_supported()
    def test_each_rank_and_document_against_explicit_coordinate_oracle(self):
        torch.use_deterministic_algorithms(True)
        cases = (
            VarlenCase(
                "explicit-r1",
                ((3,), (5,)),
                (3,),
                (1,),
                (1,),
                (False,),
                torch.float32,
                4,
                1,
                16,
                24,
            ),
            VarlenCase(
                "explicit-r2",
                ((2, 3), (3, 2)),
                (2, 2),
                (1, 2),
                (1, 1),
                (False, True),
                torch.float32,
                4,
                1,
                16,
                24,
            ),
            VarlenCase(
                "explicit-r3",
                ((2, 3, 3), (3, 2, 4)),
                (2, 2, 3),
                (1, 2, 1),
                (1, 1, 1),
                (False, True, False),
                torch.float32,
                4,
                1,
                16,
                24,
            ),
            VarlenCase(
                "explicit-r1-bf16-gqa-causal",
                ((3,), (5,)),
                (3,),
                (1,),
                (1,),
                (True,),
                torch.bfloat16,
                4,
                2,
                16,
                24,
            ),
            VarlenCase(
                "explicit-r2-bf16-gqa-causal",
                ((2, 3), (3, 2)),
                (2, 2),
                (1, 2),
                (1, 1),
                (True, True),
                torch.bfloat16,
                4,
                2,
                16,
                24,
            ),
            VarlenCase(
                "explicit-r3-bf16-gqa-causal",
                ((2, 3, 3), (3, 2, 4)),
                (2, 2, 3),
                (1, 2, 1),
                (1, 1, 1),
                (True, True, True),
                torch.bfloat16,
                4,
                2,
                16,
                24,
            ),
        )
        for case in cases:
            with self.subTest(rank=case.rank):
                self.assertTrue(
                    all(
                        any(extent == kernel for extent in extents)
                        for kernel, extents in zip(case.kernel_size, zip(*case.layouts))
                    )
                )
                total = sum(_prod(doc_layout) for doc_layout in case.layouts)
                torch.manual_seed(91 + case.rank)
                query = torch.randn(
                    total,
                    case.heads,
                    case.head_dim,
                    device="cuda",
                    dtype=case.dtype,
                    requires_grad=True,
                )
                key = torch.randn(
                    total,
                    case.heads_kv,
                    case.head_dim,
                    device="cuda",
                    dtype=case.dtype,
                    requires_grad=True,
                )
                value = torch.randn(
                    total,
                    case.heads_kv,
                    case.head_dim_v,
                    device="cuda",
                    dtype=case.dtype,
                    requires_grad=True,
                )
                query_ref = query.detach().clone().requires_grad_(True)
                key_ref = key.detach().clone().requires_grad_(True)
                value_ref = value.detach().clone().requires_grad_(True)
                layout = _make_layout(case)
                varlen_fn = _VARLEN_FN_BY_RANK[case.rank]
                output, logsumexp = varlen_fn(
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
                output_refs = []
                logsumexp_refs = []
                start = 0
                for doc_layout in case.layouts:
                    end = start + _prod(doc_layout)
                    document_output, document_lse = _explicit_oracle(
                        query_ref[start:end],
                        key_ref[start:end],
                        value_ref[start:end],
                        doc_layout,
                        case.kernel_size,
                        case.stride,
                        case.dilation,
                        case.is_causal,
                    )
                    output_refs.append(document_output)
                    logsumexp_refs.append(document_lse)
                    start = end
                output_ref = torch.cat(output_refs, dim=0)
                logsumexp_ref = torch.cat(logsumexp_refs, dim=0)
                gradient = torch.randn_like(output)
                output.backward(gradient)
                output_ref.backward(gradient.float())
                # fp32 keeps its original literals. _dq_tolerances(bf16) is too
                # tight for dQ against this from-scratch oracle in practice (an
                # observed 0.00983 max abs diff vs its 0.0078 bound on
                # explicit-r2-bf16-gqa-causal); use the same bucket for all three
                # grads instead.
                if case.dtype == torch.float32:
                    fwd_atol, fwd_rtol = 3e-4, 3e-4
                    dq_atol, dq_rtol = 4e-4, 4e-4
                    grad_atol, grad_rtol = 4e-4, 4e-4
                else:
                    fwd_atol, fwd_rtol = _tolerances(case.dtype)
                    dq_atol, dq_rtol = _tolerances(case.dtype)
                    grad_atol, grad_rtol = _tolerances(case.dtype)
                torch.testing.assert_close(
                    output.float(), output_ref, atol=fwd_atol, rtol=fwd_rtol
                )
                torch.testing.assert_close(
                    logsumexp.float(), logsumexp_ref, atol=fwd_atol, rtol=fwd_rtol
                )
                for observed, expected, atol, rtol in (
                    (query.grad, query_ref.grad, dq_atol, dq_rtol),
                    (key.grad, key_ref.grad, grad_atol, grad_rtol),
                    (value.grad, value_ref.grad, grad_atol, grad_rtol),
                ):
                    torch.testing.assert_close(observed, expected, atol=atol, rtol=rtol)
                start = 0
                for doc_layout in case.layouts:
                    end = start + _prod(doc_layout)
                    for tensor in (output, query.grad, key.grad, value.grad):
                        self.assertGreater(
                            torch.count_nonzero(tensor[start:end]).item(), 0
                        )
                    start = end

    @skip_if_libnatten_is_not_supported()
    def test_seam_and_window_negative_controls_have_signal(self):
        torch.use_deterministic_algorithms(True)
        query = torch.zeros(10, 1, 8, device="cuda", dtype=torch.float32)
        key = torch.zeros_like(query)
        value = torch.zeros_like(query)
        query[5, 0, 0] = 1
        key[4, 0, 0] = 20
        value[4, 0, 0] = 10
        split_layout = natten.VarlenLayout(((5,), (5,)), device="cuda")
        flat_layout = natten.VarlenLayout(((10,),), device="cuda")
        split_output = natten.na1d_varlen(
            query, key, value, split_layout, kernel_size=3
        )
        flat_output = natten.na1d_varlen(query, key, value, flat_layout, kernel_size=3)
        self.assertEqual(split_output[5, 0, 0].item(), 0.0)
        self.assertGreater(flat_output[5, 0, 0].item(), 9.0)

        query.zero_()
        key.zero_()
        value.zero_()
        query[3, 0, 0] = 1
        key[0, 0, 0] = 20
        value[0, 0, 0] = 10
        local_layout = natten.VarlenLayout(((10,),), device="cuda")
        full_layout = natten.VarlenLayout(((10,),), device="cuda")
        local_output = natten.na1d_varlen(
            query, key, value, local_layout, kernel_size=3
        )
        full_output = natten.na1d_varlen(query, key, value, full_layout, kernel_size=10)
        self.assertEqual(local_output[3, 0, 0].item(), 0.0)
        self.assertGreater(full_output[3, 0, 0].item(), 9.0)

    @skip_if_libnatten_is_not_supported()
    def test_forward_captures_deterministic_state_for_backward(self):
        # Scenario A: deterministic forward, flip to nondeterministic before backward.
        # ctx.deterministic is captured at forward time (varlen_fna.py), so the flip
        # must be a numeric no-op against a reference held at the forward-time value
        # (deterministic=True) throughout, never flipped itself.
        case = DEFAULT_CASES[0]
        torch.manual_seed(559 + case.rank)
        torch.use_deterministic_algorithms(True)
        layout = _make_layout(case)
        active_tokens = sum(_prod(doc_layout) for doc_layout in case.layouts)
        query = torch.randn(
            active_tokens,
            case.heads,
            case.head_dim,
            device="cuda",
            dtype=case.dtype,
            requires_grad=True,
        )
        key = torch.randn(
            active_tokens,
            case.heads_kv,
            case.head_dim,
            device="cuda",
            dtype=case.dtype,
            requires_grad=True,
        )
        value = torch.randn(
            active_tokens,
            case.heads_kv,
            case.head_dim_v,
            device="cuda",
            dtype=case.dtype,
            requires_grad=True,
        )
        query_ref = query.detach().clone().requires_grad_(True)
        key_ref = key.detach().clone().requires_grad_(True)
        value_ref = value.detach().clone().requires_grad_(True)
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
        output_ref, logsumexp_ref = _regular_reference(
            case, layout, query_ref, key_ref, value_ref
        )
        gradient = torch.randn_like(output)
        torch.use_deterministic_algorithms(False)
        output.backward(gradient)
        self.assertIsNotNone(query.grad)
        torch.use_deterministic_algorithms(True)
        output_ref.backward(gradient)
        self.assertTrue(torch.equal(output, output_ref))
        self.assertTrue(torch.equal(logsumexp, logsumexp_ref))
        self.assertTrue(torch.equal(query.grad, query_ref.grad))
        self.assertTrue(torch.equal(key.grad, key_ref.grad))
        self.assertTrue(torch.equal(value.grad, value_ref.grad))

        # Scenario B: nondeterministic forward, flip to deterministic before backward.
        # split_cap=(1,) forces uses_kv_parallelism=False, isolating
        # backward_use_pt_reduction's compute_delta_with_torch path from true
        # KV-parallel nondeterminism (a separate concern covered by Scenario C).
        pt_reduction_case = VarlenCase(
            "determinism-flip-pt-reduction",
            # Two different document sizes (non-uniform): this scenario is
            # about the varlen schedule's uses_kv_parallelism/memo state
            # specifically (checked below), which a uniform (here, a single-
            # document) layout would bypass via the uniform-dispatch branch
            # in natten.backends.varlen_fna.
            ((17,), (13,)),
            (5,),
            (1,),
            (1,),
            (False,),
            torch.float16,
            2,
            2,
            64,
            64,
            deterministic=False,
            split_cap=(1,),
            backward_use_pt_reduction=True,
        )
        torch.manual_seed(560 + pt_reduction_case.rank)
        torch.use_deterministic_algorithms(False)
        layout = _make_layout(pt_reduction_case)
        active_tokens = sum(
            _prod(doc_layout) for doc_layout in pt_reduction_case.layouts
        )
        query = torch.randn(
            active_tokens,
            pt_reduction_case.heads,
            pt_reduction_case.head_dim,
            device="cuda",
            dtype=pt_reduction_case.dtype,
            requires_grad=True,
        )
        key = torch.randn(
            active_tokens,
            pt_reduction_case.heads_kv,
            pt_reduction_case.head_dim,
            device="cuda",
            dtype=pt_reduction_case.dtype,
            requires_grad=True,
        )
        value = torch.randn(
            active_tokens,
            pt_reduction_case.heads_kv,
            pt_reduction_case.head_dim_v,
            device="cuda",
            dtype=pt_reduction_case.dtype,
            requires_grad=True,
        )
        query_ref = query.detach().clone().requires_grad_(True)
        key_ref = key.detach().clone().requires_grad_(True)
        value_ref = value.detach().clone().requires_grad_(True)
        output, logsumexp = natten.na1d_varlen(
            query,
            key,
            value,
            layout,
            kernel_size=pt_reduction_case.kernel_size,
            stride=pt_reduction_case.stride,
            dilation=pt_reduction_case.dilation,
            is_causal=pt_reduction_case.is_causal,
            backward_kv_splits=pt_reduction_case.split_cap,
            backward_use_pt_reduction=pt_reduction_case.backward_use_pt_reduction,
            return_lse=True,
        )
        self.assertFalse(_resolved_state(layout).uses_kv_parallelism)
        output_ref, logsumexp_ref = _regular_reference(
            pt_reduction_case, layout, query_ref, key_ref, value_ref
        )
        gradient = torch.randn_like(output)
        torch.use_deterministic_algorithms(True)
        output.backward(gradient)
        self.assertIsNotNone(query.grad)
        torch.use_deterministic_algorithms(False)
        output_ref.backward(gradient)
        self.assertTrue(torch.equal(output, output_ref))
        self.assertTrue(torch.equal(logsumexp, logsumexp_ref))
        dq_atol, dq_rtol = _dq_tolerances(pt_reduction_case.dtype)
        torch.testing.assert_close(
            query.grad, query_ref.grad, atol=dq_atol, rtol=dq_rtol
        )
        atol, rtol = _tolerances(pt_reduction_case.dtype)
        torch.testing.assert_close(key.grad, key_ref.grad, atol=atol, rtol=rtol)
        torch.testing.assert_close(value.grad, value_ref.grad, atol=atol, rtol=rtol)

        # Scenario C: same flip, but with true KV-parallel nondeterminism in play
        # (no split_cap), unlike Scenario B's forced split_cap=(1,).
        nondeterministic_case = NONDETERMINISTIC_CONTROL_CASE
        torch.manual_seed(561 + nondeterministic_case.rank)
        torch.use_deterministic_algorithms(False)
        layout = _make_layout(nondeterministic_case)
        active_tokens = sum(
            _prod(doc_layout) for doc_layout in nondeterministic_case.layouts
        )
        query = torch.randn(
            active_tokens,
            nondeterministic_case.heads,
            nondeterministic_case.head_dim,
            device="cuda",
            dtype=nondeterministic_case.dtype,
            requires_grad=True,
        )
        key = torch.randn(
            active_tokens,
            nondeterministic_case.heads_kv,
            nondeterministic_case.head_dim,
            device="cuda",
            dtype=nondeterministic_case.dtype,
            requires_grad=True,
        )
        value = torch.randn(
            active_tokens,
            nondeterministic_case.heads_kv,
            nondeterministic_case.head_dim_v,
            device="cuda",
            dtype=nondeterministic_case.dtype,
            requires_grad=True,
        )
        query_ref = query.detach().clone().requires_grad_(True)
        key_ref = key.detach().clone().requires_grad_(True)
        value_ref = value.detach().clone().requires_grad_(True)
        output, logsumexp = natten.na1d_varlen(
            query,
            key,
            value,
            layout,
            kernel_size=nondeterministic_case.kernel_size,
            stride=nondeterministic_case.stride,
            dilation=nondeterministic_case.dilation,
            is_causal=nondeterministic_case.is_causal,
            return_lse=True,
        )
        self.assertTrue(_resolved_state(layout).uses_kv_parallelism)
        output_ref, logsumexp_ref = _regular_reference(
            nondeterministic_case, layout, query_ref, key_ref, value_ref
        )
        gradient = torch.randn_like(output)
        torch.use_deterministic_algorithms(True)
        output.backward(gradient)
        self.assertIsNotNone(query.grad)
        torch.use_deterministic_algorithms(False)
        output_ref.backward(gradient)
        self.assertTrue(torch.equal(output, output_ref))
        self.assertTrue(torch.equal(logsumexp, logsumexp_ref))
        dq_atol, dq_rtol = _dq_tolerances(nondeterministic_case.dtype)
        torch.testing.assert_close(
            query.grad, query_ref.grad, atol=dq_atol, rtol=dq_rtol
        )
        atol, rtol = _tolerances(nondeterministic_case.dtype)
        torch.testing.assert_close(key.grad, key_ref.grad, atol=atol, rtol=rtol)
        torch.testing.assert_close(value.grad, value_ref.grad, atol=atol, rtol=rtol)

    @skip_if_libnatten_is_not_supported()
    def test_fullgraph_and_aot_autograd(self):
        case = VarlenCase(
            "compile",
            ((17,), (33,)),
            (5,),
            (1,),
            (1,),
            (False,),
            torch.float16,
            2,
            2,
            64,
            64,
        )
        torch.use_deterministic_algorithms(True)
        layout = _make_layout(case)
        total = sum(_prod(doc_layout) for doc_layout in case.layouts)

        def fn(q, k, v):
            return natten.na1d_varlen(
                q,
                k,
                v,
                layout,
                kernel_size=case.kernel_size,
                stride=case.stride,
                dilation=case.dilation,
                is_causal=case.is_causal,
            )

        torch.manual_seed(238)
        inputs = (
            torch.randn(
                total,
                case.heads,
                case.head_dim,
                device="cuda",
                dtype=case.dtype,
            )
            for _ in range(3)
        )
        inputs = tuple(inputs)
        gradient = torch.randn(
            total,
            case.heads,
            case.head_dim_v,
            device="cuda",
            dtype=case.dtype,
        )

        def run(callable_fn):
            run_inputs = tuple(
                tensor.detach().clone().requires_grad_(True) for tensor in inputs
            )
            output = callable_fn(*run_inputs)
            output.backward(gradient)
            return (output.detach(),) + tuple(
                tensor.grad.detach() for tensor in run_inputs
            )

        reference = run(fn)
        for backend in ("eager", "aot_eager"):
            with self.subTest(backend=backend):
                torch.compiler.reset()
                try:
                    with torch._dynamo.config.patch(
                        recompile_limit=1,
                        accumulated_recompile_limit=1,
                        fail_on_recompile_limit_hit=True,
                    ):
                        compiled = torch.compile(fn, backend=backend, fullgraph=True)
                        first = run(compiled)
                        second = run(compiled)
                        self.assertTrue(
                            all(
                                torch.equal(expected, observed)
                                for expected, observed in zip(reference, first)
                            )
                        )
                        self.assertTrue(
                            all(
                                torch.equal(expected, observed)
                                for expected, observed in zip(first, second)
                            )
                        )
                finally:
                    torch.compiler.reset()

    @skip_if_libnatten_is_not_supported()
    def test_fullgraph_compile_empty_documents(self):
        # torch.compile wrapper around na1d_varlen, covering (a) a mixed
        # empty/non-empty layout and (b) an all-empty layout under
        # fullgraph=True -- same shape as test_fullgraph_and_aot_autograd
        # above (eager reference, then "eager"/"aot_eager" compiled
        # backends, asserting bit-identical results across repeated calls).
        #
        # The mixed case's backward is deliberately run under
        # deterministic=False with an explicit backward_kv_splits=(1,) to
        # force single-split (no atomics) for every document: this keeps
        # the comparison isolated to what it's actually checking
        # (compiled-vs-eager correctness for the new empty-document code
        # paths), independent of KV-split reduction-order noise.
        heads, head_dim = 2, 32
        dtype = torch.float16

        def make_case(name, layouts):
            return VarlenCase(
                name,
                layouts,
                (3,),
                (1,),
                (1,),
                (False,),
                dtype,
                heads,
                heads,
                head_dim,
                head_dim,
            )

        cases = (
            (
                make_case("compile-mixed-empty", ((0,), (17,), (0,), (33,), (0,))),
                False,
                (1,),
            ),
            (make_case("compile-all-empty", ((0,), (0,))), True, None),
        )

        for case, deterministic, split_cap in cases:
            with self.subTest(case=case.name):
                previous = _set_deterministic(deterministic)
                try:
                    total = sum(_prod(doc) for doc in case.layouts)
                    layout = _make_layout(case)

                    def fn(q, k, v):
                        return natten.na1d_varlen(
                            q,
                            k,
                            v,
                            layout,
                            kernel_size=case.kernel_size,
                            backward_kv_splits=split_cap,
                            return_lse=True,
                        )

                    torch.manual_seed(4200)
                    inputs = tuple(
                        torch.randn(total, heads, head_dim, device="cuda", dtype=dtype)
                        for _ in range(3)
                    )
                    gradient = torch.randn(
                        total, heads, head_dim, device="cuda", dtype=dtype
                    )

                    def run(callable_fn):
                        q, k, v = (
                            t.detach().clone().requires_grad_(True) for t in inputs
                        )
                        out, lse = callable_fn(q, k, v)
                        out.backward(gradient)
                        return (
                            out.detach(),
                            lse.detach(),
                            q.grad.detach(),
                            k.grad.detach(),
                            v.grad.detach(),
                        )

                    reference = run(fn)  # eager, also warms the layout's memo
                    for backend in ("eager", "aot_eager"):
                        with self.subTest(case=case.name, backend=backend):
                            torch.compiler.reset()
                            try:
                                with torch._dynamo.config.patch(
                                    recompile_limit=1,
                                    accumulated_recompile_limit=1,
                                    fail_on_recompile_limit_hit=True,
                                ):
                                    compiled = torch.compile(
                                        fn, backend=backend, fullgraph=True
                                    )
                                    first = run(compiled)
                                    second = run(compiled)
                                    self.assertTrue(
                                        all(
                                            torch.equal(expected, observed)
                                            for expected, observed in zip(
                                                reference, first
                                            )
                                        )
                                    )
                                    self.assertTrue(
                                        all(
                                            torch.equal(expected, observed)
                                            for expected, observed in zip(first, second)
                                        )
                                    )
                            finally:
                                torch.compiler.reset()
                finally:
                    torch.use_deterministic_algorithms(previous)

    @skip_if_libnatten_is_not_supported()
    def test_cold_miss_default_budget_correct_and_bounded_recompile(self):
        # Unlike test_fullgraph_and_aot_autograd above (which warms its
        # layout via an eager `reference = run(fn)` call using the exact same
        # key before ever compiling), this builds a fresh VarlenLayout and
        # lets its FIRST resolve happen inside the compiled region. Checks
        # both claims against the real op, rather than against the stubbed
        # build used by test_varlen_layout.py's memo-mechanism tests:
        # correct numerics, and a bounded (<=1 extra) recompile cost from the
        # memo-membership guard flip once the miss leaves the memo warm.
        #
        # The budget is pinned explicitly to torch._dynamo.config's
        # compiled-in defaults (as of torch 2.11) rather than left ambient:
        # tests/utils.py's reset_torch_compile helper, used by sibling test
        # modules such as test_torch_compile.py, mutates
        # accumulated_recompile_limit/fail_on_recompile_limit_hit process-
        # globally without restoring them, so an un-pinned "default" budget
        # in a full-suite run can silently inherit an earlier test's tight
        # one.
        default_budget = dict(
            recompile_limit=8,
            accumulated_recompile_limit=256,
            cache_size_limit=8,
            fail_on_recompile_limit_hit=False,
        )
        case = VarlenCase(
            "cold-miss-default-budget",
            ((17,), (33,)),
            (5,),
            (1,),
            (1,),
            (False,),
            torch.float16,
            2,
            2,
            64,
            64,
        )
        total = sum(_prod(doc_layout) for doc_layout in case.layouts)
        torch.use_deterministic_algorithms(True)

        def fn(q, k, v, layout):
            return natten.na1d_varlen(
                q,
                k,
                v,
                layout,
                kernel_size=case.kernel_size,
                stride=case.stride,
                dilation=case.dilation,
                is_causal=case.is_causal,
            )

        torch.manual_seed(608)
        query = torch.randn(
            total, case.heads, case.head_dim, device="cuda", dtype=case.dtype
        )
        key = torch.randn(
            total, case.heads_kv, case.head_dim, device="cuda", dtype=case.dtype
        )
        value = torch.randn(
            total, case.heads_kv, case.head_dim_v, device="cuda", dtype=case.dtype
        )

        reference = fn(query, key, value, _make_layout(case))  # eager, fresh layout

        torch.compiler.reset()
        cold_counter = CompileCounter()
        cold_layout = _make_layout(case)  # materialized, but geometry memo is cold
        try:
            with torch._dynamo.config.patch(**default_budget):
                compiled_cold = torch.compile(fn, backend=cold_counter, fullgraph=True)
                out1 = compiled_cold(query, key, value, cold_layout)
                out2 = compiled_cold(query, key, value, cold_layout)
                out3 = compiled_cold(query, key, value, cold_layout)
        finally:
            torch.compiler.reset()

        self.assertTrue(torch.equal(out1, reference))
        self.assertTrue(torch.equal(out2, reference))
        self.assertTrue(torch.equal(out3, reference))
        cold_frame_count = cold_counter.frame_count

        # Same case, pre-warmed eagerly -- never takes the miss branch under
        # compile at all -- as the bound's reference point.
        torch.compiler.reset()
        warm_counter = CompileCounter()
        warm_layout = _make_layout(case)
        fn(query, key, value, warm_layout)  # eager prewarm
        try:
            with torch._dynamo.config.patch(**default_budget):
                compiled_warm = torch.compile(fn, backend=warm_counter, fullgraph=True)
                compiled_warm(query, key, value, warm_layout)
        finally:
            torch.compiler.reset()

        self.assertLessEqual(cold_frame_count - warm_counter.frame_count, 1)

    @skip_if_libnatten_is_not_supported()
    def test_distinct_layouts_amortize_under_torch_compile(self):
        # Distinct batch compositions amortize under torch.compile.
        # na1d_varlen/na2d_varlen/na3d_varlen are three distinct top-level
        # entry points, so compiling "once" here means three independent
        # compiled functions (one per rank), each contributing its own
        # frames. Measured per rank over the call order used below (base
        # case, variant case, variant, base):
        #   rank=1: [1, 2, 2, 2]   rank=2: [1, 2, 2, 2]   rank=3: [1, 2, 2, 2]
        # Each rank independently plateaus at 2 (one compile for its first
        # composition, one more the first time active_tokens changes, then
        # full generalization via automatic dynamic shapes -- never growing
        # again, including across the reverse-order pass): per-composition
        # amortization, not a recompile storm. Summed across the 3
        # independently-compiled functions sharing one CompileCounter, that
        # is the frame_count <= 6 asserted at the end -- a consequence of
        # having three entry points, not of weaker amortization per rank.
        base_cases = (DEFAULT_CASES[0], DEFAULT_CASES[6], DEFAULT_CASES[12])
        variant_layouts = {
            DEFAULT_CASES[0].name: ((17,), (33,)),
            DEFAULT_CASES[6].name: ((32, 32), (16, 16)),
            DEFAULT_CASES[12].name: ((10, 8, 8), (15, 8, 8)),
        }
        variants = tuple(
            replace(case, layouts=variant_layouts[case.name]) for case in base_cases
        )
        cases = base_cases + variants

        torch.use_deterministic_algorithms(True)
        torch.manual_seed(717)

        # One compiled function per RANK, not per case: na{1,2,3}d_varlen are
        # three distinct top-level entry points, and kernel_size/stride/
        # dilation/is_causal are per-call arguments rather than properties of
        # the layout -- but a case's variant only ever changes `layouts`
        # (rank, kernel_size, etc. are shared with its base case, by
        # construction above), so each rank's two cases (base, variant) can
        # share ONE compiled closure taking only (q, k, v, layout).
        by_rank: dict = {}
        for case in cases:
            active_tokens = sum(_prod(doc_layout) for doc_layout in case.layouts)
            query = torch.randn(
                active_tokens,
                case.heads,
                case.head_dim,
                device="cuda",
                dtype=case.dtype,
            )
            key = torch.randn(
                active_tokens,
                case.heads_kv,
                case.head_dim,
                device="cuda",
                dtype=case.dtype,
            )
            value = torch.randn(
                active_tokens,
                case.heads_kv,
                case.head_dim_v,
                device="cuda",
                dtype=case.dtype,
            )
            layout = _make_layout(case)
            varlen_fn = _VARLEN_FN_BY_RANK[case.rank]
            # Prewarm eagerly: a cold miss under compile costs one extra
            # recompile from the memo-membership guard flip, which would
            # inflate the frame_count this test asserts a bound on below.
            # Prewarming keeps that count isolated to the amortization
            # pattern under study.
            varlen_fn(
                query,
                key,
                value,
                layout,
                kernel_size=case.kernel_size,
                stride=case.stride,
                dilation=case.dilation,
                is_causal=case.is_causal,
            )
            by_rank.setdefault(case.rank, []).append((query, key, value, layout, case))

        def make_fn(varlen_fn, kernel_size, stride, dilation, is_causal):
            def fn(q, k, v, layout):
                return varlen_fn(
                    q,
                    k,
                    v,
                    layout,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=dilation,
                    is_causal=is_causal,
                )

            return fn

        counter = CompileCounter()
        torch.compiler.reset()
        try:
            with torch._dynamo.config.patch(
                cache_size_limit=50,
                accumulated_recompile_limit=50,
                fail_on_recompile_limit_hit=True,
            ):
                compiled_by_rank = {}
                for rank, entries in by_rank.items():
                    base_case = entries[0][4]
                    compiled_by_rank[rank] = torch.compile(
                        make_fn(
                            _VARLEN_FN_BY_RANK[rank],
                            base_case.kernel_size,
                            base_case.stride,
                            base_case.dilation,
                            base_case.is_causal,
                        ),
                        backend=counter,
                        fullgraph=True,
                    )
                for entries in by_rank.values():
                    for query, key, value, layout, case in entries:
                        compiled_by_rank[case.rank](query, key, value, layout)
                for entries in by_rank.values():
                    for query, key, value, layout, case in reversed(entries):
                        compiled_by_rank[case.rank](query, key, value, layout)
        finally:
            torch.compiler.reset()

        self.assertLessEqual(counter.frame_count, 6)

    @skip_if_not_running_extended_tests()
    @skip_if_libnatten_is_not_supported()
    def test_extended_mandatory_inductor_rank3_mqa_forward_and_backward(self):
        case = VarlenCase(
            "inductor-r3-mqa-vdim",
            ((3, 4, 5), (4, 3, 5)),
            (3, 3, 3),
            (1, 1, 1),
            (1, 1, 1),
            (False, True, False),
            torch.float16,
            4,
            1,
            16,
            24,
        )
        torch.use_deterministic_algorithms(True)
        layout = _make_layout(case)
        total = sum(_prod(doc_layout) for doc_layout in case.layouts)

        def fn(q, k, v):
            return natten.na3d_varlen(
                q,
                k,
                v,
                layout,
                kernel_size=case.kernel_size,
                stride=case.stride,
                dilation=case.dilation,
                is_causal=case.is_causal,
            )

        torch.manual_seed(832)
        inputs = (
            torch.randn(
                total,
                case.heads,
                case.head_dim,
                device="cuda",
                dtype=case.dtype,
            ),
            torch.randn(
                total,
                case.heads_kv,
                case.head_dim,
                device="cuda",
                dtype=case.dtype,
            ),
            torch.randn(
                total,
                case.heads_kv,
                case.head_dim_v,
                device="cuda",
                dtype=case.dtype,
            ),
        )
        gradient = torch.randn(
            total,
            case.heads,
            case.head_dim_v,
            device="cuda",
            dtype=case.dtype,
        )

        def run(callable_fn):
            run_inputs = tuple(
                tensor.detach().clone().requires_grad_(True) for tensor in inputs
            )
            output = callable_fn(*run_inputs)
            output.backward(gradient)
            return (output.detach(),) + tuple(
                tensor.grad.detach() for tensor in run_inputs
            )

        expected = run(fn)
        compiled = torch.compile(fn, fullgraph=True)
        observed = run(compiled)
        for expected_tensor, observed_tensor in zip(expected, observed):
            torch.testing.assert_close(observed_tensor, expected_tensor, atol=0, rtol=0)

    @skip_if_not_running_extended_tests()
    @skip_if_libnatten_is_not_supported()
    def test_extended_document_count_and_shape_interpolation(self):
        layouts = tuple((5 + 4 * (index % 8),) for index in range(4096))
        case = VarlenCase(
            "many-documents",
            layouts,
            (5,),
            (1,),
            (1,),
            (False,),
            torch.float16,
            1,
            1,
            8,
            8,
        )
        self._run_case(case)

    def _run_head_dimension_sentinel_oracle(
        self, head_dim: int, head_dim_v: int
    ) -> None:
        torch.use_deterministic_algorithms(True)
        active_tokens = 7
        token_signal = torch.linspace(
            -1, 1, active_tokens, device="cuda", dtype=torch.float16
        )
        query = torch.zeros(
            active_tokens, 1, head_dim, device="cuda", dtype=torch.float16
        )
        key = torch.zeros_like(query)
        value = torch.zeros(
            active_tokens, 1, head_dim_v, device="cuda", dtype=torch.float16
        )
        if head_dim == 65_536:
            query[:, :, -1] = 1
            key[:, :, -1] = token_signal[:, None] * 4
            feature_scale = (
                torch.arange(1, head_dim_v + 1, device="cuda", dtype=torch.float16)
                / head_dim_v
            )
            value[:, 0] = token_signal[:, None] * feature_scale
        else:
            query[:, :, 0] = 1
            key[:, :, 0] = token_signal[:, None] * 4
            value[:, :, -1] = token_signal[:, None]
        query.requires_grad_(True)
        key.requires_grad_(True)
        value.requires_grad_(True)
        query_ref = query.detach().clone().requires_grad_(True)
        key_ref = key.detach().clone().requires_grad_(True)
        value_ref = value.detach().clone().requires_grad_(True)
        layout = natten.VarlenLayout(((active_tokens,),), device="cuda")
        output, logsumexp = natten.na1d_varlen(
            query, key, value, layout, kernel_size=3, scale=1.0, return_lse=True
        )
        output_ref, logsumexp_ref = _explicit_oracle(
            query_ref,
            key_ref,
            value_ref,
            (active_tokens,),
            (3,),
            (1,),
            (1,),
            (False,),
            scale=1.0,
        )
        gradient = torch.zeros_like(output)
        if head_dim == 65_536:
            gradient[:] = 1
        else:
            gradient[:, :, -1] = 1
        output.backward(gradient)
        output_ref.backward(gradient.float())
        query_grad = cast(Tensor, query.grad)
        key_grad = cast(Tensor, key.grad)
        value_grad = cast(Tensor, value.grad)
        query_ref_grad = cast(Tensor, query_ref.grad)
        key_ref_grad = cast(Tensor, key_ref.grad)
        value_ref_grad = cast(Tensor, value_ref.grad)

        torch.testing.assert_close(output.float(), output_ref, atol=2e-3, rtol=2e-3)
        torch.testing.assert_close(logsumexp, logsumexp_ref, atol=2e-3, rtol=2e-3)
        for observed, expected in (
            (query_grad, query_ref_grad),
            (key_grad, key_ref_grad),
            (value_grad, value_ref_grad),
        ):
            torch.testing.assert_close(observed, expected, atol=2e-2, rtol=2e-2)
        if head_dim == 65_536:
            zero_signal_output, _ = _explicit_oracle(
                torch.zeros_like(query_ref),
                torch.zeros_like(key_ref),
                value_ref.detach(),
                (active_tokens,),
                (3,),
                (1,),
                (1,),
                (False,),
                scale=1.0,
            )
            self.assertGreater(
                (output_ref - zero_signal_output).abs().max().item(), 0.1
            )
            self.assertGreater(query_ref_grad[..., -1].abs().max().item(), 0.1)
            self.assertGreater(key_ref_grad[..., -1].abs().max().item(), 0.1)
        else:
            self.assertGreater(output_ref[..., -1].abs().max().item(), 0.1)
            self.assertGreater(value_ref_grad[..., -1].abs().max().item(), 0.1)

    @skip_if_not_running_extended_tests()
    @skip_if_libnatten_is_not_supported()
    def test_extended_head_dimension_limits(self):
        cases = (
            VarlenCase(
                "qk-head-dim-limit",
                ((7,), (11,)),
                (3,),
                (1,),
                (1,),
                (False,),
                torch.float16,
                1,
                1,
                65_536,
                8,
            ),
            VarlenCase(
                "value-head-dim-limit",
                ((7,), (11,)),
                (3,),
                (1,),
                (1,),
                (False,),
                torch.float16,
                1,
                1,
                8,
                65_536,
            ),
        )
        for case in cases:
            with self.subTest(case=case.name):
                self._run_case(case)
                self._run_head_dimension_sentinel_oracle(case.head_dim, case.head_dim_v)

    @skip_if_not_running_extended_tests()
    @skip_if_libnatten_is_not_supported()
    def test_extended_upstream_production_layout_interpolation(self):
        cases = (
            VarlenCase(
                "workload-1d-32k-w2k",
                ((2048,), (8192,), (32768,)),
                (2048,),
                (256,),
                (1,),
                (False,),
                torch.float16,
                1,
                1,
                64,
                64,
                False,
            ),
            VarlenCase(
                "workload-2d-flux-like",
                ((80, 80), (144, 144), (256, 256)),
                (80, 80),
                (16, 16),
                (1, 1),
                (False, False),
                torch.float16,
                1,
                1,
                64,
                64,
                False,
            ),
            VarlenCase(
                "workload-3d-hunyuan-like",
                ((18, 24, 24), (22, 32, 48), (30, 48, 80)),
                (18, 24, 24),
                (2, 8, 8),
                (1, 1, 1),
                (False, False, False),
                torch.float16,
                1,
                1,
                32,
                32,
                False,
            ),
            VarlenCase(
                "one-large-many-tiny",
                ((32768,),) + ((5,),) * 1024,
                (5,),
                (1,),
                (1,),
                (False,),
                torch.float16,
                1,
                1,
                8,
                8,
            ),
        )
        for case in cases:
            with self.subTest(case=case.name):
                self._run_case(case)

    @skip_if_not_running_extended_tests()
    @skip_if_libnatten_is_not_supported()
    def test_extended_84064_documents_cross_65536_work_item_boundary(self):
        torch.use_deterministic_algorithms(True)
        documents = 84_064
        boundary_document = 65_535
        boundary_length = 65
        layouts = (
            ((2,),) * boundary_document
            + ((boundary_length,),)
            + ((2,),) * (documents - boundary_document - 1)
        )
        active_tokens = sum(_prod(doc_layout) for doc_layout in layouts)
        layout = natten.VarlenLayout(layouts, device="cuda")
        self.assertEqual(layout.total_tokens, active_tokens)

        torch.manual_seed(406)
        query = torch.randn(active_tokens, 1, 8, device="cuda", requires_grad=True)
        key = torch.randn_like(query, requires_grad=True)
        value = torch.randn(active_tokens, 1, 16, device="cuda", requires_grad=True)
        output, logsumexp = natten.na1d_varlen(
            query,
            key,
            value,
            layout,
            kernel_size=(2,),
            q_tile_shape=(32,),
            kv_tile_shape=(128,),
            backward_q_tile_shape=(64,),
            backward_kv_tile_shape=(64,),
            return_lse=True,
        )

        state = _resolved_state(layout)
        self.assertEqual(len(state.selected_splits), documents)
        self.assertEqual(state.forward_work_count, documents + 2)
        self.assertEqual(
            state.forward_worklist[65_535:65_537].cpu().tolist(),
            [[boundary_document, 0], [boundary_document, 1]],
        )
        self.assertEqual(state.forward_worklist[-1].cpu().tolist(), [documents - 1, 0])
        self.assertEqual(
            state.backward_worklist[65_535:65_537].cpu().tolist(),
            [[boundary_document, 0], [boundary_document + 1, 0]],
        )
        self.assertEqual(state.total_backward_q_tiles, documents + 1)
        self.assertEqual(
            state.backward_q_tile_offsets[65_535:65_538].cpu().tolist(),
            [65_535, 65_537, 65_538],
        )

        def document_offset(document_id: int) -> int:
            return document_id * 2 + (boundary_length - 2) * (
                document_id > boundary_document
            )

        signal_documents = (65_534, boundary_document, 65_536, documents - 1)
        gradient = torch.zeros_like(output)
        for document_id in signal_documents:
            start = document_offset(document_id)
            length = boundary_length if document_id == boundary_document else 2
            gradient[start : start + length] = torch.randn_like(
                output[start : start + length]
            )
        output.backward(gradient)

        for document_id in signal_documents:
            start = document_offset(document_id)
            length = boundary_length if document_id == boundary_document else 2
            end = start + length
            query_ref = query[start:end].detach().clone().requires_grad_(True)
            key_ref = key[start:end].detach().clone().requires_grad_(True)
            value_ref = value[start:end].detach().clone().requires_grad_(True)
            output_ref, logsumexp_ref = _explicit_oracle(
                query_ref,
                key_ref,
                value_ref,
                (length,),
                (2,),
                (1,),
                (1,),
                (False,),
            )
            output_ref.backward(gradient[start:end])
            torch.testing.assert_close(
                output[start:end], output_ref, atol=3e-4, rtol=3e-4
            )
            torch.testing.assert_close(
                logsumexp[start:end], logsumexp_ref, atol=3e-4, rtol=3e-4
            )
            for observed, expected in (
                (query.grad[start:end], query_ref.grad),
                (key.grad[start:end], key_ref.grad),
                (value.grad[start:end], value_ref.grad),
            ):
                torch.testing.assert_close(observed, expected, atol=4e-4, rtol=4e-4)
                self.assertGreater(torch.count_nonzero(observed).item(), 0)
        boundary_end = document_offset(boundary_document) + boundary_length
        self.assertGreater(
            torch.count_nonzero(query.grad[boundary_end - 1 : boundary_end]).item(),
            0,
        )

    @skip_if_not_running_extended_tests()
    @skip_if_libnatten_is_not_supported()
    def test_extended_representative_local_tile_boundaries(self):
        cases = (
            VarlenCase(
                "tiles-r1-q-boundary",
                ((31,), (33,), (129,)),
                (2,),
                (1,),
                (1,),
                (False,),
                torch.float16,
                1,
                1,
                64,
                64,
                q_tile_shape=(32,),
                kv_tile_shape=(128,),
                backward_q_tile_shape=(64,),
                backward_kv_tile_shape=(64,),
            ),
            VarlenCase(
                "tiles-r2-wide-axis",
                ((7, 5), (9, 3), (33, 5)),
                (2, 2),
                (1, 1),
                (1, 1),
                (False, False),
                torch.float16,
                1,
                1,
                64,
                64,
                q_tile_shape=(8, 4),
                kv_tile_shape=(32, 4),
                backward_q_tile_shape=(8, 8),
                backward_kv_tile_shape=(4, 16),
            ),
            VarlenCase(
                "tiles-r2-wide-axis-transposed",
                ((5, 7), (3, 9), (5, 33)),
                (2, 2),
                (1, 1),
                (1, 1),
                (False, False),
                torch.float16,
                1,
                1,
                64,
                64,
                q_tile_shape=(4, 8),
                kv_tile_shape=(4, 32),
                backward_q_tile_shape=(8, 8),
                backward_kv_tile_shape=(16, 4),
            ),
            VarlenCase(
                "tiles-r3-every-axis",
                ((3, 5, 5), (5, 3, 3), (9, 5, 5)),
                (2, 2, 2),
                (1, 1, 1),
                (1, 1, 1),
                (False, False, False),
                torch.float16,
                1,
                1,
                64,
                64,
                q_tile_shape=(4, 4, 4),
                kv_tile_shape=(4, 4, 4),
                backward_q_tile_shape=(4, 4, 4),
                backward_kv_tile_shape=(4, 4, 4),
            ),
        )
        self.assertEqual(cases[1].q_tile_shape, cases[2].q_tile_shape[::-1])
        self.assertEqual(cases[1].kv_tile_shape, cases[2].kv_tile_shape[::-1])
        for case in cases:
            assert case.q_tile_shape is not None
            assert case.kv_tile_shape is not None
            for axis, q_tile in enumerate(case.q_tile_shape):
                extents = tuple(layout[axis] for layout in case.layouts)
                self.assertTrue(any(extent == q_tile - 1 for extent in extents))
                self.assertTrue(any(extent == q_tile + 1 for extent in extents))
                self.assertTrue(any((extent - 1) // q_tile > 0 for extent in extents))
                self.assertTrue(
                    any(extent % case.kv_tile_shape[axis] != 0 for extent in extents)
                )
            with self.subTest(case=case.name):
                self._run_case(case)

    def _run_large_scale_case(
        self,
        layouts,
        heads,
        head_dim,
        dtype=torch.float16,
        kernel_size=7,
        seed=_INT32_BOUNDARY_SEED,
    ):
        # Deterministic fwd+bwd + finiteness + last-3-position oracle spot
        # check for cases whose total element count (tokens * heads *
        # head_dim) exceeds int32. `layouts` is a tuple of single-axis
        # document extents; active_tokens is their sum. The oracle
        # spot-check covers the packed sequence's last 3 positions -- the
        # LAST document's tail -- so with len(layouts) > 1 it exercises a
        # nonzero token_start: the context slice's right edge is the true
        # right edge, so the boundary window-clamping in _window_positions
        # produces identical windows for these positions whether or not
        # other documents precede it.
        torch.use_deterministic_algorithms(True)
        active_tokens = sum(doc_layout[0] for doc_layout in layouts)
        torch.manual_seed(seed)
        query = torch.randn(
            active_tokens,
            heads,
            head_dim,
            device="cuda",
            dtype=dtype,
            requires_grad=True,
        )
        key = torch.randn(
            active_tokens,
            heads,
            head_dim,
            device="cuda",
            dtype=dtype,
            requires_grad=True,
        )
        value = torch.randn(
            active_tokens,
            heads,
            head_dim,
            device="cuda",
            dtype=dtype,
            requires_grad=True,
        )
        layout = natten.VarlenLayout(layouts, device="cuda")

        output, logsumexp = natten.na1d_varlen(
            query, key, value, layout, kernel_size=kernel_size, return_lse=True
        )
        self.assertFalse(_resolved_state(layout).uses_kv_parallelism)
        gradient = torch.randn_like(output)
        output.backward(gradient)
        torch.cuda.synchronize()

        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.isfinite(query.grad).all())
        self.assertTrue(torch.isfinite(key.grad).all())
        self.assertTrue(torch.isfinite(value.grad).all())

        context = 2 * kernel_size
        context_start = active_tokens - context
        query_context = query[context_start:active_tokens].detach().clone()
        key_context = key[context_start:active_tokens].detach().clone()
        value_context = value[context_start:active_tokens].detach().clone()
        output_oracle, logsumexp_oracle = _explicit_oracle(
            query_context,
            key_context,
            value_context,
            (context,),
            (kernel_size,),
            (1,),
            (1,),
            (False,),
        )
        atol, rtol = _tolerances(dtype)
        torch.testing.assert_close(
            output[active_tokens - 3 :].float(),
            output_oracle[-3:],
            atol=atol,
            rtol=rtol,
        )
        torch.testing.assert_close(
            logsumexp[active_tokens - 3 :].float(),
            logsumexp_oracle[-3:],
            atol=atol,
            rtol=rtol,
        )

    @skip_if_not_running_extended_tests()
    @skip_if_libnatten_is_not_supported()
    def test_extended_element_count_exact_int32_crossing(self):
        # The exact int32 element-count boundary. tokens=2_097_153,
        # heads=16, head_dim=64 -> 2_147_484_672 elements, 1_025 OVER
        # 2**31-1 (2_147_483_647) -- the smallest crossing at this
        # heads/head_dim. Legal because element count is not fenced; the
        # two counts that ARE fenced stay far under their limits here
        # (2_097_153 tokens; 16*64=1024). Two documents (not one): a single
        # document is a uniform VarlenLayout and runs on the fixed-shape
        # kernels (natten.backends.varlen_fna's uniform-dispatch branch);
        # this test is about the varlen schedule, so splitting off a small
        # second document keeps the exact total-element-count boundary
        # (active_tokens is their sum) while also exercising a nonzero
        # token_start, same as test_extended_multi_document_above_int32.
        self._run_large_scale_case(((2_096_153,), (1_000,)), heads=16, head_dim=64)

    @skip_if_not_running_extended_tests()
    @skip_if_libnatten_is_not_supported()
    def test_extended_last_safe_int32_offset(self):
        # The last SAFE length, one token below the crossing above.
        # tokens=2_097_152, heads=16, head_dim=64: the maximum element
        # index the delta kernel computes, (Q-1)*H*D + (H-1)*D + (D-1) =
        # 2_097_151*1024 + 15*64 + 63, is exactly 2_147_483_647
        # (2**31-1), the largest value int32 holds. Together with
        # test_extended_element_count_exact_int32_crossing (2_097_153,
        # the first unsafe length), this pins the precise boundary of
        # the coordinate-times-stride fix to the token. Q is active_tokens
        # (a packed-tensor-wide, document-boundary-independent flat
        # address), so splitting into two documents -- needed so this
        # uniform-shape-free layout still reaches the varlen schedule this
        # test is about, see test_extended_element_count_exact_int32_
        # crossing -- preserves the exact boundary while it's at it.
        self._run_large_scale_case(((2_096_152,), (1_000,)), heads=16, head_dim=64)

    @skip_if_not_running_extended_tests()
    @skip_if_libnatten_is_not_supported()
    def test_extended_element_count_well_above_int32(self):
        # tokens=4_000_000, heads=16, head_dim=64 -> ~4.1e9 elements, ~1.9x
        # the int32 element-count limit (2**31-1). Two documents (not
        # one): see test_extended_element_count_exact_int32_crossing.
        self._run_large_scale_case(((3_999_000,), (1_000,)), heads=16, head_dim=64)

    @skip_if_not_running_extended_tests()
    @skip_if_libnatten_is_not_supported()
    def test_extended_multi_document_above_int32(self):
        # 2 documents [1_500_000, 800_000] x 16 x 64 -> 2_355_200_000
        # elements, over 2**31-1. A single document never exercises a
        # nonzero token_start at this scale; packing a second, smaller
        # document after a 1.5M-token first document does. The oracle
        # spot-check (last 3 packed positions, see _run_large_scale_case)
        # lands in the SECOND document's tail.
        self._run_large_scale_case(((1_500_000,), (800_000,)), heads=16, head_dim=64)

    @skip_if_not_running_extended_tests()
    @skip_if_libnatten_is_not_supported()
    def test_extended_delta_kernel_many_heads_min_head_dim(self):
        # Infeasible-crossing note: the delta kernel's device loop
        # multiplies idx_q (packed active_tokens) against TWO different
        # products -- get<2>(stride_O) = dim*heads (exercised by the
        # element-count-crossing tests above) and get<2>(stride_sum_OdO) =
        # heads (NOT exercised by those, since they cross via a large
        # dim*heads product at modest heads=16). Truly crossing
        # idx_q*heads > 2**31 at the minimum head_dim=8 needs elements =
        # tokens*heads*8 >= 2**31*8 ~= 1.7e10, i.e. ~34GB/tensor, ~270GB
        # across the 8 live Q/K/V/O/dQ/dK/dV/dO tensors -- infeasible on
        # one GPU (140GB here, and the varlen op is single-device).
        # This case is the largest FEASIBLE approximation
        # instead -- and "feasible" is tighter than the naive 8-tensor
        # estimate above suggests: an earlier tokens=650_000, heads=1024
        # attempt (tokens*heads=6.66e8, ~85GB across the 8 live tensors by
        # that estimate) still hit CUDA OOM on a 140GB GPU, so the
        # per-split backward workspace at heads=1024 costs materially more
        # than the raw QKV+grad tensors. tokens=300_000, heads=512 keeps
        # the same "large heads, minimum head_dim" character an order of
        # magnitude further from the idx_q*heads > 2**31 threshold
        # (tokens*heads=1.54e8) but fits comfortably. The idx_q*heads
        # widening itself is evidenced by code review (identical
        # tuple-slot type, identical cast pattern as the dim*heads slot
        # exercised above) rather than by a direct over-threshold run.
        # Two documents (not one): see
        # test_extended_element_count_exact_int32_crossing.
        self._run_large_scale_case(((299_000,), (1_000,)), heads=512, head_dim=8)

    @skip_if_not_running_extended_tests()
    @skip_if_libnatten_is_not_supported()
    def test_extended_pt_reduction_matches_cuda(self):
        # tokens=2_097_153, heads=16, head_dim=64, fp16 (above the int32
        # element-count limit), non-deterministic (compute_delta_with_torch
        # = backward_use_pt_reduction and not deterministic -- the flag
        # only takes effect with deterministic algorithms off), run twice
        # with backward_use_pt_reduction False (CUDA delta kernel) vs True
        # (PyTorch reduction), same upstream gradient both times.
        #
        # Tolerance note: the two reduction paths disagree on ~1/3 of
        # dQ/dK elements by exactly one fp16 ULP (0.00390625) -- constant
        # across every scale swept (50_000 through this test's 2_097_153
        # tokens), with dV bit-identical throughout. That flat,
        # scale-independent profile is a pre-existing floating-point
        # reduction-order characteristic of backward_use_pt_reduction, not
        # a regression tied to the element-count scale (it reproduces at
        # 50_000 tokens, nowhere near the int32 limit) -- but it is far
        # larger than _dq_tolerances' bit-identical-reference-grade bound
        # (raw fp16 eps, no rtol slack), which is the wrong tool for a
        # cross-algorithm comparison. _tolerances (2e-2) comfortably
        # covers the real noise on all three grads while still catching
        # genuine corruption (NaN/Inf or order-of-magnitude errors, which
        # this is not).
        tokens, heads, head_dim = 2_097_153, 16, 64
        torch.use_deterministic_algorithms(False)
        torch.manual_seed(_INT32_BOUNDARY_SEED)
        base_query = torch.randn(
            tokens, heads, head_dim, device="cuda", dtype=torch.float16
        )
        base_key = torch.randn(
            tokens, heads, head_dim, device="cuda", dtype=torch.float16
        )
        base_value = torch.randn(
            tokens, heads, head_dim, device="cuda", dtype=torch.float16
        )
        shared_gradient = {}

        def run(pt_reduction):
            query = base_query.detach().clone().requires_grad_(True)
            key = base_key.detach().clone().requires_grad_(True)
            value = base_value.detach().clone().requires_grad_(True)
            # Fresh layout per call: backward_use_pt_reduction isn't part of
            # the memo key (it's a per-call dispatch flag, not a build-time
            # one), so reusing one layout across both calls would be fine
            # too -- a fresh one per call just keeps this test's two
            # branches obviously independent. Two documents (not one): a
            # single document is a uniform VarlenLayout and runs on the
            # fixed-shape kernels (natten.backends.varlen_fna's
            # uniform-dispatch branch); this test is about the varlen
            # backward_use_pt_reduction path, so it uses two documents --
            # see test_extended_element_count_exact_int32_crossing.
            layout = natten.VarlenLayout(((tokens - 1_000,), (1_000,)), device="cuda")
            output = natten.na1d_varlen(
                query,
                key,
                value,
                layout,
                kernel_size=7,
                backward_use_pt_reduction=pt_reduction,
            )
            if "gradient" not in shared_gradient:
                shared_gradient["gradient"] = torch.randn_like(output)
            output.backward(shared_gradient["gradient"])
            return query.grad, key.grad, value.grad

        dq_cuda, dk_cuda, dv_cuda = run(False)
        dq_pt, dk_pt, dv_pt = run(True)

        atol, rtol = _tolerances(torch.float16)
        torch.testing.assert_close(dq_cuda, dq_pt, atol=atol, rtol=rtol)
        torch.testing.assert_close(dk_cuda, dk_pt, atol=atol, rtol=rtol)
        torch.testing.assert_close(dv_cuda, dv_pt, atol=atol, rtol=rtol)

    @skip_if_not_running_extended_tests()
    @skip_if_libnatten_is_not_supported()
    def test_extended_backward_split_boundary(self):
        # dataclasses variant of the 2_097_153-token, heads=16, head_dim=64
        # scale (above the int32 element-count limit), non-deterministic
        # with an explicit backward_kv_tile_shape small enough (relative to
        # 2_097_153 tokens) that the default split heuristic parallelizes
        # the KV dimension -- exercising real workspace_elements_* sizing
        # at over-2**31-element scale. Routed through the shared
        # VarlenCase/_run_case harness, which asserts the resolved state's
        # uses_kv_parallelism and checks grads against the fixed-path
        # reference within tolerance for non-deterministic cases. Two
        # documents (not one): see
        # test_extended_element_count_exact_int32_crossing.
        case = VarlenCase(
            name="int32-boundary-over-fp16-kv-split",
            layouts=((2_096_153,), (1_000,)),
            kernel_size=(7,),
            stride=(1,),
            dilation=(1,),
            is_causal=(False,),
            dtype=torch.float16,
            heads=16,
            heads_kv=16,
            head_dim=64,
            head_dim_v=64,
            deterministic=False,
            backward_q_tile_shape=(128,),
            backward_kv_tile_shape=(128,),
        )
        self._run_case(case)

    @skip_if_not_running_extended_tests()
    @skip_if_libnatten_is_not_supported()
    def test_extended_deterministic_above_int32(self):
        # The same dataclasses variant of the 2_097_153-token, heads=16,
        # head_dim=64 scale, deterministic=True -- the single-split
        # workspace path, over 2**31 elements. Routed through the shared
        # VarlenCase/_run_case harness (bit-identical check against the
        # fixed-path reference for deterministic cases). Two documents
        # (not one): see test_extended_element_count_exact_int32_crossing.
        case = VarlenCase(
            name="int32-boundary-over-fp16-deterministic",
            layouts=((2_096_153,), (1_000,)),
            kernel_size=(7,),
            stride=(1,),
            dilation=(1,),
            is_causal=(False,),
            dtype=torch.float16,
            heads=16,
            heads_kv=16,
            head_dim=64,
            head_dim_v=64,
            deterministic=True,
        )
        self._run_case(case)

    @skip_if_not_running_extended_tests()
    @skip_if_libnatten_is_not_supported()
    def test_extended_many_tiny_docs_leak_loop(self):
        # Leak surface: build/teardown of the resolved-state tensors each
        # iteration, under PYTORCH_NO_CUDA_MEMORY_CACHING=1 (forced by the
        # Makefile/run_tests_parallel.sh), so every alloc/free is an uncached
        # cudaMalloc/Free round trip and the allocator's own pooling can't hide
        # a real leak. A fresh VarlenLayout every iteration (as below) starts
        # with an empty memo, so each call is a guaranteed cache miss -- the
        # build/teardown intent transfers directly, with no separate bypass
        # needed (unlike a hypothetical shared/global cache, a per-object memo
        # on a discarded object is naturally reclaimed with it).
        torch.use_deterministic_algorithms(True)
        layouts = tuple((5 + 4 * (i % 5),) for i in range(64))
        active_tokens = sum(_prod(doc_layout) for doc_layout in layouts)

        torch.manual_seed(864)
        base_inputs = tuple(
            torch.randn(active_tokens, 1, 8, device="cuda", dtype=torch.float16)
            for _ in range(3)
        )
        gradient = torch.randn(active_tokens, 1, 8, device="cuda", dtype=torch.float16)
        control_layout = natten.VarlenLayout(layouts, device="cuda")

        def run_probe():
            query, key, value = (
                tensor.detach().clone().requires_grad_(True) for tensor in base_inputs
            )
            output, logsumexp = natten.na1d_varlen(
                query, key, value, control_layout, kernel_size=5, return_lse=True
            )
            output.backward(gradient)
            return (output, logsumexp, query.grad, key.grad, value.grad)

        before = run_probe()

        torch.cuda.reset_peak_memory_stats()
        checkpoints = []
        iterations = 600
        for i in range(iterations):
            layout = natten.VarlenLayout(layouts, device="cuda")
            query, key, value = (
                tensor.detach().clone().requires_grad_(True) for tensor in base_inputs
            )
            output = natten.na1d_varlen(query, key, value, layout, kernel_size=5)
            output.backward(gradient)
            if (i + 1) % 50 == 0:
                checkpoints.append(torch.cuda.max_memory_allocated())

        after = run_probe()

        self.assertTrue(
            all(
                torch.equal(expected, observed)
                for expected, observed in zip(before, after)
            )
        )
        self.assertGreaterEqual(len(checkpoints), 6)
        mid = len(checkpoints) // 2
        mid_window = checkpoints[mid - 1 : mid + 2]
        self.assertLessEqual(max(checkpoints[-3:]), max(mid_window) * 1.05)


if __name__ == "__main__":
    unittest.main()
