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
"""kernel_size = 1 axes of varlen NA: such an axis mixes nothing (each query
attends only to tokens sharing its coordinate on that axis); is_causal and
dilation have no effect there. Lowered away in Python
(natten.backends.varlen_lowering) before reaching a CUDA kernel -- folded
(a leading run, zero-copy view) or permuted (the rest, gather in/scatter
out) -- so the kernel only ever sees kernel_size >= 2. See docs/backends.md
and CHANGELOG.md for the public description.

Primary oracle: tests/utils.py's `_explicit_oracle`, a from-scratch
per-token coordinate/softmax computation, called once per document. It
already treats kernel_size = 1 correctly with NO special-casing on this
file's part: for any axis, `axis_neighbors` degenerates to exactly the
query's own single position when kernel_size == 1 (verified by hand for
both causal and non-causal, independent of stride/dilation, since a
window of width 1 has only one possible position). This makes it a fully
independent check of the semantics -- it does not share any fold/permute/
identity code path with the implementation under test -- for any pattern
of degenerate axes, so (a)/(b)/(c)'s different degenerate-axis positions
(leading-only, non-leading-only, both) are exercised as one case list
through one shared runner rather than three separately hand-built oracles.
"""

import pickle
import unittest
from typing import Any, Callable, Dict, List, Tuple
from unittest import mock

import natten
import torch
from natten._libnatten import na1d_forward, varlen_na1d_forward
from natten.backends import cutlass_fna_generic
from natten.backends.varlen_lowering import maybe_lower_degenerate_axes
from natten.types import DimensionType
from natten.utils.testing import (
    skip_if_fewer_than_n_gpus,
    skip_if_libnatten_is_not_supported,
)
from torch._dynamo.testing import CompileCounter

from .utils import (
    _dtype_is_supported,
    _explicit_oracle,
    _make_layout,
    _prod,
    _set_deterministic,
    _tolerances,
    VarlenCase,
)
from .varlen_numerics import (
    dense_reference,
    document_comparison_rule,
    document_mask,
    dtype_interval,
    effective_kernel,
    grouped_head_prediction,
    rank_one_match,
    rank_one_scalar,
    reference_interval,
    single_key_scalar_bound,
)

_VARLEN_FN_BY_RANK: Dict[int, Callable[..., Any]] = {
    1: natten.na1d_varlen,
    2: natten.na2d_varlen,
    3: natten.na3d_varlen,
}

_FP32_FWD_ATOL = 3e-4
_FP32_GRAD_ATOL = 4e-4


def _lower_with_mock_dispatch(layout, kernel_size: DimensionType, tensors):
    """``maybe_lower_degenerate_axes`` on one pack, with the residual call it would
    make replaced by a mock.

    The mock is what makes the route observable from outside the layout: the
    identity path and every pack the lowering declines leave it uncalled, and any
    other route hands it the derived layout, geometry and tensors it built. Returns
    the pair to assert on.
    """
    query, key, value = tensors
    rank = layout.rank
    dispatch = mock.Mock(return_value=torch.zeros_like(value))
    result = maybe_lower_degenerate_axes(
        na_dim=rank,
        query=query,
        key=key,
        value=value,
        layout=layout,
        kernel_size=kernel_size,
        stride=(1,) * rank,
        dilation=(1,) * rank,
        is_causal=(False,) * rank,
        scale=None,
        backend="cutlass-fna",
        q_tile_shape=None,
        kv_tile_shape=None,
        backward_q_tile_shape=None,
        backward_kv_tile_shape=None,
        backward_kv_splits=None,
        backward_use_pt_reduction=False,
        return_lse=False,
        dispatch=dispatch,
    )
    return result, dispatch


def _assert_check(test, check: Dict[str, Any], message: str) -> None:
    """Assert one tests/varlen_numerics.py comparison record."""
    test.assertTrue(check["pass"], f"{message}: {check}")


def _assert_single_key_rows(
    test, rows: List[int], query, key, value, upstream, dq, dk, dtype, scale: float
) -> None:
    """Gate for a query whose only visible key is itself, computed in the
    kernel rather than on the Python identity path.

    Its exact gradients are zero, and what the kernel returns instead is the
    rank-1 image of one scalar per head: the residual of the same head_dim-term
    dot product taken over the backward GEMM's and the delta reduction's
    different summation orders, bounded by ``single_key_scalar_bound``.
    """
    heads = query.shape[-2]
    heads_kv = key.shape[-2]
    repeats = heads // heads_kv
    for row in rows:
        coefficients, bounds = [], []
        for head in range(heads):
            kv_head = head // repeats
            bound = single_key_scalar_bound(
                upstream[row, head], value[row, kv_head], dtype
            )
            _, coefficient = rank_one_scalar(dq[row, head], key[row, kv_head], dtype)
            label = f"row {row} head {head}"
            test.assertLessEqual(abs(coefficient / scale), bound, f"dQ {label} scalar")
            _assert_check(
                test,
                rank_one_match(dq[row, head], key[row, kv_head], coefficient, dtype),
                f"dQ {label}",
            )
            coefficients.append(coefficient)
            bounds.append(bound)
        if heads == heads_kv:
            for head in range(heads):
                _, coefficient = rank_one_scalar(dk[row, head], query[row, head], dtype)
                label = f"row {row} head {head}"
                test.assertLessEqual(
                    abs(coefficient / scale), bounds[head], f"dK {label} scalar"
                )
                _assert_check(
                    test,
                    rank_one_match(dk[row, head], query[row, head], coefficient, dtype),
                    f"dK {label}",
                )
            continue
        predicted, radius = grouped_head_prediction(
            coefficients, query[row], repeats, dtype
        )
        difference = (dk[row].detach().cpu().double() - predicted).abs()
        test.assertTrue(bool((difference <= radius).all()), f"dK row {row}")


def _run_oracle_case(case: VarlenCase) -> None:
    """Shared runner for (a)/(b)/(c): forward, lse, and dq/dk/dv against
    _explicit_oracle, called once per document (concatenated in layout
    order, exactly the packed convention). Every VarlenCase below has at
    least one kernel_size axis == 1; _explicit_oracle needs no special
    handling for that (see module docstring).
    """
    if not _dtype_is_supported(case.dtype):
        raise unittest.SkipTest(f"{case.dtype} is unavailable on this device")
    previous = _set_deterministic(case.deterministic)
    try:
        torch.manual_seed(7100 + case.rank)
        total = sum(_prod(layout) for layout in case.layouts)
        query = torch.randn(
            total, case.heads, case.head_dim, device="cuda", dtype=case.dtype
        )
        key = torch.randn(
            total, case.heads_kv, case.head_dim, device="cuda", dtype=case.dtype
        )
        value = torch.randn(
            total, case.heads_kv, case.head_dim_v, device="cuda", dtype=case.dtype
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
            return_lse=True,
        )

        output_refs = []
        logsumexp_refs = []
        start = 0
        for doc_layout in case.layouts:
            end = start + _prod(doc_layout)
            doc_output, doc_lse = _explicit_oracle(
                query_ref[start:end],
                key_ref[start:end],
                value_ref[start:end],
                doc_layout,
                case.kernel_size,
                case.stride,
                case.dilation,
                case.is_causal,
                scale=case.scale,
            )
            output_refs.append(doc_output)
            logsumexp_refs.append(doc_lse)
            start = end
        output_ref = torch.cat(output_refs, dim=0)
        logsumexp_ref = torch.cat(logsumexp_refs, dim=0)

        if case.dtype == torch.float32:
            fwd_atol, grad_atol = _FP32_FWD_ATOL, _FP32_GRAD_ATOL
        else:
            fwd_atol, _ = _tolerances(case.dtype)
            # 2x _tolerances' own bound for gradients specifically: this
            # oracle shares NO code with the implementation (see module
            # docstring), and GQA's value gradient (torch.repeat_interleave's
            # backward) adds an extra reduction step beyond what dq/dk go
            # through -- test_fna_varlen.py's own _explicit_oracle usage
            # notes the same class of from-scratch-oracle bf16 slack for
            # dQ; observed here up to ~0.06 against a 0.04 base bound on a
            # GQA+dilated case, comfortably inside 2x while still well
            # short of vacuous.
            grad_atol = 2 * fwd_atol

        torch.testing.assert_close(
            output.float(),
            output_ref,
            atol=fwd_atol,
            rtol=0,
            msg=f"{case.name}: output",
        )
        torch.testing.assert_close(
            logsumexp.float(),
            logsumexp_ref,
            atol=fwd_atol,
            rtol=0,
            msg=f"{case.name}: lse",
        )

        gradient = torch.randn_like(output)
        output.backward(gradient)
        output_ref.backward(gradient.float())
        for observed, expected, name in (
            (query.grad, query_ref.grad, "dq"),
            (key.grad, key_ref.grad, "dk"),
            (value.grad, value_ref.grad, "dv"),
        ):
            torch.testing.assert_close(
                observed.float(),
                expected.float(),
                atol=grad_atol,
                rtol=0,
                msg=f"{case.name}: {name} mismatch",
            )
    finally:
        torch.use_deterministic_algorithms(previous)


# (a) leading fold only: kernel (1, K, K) / (1, K). (b) permute only:
# kernel (K, 1, 1), causal on the kept (T) axis, one dilation=2 case. (c)
# fold + permute: kernel (1, K, 1). Each mixes document extents (some equal
# to, some above the kernel on the kept axis) so the existing per-document
# effective-kernel clamp is exercised alongside lowering, not separately.
DEGENERATE_ORACLE_CASES: Tuple[VarlenCase, ...] = (
    VarlenCase(
        "a-r3-fold-fp32-mixed-T",
        ((1, 5, 5), (3, 5, 5), (4, 6, 6), (1, 7, 4)),
        (1, 3, 3),
        (1, 1, 1),
        (1, 1, 1),
        (False, False, False),
        torch.float32,
        2,
        2,
        16,
        16,
    ),
    VarlenCase(
        "a-r3-fold-bf16-causal-hw",
        ((1, 5, 5), (3, 5, 5), (2, 6, 6)),
        (1, 3, 3),
        (1, 1, 1),
        (1, 1, 1),
        (False, True, False),
        torch.bfloat16,
        3,
        3,
        32,
        32,
    ),
    VarlenCase(
        "a-r2-fold-fp32-per-row",
        ((1, 9), (3, 9), (5, 6)),
        (1, 4),
        (1, 1),
        (1, 1),
        (False, True),
        torch.float32,
        2,
        2,
        16,
        24,
    ),
    VarlenCase(
        "b-r3-permute-fp32-causal-T",
        ((6, 4, 4), (9, 4, 4), (4, 4, 4)),
        (4, 1, 1),
        (1, 1, 1),
        (1, 1, 1),
        (True, False, False),
        torch.float32,
        2,
        2,
        16,
        16,
    ),
    VarlenCase(
        "b-r3-permute-bf16-causal-dilated-T",
        ((10, 4, 4), (9, 5, 5)),
        (3, 1, 1),
        (1, 1, 1),
        (2, 1, 1),
        (True, False, False),
        torch.bfloat16,
        4,
        2,
        32,
        32,
    ),
    VarlenCase(
        "b-r3-permute-fp32-deterministic",
        ((6, 4, 4), (7, 3, 3)),
        (4, 1, 1),
        (1, 1, 1),
        (1, 1, 1),
        (True, False, False),
        torch.float32,
        2,
        2,
        16,
        16,
        True,
    ),
    VarlenCase(
        "c-r3-fold-permute-fp32-mixed",
        ((1, 5, 6), (3, 5, 6), (2, 7, 4)),
        (1, 3, 1),
        (1, 1, 1),
        (1, 1, 1),
        (False, False, False),
        torch.float32,
        2,
        2,
        16,
        16,
    ),
    VarlenCase(
        "c-r3-fold-permute-bf16-causal-H",
        ((1, 5, 6), (4, 6, 4)),
        (1, 3, 1),
        (1, 1, 1),
        (1, 1, 1),
        (False, True, False),
        torch.bfloat16,
        3,
        3,
        32,
        32,
    ),
)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
class VarlenDegenerateAxesOracleTests(unittest.TestCase):
    def tearDown(self):
        torch.use_deterministic_algorithms(False)

    @skip_if_libnatten_is_not_supported()
    def test_fold_and_permute_against_explicit_oracle(self):
        for case in DEGENERATE_ORACLE_CASES:
            with self.subTest(case=case.name):
                _run_oracle_case(case)

    @skip_if_libnatten_is_not_supported()
    def test_heterogeneous_pack_image_column_is_a_single_key_row(self):
        # (b)'s heterogeneous-pack sub-case: kernel (K, 1, 1) permutes H, W
        # to the front, turning each document into H*W length-T 1-D
        # documents. Every column of the image document (T = 1) then holds a
        # single token, so the kernel computes it alongside the videos as a
        # single-key row: output is V, dV is the upstream gradient, and dQ/dK
        # are the rank-1 residual _assert_single_key_rows describes.
        torch.manual_seed(7200)
        dtype = torch.float32
        heads, head_dim, head_dim_v = 2, 16, 16
        kernel_size, dilation, is_causal = (4, 1, 1), (1, 1, 1), (True, False, False)
        video_shape_a = (6, 4, 4)
        video_shape_b = (5, 4, 4)
        image_shape = (1, 4, 4)
        layouts: Tuple[DimensionType, ...] = (video_shape_a, video_shape_b, image_shape)
        total = sum(_prod(s) for s in layouts)
        image_start = _prod(video_shape_a) + _prod(video_shape_b)
        image_end = image_start + _prod(image_shape)

        layout = natten.VarlenLayout(layouts, device="cuda")
        query = torch.randn(
            total, heads, head_dim, device="cuda", dtype=dtype, requires_grad=True
        )
        key = torch.randn(
            total, heads, head_dim, device="cuda", dtype=dtype, requires_grad=True
        )
        value = torch.randn(
            total, heads, head_dim_v, device="cuda", dtype=dtype, requires_grad=True
        )

        output, lse = natten.na3d_varlen(
            query,
            key,
            value,
            layout,
            kernel_size=kernel_size,
            dilation=dilation,
            is_causal=is_causal,
            return_lse=True,
        )
        scale = head_dim**-0.5
        expected_lse = scale * (
            query[image_start:image_end] * key[image_start:image_end]
        ).sum(-1)
        rows = list(range(image_start, image_end))
        _assert_check(
            self,
            dtype_interval(output[rows], value[rows], dtype),
            "image output is V",
        )
        gradient = torch.randn_like(output)
        dq, dk, dv = torch.autograd.grad(output, (query, key, value), gradient)
        _assert_check(
            self,
            dtype_interval(dv[rows], gradient[rows], dtype),
            "image dV is the upstream gradient",
        )
        _assert_single_key_rows(
            self, rows, query, key, value, gradient, dq, dk, dtype, scale
        )
        torch.testing.assert_close(
            lse[image_start:image_end], expected_lse, atol=_FP32_FWD_ATOL, rtol=0
        )


# Documents narrower than the kernel on some axis, packed next to documents
# that are not. The pack is dispatched in ONE launch and each document's
# window is clamped to its own extent inside the kernel; the same document
# called on its own is a uniform layout, so it clamps in Python instead and
# lands on a lower-rank kernel. The two agree on the mathematics, not bit for
# bit -- documents that clamp are judged against a dense float64 reference,
# documents that do not must still match their isolated call exactly.
@unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
class VarlenMixedPackTests(unittest.TestCase):
    def setUp(self):
        self.previous_deterministic = _set_deterministic(True)

    def tearDown(self):
        torch.use_deterministic_algorithms(self.previous_deterministic)

    @skip_if_libnatten_is_not_supported()
    def test_mixed_pack_inference_cache_and_pickle_support_backward(self):
        torch.manual_seed(1911)
        shapes = ((1, 5), (7, 5), (1, 5))
        layout = natten.VarlenLayout(shapes)
        inputs = tuple(
            torch.randn(45, 2, 32, device="cuda", dtype=torch.bfloat16)[:, :, ::2]
            for _ in range(3)
        )

        def run(lay):
            leaves = tuple(x.detach().requires_grad_(True) for x in inputs)
            out = natten.na2d_varlen(*leaves, lay, kernel_size=(5, 3), stride=(3, 1))
            return (out, *torch.autograd.grad(out, leaves, gradient))

        with torch.inference_mode():
            natten.na2d_varlen(*inputs, layout, kernel_size=(5, 3), stride=(3, 1))
        restored = pickle.loads(pickle.dumps(layout))
        self.assertIsNone(restored.device)
        gradient = torch.randn_like(inputs[0])
        reference = run(natten.VarlenLayout(shapes))
        for lay in (layout, restored):
            for actual, expected in zip(run(lay), reference):
                self.assertTrue(torch.equal(actual, expected))

    @skip_if_libnatten_is_not_supported()
    def test_mixed_pack_preserves_dilated_axis_fit_check(self):
        shapes = ((1, 1), (7, 5), (1, 5))
        layout = natten.VarlenLayout(shapes)
        inputs = tuple(torch.zeros(41, 1, 16, device="cuda") for _ in range(3))
        with self.assertRaisesRegex(ValueError, "kernel_size \\* dilation must fit"):
            natten.na2d_varlen(*inputs, layout, kernel_size=(3, 3), dilation=(2, 1))

    @skip_if_libnatten_is_not_supported()
    def test_mixed_single_token_documents_are_single_key_rows(self):
        for seed in (0, 1907):
            for dtype in (torch.float32, torch.float16, torch.bfloat16):
                for heads, heads_kv in ((1, 1), (4, 2)):
                    with self.subTest(seed=seed, dtype=dtype, heads=heads):
                        if not _dtype_is_supported(dtype):
                            self.skipTest(f"{dtype} is unavailable on this device")
                        torch.manual_seed(seed)
                        # Interleaved singleton and empty documents exercise
                        # restoration of both document order and gradients.
                        shapes = ((1,), (5,), (0,), (1,), (7,))
                        layout = natten.VarlenLayout(shapes)
                        q = torch.randn(
                            14,
                            heads,
                            16,
                            device="cuda",
                            dtype=dtype,
                            requires_grad=True,
                        )
                        k = torch.randn(
                            14,
                            heads_kv,
                            16,
                            device="cuda",
                            dtype=dtype,
                            requires_grad=True,
                        )
                        v = torch.randn(
                            14,
                            heads_kv,
                            24,
                            device="cuda",
                            dtype=dtype,
                            requires_grad=True,
                        )
                        out, lse = natten.na1d_varlen(
                            q,
                            k,
                            v,
                            layout,
                            kernel_size=5,
                            is_causal=True,
                            return_lse=True,
                        )
                        grad = torch.zeros_like(out)
                        grad[0] = torch.randn_like(grad[0])
                        grad[6] = torch.randn_like(grad[6])
                        dq, dk, dv = torch.autograd.grad(
                            out, (q, k, v), grad, retain_graph=True
                        )
                        rows = [0, 6]
                        others = [i for i in range(14) if i not in rows]
                        repeats = heads // heads_kv
                        expected_v = v.repeat_interleave(repeats, dim=1)
                        grouped = grad.reshape(14, heads_kv, repeats, 24)
                        expected_dv = grouped.sum(2)
                        _assert_check(
                            self,
                            dtype_interval(out[rows], expected_v[rows], dtype),
                            "output is V",
                        )
                        _assert_check(
                            self,
                            dtype_interval(
                                dv,
                                expected_dv,
                                dtype,
                                magnitude=grouped.abs().sum(2),
                            ),
                            "dV routes grad_output",
                        )
                        # Every other query's upstream gradient is zero, so
                        # its dQ/dK stay exactly zero whatever this row does.
                        self.assertEqual(int(torch.count_nonzero(dq[others])), 0)
                        self.assertEqual(int(torch.count_nonzero(dk[others])), 0)
                        _assert_single_key_rows(
                            self, rows, q, k, v, grad, dq, dk, dtype, 16**-0.5
                        )
                        # logsumexp is differentiable and its gradient is
                        # ignored, the same contract as the rest of the family.
                        for tensor in torch.autograd.grad(
                            lse, (q, k, v), torch.randn_like(lse)
                        ):
                            self.assertTrue(
                                torch.equal(tensor, torch.zeros_like(tensor))
                            )

    @skip_if_libnatten_is_not_supported()
    def test_mixed_pack_documents_against_dense_reference(self):
        cases = (
            (((1, 9, 11), (7, 9, 11)), (5, 4, 4), (True, False, False), 1, 1),
            (
                ((7, 9, 11), (1, 9, 11), (7, 1, 11), (0, 9, 11), (1, 9, 11)),
                (5, 4, 4),
                (True, False, False),
                1,
                1,
            ),
            (((9, 1), (9, 11), (1, 11)), (4, 4), (False, False), 1, 1),
            (((1, 9, 11), (1, 8, 10)), (5, 4, 4), (True, False, False), 1, 1),
            (((1, 9, 11), (7, 9, 11)), (5, 4, 4), (True, False, False), 4, 2),
        )
        names = ("out", "lse", "dq", "dk", "dv")
        for seed in (0, 1907):
            for dtype in (torch.float32, torch.float16, torch.bfloat16):
                for shapes, kernel, causal, heads, heads_kv in cases:
                    with self.subTest(
                        seed=seed, dtype=dtype, shapes=shapes, heads=heads
                    ):
                        if not _dtype_is_supported(dtype):
                            self.skipTest(f"{dtype} is unavailable on this device")
                        torch.manual_seed(seed)
                        layout = natten.VarlenLayout(shapes)
                        total = layout.total_tokens
                        inputs = tuple(
                            torch.randn(
                                total,
                                head_count,
                                16,
                                device="cuda",
                                dtype=dtype,
                                requires_grad=True,
                            )
                            for head_count in (heads, heads_kv, heads_kv)
                        )
                        fn = _VARLEN_FN_BY_RANK[len(kernel)]
                        out, lse = fn(
                            *inputs,
                            layout,
                            kernel_size=kernel,
                            is_causal=causal,
                            return_lse=True,
                        )
                        grad = torch.randn_like(out)
                        packed = (out, lse, *torch.autograd.grad(out, inputs, grad))
                        ones = (1,) * len(kernel)
                        start = 0
                        for shape in shapes:
                            length = _prod(shape)
                            if not length:
                                continue
                            window = slice(start, start + length)
                            start += length
                            leaves = tuple(
                                x[window].detach().clone().requires_grad_(True)
                                for x in inputs
                            )
                            ref_out, ref_lse = fn(
                                *leaves,
                                natten.VarlenLayout((shape,)),
                                kernel_size=kernel,
                                is_causal=causal,
                                return_lse=True,
                            )
                            isolated = (
                                ref_out,
                                ref_lse,
                                *torch.autograd.grad(ref_out, leaves, grad[window]),
                            )
                            rule = document_comparison_rule(kernel, ones, shapes, shape)
                            if rule == "bitwise":
                                # Same kernel family either way, so the pack
                                # must not perturb this document at all.
                                for name, actual, expected in zip(
                                    names, packed, isolated
                                ):
                                    self.assertTrue(
                                        torch.equal(actual[window], expected),
                                        f"{name} of {shape}",
                                    )
                                continue
                            # The third rule (a document the pack computes as
                            # single-key rows while its isolated call answers
                            # exactly) needs the rank-1 residual gate instead
                            # of a reference interval, and no case here asks
                            # for it -- one that does must bring that gate.
                            self.assertEqual(rule, "reference-interval", str(shape))
                            mask = document_mask(shape, kernel, ones, ones, causal)
                            reference = dense_reference(*leaves, grad[window], mask)
                            for name, actual, expected in zip(names, packed, isolated):
                                _assert_check(
                                    self,
                                    reference_interval(
                                        actual[window], expected, reference[name]
                                    ),
                                    f"{name} of {shape}",
                                )


# (e) Uniform layouts with degenerate axes: the uniform-dispatch branch and
# degenerate-axis lowering compose -- the derived (folded/permuted) layout
# is itself uniform, so it lands back on the fixed-shape kernels, and the
# whole call must equal na2d bit-for-bit (torch.equal, not a tolerance).
@unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
class VarlenDegenerateAxesUniformTests(unittest.TestCase):
    @skip_if_libnatten_is_not_supported()
    def test_uniform_video_pack_matches_fixed_na2d_bit_for_bit(self):
        torch.manual_seed(7300)
        num_docs, shape = 3, (5, 6, 6)
        heads, head_dim = 2, 32
        kernel_size = (1, 3, 3)
        total = num_docs * _prod(shape)

        layout = natten.VarlenLayout((shape,) * num_docs, device="cuda")
        query = torch.randn(total, heads, head_dim, device="cuda", dtype=torch.float16)
        key = torch.randn(total, heads, head_dim, device="cuda", dtype=torch.float16)
        value = torch.randn(total, heads, head_dim, device="cuda", dtype=torch.float16)
        query.requires_grad_(True)
        key.requires_grad_(True)
        value.requires_grad_(True)
        query_ref = query.detach().clone().requires_grad_(True)
        key_ref = key.detach().clone().requires_grad_(True)
        value_ref = value.detach().clone().requires_grad_(True)

        output, lse = natten.na3d_varlen(
            query, key, value, layout, kernel_size=kernel_size, return_lse=True
        )
        # equals na2d on the [n * T, H, W] view: kernel (1, 3, 3) folds T
        # away entirely (all axes after axis 0 kept), so the derived
        # layout's rank-2 view IS that reshape.
        view_shape = (num_docs * shape[0], shape[1], shape[2])
        q_view = query_ref.view(*view_shape, heads, head_dim)
        k_view = key_ref.view(*view_shape, heads, head_dim)
        v_view = value_ref.view(*view_shape, heads, head_dim)
        output_ref, lse_ref = cutlass_fna_generic(
            q_view, k_view, v_view, kernel_size=(3, 3), return_lse=True
        )
        output_ref = output_ref.reshape(total, heads, head_dim)
        lse_ref = lse_ref.reshape(total, heads)

        self.assertTrue(torch.equal(output, output_ref))
        self.assertTrue(torch.equal(lse, lse_ref))
        gradient = torch.randn_like(output)
        output.backward(gradient)
        output_ref.backward(gradient)
        self.assertTrue(torch.equal(query.grad, query_ref.grad))
        self.assertTrue(torch.equal(key.grad, key_ref.grad))
        self.assertTrue(torch.equal(value.grad, value_ref.grad))

    @skip_if_libnatten_is_not_supported()
    def test_uniform_image_pack_clamps_and_matches_fixed_na2d_bit_for_bit(self):
        # Uniform image pack (T=1) under a video kernel (5, K, K): the
        # per-axis clamp reduces it to (1, K, K) before lowering ever sees
        # it, so this still folds and lands on na2d bit-for-bit.
        torch.manual_seed(7301)
        num_docs, shape = 4, (1, 6, 6)
        heads, head_dim = 2, 32
        kernel_size = (5, 3, 3)
        total = num_docs * _prod(shape)

        layout = natten.VarlenLayout((shape,) * num_docs, device="cuda")
        query = torch.randn(total, heads, head_dim, device="cuda", dtype=torch.float16)
        key = torch.randn(total, heads, head_dim, device="cuda", dtype=torch.float16)
        value = torch.randn(total, heads, head_dim, device="cuda", dtype=torch.float16)

        output, lse = natten.na3d_varlen(
            query, key, value, layout, kernel_size=kernel_size, return_lse=True
        )
        view_shape = (num_docs * shape[0], shape[1], shape[2])
        output_ref, lse_ref = cutlass_fna_generic(
            query.view(*view_shape, heads, head_dim),
            key.view(*view_shape, heads, head_dim),
            value.view(*view_shape, heads, head_dim),
            kernel_size=(3, 3),
            return_lse=True,
        )
        self.assertTrue(torch.equal(output, output_ref.reshape(total, heads, head_dim)))
        self.assertTrue(torch.equal(lse, lse_ref.reshape(total, heads)))


# (d) All axes degenerate: the identity fast path.
@unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
class VarlenDegenerateAxesIdentityTests(unittest.TestCase):
    def _run_identity_case(
        self, rank: int, dtype: torch.dtype, heads: int, heads_kv: int
    ) -> None:
        if not _dtype_is_supported(dtype):
            self.skipTest(f"{dtype} is unavailable on this device")
        torch.manual_seed(7400 + rank)
        head_dim, head_dim_v = 16, 24
        layouts: Tuple[DimensionType, ...] = (
            ((3,), (5,))
            if rank == 1
            else ((2, 3), (3, 2))
            if rank == 2
            else ((2, 2, 3), (1, 3, 2))
        )
        kernel_size = (1,) * rank
        total = sum(_prod(s) for s in layouts)
        layout = natten.VarlenLayout(layouts, device="cuda")
        query = torch.randn(total, heads, head_dim, device="cuda", dtype=dtype)
        key = torch.randn(total, heads_kv, head_dim, device="cuda", dtype=dtype)
        value = torch.randn(total, heads_kv, head_dim_v, device="cuda", dtype=dtype)
        query.requires_grad_(True)
        key.requires_grad_(True)
        value.requires_grad_(True)

        varlen_fn = _VARLEN_FN_BY_RANK[rank]
        output, lse = varlen_fn(
            query, key, value, layout, kernel_size=kernel_size, return_lse=True
        )
        # The identity path answers the whole call itself: the residual
        # dispatch the lowering would otherwise make is never reached.
        _, dispatch = _lower_with_mock_dispatch(
            layout, kernel_size, (query, key, value)
        )
        dispatch.assert_not_called()

        repeats = heads // heads_kv
        if heads != heads_kv:
            key_g = torch.repeat_interleave(
                key, repeats=repeats, dim=-2, output_size=heads
            )
            value_g = torch.repeat_interleave(
                value, repeats=repeats, dim=-2, output_size=heads
            )
        else:
            key_g = key
            value_g = value
        scale = head_dim**-0.5
        expected_lse = scale * (query.float() * key_g.float()).sum(-1)

        self.assertTrue(torch.equal(output, value_g))
        torch.testing.assert_close(lse.float(), expected_lse, atol=1e-6, rtol=0)

        gradient = torch.randn_like(output)
        output.backward(gradient)
        # GQA: value.grad accumulates through repeat_interleave's backward
        # (a sum across each kv head's repeated group), not a literal copy
        # of `gradient` -- same reshape-and-sum the interleave itself uses.
        if heads != heads_kv:
            expected_value_grad = gradient.reshape(
                gradient.shape[0], heads_kv, repeats, gradient.shape[-1]
            ).sum(dim=2)
        else:
            expected_value_grad = gradient
        self.assertTrue(torch.equal(value.grad, expected_value_grad))
        self.assertTrue(torch.equal(query.grad, torch.zeros_like(query)))
        self.assertTrue(torch.equal(key.grad, torch.zeros_like(key)))

    @skip_if_libnatten_is_not_supported()
    def test_all_ones_kernel_rank1(self):
        self._run_identity_case(1, torch.float32, heads=2, heads_kv=2)

    @skip_if_libnatten_is_not_supported()
    def test_all_ones_kernel_rank2(self):
        self._run_identity_case(2, torch.float32, heads=4, heads_kv=2)

    @skip_if_libnatten_is_not_supported()
    def test_all_ones_kernel_rank3(self):
        self._run_identity_case(3, torch.bfloat16, heads=4, heads_kv=4)

    @skip_if_libnatten_is_not_supported()
    def test_identity_launches_no_kernel(self):
        torch.manual_seed(7410)
        layout = natten.VarlenLayout(((3,), (4,)), device="cuda")
        query = torch.randn(
            7, 2, 16, device="cuda", dtype=torch.float16, requires_grad=True
        )
        key = torch.randn(
            7, 2, 16, device="cuda", dtype=torch.float16, requires_grad=True
        )
        value = torch.randn(
            7, 2, 16, device="cuda", dtype=torch.float16, requires_grad=True
        )

        with (
            mock.patch(
                "natten.backends.varlen_fna.varlen_na1d_forward",
                side_effect=AssertionError("varlen raw op should not be called"),
            ),
            mock.patch(
                "natten.backends.fna.na1d_forward",
                side_effect=AssertionError("fixed raw op should not be called"),
            ),
        ):
            output = natten.na1d_varlen(query, key, value, layout, kernel_size=1)
            output.sum().backward()
        _, dispatch = _lower_with_mock_dispatch(layout, (1,), (query, key, value))
        dispatch.assert_not_called()
        # Sanity: the mocks above are wired to the same symbols the
        # production code imports (not a stale/unused patch target).
        self.assertIs(varlen_na1d_forward, natten._libnatten.varlen_na1d_forward)
        self.assertIs(na1d_forward, natten._libnatten.na1d_forward)


# (f) Production-shape smoke: a small mixed video/image pack, two kernels.
@unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
class VarlenDegenerateAxesProductionSmokeTests(unittest.TestCase):
    @skip_if_libnatten_is_not_supported()
    def test_production_shape_smoke(self):
        torch.manual_seed(7500)
        dtype = torch.bfloat16
        heads, head_dim = 4, 32
        layouts: Tuple[DimensionType, ...] = (
            (9, 32, 32),
            (9, 32, 32),
            (1, 48, 64),
        )
        total = sum(_prod(s) for s in layouts)

        for kernel_size, dilation, is_causal, name in (
            ((1, 8, 8), (1, 1, 1), (False, False, False), "spatial"),
            ((5, 1, 1), (1, 1, 1), (True, False, False), "temporal-causal"),
        ):
            with self.subTest(name=name):
                if not _dtype_is_supported(dtype):
                    self.skipTest(f"{dtype} is unavailable on this device")
                layout = natten.VarlenLayout(layouts, device="cuda")
                query = torch.randn(
                    total,
                    heads,
                    head_dim,
                    device="cuda",
                    dtype=dtype,
                    requires_grad=True,
                )
                key = torch.randn(
                    total,
                    heads,
                    head_dim,
                    device="cuda",
                    dtype=dtype,
                    requires_grad=True,
                )
                value = torch.randn(
                    total,
                    heads,
                    head_dim,
                    device="cuda",
                    dtype=dtype,
                    requires_grad=True,
                )
                output, lse = _VARLEN_FN_BY_RANK[3](
                    query,
                    key,
                    value,
                    layout,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    is_causal=is_causal,
                    return_lse=True,
                )
                self.assertTrue(torch.isfinite(output).all())
                self.assertTrue(torch.isfinite(lse).all())
                output.sum().backward()
                self.assertTrue(torch.isfinite(query.grad).all())
                self.assertTrue(torch.isfinite(key.grad).all())
                self.assertTrue(torch.isfinite(value.grad).all())


# (g) torch.compile(fullgraph=True): fold, permute, and identity paths.
@unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
class VarlenDegenerateAxesCompileTests(unittest.TestCase):
    def _run_compile_case(
        self,
        rank,
        layouts,
        kernel_size,
        second_layouts,
        second_kernel_size,
        cold_layout=False,
    ) -> None:
        # `call` takes layout/kernel_size as plain arguments (not closure-
        # captured): the SAME compiled wrapper is reused for both
        # geometries below, so the "second geometry" check exercises
        # dynamo's own guard/recompile decision on a real repeat call,
        # rather than compiling two separately-identified functions (which
        # would trivially recompile regardless of geometry).
        def call(q, k, v, lay, ks):
            return _VARLEN_FN_BY_RANK[rank](q, k, v, lay, kernel_size=ks)

        dtype = torch.float16
        heads, head_dim = 2, 16
        layout = natten.VarlenLayout(layouts, device="cuda")
        total = sum(_prod(s) for s in layouts)

        torch.manual_seed(7600)
        inputs = tuple(
            torch.randn(total, heads, head_dim, device="cuda", dtype=dtype)
            for _ in range(3)
        )
        gradient = torch.randn(total, heads, head_dim, device="cuda", dtype=dtype)

        def run(callable_fn, lay, ks):
            run_inputs = tuple(
                tensor.detach().clone().requires_grad_(True) for tensor in inputs
            )
            output = callable_fn(*run_inputs, lay, ks)
            output.backward(gradient)
            return (output.detach(),) + tuple(
                tensor.grad.detach() for tensor in run_inputs
            )

        def run_fwd_bwd_only(callable_fn, lay, ks, call_inputs):
            run_inputs = tuple(
                tensor.detach().clone().requires_grad_(True) for tensor in call_inputs
            )
            callable_fn(*run_inputs, lay, ks).sum().backward()

        torch.use_deterministic_algorithms(True)
        try:
            reference = run(call, layout, kernel_size)
            if cold_layout:
                layout = natten.VarlenLayout(layouts)
            torch.compiler.reset()
            try:
                compiled = torch.compile(call, fullgraph=True)
                # A cold layout may add one graph when its device/memo
                # guards become warm, matching the existing cold-miss
                # contract in test_fna_varlen.py.
                graph_budget = 2 if cold_layout else 1
                with torch._dynamo.config.patch(
                    recompile_limit=graph_budget,
                    accumulated_recompile_limit=graph_budget,
                    fail_on_recompile_limit_hit=True,
                ):
                    observed = run(compiled, layout, kernel_size)
                    observed_again = run(compiled, layout, kernel_size)
                    observed_third = run(compiled, layout, kernel_size)
                for expected, actual in zip(reference, observed):
                    self.assertTrue(torch.equal(expected, actual))
                for expected, actual in zip(reference, observed_again):
                    self.assertTrue(torch.equal(expected, actual))
                for expected, actual in zip(reference, observed_third):
                    self.assertTrue(torch.equal(expected, actual))
            finally:
                torch.compiler.reset()

            # A second, different geometry -- through the SAME compiled
            # callable -- adds at most one recompile.
            second_layout = natten.VarlenLayout(second_layouts, device="cuda")
            second_total = sum(_prod(s) for s in second_layouts)
            second_inputs = tuple(
                torch.randn(second_total, heads, head_dim, device="cuda", dtype=dtype)
                for _ in range(3)
            )

            torch.compiler.reset()
            counter = CompileCounter()
            compiled_counted = torch.compile(call, backend=counter, fullgraph=True)
            run_fwd_bwd_only(compiled_counted, layout, kernel_size, inputs)
            frames_after_first = counter.frame_count
            run_fwd_bwd_only(
                compiled_counted, second_layout, second_kernel_size, second_inputs
            )
            self.assertLessEqual(counter.frame_count - frames_after_first, 1)
            torch.compiler.reset()
        finally:
            torch.use_deterministic_algorithms(False)

    @skip_if_libnatten_is_not_supported()
    def test_compile_mixed_pack_cold_layout(self):
        self._run_compile_case(
            rank=3,
            layouts=((1, 5, 5), (3, 5, 5), (1, 5, 5)),
            kernel_size=(3, 3, 3),
            second_layouts=((3, 6, 6), (1, 6, 6), (1, 5, 5)),
            second_kernel_size=(3, 3, 3),
            cold_layout=True,
        )

    @skip_if_libnatten_is_not_supported()
    def test_compile_mixed_permute_cold_layout(self):
        self._run_compile_case(
            rank=3,
            layouts=((1, 4, 4), (5, 4, 4), (1, 3, 4)),
            kernel_size=(3, 1, 1),
            second_layouts=((5, 4, 4), (1, 4, 4), (1, 3, 4)),
            second_kernel_size=(3, 1, 1),
            cold_layout=True,
        )

    @skip_if_libnatten_is_not_supported()
    def test_compile_fold_path(self):
        self._run_compile_case(
            rank=3,
            layouts=((1, 5, 5), (3, 5, 5)),
            kernel_size=(1, 3, 3),
            second_layouts=((1, 6, 6), (2, 6, 6)),
            second_kernel_size=(1, 3, 3),
        )

    @skip_if_libnatten_is_not_supported()
    def test_compile_permute_path(self):
        self._run_compile_case(
            rank=3,
            layouts=((6, 4, 4), (7, 4, 4)),
            kernel_size=(3, 1, 1),
            second_layouts=((5, 4, 4), (6, 4, 4)),
            second_kernel_size=(3, 1, 1),
        )

    @skip_if_libnatten_is_not_supported()
    def test_compile_identity_path(self):
        self._run_compile_case(
            rank=2,
            layouts=((2, 3), (3, 2)),
            kernel_size=(1, 1),
            second_layouts=((3, 4), (2, 2)),
            second_kernel_size=(1, 1),
        )


# (h) Errors and edge cases.
@unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
class VarlenDegenerateAxesErrorTests(unittest.TestCase):
    @skip_if_libnatten_is_not_supported()
    def test_tile_knobs_with_degenerate_axis_raise(self):
        layout = natten.VarlenLayout(((4, 4), (5, 5)), device="cuda")
        query = torch.zeros(41, 1, 16, device="cuda", dtype=torch.float16)
        common = dict(
            query=query, key=query, value=query, layout=layout, kernel_size=(1, 3)
        )
        for kwargs in (
            {"q_tile_shape": (1, 1), "kv_tile_shape": (1, 1)},
            {"backward_q_tile_shape": (1, 1), "backward_kv_tile_shape": (1, 1)},
            {"backward_kv_splits": (1, 1)},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(ValueError, "not supported together"):
                    natten.na2d_varlen(**common, **kwargs)

    @skip_if_libnatten_is_not_supported()
    def test_kernel_size_zero_or_negative_still_rejected(self):
        layout = natten.VarlenLayout(((4,), (5,)), device="cuda")
        query = torch.zeros(9, 1, 16, device="cuda", dtype=torch.float16)
        for bad in (0, -1, -3):
            with self.subTest(kernel_size=bad):
                with self.assertRaises(ValueError):
                    natten.na1d_varlen(query, query, query, layout, kernel_size=bad)

    def test_pickle_after_lowering_drops_derived_state(self):
        layout = natten.VarlenLayout(((1, 5, 5), (3, 5, 5)))
        folded = layout._folded(1)
        self.assertEqual(len(layout._fold_memo), 1)
        restored = pickle.loads(pickle.dumps(layout))
        self.assertEqual(restored.rank, layout.rank)
        self.assertEqual(restored.shapes, layout.shapes)
        self.assertEqual(restored.total_tokens, layout.total_tokens)
        self.assertEqual(len(restored._fold_memo), 0)
        self.assertEqual(len(restored._permute_memo), 0)
        self.assertIsNot(restored, layout)
        self.assertIsNot(restored, folded)


# (i) Memo / derived-state bookkeeping.
@unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
class VarlenDegenerateAxesBookkeepingTests(unittest.TestCase):
    @skip_if_libnatten_is_not_supported()
    def test_fold_and_permute_state_cached_across_calls(self):
        # Second call with the same geometry builds nothing new: the
        # derived layout and perm/inv tensors are the SAME objects (an
        # actual rebuild would allocate fresh ones), not merely
        # value-equal.
        torch.manual_seed(7700)
        layout = natten.VarlenLayout(((1, 5, 6), (3, 5, 6)), device="cuda")
        kernel_size = (1, 3, 1)
        total = sum(_prod(s) for s in layout.shapes)
        heads, head_dim = 2, 16

        def run():
            q = torch.randn(total, heads, head_dim, device="cuda", dtype=torch.float16)
            natten.na3d_varlen(q, q, q, layout, kernel_size=kernel_size)

        run()
        self.assertEqual(len(layout._fold_memo), 1)
        folded = layout._fold_memo[1]
        self.assertEqual(len(folded._permute_memo), 1)
        perm_key = next(iter(folded._permute_memo))
        _, perm, inv = folded._permute_memo[perm_key]

        run()
        self.assertEqual(len(layout._fold_memo), 1)
        self.assertIs(layout._fold_memo[1], folded)
        self.assertEqual(len(folded._permute_memo), 1)
        _, perm2, inv2 = folded._permute_memo[perm_key]
        self.assertIs(perm2, perm)
        self.assertIs(inv2, inv)

    @skip_if_libnatten_is_not_supported()
    def test_permute_wrong_device_raises_before_building(self):
        layout = natten.VarlenLayout(((6, 4, 4), (7, 4, 4)), device="cuda")
        # layout.device (not a bare "cuda") -- construction pins to the
        # index-qualified device an actually allocated tensor reports (see
        # VarlenLayout._materialize), which a bare "cuda" does not compare
        # equal to.
        assert layout.device is not None
        layout._permuted((1, 2), layout.device)
        with self.assertRaisesRegex(ValueError, "pinned to device"):
            layout._permuted((1, 2), torch.device("cpu"))

    @skip_if_fewer_than_n_gpus(2)
    @skip_if_libnatten_is_not_supported()
    def test_lower_degenerate_axes_fold_path_wrong_device_raises_before_building(self):
        # maybe_lower_degenerate_axes is called directly here, bypassing
        # _neighborhood_attention_varlen_generic's own layout._check_device_pin
        # call, so this exercises the function's own check -- same reason
        # test_permute_wrong_device_raises_before_building calls
        # layout._permuted directly instead of going through the public
        # entry point.
        layout = natten.VarlenLayout(((5, 5, 5), (6, 5, 5)), device="cuda:0")
        total = sum(_prod(s) for s in layout.shapes)
        query = torch.randn(total, 1, 16, device="cuda:1", dtype=torch.float16)
        dispatch = mock.Mock()
        with self.assertRaisesRegex(ValueError, "pinned to device"):
            maybe_lower_degenerate_axes(
                na_dim=3,
                query=query,
                key=query,
                value=query,
                layout=layout,
                kernel_size=(1, 3, 3),
                stride=(1, 1, 1),
                dilation=(1, 1, 1),
                is_causal=(False, False, False),
                scale=None,
                backend="cutlass-fna",
                q_tile_shape=None,
                kv_tile_shape=None,
                backward_q_tile_shape=None,
                backward_kv_tile_shape=None,
                backward_kv_splits=None,
                backward_use_pt_reduction=False,
                return_lse=False,
                dispatch=dispatch,
            )
        dispatch.assert_not_called()
        self.assertEqual(len(layout._fold_memo), 0)

    @skip_if_libnatten_is_not_supported()
    def test_lower_degenerate_axes_identity_path_wrong_device_raises_before_building(
        self,
    ):
        # Same reasoning as the fold-path test above, for the path that
        # never calls _folded or _permuted at all (every axis degenerate):
        # the check still has to run, since nothing else downstream of it
        # would otherwise catch a mismatched device before _identity_output
        # silently computes on query's own device.
        layout = natten.VarlenLayout(((4,), (5,)), device="cuda")
        total = layout.total_tokens
        query = torch.randn(total, 1, 16, device="cpu", dtype=torch.float16)
        dispatch = mock.Mock()
        with self.assertRaisesRegex(ValueError, "pinned to device"):
            maybe_lower_degenerate_axes(
                na_dim=1,
                query=query,
                key=query,
                value=query,
                layout=layout,
                kernel_size=(1,),
                stride=(1,),
                dilation=(1,),
                is_causal=(False,),
                scale=None,
                backend="cutlass-fna",
                q_tile_shape=None,
                kv_tile_shape=None,
                backward_q_tile_shape=None,
                backward_kv_tile_shape=None,
                backward_kv_splits=None,
                backward_use_pt_reduction=False,
                return_lse=False,
                dispatch=dispatch,
            )
        dispatch.assert_not_called()


# (j) Empty documents are inert. A zero-token document has nothing to attend
# over, so uniformity for lowering is judged over the documents that do carry
# tokens (VarlenLayout.uniform_shape): identity, fold and permute lower a pack
# padded with empty documents exactly as they lower the same pack without
# them, and output, LSE and all three gradients come out bitwise identical.
def _empty_document_shapes(rank: int) -> Tuple[DimensionType, ...]:
    """Three zero-token document shapes: a zero on the leading axis, a zero
    on the trailing axis, and every axis zero -- the three ways a derived
    (folded/permuted) layout can meet an empty document, since an axis it
    groups over and an axis it keeps repartition a zero extent differently.
    """
    if rank == 1:
        return ((0,),) * 3
    leading: Any = (0,) + (1,) * (rank - 1)
    trailing: Any = (1,) * (rank - 1) + (0,)
    everywhere: Any = (0,) * rank
    return (leading, trailing, everywhere)


def _empty_insertions(
    shapes: Tuple[DimensionType, ...], rank: int
) -> Tuple[Tuple[str, Tuple[DimensionType, ...]], ...]:
    """``shapes`` with empty documents spliced in, one variant per position."""
    leading, trailing, everywhere = _empty_document_shapes(rank)
    return (
        ("front", (leading,) + shapes),
        ("back", shapes + (trailing,)),
        ("middle", shapes[:1] + (everywhere,) + shapes[1:]),
        (
            "front-middle-back",
            (leading,) + shapes[:1] + (everywhere,) + shapes[1:] + (trailing,),
        ),
    )


# name, layouts, kernel_size, lowering route the clamp resolves to.
_EMPTY_INSERTION_CASES = (
    ("identity-r1", ((1,), (1,)), (3,), "identity"),
    ("identity-r2", ((1, 1), (1, 1)), (3, 3), "identity"),
    ("identity-r3", ((1, 1, 1), (1, 1, 1)), (3, 3, 3), "identity"),
    ("fold-r2", ((1, 6), (1, 6)), (3, 3), "fold"),
    ("fold-r3", ((1, 4, 4), (1, 4, 4)), (3, 3, 3), "fold"),
    ("permute-r2", ((6, 1), (6, 1)), (3, 3), "permute"),
    ("permute-r3", ((4, 1, 4), (4, 1, 4)), (3, 3, 3), "permute"),
    ("fold-then-permute-r3", ((1, 4, 1), (1, 4, 1)), (3, 3, 3), "fold+permute"),
)

_EMPTY_INSERTION_RESULT_NAMES = ("output", "lse", "dq", "dk", "dv")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
class VarlenEmptyDocumentInsertionTests(unittest.TestCase):
    def _call(
        self,
        shapes: Tuple[DimensionType, ...],
        kernel_size: DimensionType,
        rank: int,
        tensors: List[torch.Tensor],
        grad_output: torch.Tensor,
        grad_lse: torch.Tensor,
    ) -> Tuple[Any, Tuple[torch.Tensor, ...]]:
        leaves = [tensor.detach().clone().requires_grad_(True) for tensor in tensors]
        layout = natten.VarlenLayout(shapes, device="cuda")
        output, lse = _VARLEN_FN_BY_RANK[rank](
            *leaves, layout, kernel_size=kernel_size, return_lse=True
        )
        # LSE is differentiable but its upstream gradient is ignored (the FNA
        # contract); passing a non-zero one exercises that both packs ignore
        # it the same way instead of only that neither crashes.
        grads = torch.autograd.grad((output, lse), leaves, (grad_output, grad_lse))
        results = (output.detach(), lse.detach()) + tuple(g.detach() for g in grads)
        return layout, results

    def _assert_route(
        self,
        layout,
        kernel_size: DimensionType,
        route: str,
        tensors: List[torch.Tensor],
    ) -> None:
        """The padded pack must lower the same way the unpadded one does --
        bitwise equality alone would also hold if BOTH packs had lost the
        clamp and gone to the kernel.

        The route is read off one mocked ``dispatch``: the identity path never
        reaches it, a fold hands it the caller's own tokens (a metadata
        repartition, no copy) and a permute hands it gathered ones, each with
        the residual rank and derived shapes its consumed axes leave behind.
        """
        query, _, value = tensors
        rank = layout.rank
        result, dispatch = _lower_with_mock_dispatch(layout, kernel_size, tensors)
        # This class's cases never pass dilation to _call, so it is (1,) *
        # rank throughout -- effective_kernel's dilation branch is inert here.
        effective = effective_kernel(kernel_size, (1,) * rank, layout.uniform_shape)
        keep = tuple(axis for axis, k in enumerate(effective) if k > 1)

        if route == "identity":
            dispatch.assert_not_called()
            self.assertTrue(torch.equal(result, value))
            return

        dispatch.assert_called_once()
        call = dispatch.call_args.kwargs
        derived = call["layout"]
        self.assertEqual(call["na_dim"], len(keep))
        self.assertEqual(derived.rank, len(keep))
        self.assertEqual(derived.total_tokens, layout.total_tokens)
        self.assertEqual(call["kernel_size"], tuple(effective[a] for a in keep))
        # Every derived document that carries tokens is a source document
        # restricted to the axes the lowering kept -- for a fold that is a
        # suffix of its shape, for a permute a reordered selection.
        self.assertEqual(
            {shape for shape in derived.shapes if _prod(shape)},
            {tuple(shape[a] for a in keep) for shape in layout.shapes if _prod(shape)},
        )
        # A fold repartitions metadata only, so the kernel sees the caller's
        # own tensor; a permute gathers the tokens into a new one first.
        self.assertEqual(call["query"] is query, route == "fold")
        # The leading run of degenerate axes is folded away first, and
        # whatever is still degenerate after it is what the permute gathers,
        # so the two steps are named separately in the case list.
        folded = 0
        while folded < rank and effective[folded] == 1:
            folded += 1
        self.assertEqual(folded > 0, route in ("fold", "fold+permute"))
        self.assertEqual(
            rank - folded - len(keep) > 0, route in ("permute", "fold+permute")
        )

    def _run_case(
        self,
        shapes: Tuple[DimensionType, ...],
        kernel_size: DimensionType,
        route: str,
        dtype: torch.dtype,
    ) -> None:
        if not _dtype_is_supported(dtype):
            self.skipTest(f"{dtype} is unavailable on this device")
        rank = len(shapes[0])
        previous = _set_deterministic(True)
        try:
            torch.manual_seed(7600 + rank)
            heads, head_dim = 2, 32
            total = sum(_prod(shape) for shape in shapes)
            tensors = [
                torch.randn(total, heads, head_dim, device="cuda", dtype=dtype)
                for _ in range(3)
            ]
            grad_output = torch.randn(
                total, heads, head_dim, device="cuda", dtype=dtype
            )
            grad_lse = torch.randn(total, heads, device="cuda", dtype=torch.float32)

            layout, expected = self._call(
                shapes, kernel_size, rank, tensors, grad_output, grad_lse
            )
            self.assertTrue(layout.is_uniform)
            self.assertEqual(layout.uniform_shape, shapes[0])
            self._assert_route(layout, kernel_size, route, tensors)

            for placement, padded_shapes in _empty_insertions(shapes, rank):
                with self.subTest(placement=placement):
                    padded_layout, actual = self._call(
                        padded_shapes, kernel_size, rank, tensors, grad_output, grad_lse
                    )
                    self.assertFalse(padded_layout.is_uniform)
                    self.assertEqual(padded_layout.uniform_shape, shapes[0])
                    self.assertEqual(padded_layout.total_tokens, total)
                    self._assert_route(padded_layout, kernel_size, route, tensors)
                    for name, want, got in zip(
                        _EMPTY_INSERTION_RESULT_NAMES, expected, actual
                    ):
                        self.assertTrue(
                            torch.equal(want, got),
                            f"{name} changed when empty documents were inserted "
                            f"({placement}): {padded_shapes} vs {shapes}",
                        )
        finally:
            _set_deterministic(previous)

    @skip_if_libnatten_is_not_supported()
    def test_empty_documents_do_not_change_results_fp16(self):
        for name, shapes, kernel_size, route in _EMPTY_INSERTION_CASES:
            with self.subTest(case=name):
                self._run_case(shapes, kernel_size, route, torch.float16)

    @skip_if_libnatten_is_not_supported()
    def test_empty_documents_do_not_change_results_fp32(self):
        for name, shapes, kernel_size, route in _EMPTY_INSERTION_CASES:
            with self.subTest(case=name):
                self._run_case(shapes, kernel_size, route, torch.float32)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
class VarlenAllEmptyPackLoweringTests(unittest.TestCase):
    @skip_if_libnatten_is_not_supported()
    def test_all_empty_pack_takes_the_fast_path(self):
        # No document carries a token, so there is no shape to clamp against
        # (uniform_shape is None) and the call belongs to the all-empty fast
        # path -- no lowering, no schedule, no device pin.
        for shapes in (((0,), (0,)), ((0, 4), (4, 0)), ((0, 0, 0),)):
            with self.subTest(shapes=shapes):
                rank = len(shapes[0])
                layout = natten.VarlenLayout(shapes)
                self.assertIsNone(layout.uniform_shape)
                query = torch.zeros(
                    0, 2, 32, device="cuda", dtype=torch.float16, requires_grad=True
                )
                key = query.detach().clone().requires_grad_(True)
                value = query.detach().clone().requires_grad_(True)
                output, lse = _VARLEN_FN_BY_RANK[rank](
                    query,
                    key,
                    value,
                    layout,
                    kernel_size=(3,) * rank,
                    return_lse=True,
                )
                self.assertEqual(output.shape, (0, 2, 32))
                self.assertEqual(lse.shape, (0, 2))
                output.sum().backward()
                for tensor in (query, key, value):
                    self.assertEqual(tensor.grad.shape, tensor.shape)
                # Nothing was lowered and nothing was scheduled: the
                # lowering declines the pack (returns None instead of an
                # answer) whether or not the caller's kernel_size is
                # degenerate, and its residual dispatch is never reached.
                # The layout is left unpinned, which is the fast path's own
                # contract -- an all-empty call never materializes it.
                for probe in ((3,) * rank, (1,) * rank):
                    result, dispatch = _lower_with_mock_dispatch(
                        layout, probe, (query, key, value)
                    )
                    self.assertIsNone(result)
                    dispatch.assert_not_called()
                self.assertIsNone(layout.device)


if __name__ == "__main__":
    unittest.main()
