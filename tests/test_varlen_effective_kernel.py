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
"""Variable-length CUTLASS FNA's per-document effective-kernel clamp: a
document narrower than kernel_size on some axis attends over its whole
extent on that axis instead (effective_kernel = min(kernel_size, extent)),
for every axis with dilation == 1. See docs/backends.md and CHANGELOG.md for
the public description.

Oracle: for each document, an axis with extent 1 has exactly one valid
window position (itself) no matter the kernel size, stride, or causal
setting on that axis, so it is dropped and the remaining axes are handed to
the fixed (non-varlen) CUTLASS kernel at the reduced rank with the per-axis
clamped kernel size -- an exact equivalence, not an approximation, since
the fixed kernel already accepts any kernel_size <= extent. A document
collapsed on every axis this way attends only to itself: output is exactly
`v`, logsumexp is exactly `scale * (q * k).sum(-1)`. This is the same
per-document-fixed-call oracle pattern as `_regular_reference` in
tests/utils.py / test_fna_varlen.py, extended with that rank reduction so
it also covers documents the fixed kernel's own kernel_size > 1 gate would
otherwise reject.
"""

import unittest
from typing import Any, Callable, Dict, Optional, Tuple

import natten
import torch
from natten.backends import cutlass_fna_generic
from natten.types import CausalArgType, DimensionType
from natten.utils.testing import skip_if_libnatten_is_not_supported
from torch import Tensor

from .utils import (
    _dtype_is_supported,
    _make_layout,
    _prod,
    _set_deterministic,
    _tolerances,
    VarlenCase,
)

_VARLEN_FN_BY_RANK: Dict[int, Callable[..., Any]] = {
    1: natten.na1d_varlen,
    2: natten.na2d_varlen,
    3: natten.na3d_varlen,
}

# fp32 tolerances: the same "3e-4-class" forward / 4e-4-class gradient bound
# already used for the explicit-oracle comparison in
# test_fna_varlen.py::test_each_rank_and_document_against_explicit_coordinate_oracle.
# bf16 reuses tests/utils.py's _tolerances for both forward and gradients (see
# that test's comment: bf16 dQ against a from-scratch oracle needs the wider
# bound, not _dq_tolerances' bit-identical-reference-grade one). Every
# comparison here is atol-only (rtol=0).
_FP32_FWD_ATOL = 3e-4
_FP32_GRAD_ATOL = 4e-4


def _effective_kernel(
    doc_layout: DimensionType,
    kernel_size: DimensionType,
    dilation: DimensionType,
) -> Tuple[int, ...]:
    return tuple(
        k if d > 1 else min(k, e) for k, e, d in zip(kernel_size, doc_layout, dilation)
    )


def _clamped_document_reference(
    doc_layout: DimensionType,
    kernel_size: DimensionType,
    stride: DimensionType,
    dilation: DimensionType,
    is_causal: CausalArgType,
    scale: Optional[float],
    q_doc: Tensor,
    k_doc: Tensor,
    v_doc: Tensor,
    heads: int,
    heads_kv: int,
    head_dim_v: int,
) -> Tuple[Tensor, Tensor]:
    """fp32 forward+backward reference for one document under the
    effective-kernel clamp; see the module docstring for the method."""
    effective_kernel = _effective_kernel(doc_layout, kernel_size, dilation)
    keep_axes = tuple(i for i, e in enumerate(doc_layout) if e > 1)
    head_dim = q_doc.shape[-1]
    scale_value = scale if scale is not None else head_dim**-0.5

    if not keep_axes:
        # Every axis has extent 1: the only key is the token itself.
        k_expanded = k_doc.float()
        v_expanded = v_doc.float()
        if heads != heads_kv:
            repeats = heads // heads_kv
            k_expanded = k_expanded.repeat_interleave(repeats, dim=1)
            v_expanded = v_expanded.repeat_interleave(repeats, dim=1)
        lse = scale_value * (q_doc.float() * k_expanded).sum(-1)
        return v_expanded, lse

    reduced_kernel = tuple(effective_kernel[i] for i in keep_axes)
    reduced_stride = tuple(stride[i] for i in keep_axes)
    reduced_dilation = tuple(dilation[i] for i in keep_axes)
    reduced_causal = tuple(is_causal[i] for i in keep_axes)

    q_reshaped = q_doc.reshape(1, *doc_layout, heads, head_dim)
    k_reshaped = k_doc.reshape(1, *doc_layout, heads_kv, head_dim)
    v_reshaped = v_doc.reshape(1, *doc_layout, heads_kv, head_dim_v)
    for axis in sorted(range(len(doc_layout)), reverse=True):
        if axis not in keep_axes:
            q_reshaped = q_reshaped.squeeze(axis + 1)
            k_reshaped = k_reshaped.squeeze(axis + 1)
            v_reshaped = v_reshaped.squeeze(axis + 1)

    output, lse = cutlass_fna_generic(
        q_reshaped,
        k_reshaped,
        v_reshaped,
        kernel_size=reduced_kernel,
        stride=reduced_stride,
        dilation=reduced_dilation,
        is_causal=reduced_causal,
        scale=scale,
        return_lse=True,
    )
    return output.reshape(-1, heads, head_dim_v).float(), lse.reshape(-1, heads)


def _case_reference(
    case: VarlenCase, query: Tensor, key: Tensor, value: Tensor
) -> Tuple[Tensor, Tensor]:
    outputs = []
    logsumexps = []
    start = 0
    for doc_layout in case.layouts:
        length = _prod(doc_layout)
        end = start + length
        if length == 0:
            start = end
            continue
        doc_output, doc_lse = _clamped_document_reference(
            doc_layout,
            case.kernel_size,
            case.stride,
            case.dilation,
            case.is_causal,
            case.scale,
            query[start:end],
            key[start:end],
            value[start:end],
            case.heads,
            case.heads_kv,
            case.head_dim_v,
        )
        outputs.append(doc_output)
        logsumexps.append(doc_lse)
        start = end
    return torch.cat(outputs, dim=0), torch.cat(logsumexps, dim=0)


# Each case mixes documents whose extent on one or more axes is below /
# equal to / above kernel_size, so the clamp is exercised alongside its own
# no-op boundary in the same call. ek-r1-mixed-fp32-noncausal-deterministic
# is the deterministic-mode variant of ek-r1-mixed-fp32-noncausal (same
# layout/geometry, forced single-KV-split schedule).
EFFECTIVE_KERNEL_CASES: Tuple[VarlenCase, ...] = (
    VarlenCase(
        "ek-r1-mixed-fp32-noncausal",
        ((1,), (3,), (5,), (9,)),
        (5,),
        (1,),
        (1,),
        (False,),
        torch.float32,
        3,
        3,
        16,
        16,
        False,
    ),
    VarlenCase(
        "ek-r1-mixed-fp32-noncausal-deterministic",
        ((1,), (3,), (5,), (9,)),
        (5,),
        (1,),
        (1,),
        (False,),
        torch.float32,
        3,
        3,
        16,
        16,
        True,
    ),
    VarlenCase(
        "ek-r1-mixed-bf16-causal-gqa",
        ((1,), (3,), (5,), (9,)),
        (5,),
        (1,),
        (1,),
        (True,),
        torch.bfloat16,
        4,
        2,
        32,
        32,
        False,
    ),
    VarlenCase(
        "ek-r2-mixed-fp32-noncausal",
        ((1, 7), (3, 3), (5, 9), (2, 2)),
        (5, 3),
        (1, 1),
        (1, 1),
        (False, False),
        torch.float32,
        2,
        2,
        16,
        24,
        False,
    ),
    VarlenCase(
        "ek-r2-mixed-bf16-causal",
        ((1, 7), (3, 3), (5, 9), (2, 2)),
        (5, 3),
        (1, 1),
        (1, 1),
        (True, False),
        torch.bfloat16,
        3,
        3,
        32,
        32,
        False,
    ),
    VarlenCase(
        "ek-r2-dilated-axis-fp32",
        ((2, 9), (8, 7)),
        (3, 3),
        (1, 1),
        (1, 2),
        (False, False),
        torch.float32,
        2,
        2,
        16,
        16,
        False,
    ),
    VarlenCase(
        "ek-r3-mixed-fp32-noncausal",
        ((1, 4, 4), (3, 4, 4), (5, 7, 7), (2, 2, 2)),
        (5, 3, 3),
        (1, 1, 1),
        (1, 1, 1),
        (False, False, False),
        torch.float32,
        2,
        2,
        16,
        16,
        False,
    ),
    VarlenCase(
        "ek-r3-mixed-bf16-causal",
        ((1, 4, 4), (3, 4, 4), (5, 7, 7), (2, 2, 2)),
        (5, 3, 3),
        (1, 1, 1),
        (1, 1, 1),
        (True, False, False),
        torch.bfloat16,
        2,
        2,
        32,
        32,
        False,
    ),
)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
class VarlenEffectiveKernelTests(unittest.TestCase):
    def tearDown(self):
        torch.use_deterministic_algorithms(False)

    def _run_case(self, case: VarlenCase) -> None:
        if not _dtype_is_supported(case.dtype):
            self.skipTest(f"{case.dtype} is unavailable on this device")
        previous = _set_deterministic(case.deterministic)
        try:
            torch.manual_seed(4900 + case.rank)
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
                return_lse=True,
            )

            output_ref, logsumexp_ref = _case_reference(
                case, query_ref, key_ref, value_ref
            )

            if case.dtype == torch.float32:
                fwd_atol, grad_atol = _FP32_FWD_ATOL, _FP32_GRAD_ATOL
            else:
                fwd_atol, _ = _tolerances(case.dtype)
                grad_atol = fwd_atol

            torch.testing.assert_close(
                output.float(), output_ref, atol=fwd_atol, rtol=0
            )
            torch.testing.assert_close(
                logsumexp.float(), logsumexp_ref, atol=fwd_atol, rtol=0
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

    @skip_if_libnatten_is_not_supported()
    def test_mixed_extents_against_per_document_oracle(self):
        for case in EFFECTIVE_KERNEL_CASES:
            with self.subTest(case=case.name):
                self._run_case(case)

    @skip_if_libnatten_is_not_supported()
    def test_production_scale_mixed_pack_regression(self):
        # 4 documents (1, 64, 64) + 4 documents (8, 64, 64), kernel
        # (5, 13, 13): the T=1 documents clamp the T axis to 1 while the
        # T=8 documents are entirely unaffected (extent 8 >= kernel 5).
        # Before the clamp, the T=1 documents' out-of-range window read
        # past the document's real K/V rows -- for the outer (T) axis this
        # steps by a whole spatial-plane's worth of tokens per unit of
        # overflow, which is what produced the illegal memory access this
        # is the regression test for (phase3/k_gt_extent_trace.md Q2).
        layouts: Tuple[DimensionType, ...] = ((1, 64, 64),) * 4 + ((8, 64, 64),) * 4
        kernel_size = (5, 13, 13)
        stride = (1, 1, 1)
        dilation = (1, 1, 1)
        dtype = torch.bfloat16
        heads, head_dim, head_dim_v = 8, 64, 64
        for is_causal in ((True, False, False), (False, False, False)):
            with self.subTest(is_causal=is_causal):
                case = VarlenCase(
                    f"ek-production-scale-causal-{is_causal}",
                    layouts,
                    kernel_size,
                    stride,
                    dilation,
                    is_causal,
                    dtype,
                    heads,
                    heads,
                    head_dim,
                    head_dim_v,
                    True,
                )
                self._run_case(case)

    @skip_if_libnatten_is_not_supported()
    def test_phantom_key_minimal_1d_length_one(self):
        # The only valid window for a length-1 document is the token
        # itself: output must equal v and logsumexp must equal
        # scale * (q . k) exactly (up to float rounding). This is the
        # minimal repro for the phantom-key bug
        # (phase3/k_gt_extent_trace.md Q2: L=1, K=3 produced two phantom
        # key slots before the clamp).
        torch.manual_seed(5300)
        dtype = torch.float32
        heads, head_dim, head_dim_v = 2, 16, 24
        layout = natten.VarlenLayout(((1,),), device="cuda")
        q = torch.randn(
            1, heads, head_dim, device="cuda", dtype=dtype, requires_grad=True
        )
        k = torch.randn(
            1, heads, head_dim, device="cuda", dtype=dtype, requires_grad=True
        )
        v = torch.randn(
            1, heads, head_dim_v, device="cuda", dtype=dtype, requires_grad=True
        )

        output, lse = natten.na1d_varlen(
            q, k, v, layout, kernel_size=3, return_lse=True
        )

        scale = head_dim**-0.5
        expected_lse = scale * (q * k).sum(-1)
        torch.testing.assert_close(output, v, atol=_FP32_FWD_ATOL, rtol=0)
        torch.testing.assert_close(lse, expected_lse, atol=_FP32_FWD_ATOL, rtol=0)

        gradient = torch.randn_like(output)
        output.backward(gradient)
        # output == v exactly, so d(output)/dv is the identity: dv ==
        # gradient. output does not depend on q or k at all for a
        # single-key window (the softmax weight is 1.0 regardless of the
        # score), so dq and dk are exactly zero -- a phantom read
        # contaminating either would show up here as spurious noise.
        torch.testing.assert_close(v.grad, gradient, atol=_FP32_GRAD_ATOL, rtol=0)
        torch.testing.assert_close(
            q.grad, torch.zeros_like(q.grad), atol=_FP32_GRAD_ATOL, rtol=0
        )
        torch.testing.assert_close(
            k.grad, torch.zeros_like(k.grad), atol=_FP32_GRAD_ATOL, rtol=0
        )

    @skip_if_libnatten_is_not_supported()
    def test_phantom_key_minimal_3d_mixed_axes(self):
        # (1, 2, 2) with kernel (3, 2, 2): axis 0 clamps to a trivial
        # single-key window (extent 1) while axes 1 and 2 each keep a
        # genuine dense window (extent == kernel == 2, every query attends
        # to both positions) -- checks that clamping one axis leaves the
        # others' (non-trivial) windows untouched.
        torch.manual_seed(5301)
        dtype = torch.float32
        heads, head_dim, head_dim_v = 2, 16, 24
        doc_layout = (1, 2, 2)
        kernel_size = (3, 2, 2)
        stride = (1, 1, 1)
        dilation = (1, 1, 1)
        is_causal = (False, False, False)
        total = _prod(doc_layout)
        layout = natten.VarlenLayout((doc_layout,), device="cuda")

        q = torch.randn(
            total, heads, head_dim, device="cuda", dtype=dtype, requires_grad=True
        )
        k = torch.randn(
            total, heads, head_dim, device="cuda", dtype=dtype, requires_grad=True
        )
        v = torch.randn(
            total, heads, head_dim_v, device="cuda", dtype=dtype, requires_grad=True
        )
        q_ref = q.detach().clone().requires_grad_(True)
        k_ref = k.detach().clone().requires_grad_(True)
        v_ref = v.detach().clone().requires_grad_(True)

        output, lse = natten.na3d_varlen(
            q,
            k,
            v,
            layout,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            is_causal=is_causal,
            return_lse=True,
        )
        output_ref, lse_ref = _clamped_document_reference(
            doc_layout,
            kernel_size,
            stride,
            dilation,
            is_causal,
            None,
            q_ref,
            k_ref,
            v_ref,
            heads,
            heads,
            head_dim_v,
        )
        torch.testing.assert_close(
            output.float(), output_ref, atol=_FP32_FWD_ATOL, rtol=0
        )
        torch.testing.assert_close(lse.float(), lse_ref, atol=_FP32_FWD_ATOL, rtol=0)

        gradient = torch.randn_like(output)
        output.backward(gradient)
        output_ref.backward(gradient.float())
        for observed, expected in (
            (q.grad, q_ref.grad),
            (k.grad, k_ref.grad),
            (v.grad, v_ref.grad),
        ):
            torch.testing.assert_close(
                observed.float(), expected, atol=_FP32_GRAD_ATOL, rtol=0
            )
            self.assertGreater(torch.count_nonzero(observed).item(), 0)

    @skip_if_libnatten_is_not_supported()
    def test_empty_documents_mixed_with_short_documents(self):
        # A short (clamped) document mixed with empty documents: an empty
        # document must still be an exact no-op -- identical packed rows,
        # mixed-with-empty vs. empties-removed -- exactly as it is for
        # unclamped documents (existing semantics, see
        # test_fna_varlen.py::_run_empty_document_case).
        cases = (
            (1, ((0,), (1,), (0,), (5,), (0,)), (5,), (1,), (1,), (False,)),
            (
                2,
                ((0, 6), (1, 6), (3, 3), (0, 4)),
                (5, 3),
                (1, 1),
                (1, 1),
                (True, False),
            ),
        )
        dtype = torch.float16
        heads, head_dim = 2, 16
        for rank, layouts_with_empty, kernel_size, stride, dilation, is_causal in cases:
            with self.subTest(rank=rank):
                reduced_layouts = tuple(
                    doc for doc in layouts_with_empty if _prod(doc) > 0
                )
                total = sum(_prod(doc) for doc in layouts_with_empty)
                varlen_fn = _VARLEN_FN_BY_RANK[rank]

                torch.manual_seed(5100 + rank)
                query = torch.randn(total, heads, head_dim, device="cuda", dtype=dtype)
                key = torch.randn(total, heads, head_dim, device="cuda", dtype=dtype)
                value = torch.randn(total, heads, head_dim, device="cuda", dtype=dtype)
                gradient = torch.randn(
                    total, heads, head_dim, device="cuda", dtype=dtype
                )

                def run(layouts, deterministic):
                    previous = _set_deterministic(deterministic)
                    try:
                        doc_layout = natten.VarlenLayout(layouts, device="cuda")
                        q = query.detach().clone().requires_grad_(True)
                        k = key.detach().clone().requires_grad_(True)
                        v = value.detach().clone().requires_grad_(True)
                        output, logsumexp = varlen_fn(
                            q,
                            k,
                            v,
                            doc_layout,
                            kernel_size=kernel_size,
                            stride=stride,
                            dilation=dilation,
                            is_causal=is_causal,
                            return_lse=True,
                        )
                        return output, logsumexp, q, k, v
                    finally:
                        torch.use_deterministic_algorithms(previous)

                for deterministic in (True, False):
                    with self.subTest(
                        rank=rank, deterministic=deterministic, part="forward"
                    ):
                        out_mixed, lse_mixed, *_ = run(
                            layouts_with_empty, deterministic
                        )
                        out_reduced, lse_reduced, *_ = run(
                            reduced_layouts, deterministic
                        )
                        self.assertTrue(torch.equal(out_mixed, out_reduced))
                        self.assertTrue(torch.equal(lse_mixed, lse_reduced))

                with self.subTest(rank=rank, part="backward"):
                    out_mixed, lse_mixed, qm, km, vm = run(layouts_with_empty, True)
                    out_mixed.backward(gradient)
                    out_reduced, lse_reduced, qr, kr, vr = run(reduced_layouts, True)
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
    def test_dilation_above_one_still_requires_fit(self):
        # dilation == 1 axes never raise (clamped instead); an axis with
        # dilation > 1 is not clamped and must still fit
        # kernel_size * dilation, with a message that mentions dilation.
        layout = natten.VarlenLayout(((5,),), device="cuda")
        query = torch.zeros(5, 1, 16, device="cuda", dtype=torch.float16)
        with self.assertRaisesRegex(ValueError, "dilation"):
            natten.na1d_varlen(query, query, query, layout, kernel_size=3, dilation=2)

        # Per-axis independence: axis 0 (dilation 1, extent 2 < kernel 5)
        # clamps fine on its own; axis 1 (dilation 2, extent 4 < kernel 3 *
        # dilation 2 = 6) must still raise.
        mixed_layout = natten.VarlenLayout(((2, 4),), device="cuda")
        query2 = torch.zeros(8, 1, 16, device="cuda", dtype=torch.float16)
        with self.assertRaisesRegex(ValueError, "dilation"):
            natten.na2d_varlen(
                query2,
                query2,
                query2,
                mixed_layout,
                kernel_size=(5, 3),
                dilation=(1, 2),
            )

    @skip_if_libnatten_is_not_supported()
    def test_torch_compile_fullgraph_matches_eager(self):
        layouts: Tuple[DimensionType, ...] = ((1, 7), (3, 3), (5, 9))
        kernel_size, stride, dilation = (5, 3), (1, 1), (1, 1)
        is_causal = (False, False)
        dtype = torch.float16
        heads, head_dim = 2, 16
        layout = natten.VarlenLayout(layouts, device="cuda")
        total = sum(_prod(doc) for doc in layouts)

        def fn(q, k, v):
            return natten.na2d_varlen(
                q,
                k,
                v,
                layout,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                is_causal=is_causal,
            )

        torch.manual_seed(5200)
        inputs = tuple(
            torch.randn(total, heads, head_dim, device="cuda", dtype=dtype)
            for _ in range(3)
        )
        gradient = torch.randn(total, heads, head_dim, device="cuda", dtype=dtype)

        def run(callable_fn):
            run_inputs = tuple(
                tensor.detach().clone().requires_grad_(True) for tensor in inputs
            )
            output = callable_fn(*run_inputs)
            output.backward(gradient)
            return (output.detach(),) + tuple(
                tensor.grad.detach() for tensor in run_inputs
            )

        torch.use_deterministic_algorithms(True)
        try:
            reference = run(fn)
            torch.compiler.reset()
            try:
                with torch._dynamo.config.patch(
                    recompile_limit=1,
                    accumulated_recompile_limit=1,
                    fail_on_recompile_limit_hit=True,
                ):
                    compiled = torch.compile(fn, fullgraph=True)
                    observed = run(compiled)
                    for expected, actual in zip(reference, observed):
                        self.assertTrue(torch.equal(expected, actual))
            finally:
                torch.compiler.reset()
        finally:
            torch.use_deterministic_algorithms(False)


if __name__ == "__main__":
    unittest.main()
