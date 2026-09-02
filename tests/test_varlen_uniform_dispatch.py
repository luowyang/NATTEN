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
"""Uniform-layout dispatch: a VarlenLayout whose documents all share the same
shape runs on the fixed-shape CUTLASS FNA kernels (natten.backends.fna.
cutlass_fna_generic) instead of building a varlen schedule. See
docs/backends.md and CHANGELOG.md for the public description.

Oracle: `cutlass_fna_generic(view, kernel_size=k_eff, ...)` on the packed
tensor reshaped to `[num_docs, *shape, heads, head_dim]` -- the exact
backend-private call the dispatch itself makes internally (see the layering
rule in its docstring), so the comparison is `torch.equal`, not a tolerance:
both paths reach the identical kernel launch with identical bytes. The
public `na{1,2,3}d` is deliberately not used as the oracle: for a fully
clamped axis (`k_eff == extent`), its own `is_self_attention` optimization
reroutes 1-D (and causal-free full-window 2-D/3-D) calls to a different
(FMHA) kernel regardless of the `backend=` hint -- a pre-existing, orthogonal
optimization in that public wrapper, not part of what "runs on the
fixed-shape CUTLASS FNA kernel" means here.
"""

import unittest
from typing import Any, Callable, Dict, Tuple

import natten
import torch
from natten.backends import cutlass_fna_generic
from natten.types import DimensionType
from natten.utils.testing import skip_if_libnatten_is_not_supported
from torch import Tensor

from .utils import (
    _dtype_is_supported,
    _make_layout,
    _prod,
    _set_deterministic,
    VarlenCase,
)

_VARLEN_FN_BY_RANK: Dict[int, Callable[..., Any]] = {
    1: natten.na1d_varlen,
    2: natten.na2d_varlen,
    3: natten.na3d_varlen,
}


def _effective_kernel(
    shape: DimensionType, kernel_size: DimensionType, dilation: DimensionType
) -> Tuple[int, ...]:
    return tuple(
        k if d > 1 else min(k, e) for k, e, d in zip(kernel_size, shape, dilation)
    )


# Each case is a uniform layout (every document sharing `case.layouts[0]`);
# num_docs is implied by len(case.layouts). Covers both ranks and dtypes,
# causal and non-causal, a dilation > 1 axis, a document shape shorter than
# kernel_size on one axis (k_eff < kernel_size), n=1 doc, and one
# deterministic-mode case.
UNIFORM_CASES: Tuple[VarlenCase, ...] = (
    VarlenCase(
        "u-r1-fp32-noncausal",
        ((9,), (9,)),
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
        "u-r1-bf16-causal-single-doc-short",
        ((3,),),
        (5,),
        (1,),
        (1,),
        (True,),
        torch.bfloat16,
        4,
        4,
        32,
        32,
        True,
    ),
    VarlenCase(
        "u-r2-fp32-mixed-causal-dilated",
        ((4, 12), (4, 12), (4, 12)),
        (3, 5),
        (1, 1),
        (1, 2),
        (False, True),
        torch.float32,
        2,
        2,
        16,
        24,
        True,
    ),
    VarlenCase(
        "u-r2-bf16-causal-deterministic",
        ((8, 8), (8, 8)),
        (4, 4),
        (1, 1),
        (1, 1),
        (True, False),
        torch.bfloat16,
        3,
        3,
        32,
        32,
        True,
    ),
    VarlenCase(
        "u-r3-fp32-noncausal",
        ((6, 6, 6), (6, 6, 6)),
        (3, 3, 3),
        (1, 1, 1),
        (1, 1, 1),
        (False, False, False),
        torch.float32,
        2,
        2,
        16,
        16,
        True,
    ),
    VarlenCase(
        "u-r3-bf16-causal-short-temporal",
        ((2, 7, 7),) * 4,
        (5, 3, 3),
        (1, 1, 1),
        (1, 1, 1),
        (True, False, False),
        torch.bfloat16,
        2,
        2,
        32,
        32,
        True,
    ),
)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
class VarlenUniformDispatchTests(unittest.TestCase):
    def tearDown(self):
        torch.use_deterministic_algorithms(False)

    def _run_case(self, case: VarlenCase) -> None:
        if not _dtype_is_supported(case.dtype):
            self.skipTest(f"{case.dtype} is unavailable on this device")
        previous = _set_deterministic(case.deterministic)
        try:
            torch.manual_seed(6100 + case.rank)
            shape = case.layouts[0]
            num_docs = len(case.layouts)
            total = num_docs * _prod(shape)

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
                return_lse=True,
            )

            # Uniform layouts build no varlen schedule: the memo stays empty.
            self.assertEqual(len(layout._memo), 0)

            k_eff = _effective_kernel(shape, case.kernel_size, case.dilation)
            q_view = query_ref.view(num_docs, *shape, case.heads, case.head_dim)
            k_view = key_ref.view(num_docs, *shape, case.heads_kv, case.head_dim)
            v_view = value_ref.view(num_docs, *shape, case.heads_kv, case.head_dim_v)
            output_ref, logsumexp_ref = cutlass_fna_generic(
                q_view,
                k_view,
                v_view,
                kernel_size=k_eff,
                stride=case.stride,
                dilation=case.dilation,
                is_causal=case.is_causal,
                return_lse=True,
            )
            output_ref = output_ref.reshape(total, case.heads, case.head_dim_v)
            logsumexp_ref = logsumexp_ref.reshape(total, case.heads)

            self.assertTrue(
                torch.equal(output, output_ref), msg=f"{case.name}: output mismatch"
            )
            self.assertTrue(
                torch.equal(logsumexp, logsumexp_ref), msg=f"{case.name}: lse mismatch"
            )

            gradient = torch.randn_like(output)
            output.backward(gradient)
            output_ref.backward(gradient)

            for observed, expected, name in (
                (query.grad, query_ref.grad, "dq"),
                (key.grad, key_ref.grad, "dk"),
                (value.grad, value_ref.grad, "dv"),
            ):
                self.assertTrue(
                    torch.equal(observed, expected), msg=f"{case.name}: {name} mismatch"
                )
        finally:
            torch.use_deterministic_algorithms(previous)

    @skip_if_libnatten_is_not_supported()
    def test_uniform_layout_matches_fixed_bit_for_bit(self):
        for case in UNIFORM_CASES:
            with self.subTest(case=case.name):
                self._run_case(case)

    @skip_if_libnatten_is_not_supported()
    def test_return_lse_false_returns_tensor_only(self):
        torch.manual_seed(6200)
        layout = natten.VarlenLayout(((6,), (6,)), device="cuda")
        query = torch.randn(12, 2, 16, device="cuda", dtype=torch.float16)
        key = torch.randn(12, 2, 16, device="cuda", dtype=torch.float16)
        value = torch.randn(12, 2, 16, device="cuda", dtype=torch.float16)
        output = natten.na1d_varlen(query, key, value, layout, kernel_size=3)
        self.assertIsInstance(output, Tensor)
        self.assertEqual(output.shape, (12, 2, 16))

    @skip_if_libnatten_is_not_supported()
    def test_explicit_tile_shapes_pass_through(self):
        # Tile shapes proven valid for this dtype/head_dim/heads combination
        # by tests/utils.py's DEFAULT_CASES (r1-prod-causal-fp16).
        torch.manual_seed(6300)
        shape = (49,)
        num_docs = 2
        total = num_docs * _prod(shape)
        heads, head_dim = 3, 64
        kernel_size, stride, dilation, is_causal = (5,), (1,), (1,), (True,)
        q_tile_shape, kv_tile_shape = (32,), (128,)
        backward_q_tile_shape, backward_kv_tile_shape = (64,), (64,)

        query = torch.randn(total, heads, head_dim, device="cuda", dtype=torch.float16)
        key = torch.randn(total, heads, head_dim, device="cuda", dtype=torch.float16)
        value = torch.randn(total, heads, head_dim, device="cuda", dtype=torch.float16)
        query.requires_grad_(True)
        key.requires_grad_(True)
        value.requires_grad_(True)
        query_ref = query.detach().clone().requires_grad_(True)
        key_ref = key.detach().clone().requires_grad_(True)
        value_ref = value.detach().clone().requires_grad_(True)

        layout = natten.VarlenLayout((shape,) * num_docs, device="cuda")
        output, lse = natten.na1d_varlen(
            query,
            key,
            value,
            layout,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            is_causal=is_causal,
            q_tile_shape=q_tile_shape,
            kv_tile_shape=kv_tile_shape,
            backward_q_tile_shape=backward_q_tile_shape,
            backward_kv_tile_shape=backward_kv_tile_shape,
            return_lse=True,
        )
        self.assertEqual(len(layout._memo), 0)

        q_view = query_ref.view(num_docs, *shape, heads, head_dim)
        k_view = key_ref.view(num_docs, *shape, heads, head_dim)
        v_view = value_ref.view(num_docs, *shape, heads, head_dim)
        output_ref, lse_ref = cutlass_fna_generic(
            q_view,
            k_view,
            v_view,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            is_causal=is_causal,
            q_tile_shape=q_tile_shape,
            kv_tile_shape=kv_tile_shape,
            backward_q_tile_shape=backward_q_tile_shape,
            backward_kv_tile_shape=backward_kv_tile_shape,
            return_lse=True,
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
    def test_non_uniform_layout_still_takes_varlen_path(self):
        torch.manual_seed(6400)
        layout = natten.VarlenLayout(((5,), (9,)), device="cuda")
        self.assertFalse(layout.is_uniform)
        query = torch.randn(14, 2, 16, device="cuda", dtype=torch.float16)
        key = torch.randn(14, 2, 16, device="cuda", dtype=torch.float16)
        value = torch.randn(14, 2, 16, device="cuda", dtype=torch.float16)
        natten.na1d_varlen(query, key, value, layout, kernel_size=3)
        self.assertEqual(len(layout._memo), 1)

    @skip_if_libnatten_is_not_supported()
    def test_torch_compile_fullgraph_matches_eager(self):
        shape = (4, 12)
        num_docs = 3
        total = num_docs * _prod(shape)
        heads, head_dim = 2, 16
        kernel_size, stride, dilation = (3, 5), (1, 1), (1, 1)
        is_causal = (False, True)
        dtype = torch.float16
        layout = natten.VarlenLayout((shape,) * num_docs, device="cuda")

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

        torch.manual_seed(6500)
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
                    # Uniform dispatch never touches the memo, so a second
                    # call with the same layout must not recompile either.
                    observed2 = run(compiled)
                    for expected, actual in zip(reference, observed):
                        self.assertTrue(torch.equal(expected, actual))
                    for expected, actual in zip(reference, observed2):
                        self.assertTrue(torch.equal(expected, actual))
            finally:
                torch.compiler.reset()
        finally:
            torch.use_deterministic_algorithms(False)


if __name__ == "__main__":
    unittest.main()
