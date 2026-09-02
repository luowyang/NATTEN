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
file's part: for any axis, `_window_positions` degenerates to exactly the
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
from typing import Any, Callable, Dict, Tuple
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

_VARLEN_FN_BY_RANK: Dict[int, Callable[..., Any]] = {
    1: natten.na1d_varlen,
    2: natten.na2d_varlen,
    3: natten.na3d_varlen,
}

_FP32_FWD_ATOL = 3e-4
_FP32_GRAD_ATOL = 4e-4


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
    def test_heterogeneous_pack_image_column_clamps_to_identity(self):
        # (b)'s heterogeneous-pack sub-case: kernel (K, 1, 1) permutes H, W
        # to the front, turning each document into H*W length-T 1-D
        # documents. The image document (T=1) has columns of length 1 < K:
        # the per-document device-side clamp (unrelated to the Python-level
        # fold/permute lowering under test here) reduces ITS effective
        # kernel to 1, giving an exact identity for those rows specifically
        # -- checked directly here, not just within the oracle's tolerance.
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
        query = torch.randn(total, heads, head_dim, device="cuda", dtype=dtype)
        key = torch.randn(total, heads, head_dim, device="cuda", dtype=dtype)
        value = torch.randn(total, heads, head_dim_v, device="cuda", dtype=dtype)

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
        torch.testing.assert_close(
            output[image_start:image_end],
            value[image_start:image_end],
            atol=_FP32_FWD_ATOL,
            rtol=0,
        )
        torch.testing.assert_close(
            lse[image_start:image_end], expected_lse, atol=_FP32_FWD_ATOL, rtol=0
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
            else ((2, 3), (3, 2)) if rank == 2 else ((2, 2, 3), (1, 3, 2))
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
        self.assertEqual(len(layout._memo), 0)

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
        self.assertEqual(len(layout._memo), 0)
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
        self, rank, layouts, kernel_size, second_layouts, second_kernel_size
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
            torch.compiler.reset()
            try:
                compiled = torch.compile(call, fullgraph=True)
                with torch._dynamo.config.patch(
                    recompile_limit=1,
                    accumulated_recompile_limit=1,
                    fail_on_recompile_limit_hit=True,
                ):
                    observed = run(compiled, layout, kernel_size)
                    observed_again = run(compiled, layout, kernel_size)
                for expected, actual in zip(reference, observed):
                    self.assertTrue(torch.equal(expected, actual))
                for expected, actual in zip(reference, observed_again):
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

    @skip_if_libnatten_is_not_supported()
    def test_fixed_family_still_rejects_kernel_size_one(self):
        query = torch.zeros(1, 4, 4, 1, 16, device="cuda", dtype=torch.float16)
        with self.assertRaises(ValueError):
            natten.na3d(query, query, query, kernel_size=(1, 3, 3))

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


if __name__ == "__main__":
    unittest.main()
