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
"""Tests for `kernel_size == 1` axes in the fixed-shape `na1d`/`na2d`/`na3d` family: such an
axis is lowered away in Python (folded and/or permuted into the batch dimension, or
short-circuited to an identity) before any backend is chosen, so the CUDA kernels never see it.

Oracles are independent, hand-written `reshape`/`permute` views run back through `na1d`/`na2d`
at the reduced rank -- not the lowering code under test -- per the cases specified for this
feature. Every numeric test is run with `backend="cutlass-fna"` pinned and with `backend=None`
(default selection; on SM90 this picks `hopper-fna` for float16/bfloat16 and `cutlass-fna` for
float32), to check the lowering is backend-agnostic.
"""

import random
import unittest
from unittest import mock

import torch
from natten.functional import na1d, na2d, na3d
from natten.utils.checks import check_kernel_size_arg
from natten.utils.testing import skip_if_libnatten_is_not_supported, supports_bfloat16

from .utils import logger, reset_torch_compile


def _reset_everything(random_seed: int = 42, torch_seed: int = 42):
    from natten.context import (
        NattenContext,
        set_memory_usage_preference,
        use_kv_parallelism_in_fused_na,
    )

    NattenContext.reset()
    set_memory_usage_preference("unrestricted")
    use_kv_parallelism_in_fused_na(True)

    random.seed(random_seed)
    torch.manual_seed(torch_seed)
    logger.debug(f"Reset seeds: {random_seed=}, {torch_seed=}")
    torch.cuda.empty_cache()
    torch.use_deterministic_algorithms(False)


# Tolerance classes, reusing upstream's own FNA test conventions (tests/test_fna.py's
# ALLOWED_DTYPES): float32 forward atol is upstream's own 1e-4 class (we use 3e-4 for a little
# headroom, since our oracle is a second full attention call, not a bitwise-identical replay);
# backward reuses upstream's float32 tuple verbatim. bfloat16 reuses upstream's tolerances
# verbatim, both forward and backward.
FP32_FWD_ATOL = 3e-4
FP32_BWD_ATOL = (1e-2, 1e-4, 1e-4)  # dq, dk, dv
BF16_FWD_ATOL = 5e-2
BF16_BWD_ATOL = (1e-2, 1e-2, 1e-2)

# When the backend is pinned, the lowered call and the oracle call run the identical kernel on
# equivalent (reshaped/permuted) views of the same data, so forward is bitwise-exact (checked
# with torch.equal) and backward -- which may use a nondeterministic reduction -- is exact to a
# tight tolerance rather than 0.
EXACT_BWD_ATOL = (1e-4, 1e-4, 1e-4)


class _DegenerateAxesTestBase(unittest.TestCase):
    def setUp(self):
        _reset_everything()

    def tearDown(self):
        _reset_everything()

    def _assert_fwd(self, out, out_ref, lse, lse_ref, exact: bool, atol: float):
        if exact:
            self.assertTrue(
                torch.equal(out, out_ref),
                "output should be bit-exact against the oracle",
            )
            self.assertTrue(
                torch.equal(lse, lse_ref),
                "logsumexp should be bit-exact against the oracle",
            )
        else:
            torch.testing.assert_close(out.float(), out_ref.float(), atol=atol, rtol=0)
            torch.testing.assert_close(lse.float(), lse_ref.float(), atol=atol, rtol=0)

    def _assert_bwd(self, grads, grads_ref, atol):
        dq, dk, dv = grads
        dq_ref, dk_ref, dv_ref = grads_ref
        dq_atol, dk_atol, dv_atol = atol
        torch.testing.assert_close(dq.float(), dq_ref.float(), atol=dq_atol, rtol=0)
        torch.testing.assert_close(dk.float(), dk_ref.float(), atol=dk_atol, rtol=0)
        torch.testing.assert_close(dv.float(), dv_ref.float(), atol=dv_atol, rtol=0)

    def _dtypes(self):
        dtypes = [(torch.float32, FP32_FWD_ATOL, FP32_BWD_ATOL)]
        if supports_bfloat16(torch.device("cuda")):
            dtypes.append((torch.bfloat16, BF16_FWD_ATOL, BF16_BWD_ATOL))
        return dtypes


# -----------------------------------------------------------------------------------------------
# Leading fold: a leading run of kernel_size == 1 axes is folded into the batch dimension by a
# reshape (pure view).
# -----------------------------------------------------------------------------------------------


class LeadingFoldTest(_DegenerateAxesTestBase):
    def _run_rank3(self, backend, dtype, fwd_atol, bwd_atol, exact_fwd):
        device = "cuda"
        B, T, H, W = 2, 3, 12, 14
        heads, head_dim, head_dim_v = 2, 32, 40
        K = 5

        q = torch.randn(
            B, T, H, W, heads, head_dim, device=device, dtype=dtype
        ).requires_grad_(True)
        k = torch.randn(
            B, T, H, W, heads, head_dim, device=device, dtype=dtype
        ).requires_grad_(True)
        v = torch.randn(
            B, T, H, W, heads, head_dim_v, device=device, dtype=dtype
        ).requires_grad_(True)
        q_ref = q.detach().clone().requires_grad_(True)
        k_ref = k.detach().clone().requires_grad_(True)
        v_ref = v.detach().clone().requires_grad_(True)

        # T is degenerate and leading; H and W are kept, with a non-trivial stride on W.
        out, lse = na3d(
            q,
            k,
            v,
            kernel_size=(1, K, K),
            stride=(1, 1, 2),
            dilation=(1, 1, 1),
            is_causal=(False, False, False),
            scale=0.2,
            backend=backend,
            return_lse=True,
        )
        self.assertEqual(out.shape, (B, T, H, W, heads, head_dim_v))
        self.assertEqual(lse.shape, (B, T, H, W, heads))

        q2 = q_ref.reshape(B * T, H, W, heads, head_dim)
        k2 = k_ref.reshape(B * T, H, W, heads, head_dim)
        v2 = v_ref.reshape(B * T, H, W, heads, head_dim_v)
        out_ref2, lse_ref2 = na2d(
            q2,
            k2,
            v2,
            kernel_size=K,
            stride=(1, 2),
            dilation=(1, 1),
            is_causal=(False, False),
            scale=0.2,
            backend=backend,
            return_lse=True,
        )
        out_ref = out_ref2.reshape(B, T, H, W, heads, head_dim_v)
        lse_ref = lse_ref2.reshape(B, T, H, W, heads)

        self._assert_fwd(out, out_ref, lse, lse_ref, exact_fwd, fwd_atol)

        grad_out = torch.randn_like(out) * 0.1
        out.backward(grad_out)
        out_ref.backward(grad_out.detach().clone())
        self._assert_bwd(
            (q.grad, k.grad, v.grad), (q_ref.grad, k_ref.grad, v_ref.grad), bwd_atol
        )

    def _run_rank2(self, backend, dtype, fwd_atol, bwd_atol, exact_fwd):
        device = "cuda"
        B, s0, s1 = 2, 4, 20
        heads, head_dim, head_dim_v = 3, 16, 24
        K = 6

        q = torch.randn(
            B, s0, s1, heads, head_dim, device=device, dtype=dtype
        ).requires_grad_(True)
        k = torch.randn(
            B, s0, s1, heads, head_dim, device=device, dtype=dtype
        ).requires_grad_(True)
        v = torch.randn(
            B, s0, s1, heads, head_dim_v, device=device, dtype=dtype
        ).requires_grad_(True)
        q_ref = q.detach().clone().requires_grad_(True)
        k_ref = k.detach().clone().requires_grad_(True)
        v_ref = v.detach().clone().requires_grad_(True)

        out, lse = na2d(
            q,
            k,
            v,
            kernel_size=(1, K),
            stride=(1, 2),
            dilation=(1, 1),
            is_causal=(False, True),
            backend=backend,
            return_lse=True,
        )
        self.assertEqual(out.shape, (B, s0, s1, heads, head_dim_v))

        q2 = q_ref.reshape(B * s0, s1, heads, head_dim)
        k2 = k_ref.reshape(B * s0, s1, heads, head_dim)
        v2 = v_ref.reshape(B * s0, s1, heads, head_dim_v)
        out_ref2, lse_ref2 = na1d(
            q2,
            k2,
            v2,
            kernel_size=(K,),
            stride=(2,),
            dilation=(1,),
            is_causal=(True,),
            backend=backend,
            return_lse=True,
        )
        out_ref = out_ref2.reshape(B, s0, s1, heads, head_dim_v)
        lse_ref = lse_ref2.reshape(B, s0, s1, heads)

        self._assert_fwd(out, out_ref, lse, lse_ref, exact_fwd, fwd_atol)

        grad_out = torch.randn_like(out) * 0.1
        out.backward(grad_out)
        out_ref.backward(grad_out.detach().clone())
        self._assert_bwd(
            (q.grad, k.grad, v.grad), (q_ref.grad, k_ref.grad, v_ref.grad), bwd_atol
        )

    @skip_if_libnatten_is_not_supported()
    def test_rank3_leading_fold(self):
        for dtype, fwd_atol, bwd_atol in self._dtypes():
            for backend in ["cutlass-fna", None]:
                _reset_everything()
                exact = backend is not None and dtype == torch.float32
                self._run_rank3(
                    backend,
                    dtype,
                    fwd_atol,
                    EXACT_BWD_ATOL if exact else bwd_atol,
                    exact_fwd=exact,
                )

    @skip_if_libnatten_is_not_supported()
    def test_rank2_leading_fold(self):
        for dtype, fwd_atol, bwd_atol in self._dtypes():
            for backend in ["cutlass-fna", None]:
                _reset_everything()
                exact = backend is not None and dtype == torch.float32
                self._run_rank2(
                    backend,
                    dtype,
                    fwd_atol,
                    EXACT_BWD_ATOL if exact else bwd_atol,
                    exact_fwd=exact,
                )


# -----------------------------------------------------------------------------------------------
# Permute: non-leading kernel_size == 1 axes are moved in front of the kept axes and folded.
# -----------------------------------------------------------------------------------------------


class PermuteTest(_DegenerateAxesTestBase):
    def _run_k11(self, backend, dtype, fwd_atol, bwd_atol, exact_fwd):
        # kernel_size = (K, 1, 1): T (axis 0) is kept -- not leading -- H and W are degenerate.
        # Exercises is_causal and dilation on the kept axis together with two degenerate axes.
        device = "cuda"
        B, T, H, W = 2, 10, 6, 7
        heads, head_dim, head_dim_v = 2, 32, 40
        K = 4

        q = torch.randn(
            B, T, H, W, heads, head_dim, device=device, dtype=dtype
        ).requires_grad_(True)
        k = torch.randn(
            B, T, H, W, heads, head_dim, device=device, dtype=dtype
        ).requires_grad_(True)
        v = torch.randn(
            B, T, H, W, heads, head_dim_v, device=device, dtype=dtype
        ).requires_grad_(True)
        q_ref = q.detach().clone().requires_grad_(True)
        k_ref = k.detach().clone().requires_grad_(True)
        v_ref = v.detach().clone().requires_grad_(True)

        out, lse = na3d(
            q,
            k,
            v,
            kernel_size=(K, 1, 1),
            stride=(1, 1, 1),
            dilation=(2, 1, 1),
            is_causal=(True, False, False),
            backend=backend,
            return_lse=True,
        )
        self.assertEqual(out.shape, (B, T, H, W, heads, head_dim_v))

        q2 = (
            q_ref.permute(0, 2, 3, 1, 4, 5)
            .contiguous()
            .reshape(B * H * W, T, heads, head_dim)
        )
        k2 = (
            k_ref.permute(0, 2, 3, 1, 4, 5)
            .contiguous()
            .reshape(B * H * W, T, heads, head_dim)
        )
        v2 = (
            v_ref.permute(0, 2, 3, 1, 4, 5)
            .contiguous()
            .reshape(B * H * W, T, heads, head_dim_v)
        )
        out_ref2, lse_ref2 = na1d(
            q2,
            k2,
            v2,
            kernel_size=(K,),
            stride=(1,),
            dilation=(2,),
            is_causal=(True,),
            backend=backend,
            return_lse=True,
        )
        out_ref = out_ref2.reshape(B, H, W, T, heads, head_dim_v).permute(
            0, 3, 1, 2, 4, 5
        )
        lse_ref = lse_ref2.reshape(B, H, W, T, heads).permute(0, 3, 1, 2, 4)

        self._assert_fwd(out, out_ref, lse, lse_ref, exact_fwd, fwd_atol)

        grad_out = torch.randn_like(out) * 0.1
        out.backward(grad_out)
        out_ref.backward(grad_out.detach().clone())
        self._assert_bwd(
            (q.grad, k.grad, v.grad), (q_ref.grad, k_ref.grad, v_ref.grad), bwd_atol
        )

    def _run_1k1(self, backend, dtype, fwd_atol, bwd_atol, exact_fwd):
        # kernel_size = (1, K, 1): T (leading) is folded first; W is then permuted in front of
        # the kept axis H.
        device = "cuda"
        B, T, H, W = 2, 5, 9, 4
        heads, head_dim = 2, 16
        K = 3

        q = torch.randn(
            B, T, H, W, heads, head_dim, device=device, dtype=dtype
        ).requires_grad_(True)
        k = torch.randn(
            B, T, H, W, heads, head_dim, device=device, dtype=dtype
        ).requires_grad_(True)
        v = torch.randn(
            B, T, H, W, heads, head_dim, device=device, dtype=dtype
        ).requires_grad_(True)
        q_ref = q.detach().clone().requires_grad_(True)
        k_ref = k.detach().clone().requires_grad_(True)
        v_ref = v.detach().clone().requires_grad_(True)

        out, lse = na3d(
            q, k, v, kernel_size=(1, K, 1), backend=backend, return_lse=True
        )
        self.assertEqual(out.shape, (B, T, H, W, heads, head_dim))

        def to_oracle_input(t):
            return (
                t.reshape(B * T, H, W, heads, head_dim)
                .permute(0, 2, 1, 3, 4)
                .contiguous()
                .reshape(B * T * W, H, heads, head_dim)
            )

        q2, k2, v2 = (
            to_oracle_input(q_ref),
            to_oracle_input(k_ref),
            to_oracle_input(v_ref),
        )
        out_ref2, lse_ref2 = na1d(
            q2, k2, v2, kernel_size=(K,), backend=backend, return_lse=True
        )
        out_ref = (
            out_ref2.reshape(B * T, W, H, heads, head_dim)
            .permute(0, 2, 1, 3, 4)
            .reshape(B, T, H, W, heads, head_dim)
        )
        lse_ref = (
            lse_ref2.reshape(B * T, W, H, heads)
            .permute(0, 2, 1, 3)
            .reshape(B, T, H, W, heads)
        )

        self._assert_fwd(out, out_ref, lse, lse_ref, exact_fwd, fwd_atol)

        grad_out = torch.randn_like(out) * 0.1
        out.backward(grad_out)
        out_ref.backward(grad_out.detach().clone())
        self._assert_bwd(
            (q.grad, k.grad, v.grad), (q_ref.grad, k_ref.grad, v_ref.grad), bwd_atol
        )

    @skip_if_libnatten_is_not_supported()
    def test_rank3_k11_causal_dilated(self):
        for dtype, fwd_atol, bwd_atol in self._dtypes():
            for backend in ["cutlass-fna", None]:
                _reset_everything()
                exact = backend is not None and dtype == torch.float32
                self._run_k11(
                    backend,
                    dtype,
                    fwd_atol,
                    EXACT_BWD_ATOL if exact else bwd_atol,
                    exact_fwd=exact,
                )

    @skip_if_libnatten_is_not_supported()
    def test_rank3_1k1(self):
        for dtype, fwd_atol, bwd_atol in self._dtypes():
            for backend in ["cutlass-fna", None]:
                _reset_everything()
                exact = backend is not None and dtype == torch.float32
                self._run_1k1(
                    backend,
                    dtype,
                    fwd_atol,
                    EXACT_BWD_ATOL if exact else bwd_atol,
                    exact_fwd=exact,
                )


# -----------------------------------------------------------------------------------------------
# Identity: every axis degenerate. No backend is ever chosen.
# -----------------------------------------------------------------------------------------------


class IdentityTest(_DegenerateAxesTestBase):
    @skip_if_libnatten_is_not_supported()
    def test_rank1_gqa_identity_and_backward(self):
        device = "cuda"
        B, S = 2, 7
        heads, heads_kv, head_dim, head_dim_v = 4, 2, 16, 24
        repeats = heads // heads_kv

        for backend in ["cutlass-fna", None]:
            q = torch.randn(B, S, heads, head_dim, device=device).requires_grad_(True)
            k = torch.randn(B, S, heads_kv, head_dim, device=device).requires_grad_(
                True
            )
            v = torch.randn(B, S, heads_kv, head_dim_v, device=device).requires_grad_(
                True
            )

            with mock.patch(
                "natten.functional.choose_backend"
            ) as mocked_choose_backend:
                out, lse = na1d(
                    q, k, v, kernel_size=(1,), backend=backend, return_lse=True
                )
                mocked_choose_backend.assert_not_called()

            v_rep = torch.repeat_interleave(
                v, repeats=repeats, dim=-2, output_size=heads
            )
            self.assertTrue(torch.equal(out, v_rep))

            k_rep = torch.repeat_interleave(
                k, repeats=repeats, dim=-2, output_size=heads
            )
            expected_lse = (q.float() * k_rep.float()).sum(-1) * (head_dim**-0.5)
            self.assertTrue(torch.equal(lse, expected_lse))

            grad_out = torch.randn_like(out)
            out.backward(grad_out)
            expected_dv = grad_out.reshape(B, S, heads_kv, repeats, head_dim_v).sum(3)
            self.assertTrue(torch.equal(v.grad, expected_dv))
            self.assertTrue(torch.equal(q.grad, torch.zeros_like(q.grad)))
            self.assertTrue(torch.equal(k.grad, torch.zeros_like(k.grad)))

    @skip_if_libnatten_is_not_supported()
    def test_rank2_and_rank3_identity(self):
        device = "cuda"

        q = torch.randn(2, 3, 5, 2, 16, device=device).requires_grad_(True)
        k = torch.randn(2, 3, 5, 2, 16, device=device).requires_grad_(True)
        v = torch.randn(2, 3, 5, 2, 16, device=device).requires_grad_(True)
        with mock.patch("natten.functional.choose_backend") as mocked_choose_backend:
            out, lse = na2d(q, k, v, kernel_size=(1, 1), backend=None, return_lse=True)
            mocked_choose_backend.assert_not_called()
        self.assertTrue(torch.equal(out, v))
        expected_lse = (q.float() * k.float()).sum(-1) * (16**-0.5)
        self.assertTrue(torch.equal(lse, expected_lse))
        grad_out = torch.randn_like(out)
        out.backward(grad_out)
        self.assertTrue(torch.equal(v.grad, grad_out))
        self.assertTrue(torch.equal(q.grad, torch.zeros_like(q.grad)))
        self.assertTrue(torch.equal(k.grad, torch.zeros_like(k.grad)))

        q = torch.randn(1, 2, 3, 4, 2, 8, device=device).requires_grad_(True)
        k = torch.randn(1, 2, 3, 4, 2, 8, device=device).requires_grad_(True)
        v = torch.randn(1, 2, 3, 4, 2, 8, device=device).requires_grad_(True)
        with mock.patch("natten.functional.choose_backend") as mocked_choose_backend:
            out, lse = na3d(
                q, k, v, kernel_size=(1, 1, 1), backend="cutlass-fna", return_lse=True
            )
            mocked_choose_backend.assert_not_called()
        self.assertTrue(torch.equal(out, v))
        expected_lse = (q.float() * k.float()).sum(-1) * (8**-0.5)
        self.assertTrue(torch.equal(lse, expected_lse))
        grad_out = torch.randn_like(out)
        out.backward(grad_out)
        self.assertTrue(torch.equal(v.grad, grad_out))
        self.assertTrue(torch.equal(q.grad, torch.zeros_like(q.grad)))
        self.assertTrue(torch.equal(k.grad, torch.zeros_like(k.grad)))

    @skip_if_libnatten_is_not_supported()
    def test_requires_grad_parity(self):
        # Only `query` requires grad; output/logsumexp must still require grad through the
        # zero-weighted link to key/value.
        device = "cuda"
        q = torch.randn(2, 7, 2, 16, device=device, requires_grad=True)
        k = torch.randn(2, 7, 2, 16, device=device, requires_grad=False)
        v = torch.randn(2, 7, 2, 16, device=device, requires_grad=False)
        out, lse = na1d(q, k, v, kernel_size=(1,), backend=None, return_lse=True)
        self.assertTrue(out.requires_grad)
        self.assertTrue(lse.requires_grad)

        # None of q/k/v require grad: output/logsumexp must not either.
        q2 = torch.randn(2, 7, 2, 16, device=device, requires_grad=False)
        k2 = torch.randn(2, 7, 2, 16, device=device, requires_grad=False)
        v2 = torch.randn(2, 7, 2, 16, device=device, requires_grad=False)
        out2, lse2 = na1d(q2, k2, v2, kernel_size=(1,), backend=None, return_lse=True)
        self.assertFalse(out2.requires_grad)
        self.assertFalse(lse2.requires_grad)


# -----------------------------------------------------------------------------------------------
# Errors
# -----------------------------------------------------------------------------------------------


class DegenerateAxisErrorTest(unittest.TestCase):
    """All of these raise before any backend or device is touched, so plain CPU tensors and no
    gating decorator are enough.
    """

    def _make_qkv(self, shape=(1, 4, 6, 2, 8)):
        q = torch.randn(*shape)
        k = torch.randn(*shape)
        v = torch.randn(*shape)
        return q, k, v

    def test_tile_knobs_with_degenerate_axis_raise(self):
        q, k, v = self._make_qkv()
        with self.assertRaises(ValueError):
            na2d(q, k, v, kernel_size=(1, 4), q_tile_shape=(1, 4))
        with self.assertRaises(ValueError):
            na2d(q, k, v, kernel_size=(1, 4), kv_tile_shape=(1, 4))
        with self.assertRaises(ValueError):
            na2d(q, k, v, kernel_size=(1, 4), backward_q_tile_shape=(1, 4))
        with self.assertRaises(ValueError):
            na2d(q, k, v, kernel_size=(1, 4), backward_kv_tile_shape=(1, 4))
        with self.assertRaises(ValueError):
            na2d(q, k, v, kernel_size=(1, 4), backward_kv_splits=(1, 1))

        # Sanity: the same tile knob does NOT raise without a degenerate axis.
        out = na2d(q, k, v, kernel_size=(4, 4), q_tile_shape=None)
        self.assertEqual(out.shape, q.shape)

    def test_additional_kv_with_degenerate_axis_raises(self):
        q, k, v = self._make_qkv()
        add_k = torch.randn(1, 3, 2, 8)
        add_v = torch.randn(1, 3, 2, 8)
        with self.assertRaises(ValueError):
            na2d(
                q,
                k,
                v,
                kernel_size=(1, 4),
                additional_keys=add_k,
                additional_values=add_v,
            )

        # Sanity: the same additional context does NOT raise without a degenerate axis.
        out = na2d(
            q, k, v, kernel_size=(4, 4), additional_keys=add_k, additional_values=add_v
        )
        self.assertEqual(out.shape, q.shape)

    def test_attention_kwargs_with_degenerate_axis_raises(self):
        q, k, v = self._make_qkv()
        with self.assertRaises(ValueError):
            na2d(
                q,
                k,
                v,
                kernel_size=(1, 4),
                attention_kwargs={"backend": "cutlass-fmha"},
            )

    def test_kernel_size_zero_rejected(self):
        q, k, v = self._make_qkv()
        with self.assertRaises(ValueError):
            na2d(q, k, v, kernel_size=(0, 4))
        with self.assertRaises(ValueError):
            check_kernel_size_arg(2, (0, 4), allow_ones=True)
        with self.assertRaises(ValueError):
            check_kernel_size_arg(2, (0, 4), allow_ones=False)

    def test_kernel_size_negative_rejected(self):
        q, k, v = self._make_qkv()
        with self.assertRaises(ValueError):
            na2d(q, k, v, kernel_size=(-1, 4))

    def test_kernel_size_larger_than_axis_rejected(self):
        q, k, v = self._make_qkv(shape=(1, 4, 8, 2, 8))
        with self.assertRaises(ValueError):
            na2d(q, k, v, kernel_size=(1, 20))

    def test_stride_larger_than_degenerate_kernel_rejected(self):
        q, k, v = self._make_qkv()
        with self.assertRaises(ValueError):
            na2d(q, k, v, kernel_size=(1, 4), stride=(2, 1))

    def test_allow_ones_default_still_rejects_ones_elsewhere(self):
        # check_all_args/check_kernel_size_arg default (allow_ones=False) is unchanged: only
        # neighborhood_attention_generic's own pre-lowering normalization passes allow_ones=True.
        with self.assertRaises(ValueError):
            check_kernel_size_arg(2, (1, 4))
        self.assertEqual(check_kernel_size_arg(2, (1, 4), allow_ones=True), (1, 4))


# -----------------------------------------------------------------------------------------------
# torch.compile(fullgraph=True) smoke tests
# -----------------------------------------------------------------------------------------------


class DegenerateAxesCompileTest(unittest.TestCase):
    def setUp(self):
        _reset_everything()
        reset_torch_compile(cache_size_limit=4, recompile_limit=16)

    def tearDown(self):
        _reset_everything()

    def _run(self, fn, shape):
        device = "cuda"
        dtype = torch.float32

        q = torch.randn(*shape, device=device, dtype=dtype).requires_grad_(True)
        k = torch.randn(*shape, device=device, dtype=dtype).requires_grad_(True)
        v = torch.randn(*shape, device=device, dtype=dtype).requires_grad_(True)
        out_eager, lse_eager = fn(q, k, v)

        q_c = q.detach().clone().requires_grad_(True)
        k_c = k.detach().clone().requires_grad_(True)
        v_c = v.detach().clone().requires_grad_(True)
        compiled = torch.compile(fn, fullgraph=True, backend="inductor")
        out_c, lse_c = compiled(q_c, k_c, v_c)

        torch.testing.assert_close(out_c, out_eager, atol=1e-5, rtol=0)
        torch.testing.assert_close(lse_c, lse_eager, atol=1e-5, rtol=0)

        grad_out = torch.randn_like(out_eager) * 0.1
        out_eager.backward(grad_out)
        out_c.backward(grad_out.detach().clone())
        torch.testing.assert_close(q.grad, q_c.grad, atol=1e-4, rtol=0)
        torch.testing.assert_close(k.grad, k_c.grad, atol=1e-4, rtol=0)
        torch.testing.assert_close(v.grad, v_c.grad, atol=1e-4, rtol=0)

        # Second call: must not recompile or crash (accumulated_recompile_limit is set to fail).
        out_c2, lse_c2 = compiled(q_c, k_c, v_c)
        self.assertEqual(out_c2.shape, out_eager.shape)

    @skip_if_libnatten_is_not_supported()
    def test_compile_leading_fold(self):
        def call(q, k, v):
            return na3d(
                q, k, v, kernel_size=(1, 4, 4), backend="cutlass-fna", return_lse=True
            )

        self._run(call, (2, 3, 10, 10, 2, 16))

    @skip_if_libnatten_is_not_supported()
    def test_compile_permute(self):
        def call(q, k, v):
            return na3d(
                q, k, v, kernel_size=(4, 1, 1), backend="cutlass-fna", return_lse=True
            )

        self._run(call, (2, 10, 6, 7, 2, 16))

    @skip_if_libnatten_is_not_supported()
    def test_compile_identity(self):
        def call(q, k, v):
            return na2d(
                q, k, v, kernel_size=(1, 1), backend="cutlass-fna", return_lse=True
            )

        self._run(call, (2, 5, 6, 2, 16))


if __name__ == "__main__":
    unittest.main()
