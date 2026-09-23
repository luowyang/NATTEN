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

import random
import unittest

import torch
from natten.functional import na1d, na2d
from natten.utils.testing import (
    skip_if_libnatten_is_not_supported,
    skip_if_not_running_extended_tests,
    supports_bfloat16,
)

from .utils import logger

# CUDA caps gridDim.z, where fixed-shape CUTLASS FNA puts batch, at this value.
# It is also the chunk size the library launches with, and therefore the chunk
# size a caller would have to use to get the same launches by hand.
GRID_Z_LIMIT = 65535


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
    torch.cuda.empty_cache()
    torch.use_deterministic_algorithms(False)


def _bitwise_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    # torch.equal compares values, so it calls two different NaN encodings equal
    # and misses a sign difference on zero. Compare the encodings instead.
    assert a.shape == b.shape and a.dtype == b.dtype
    return torch.equal(
        a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
    )


class LargeBatchFNATest(unittest.TestCase):
    """Batches past CUDA's gridDim.z limit against a caller that chunks by hand.

    Fixed-shape CUTLASS FNA launches one grid per chunk of at most GRID_Z_LIMIT
    batches. Every block's work is fixed by the (batch, head, dilation, query
    tile) it lands on, so chunking is expected to be bit-for-bit invisible: the
    reference here is the same call issued by a caller who chunks the batch
    itself, which is what downstream code had to do before the library did.
    """

    def setUp(self):
        _reset_everything()

    def tearDown(self):
        _reset_everything()

    def _run_and_grad(self, na_op, q, k, v, d_out, kernel_size, **kwargs):
        q = q.clone().requires_grad_(True)
        k = k.clone().requires_grad_(True)
        v = v.clone().requires_grad_(True)
        out, lse = na_op(
            q,
            k,
            v,
            kernel_size=kernel_size,
            backend="cutlass-fna",
            return_lse=True,
            **kwargs,
        )
        out.backward(d_out)
        assert q.grad is not None and k.grad is not None and v.grad is not None
        return out.detach(), lse.detach(), q.grad, k.grad, v.grad

    def _chunked_reference(self, na_op, q, k, v, d_out, kernel_size, **kwargs):
        # Fresh leaves per chunk, so every gradient tensor is written by exactly
        # one backward launch and nothing is accumulated into.
        outs, lses, dqs, dks, dvs = [], [], [], [], []
        for start in range(0, q.shape[0], GRID_Z_LIMIT):
            stop = min(start + GRID_Z_LIMIT, q.shape[0])
            out, lse, dq, dk, dv = self._run_and_grad(
                na_op,
                q[start:stop],
                k[start:stop],
                v[start:stop],
                d_out[start:stop],
                kernel_size,
                **kwargs,
            )
            outs.append(out)
            lses.append(lse)
            dqs.append(dq)
            dks.append(dk)
            dvs.append(dv)
        return (
            torch.cat(outs, dim=0),
            torch.cat(lses, dim=0),
            torch.cat(dqs, dim=0),
            torch.cat(dks, dim=0),
            torch.cat(dvs, dim=0),
        )

    def _make_inputs(self, batch, input_shape, heads, head_dim, dtype):
        shape = (batch, *input_shape, heads, head_dim)
        with torch.no_grad():
            q = torch.randn(shape, device="cuda", dtype=dtype)
            k = torch.randn(shape, device="cuda", dtype=dtype)
            v = torch.randn(shape, device="cuda", dtype=dtype)
            d_out = torch.randn(shape, device="cuda", dtype=dtype) * 0.05
        return q, k, v, d_out

    def _test_bitwise_against_chunked_caller(
        self, na_op, batch, input_shape, kernel_size, dtype
    ):
        heads, head_dim = 1, 32
        torch.use_deterministic_algorithms(True)
        torch.cuda.reset_peak_memory_stats()

        q, k, v, d_out = self._make_inputs(batch, input_shape, heads, head_dim, dtype)

        out, lse, dq, dk, dv = self._run_and_grad(na_op, q, k, v, d_out, kernel_size)
        ref = self._chunked_reference(na_op, q, k, v, d_out, kernel_size)
        out_ref, lse_ref, dq_ref, dk_ref, dv_ref = ref

        logger.info(
            f"na{len(input_shape)}d {batch=} {input_shape=} {dtype=}: peak memory "
            f"{torch.cuda.max_memory_allocated() / (1 << 30):.2f} GiB"
        )

        self.assertTrue(_bitwise_equal(out, out_ref), "output is not bitwise equal")
        self.assertTrue(_bitwise_equal(lse, lse_ref), "logsumexp is not bitwise equal")
        self.assertTrue(_bitwise_equal(dq, dq_ref), "dQ is not bitwise equal")
        self.assertTrue(_bitwise_equal(dk, dk_ref), "dK is not bitwise equal")
        self.assertTrue(_bitwise_equal(dv, dv_ref), "dV is not bitwise equal")

    def _dtypes(self):
        dtypes = [torch.float32]
        if supports_bfloat16(torch.device("cuda")):
            dtypes.append(torch.bfloat16)
        return dtypes

    # The other end of the chunk loop: no chunk at all. CUDA rejects a
    # gridDim.z of 0, so this was an error before the loop existed, and it
    # stays one.
    @skip_if_libnatten_is_not_supported()
    def test_empty_batch_is_rejected(self):
        for na_op, input_shape, kernel_size in (
            (na1d, (32,), (5,)),
            (na2d, (4, 8), (3, 3)),
        ):
            for dtype in self._dtypes():
                with self.subTest(op=na_op.__name__, dtype=dtype):
                    q, k, v, _ = self._make_inputs(0, input_shape, 1, 32, dtype)
                    with self.assertRaisesRegex(RuntimeError, "non-empty batch"):
                        na_op(
                            q,
                            k,
                            v,
                            kernel_size=kernel_size,
                            backend="cutlass-fna",
                        )

    # The batches that fit in one launch, and the smallest that does not. These
    # two together are what says "behavior below the limit is unchanged, and the
    # first batch past it agrees with the chunked caller".
    @skip_if_libnatten_is_not_supported()
    def test_1d_at_and_past_grid_z_limit(self):
        for batch in (65535, 65536):
            for dtype in self._dtypes():
                with self.subTest(batch=batch, dtype=dtype):
                    _reset_everything()
                    self._test_bitwise_against_chunked_caller(
                        na1d, batch, (32,), (5,), dtype
                    )

    @skip_if_libnatten_is_not_supported()
    def test_1d_uneven_last_chunk(self):
        for dtype in self._dtypes():
            with self.subTest(dtype=dtype):
                _reset_everything()
                self._test_bitwise_against_chunked_caller(
                    na1d, 70000, (32,), (5,), dtype
                )

    @skip_if_libnatten_is_not_supported()
    @skip_if_not_running_extended_tests()
    def test_1d_chunk_boundaries(self):
        # 131070 is exactly two chunks; 131071 adds a third chunk of one batch.
        for batch in (131070, 131071):
            for dtype in self._dtypes():
                with self.subTest(batch=batch, dtype=dtype):
                    _reset_everything()
                    self._test_bitwise_against_chunked_caller(
                        na1d, batch, (32,), (5,), dtype
                    )

    @skip_if_libnatten_is_not_supported()
    @skip_if_not_running_extended_tests()
    def test_2d(self):
        for batch in (65535, 65536, 70000, 131070, 131071):
            for dtype in self._dtypes():
                with self.subTest(batch=batch, dtype=dtype):
                    _reset_everything()
                    self._test_bitwise_against_chunked_caller(
                        na2d, batch, (4, 8), (3, 3), dtype
                    )

    @skip_if_libnatten_is_not_supported()
    @skip_if_not_running_extended_tests()
    def test_1d_kv_parallel_backward(self):
        # KV parallelism makes dQ's accumulation order depend on scheduling, so
        # this one compares within tolerance. It is here to show that splitting
        # the batch across launches does not disturb the split-key slots of the
        # per-(batch, head, dilation) workspace, which is allocated per chunk.
        if not supports_bfloat16(torch.device("cuda")):
            self.skipTest("bfloat16 is not supported on this device.")

        batch, input_shape, kernel_size = 65536, (128,), (5,)
        heads, head_dim = 1, 32
        # 128 queries over a 64-wide KV tile is two tiles, which is the minimum
        # that admits two KV splits (Kernel::check_supported, kernel_backward.h).
        kv_split_kwargs = dict(
            backward_q_tile_shape=(64,),
            backward_kv_tile_shape=(64,),
            backward_kv_splits=(2,),
        )
        torch.cuda.reset_peak_memory_stats()

        q, k, v, d_out = self._make_inputs(
            batch, input_shape, heads, head_dim, torch.bfloat16
        )

        out, lse, dq, dk, dv = self._run_and_grad(
            na1d, q, k, v, d_out, kernel_size, **kv_split_kwargs
        )
        ref = self._chunked_reference(
            na1d, q, k, v, d_out, kernel_size, **kv_split_kwargs
        )
        out_ref, lse_ref, dq_ref, dk_ref, dv_ref = ref

        logger.info(
            f"na1d kv-parallel {batch=} {input_shape=}: peak memory "
            f"{torch.cuda.max_memory_allocated() / (1 << 30):.2f} GiB"
        )

        # The forward does not split keys, so it is still bitwise.
        self.assertTrue(_bitwise_equal(out, out_ref), "output is not bitwise equal")
        self.assertTrue(_bitwise_equal(lse, lse_ref), "logsumexp is not bitwise equal")
        torch.testing.assert_close(dq.float(), dq_ref.float(), atol=1e-2, rtol=0)
        torch.testing.assert_close(dk.float(), dk_ref.float(), atol=1e-2, rtol=0)
        torch.testing.assert_close(dv.float(), dv_ref.float(), atol=1e-2, rtol=0)


if __name__ == "__main__":
    unittest.main()
