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
"""The self-attention shortcut in the public `na{1,2,3}d` entry points is taken
only when the caller names no `backend`.

A neighborhood attention problem whose window covers a whole axis is equivalent
to (causal, for 1-D) self attention, so `neighborhood_attention_generic` can
answer it with `natten.attention` and one of the FMHA backends. That is a
different kernel family from any `backend=` names, and the two agree only to
within rounding, so a caller who names a backend gets that backend's kernel.

Oracle for a named backend: `na{1,2,3}d_varlen` on a single uniform document,
whose uniform-layout dispatch reaches `cutlass_fna_generic` directly, with no
shortcut of its own (see tests/test_varlen_uniform_dispatch.py). Oracle for an
unnamed backend: `natten.attention` on the same tensors, which is what the
shortcut itself calls.
"""

import unittest
from typing import Tuple
from unittest import mock

import natten
import torch
from natten.utils.testing import skip_if_libnatten_is_not_supported

from .utils import _dtype_is_supported, _prod, _set_deterministic

# The two temporal attention layers of a packed video encoder, where one video
# sample is one uniform document: the window covers the whole time axis, and the
# spatial axes are degenerate. This is the shape that first showed the two entry
# points disagreeing in BF16.
_FULL_WINDOW_3D = (
    ((5, 32, 40), (5, 1, 1), (True, False, False)),
    ((3, 16, 20), (3, 1, 1), (False, False, False)),
)

_HEADS = 3
_HEAD_DIM = 64


def _leaves(
    total: int, dtype: torch.dtype, count: int = 3
) -> Tuple[torch.Tensor, ...]:
    return tuple(
        torch.randn(total, _HEADS, _HEAD_DIM, device="cuda", dtype=dtype)
        for _ in range(count)
    )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
class SelfAttentionBackendHintTests(unittest.TestCase):
    def tearDown(self):
        torch.use_deterministic_algorithms(False)

    def _assert_bitwise(self, observed, expected, name: str) -> None:
        self.assertTrue(torch.equal(observed, expected), msg=f"{name} mismatch")

    @skip_if_libnatten_is_not_supported()
    def test_named_backend_runs_the_named_kernel_family(self):
        for dtype in (torch.bfloat16, torch.float32):
            if not _dtype_is_supported(dtype):
                continue
            for shape, kernel_size, is_causal in _FULL_WINDOW_3D:
                with self.subTest(dtype=dtype, shape=shape):
                    previous = _set_deterministic(True)
                    try:
                        torch.manual_seed(7100 + shape[0])
                        total = _prod(shape)
                        inputs = _leaves(total, dtype)
                        query, key, value = (
                            tensor.detach().clone().requires_grad_(True)
                            for tensor in inputs
                        )
                        reference = tuple(
                            tensor.detach().clone().requires_grad_(True)
                            for tensor in inputs
                        )

                        with mock.patch(
                            "natten.functional.attention",
                            wraps=natten.functional.attention,
                        ) as shortcut, mock.patch(
                            "natten.functional.cutlass_fna_generic",
                            wraps=natten.functional.cutlass_fna_generic,
                        ) as fna:
                            output, logsumexp = natten.na3d(
                                *(
                                    tensor.view(
                                        1, *shape, _HEADS, _HEAD_DIM
                                    )
                                    for tensor in reference
                                ),
                                kernel_size=kernel_size,
                                is_causal=is_causal,
                                backend="cutlass-fna",
                                return_lse=True,
                            )
                        shortcut.assert_not_called()
                        fna.assert_called_once()

                        layout = natten.VarlenLayout((shape,), device="cuda")
                        output_ref, logsumexp_ref = natten.na3d_varlen(
                            query,
                            key,
                            value,
                            layout,
                            kernel_size=kernel_size,
                            is_causal=is_causal,
                            return_lse=True,
                        )

                        output = output.reshape(total, _HEADS, _HEAD_DIM)
                        logsumexp = logsumexp.reshape(total, _HEADS)
                        self._assert_bitwise(output, output_ref, "output")
                        self._assert_bitwise(logsumexp, logsumexp_ref, "lse")

                        gradient = torch.randn_like(output)
                        output.backward(gradient)
                        output_ref.backward(gradient)
                        self._assert_bitwise(reference[0].grad, query.grad, "dq")
                        self._assert_bitwise(reference[1].grad, key.grad, "dk")
                        self._assert_bitwise(reference[2].grad, value.grad, "dv")
                    finally:
                        torch.use_deterministic_algorithms(previous)

    @skip_if_libnatten_is_not_supported()
    def test_unnamed_backend_keeps_the_shortcut(self):
        # 1-D, window over the whole extent, no degenerate axis to lower away:
        # the shortcut's own call is then `attention` on these very tensors, so
        # it doubles as the oracle.
        extent = 32
        for is_causal in (False, True):
            with self.subTest(is_causal=is_causal):
                torch.manual_seed(7200)
                query, key, value = (
                    tensor.view(1, extent, _HEADS, _HEAD_DIM)
                    for tensor in _leaves(extent, torch.bfloat16)
                )

                with mock.patch(
                    "natten.functional.attention",
                    wraps=natten.functional.attention,
                ) as shortcut:
                    output, logsumexp = natten.na1d(
                        query,
                        key,
                        value,
                        kernel_size=(extent,),
                        is_causal=(is_causal,),
                        return_lse=True,
                    )
                shortcut.assert_called_once()

                output_ref, logsumexp_ref = natten.attention(
                    query,
                    key,
                    value,
                    is_causal=is_causal,
                    scale=_HEAD_DIM**-0.5,
                    return_lse=True,
                )
                self._assert_bitwise(output, output_ref, "output")
                self._assert_bitwise(logsumexp, logsumexp_ref, "lse")

    @skip_if_libnatten_is_not_supported()
    def test_partial_window_with_named_backend_is_unchanged(self):
        # Control: the same call one kernel element short of the full window was
        # never eligible for the shortcut, so naming a backend changes nothing
        # about it.
        shape, kernel_size, is_causal = (5, 32, 40), (4, 1, 1), (True, False, False)
        previous = _set_deterministic(True)
        try:
            torch.manual_seed(7300)
            total = _prod(shape)
            inputs = _leaves(total, torch.bfloat16)
            query, key, value = (
                tensor.detach().clone().requires_grad_(True) for tensor in inputs
            )
            reference = tuple(
                tensor.detach().clone().requires_grad_(True) for tensor in inputs
            )

            with mock.patch(
                "natten.functional.attention",
                wraps=natten.functional.attention,
            ) as shortcut, mock.patch(
                "natten.functional.cutlass_fna_generic",
                wraps=natten.functional.cutlass_fna_generic,
            ) as fna:
                output, logsumexp = natten.na3d(
                    *(
                        tensor.view(1, *shape, _HEADS, _HEAD_DIM)
                        for tensor in reference
                    ),
                    kernel_size=kernel_size,
                    is_causal=is_causal,
                    backend="cutlass-fna",
                    return_lse=True,
                )
            shortcut.assert_not_called()
            fna.assert_called_once()

            layout = natten.VarlenLayout((shape,), device="cuda")
            output_ref, logsumexp_ref = natten.na3d_varlen(
                query,
                key,
                value,
                layout,
                kernel_size=kernel_size,
                is_causal=is_causal,
                return_lse=True,
            )
            self._assert_bitwise(
                output.reshape(total, _HEADS, _HEAD_DIM), output_ref, "output"
            )
            self._assert_bitwise(
                logsumexp.reshape(total, _HEADS), logsumexp_ref, "lse"
            )
        finally:
            torch.use_deterministic_algorithms(previous)
