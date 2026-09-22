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

import math
import random
import unittest
from itertools import product

import pytest
import torch
from natten._environment import _NUM_RAND_SWEEP_TESTS as RAND_SWEEP_TESTS
from natten.backends.configs.cutlass_hopper import (
    get_all_backward_configs,
    get_all_forward_configs,
)
from natten.functional import na1d
from natten.utils.testing import (
    skip_if_hopper_kernels_not_supported,
    skip_if_libnatten_is_not_supported,
    skip_if_not_running_extended_tests,
    supports_float16,
)

from .utils import logger, NattenBackendTester


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


class HopperFNABackendTest(unittest.TestCase):
    def setUp(self):
        _reset_everything()

    def tearDown(self):
        _reset_everything()

    def _test_all_dtypes_against_cutlass_2x_fna(
        self,
        batch,
        heads,
        head_dim,
        input_shape,
        kernel_size,
        stride,
        dilation,
        is_causal=None,
        configs_to_test=None,
        heads_kv=None,
    ):
        torch.set_default_device("cuda")
        assert isinstance(input_shape, tuple)
        na_dim = len(input_shape)
        assert na_dim in [1, 2, 3], "Only supports NA1D, 2D, 3D."

        tester = NattenBackendTester(
            batch=batch,
            heads=heads,
            heads_kv=heads_kv,
            head_dim=head_dim,
            input_shape=input_shape,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            is_causal=is_causal,
            test_backprop=True,
            reference_backend="cutlass-fna",
            reference_fmha_backend="cutlass-fmha",
            dtype=torch.float32,
        )

        ALLOWED_DTYPES_DEFAULT = [
            (torch.float16, (1e-2, (1e-2, 1e-2, 1e-2))),
            (torch.bfloat16, (5e-2, (1e-2, 1e-2, 1e-2))),
        ]

        ALLOWED_DTYPES_GQA = [
            (torch.float16, (1e-2, (3e-2, 4e-2, 4e-2))),
            (torch.bfloat16, (5e-2, (3e-2, 4e-2, 4e-2))),
        ]

        ALLOWED_DTYPES = (
            ALLOWED_DTYPES_DEFAULT
            if heads_kv is None or heads_kv == heads
            else ALLOWED_DTYPES_GQA
        )

        test_id = 0
        for dtype, atol in ALLOWED_DTYPES:

            dummy = torch.randn(
                (batch, *input_shape, heads, head_dim), device="cuda", dtype=dtype
            )
            forward_configs = get_all_forward_configs(dummy)
            backward_configs = get_all_backward_configs(dummy)
            assert len(forward_configs) > 0
            assert len(backward_configs) > 0

            random.shuffle(forward_configs)
            random.shuffle(backward_configs)

            for i in range(max(len(forward_configs), len(backward_configs))):
                (q_tile_shape, kv_tile_shape), kernel_schedule = forward_configs[
                    i % len(forward_configs)
                ]
                backward_q_tile_shape, backward_kv_tile_shape = backward_configs[
                    i % len(backward_configs)
                ]

                tester.test(
                    eps=atol,
                    dtype=dtype,
                    target_backend="hopper-fna",
                    target_fmha_backend="hopper-fmha",
                    q_tile_shape=q_tile_shape,
                    kv_tile_shape=kv_tile_shape,
                    backward_q_tile_shape=backward_q_tile_shape,
                    backward_kv_tile_shape=backward_kv_tile_shape,
                    kernel_schedule=kernel_schedule,
                )
                test_id += 1
                if configs_to_test is not None and test_id > configs_to_test:
                    return

    @skip_if_libnatten_is_not_supported()
    @skip_if_hopper_kernels_not_supported()
    def test_1d_against_cutlass_2x(self):
        problem_sizes = [
            (2, 4, 4, 128, (128,), (63,), (31,), (1)),
            (2, 4, 2, 128, (128,), (63,), (31,), (1)),
            (2, 4, 1, 128, (128,), (63,), (31,), (1)),
            (4, 3, 3, 128, (256,), (255,), (82,), (1)),
            (4, 3, 1, 128, (256,), (255,), (82,), (1)),
            (1, 1, 1, 128, (128,), (8,), (1,), (1)),
            (1, 1, 1, 128, (256,), (160,), (1,), (1)),
            (1, 1, 1, 128, (256,), (150,), (1,), (1)),
            (1, 1, 1, 128, (256,), (96,), (1,), (1)),
            (1, 1, 1, 128, (256,), (8,), (1,), (1)),
            (1, 1, 1, 128, (512,), (8,), (1,), (1)),
            (1, 1, 1, 128, (32768,), (63,), (1,), (4,)),
            (1, 1, 1, 128, (32768,), (63,), (1,), (8,)),
            (1, 1, 1, 128, (32768,), (63,), (1,), (16,)),
            (1, 1, 1, 128, (32768,), (2048,), (1,), (1)),
            (1, 1, 1, 128, (32768,), (2048,), (128,), (1)),
            (1, 1, 1, 128, (32768,), (2048,), (256,), (1)),
            (1, 1, 1, 128, (32768,), (2048,), (2048,), (1)),
            (1, 1, 1, 128, (64,), (8,), (1,), (1)),
            (1, 2, 2, 128, (128,), (8,), (1,), (1)),
            (1, 1, 1, 32, (64,), (8,), (1,), (1)),
            (1, 2, 2, 32, (64,), (8,), (1,), (1)),
            (1, 2, 2, 32, (64,), (3,), (1,), (1)),
            (1, 2, 2, 32, (64,), (3,), (1,), (1)),
            (1, 1, 1, 32, (64,), (3,), (1,), (1)),
            (1, 1, 1, 32, (67,), (8,), (1,), (1)),
            (1, 1, 1, 32, (69,), (12,), (1,), (1)),
            (1, 1, 1, 32, (125,), (12,), (1,), (1)),
            (1, 1, 1, 32, (17,), (12,), (1,), (1)),
            (1, 1, 1, 32, (128,), (3,), (1,), (2)),
            (1, 1, 1, 32, (128,), (3,), (2,), (5)),
            (1, 1, 1, 128, (256,), (128,), (128,), (1)),
            (1, 1, 1, 128, (256,), (136,), (136,), (1)),
            (1, 1, 1, 128, (256,), (128,), (127,), (1)),
            (1, 1, 1, 32, (256,), (4,), (1,), (1)),
            (1, 1, 1, 128, (256,), (3,), (1,), (1)),
            (1, 1, 1, 32, (256,), (64,), (1,), (1)),
            (1, 1, 1, 32, (256,), (128,), (1,), (1)),
            (1, 1, 1, 128, (256,), (128,), (128,), (1)),
            (1, 1, 1, 32, (128,), (127,), (1,), (1)),
            (1, 1, 1, 32, (128,), (3,), (1,), (1)),
            (1, 2, 2, 64, (128,), (15,), (1,), (1)),
            (1, 1, 1, 32, (128,), (3,), (2,), (10)),
            (1, 1, 1, 64, (128,), (8,), (7,), (5)),
            (1, 1, 1, 128, (128,), (61,), (33,), (1)),
            (1, 1, 1, 32, (125,), (3,), (1,), (1)),
            (1, 2, 2, 64, (125,), (15,), (1,), (1)),
            (1, 1, 1, 128, (256,), (3,), (2,), (10)),
        ]
        for i, (
            batch,
            heads,
            heads_kv,
            head_dim,
            input_shape,
            kernel_size,
            stride,
            dilation,
        ) in enumerate(problem_sizes):
            _reset_everything(random_seed=i, torch_seed=i)
            for causal in [True, False]:
                is_causal = (causal,)
                self._test_all_dtypes_against_cutlass_2x_fna(
                    batch=batch,
                    heads=heads,
                    heads_kv=heads_kv,
                    head_dim=head_dim,
                    input_shape=input_shape,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=dilation,
                    is_causal=is_causal,
                )

    @skip_if_libnatten_is_not_supported()
    @skip_if_hopper_kernels_not_supported()
    def test_2d_against_cutlass_2x(self):
        problem_sizes = [
            (1, 2, 1, 32, (32, 16), (3, 3), (1, 1), (1, 1)),
            (1, 3, 1, 32, (5, 25), (5, 16), (1, 16), (1, 1)),
            (1, 1, 1, 32, (84, 19), (7, 3), (1, 1), (5, 1)),
            (1, 1, 1, 128, (19, 29), (8, 8), (1, 1), (2, 3)),
            (1, 1, 1, 128, (48, 17), (24, 16), (1, 1), (2, 1)),
            (1, 1, 1, 128, (67, 80), (12, 7), (8, 4), (5, 11)),
            (1, 1, 1, 128, (8, 8), (8, 8), (1, 1), (1, 1)),
            (1, 1, 1, 128, (33, 33), (24, 16), (1, 1), (1, 1)),
            (1, 1, 1, 32, (16, 16), (16, 16), (1, 1), (1, 1)),
            (1, 1, 1, 128, (44, 80), (44, 80), (1, 1), (1, 1)),
            (1, 1, 1, 32, (40, 20), (3, 7), (1, 1), (1, 1)),
            (1, 1, 1, 32, (16, 16), (3, 3), (1, 1), (1, 1)),
            (1, 1, 1, 128, (44, 80), (9, 10), (1, 1), (1, 1)),
            (1, 1, 1, 64, (28, 40), (17, 31), (1, 1), (1, 1)),
            (1, 1, 1, 64, (36, 40), (36, 40), (12, 13), (1, 1)),
            (1, 1, 1, 128, (44, 80), (44, 80), (4, 8), (1, 1)),
        ]
        for i, (
            batch,
            heads,
            heads_kv,
            head_dim,
            input_shape,
            kernel_size,
            stride,
            dilation,
        ) in enumerate(problem_sizes):
            _reset_everything(random_seed=i, torch_seed=i)
            for causal_x, causal_y in product([False, True], [False, True]):
                is_causal = (causal_x, causal_y)
                self._test_all_dtypes_against_cutlass_2x_fna(
                    batch=batch,
                    heads=heads,
                    heads_kv=heads_kv,
                    head_dim=head_dim,
                    input_shape=input_shape,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=dilation,
                    is_causal=is_causal,
                )

    @skip_if_not_running_extended_tests()
    @skip_if_libnatten_is_not_supported()
    @skip_if_hopper_kernels_not_supported()
    def test_2d_against_cutlass_2x_extended(self):
        problem_sizes = [
            (1, 2, 1, 32, (5, 25), (5, 16), (1, 16), (1, 1)),
            (1, 2, 2, 32, (5, 25), (5, 16), (4, 16), (1, 1)),
            (1, 1, 1, 32, (84, 69), (7, 68), (1, 1), (5, 1)),
            (1, 1, 1, 32, (84, 69), (7, 23), (1, 1), (5, 1)),
            (1, 1, 1, 32, (84, 69), (7, 20), (1, 6), (5, 1)),
            (1, 1, 1, 128, (128, 128), (8, 8), (1, 1), (1, 1)),
            (1, 1, 1, 128, (128, 128), (8, 8), (1, 1), (4, 4)),
            (1, 1, 1, 128, (64, 64), (32, 32), (1, 1), (2, 2)),
            (1, 1, 1, 128, (64, 64), (32, 32), (1, 1), (1, 2)),
            (1, 1, 1, 128, (48, 48), (24, 24), (1, 1), (1, 2)),
            (1, 1, 1, 128, (17, 48), (16, 24), (1, 1), (1, 2)),
            (1, 1, 1, 128, (48, 48), (24, 24), (1, 1), (2, 2)),
            (1, 1, 1, 128, (48, 17), (24, 16), (1, 1), (1, 1)),
            (1, 1, 1, 128, (72, 80), (24, 16), (1, 1), (3, 5)),
            (1, 1, 1, 128, (48, 17), (24, 16), (1, 1), (1, 1)),
            (1, 1, 1, 128, (44, 80), (32, 32), (22, 16), (1, 1)),
            (1, 1, 1, 128, (44, 80), (24, 24), (1, 1), (1, 1)),
            (1, 1, 1, 128, (44, 80), (24, 24), (8, 16), (1, 1)),
            (1, 1, 1, 128, (44, 80), (24, 16), (1, 1), (1, 1)),
            (1, 1, 1, 128, (44, 80), (24, 16), (8, 16), (1, 1)),
            (1, 1, 1, 64, (28, 40), (28, 40), (1, 1), (1, 1)),
            (1, 1, 1, 32, (16, 16), (16, 16), (4, 5), (1, 1)),
            (1, 1, 1, 32, (16, 16), (16, 16), (4, 8), (1, 1)),
            (1, 1, 1, 32, (16, 16), (15, 15), (1, 1), (1, 1)),
            (1, 1, 1, 32, (16, 16), (14, 14), (1, 1), (1, 1)),
            (1, 1, 1, 64, (36, 40), (36, 40), (1, 1), (1, 1)),
            (1, 1, 1, 128, (48, 80), (24, 24), (8, 8), (1, 1)),
        ]
        for i, (
            batch,
            heads,
            heads_kv,
            head_dim,
            input_shape,
            kernel_size,
            stride,
            dilation,
        ) in enumerate(problem_sizes):
            _reset_everything(random_seed=i, torch_seed=i)
            for causal_x, causal_y in product([True, False], [True, False]):
                is_causal = (causal_x, causal_y)
                self._test_all_dtypes_against_cutlass_2x_fna(
                    batch=batch,
                    heads=heads,
                    heads_kv=heads_kv,
                    head_dim=head_dim,
                    input_shape=input_shape,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=dilation,
                    is_causal=is_causal,
                )

    @skip_if_libnatten_is_not_supported()
    @skip_if_hopper_kernels_not_supported()
    def test_3d_against_cutlass_2x(self):
        problem_sizes = [
            (2, 2, 1, 128, (68, 64, 8), (14, 32, 2), (3, 4, 1), (1, 2, 3)),
            (4, 8, 2, 64, (32, 10, 10), (7, 3, 3), (5, 1, 1), (1, 2, 3)),
            (1, 1, 1, 64, (18, 37, 12), (14, 16, 12), (12, 8, 1), (1, 2, 1)),
            (1, 1, 1, 32, (13, 11, 9), (3, 4, 3), (2, 3, 3), (3, 2, 2)),
            (1, 4, 4, 32, (8, 8, 16), (3, 3, 3), (2, 1, 2), (2, 2, 4)),
            (1, 1, 1, 64, (18, 37, 12), (14, 16, 12), (12, 8, 1), (1, 2, 1)),
            (1, 1, 1, 128, (57, 20, 88), (20, 4, 6), (1, 1, 1), (2, 1, 1)),
            (1, 1, 1, 128, (57, 32, 32), (10, 32, 32), (1, 1, 1), (1, 1, 1)),
            (1, 1, 1, 128, (32, 64, 64), (16, 16, 16), (1, 2, 2), (1, 1, 1)),
            (1, 1, 1, 128, (30, 48, 80), (18, 24, 24), (16, 8, 8), (1, 1, 1)),
            (1, 1, 1, 128, (16, 44, 80), (12, 32, 32), (8, 22, 16), (1, 1, 1)),
            (1, 1, 1, 128, (31, 32, 32), (10, 32, 32), (1, 1, 1), (1, 1, 1)),
            (1, 1, 1, 128, (30, 32, 32), (10, 32, 32), (1, 1, 1), (1, 1, 1)),
            (1, 1, 1, 32, (16, 16, 16), (16, 16, 16), (1, 1, 1), (1, 1, 1)),
            (1, 1, 1, 32, (8, 4, 8), (7, 3, 7), (1, 1, 1), (1, 1, 1)),
            (1, 1, 1, 32, (16, 16, 16), (15, 15, 15), (1, 1, 1), (1, 1, 1)),
            (1, 1, 1, 64, (24, 28, 40), (24, 28, 40), (1, 1, 1), (1, 1, 1)),
            (1, 1, 1, 32, (16, 16, 16), (16, 16, 16), (2, 4, 5), (1, 1, 1)),
        ]
        for i, (
            batch,
            heads,
            heads_kv,
            head_dim,
            input_shape,
            kernel_size,
            stride,
            dilation,
        ) in enumerate(problem_sizes):
            _reset_everything(random_seed=i, torch_seed=i)
            for causal_x, causal_y, causal_z in product(
                [True, False], [True, False], [True, False]
            ):
                is_causal = (causal_x, causal_y, causal_z)
                self._test_all_dtypes_against_cutlass_2x_fna(
                    batch=batch,
                    heads=heads,
                    heads_kv=heads_kv,
                    head_dim=head_dim,
                    input_shape=input_shape,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=dilation,
                    is_causal=is_causal,
                )

    @skip_if_not_running_extended_tests()
    @skip_if_libnatten_is_not_supported()
    @skip_if_hopper_kernels_not_supported()
    def test_3d_against_cutlass_2x_extended(self):
        problem_sizes = [
            (1, 12, 3, 64, (14, 8, 8), (7, 5, 5), (2, 1, 3), (2, 1, 1)),
            (1, 12, 4, 64, (14, 8, 8), (7, 5, 5), (2, 1, 3), (2, 1, 1)),
            (1, 12, 6, 64, (14, 8, 8), (7, 5, 5), (2, 1, 3), (2, 1, 1)),
            (1, 12, 12, 64, (32, 8, 8), (7, 5, 5), (2, 1, 3), (2, 1, 1)),
            (1, 1, 1, 64, (18, 37, 12), (14, 16, 12), (12, 8, 1), (1, 1, 1)),
            (1, 1, 1, 32, (13, 11, 9), (3, 4, 3), (1, 1, 1), (3, 2, 2)),
            (1, 1, 1, 32, (8, 8, 4), (3, 4, 3), (1, 1, 1), (1, 1, 1)),
            (2, 2, 2, 32, (8, 8, 10), (3, 4, 3), (3, 4, 1), (1, 1, 1)),
            (1, 2, 2, 32, (8, 8, 12), (5, 8, 11), (2, 3, 4), (1, 1, 1)),
            (1, 1, 1, 64, (18, 37, 12), (14, 16, 12), (12, 8, 1), (1, 1, 1)),
            (2, 3, 3, 32, (5, 45, 73), (2, 7, 32), (1, 4, 32), (2, 2, 2)),
            (1, 1, 1, 128, (57, 20, 88), (19, 4, 6), (13, 1, 6), (2, 2, 7)),
            (2, 3, 3, 128, (57, 20, 88), (19, 4, 6), (13, 1, 6), (2, 2, 7)),
            (1, 1, 1, 128, (32, 64, 64), (16, 16, 16), (2, 1, 1), (2, 2, 3)),
            (1, 1, 1, 128, (61, 61, 61), (10, 10, 10), (1, 1, 1), (1, 1, 2)),
            (1, 1, 1, 128, (61, 61, 61), (10, 10, 10), (1, 1, 1), (1, 2, 1)),
            (1, 1, 1, 128, (61, 61, 61), (10, 10, 10), (1, 1, 1), (2, 2, 2)),
            (1, 1, 1, 128, (32, 64, 64), (16, 16, 16), (2, 1, 1), (1, 1, 1)),
            (1, 1, 1, 128, (32, 64, 64), (16, 16, 16), (1, 1, 2), (1, 1, 1)),
            (1, 1, 1, 32, (16, 16, 16), (16, 16, 16), (8, 4, 8), (1, 1, 1)),
            (1, 1, 1, 64, (24, 36, 40), (24, 36, 40), (10, 12, 13), (1, 1, 1)),
            (1, 1, 1, 128, (16, 44, 80), (16, 44, 80), (8, 4, 8), (1, 1, 1)),
            (1, 1, 1, 128, (30, 44, 80), (18, 24, 24), (1, 1, 1), (1, 1, 1)),
            (1, 1, 1, 128, (30, 44, 80), (18, 24, 24), (2, 8, 16), (1, 1, 1)),
            (1, 1, 1, 128, (30, 44, 80), (24, 24, 16), (1, 1, 1), (1, 1, 1)),
            (1, 1, 1, 128, (30, 44, 80), (24, 24, 16), (2, 8, 16), (1, 1, 1)),
            (1, 1, 1, 32, (16, 16, 16), (14, 14, 14), (1, 1, 1), (1, 1, 1)),
            (1, 1, 1, 32, (16, 16, 16), (3, 3, 3), (1, 1, 1), (1, 1, 1)),
            (1, 1, 1, 128, (16, 44, 80), (8, 9, 10), (1, 1, 1), (1, 1, 1)),
            (1, 1, 1, 64, (24, 28, 40), (11, 17, 31), (1, 1, 1), (1, 1, 1)),
            (1, 1, 1, 64, (24, 36, 40), (24, 36, 40), (1, 1, 1), (1, 1, 1)),
            (1, 1, 1, 128, (16, 44, 80), (16, 44, 80), (1, 1, 1), (1, 1, 1)),
            (1, 1, 1, 128, (30, 48, 18), (30, 48, 18), (1, 1, 1), (1, 1, 1)),
            (1, 1, 1, 128, (8, 8, 8), (8, 8, 8), (1, 1, 1), (1, 1, 1)),
            (1, 1, 1, 128, (30, 48, 17), (18, 24, 16), (1, 1, 1), (1, 1, 1)),
            (1, 1, 1, 128, (33, 33, 33), (18, 24, 16), (1, 1, 1), (1, 1, 1)),
            (1, 1, 1, 128, (30, 48, 18), (3, 2, 8), (1, 1, 1), (8, 18, 2)),
            (1, 1, 1, 128, (61, 32, 32), (10, 32, 32), (1, 1, 1), (2, 1, 1)),
            (1, 1, 1, 128, (57, 32, 32), (10, 32, 32), (1, 1, 1), (2, 1, 1)),
            (1, 1, 1, 128, (57, 20, 88), (20, 8, 16), (1, 1, 1), (2, 1, 1)),
            (3, 1, 1, 64, (18, 37, 12), (14, 16, 12), (12, 8, 6), (1, 2, 1)),
            (1, 1, 1, 128, (32, 64, 64), (16, 16, 16), (1, 1, 2), (1, 3, 2)),
            (1, 1, 1, 128, (48, 64, 64), (7, 15, 11), (1, 2, 2), (5, 3, 2)),
        ]
        for i, (
            batch,
            heads,
            heads_kv,
            head_dim,
            input_shape,
            kernel_size,
            stride,
            dilation,
        ) in enumerate(problem_sizes):
            _reset_everything(random_seed=i, torch_seed=i)
            for causal_x, causal_y, causal_z in product(
                [True, False], [True, False], [True, False]
            ):
                is_causal = (causal_x, causal_y, causal_z)
                self._test_all_dtypes_against_cutlass_2x_fna(
                    batch=batch,
                    heads=heads,
                    heads_kv=heads_kv,
                    head_dim=head_dim,
                    input_shape=input_shape,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=dilation,
                    is_causal=is_causal,
                )

    def _test_randsweep_against_cutlass_2x(
        self, na_dim, max_tests=1000, configs_to_test=None
    ):
        max_seqlen = 2**17
        # max size per-dim for different profiles
        max_size = {1: 2**15, 2: 128, 3: 96}

        # seqlen limit for freely choosing batch and heads
        seqlen_limit_batched = 2**13

        for i in range(max_tests):
            # to help with reproducibility of use cases
            _reset_everything(random_seed=i, torch_seed=i)

            input_shape = []
            for j in range(na_dim):
                input_shape.append(random.choice(range(4, max_size[na_dim] + 1)))

            while math.prod(input_shape) > max_seqlen:
                dim_to_cut = random.choice(range(na_dim))
                input_shape[dim_to_cut] = max(4, int(input_shape[dim_to_cut] * 0.1))

            input_shape = tuple(input_shape)
            assert math.prod(input_shape) <= max_seqlen

            max_heads = min(
                max(1, (seqlen_limit_batched // math.prod(input_shape)) * 4), 4
            )
            heads = random.choice(range(1, max_heads + 1))

            max_batch = min(
                max(1, ((seqlen_limit_batched // math.prod(input_shape)) * 4) // heads),
                4,
            )
            batch = random.choice(range(1, max_batch + 1))

            heads_kv = random.choice([i for i in range(1, heads + 1) if heads % i == 0])
            head_dim = random.choice([32, 64, 128])

            input_shape = tuple(input_shape)
            kernel_size = tuple(random.choice(range(2, x)) for x in input_shape)
            stride = tuple(random.choice(range(1, k + 1)) for k in kernel_size)
            dilation = tuple(
                random.choice(range(1, x // k + 1))
                for x, k in zip(input_shape, kernel_size)
            )
            is_causal = tuple(random.choice([False, True]) for _ in range(na_dim))

            self._test_all_dtypes_against_cutlass_2x_fna(
                batch=batch,
                heads=heads,
                heads_kv=heads_kv,
                head_dim=head_dim,
                input_shape=input_shape,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                is_causal=is_causal,
                configs_to_test=configs_to_test,
            )

    @skip_if_libnatten_is_not_supported()
    @skip_if_hopper_kernels_not_supported()
    def test_randsweep_1d_against_cutlass_2x_quick(self):
        self._test_randsweep_against_cutlass_2x(1, max_tests=10, configs_to_test=3)

    @skip_if_libnatten_is_not_supported()
    @skip_if_hopper_kernels_not_supported()
    def test_randsweep_2d_against_cutlass_2x_quick(self):
        self._test_randsweep_against_cutlass_2x(2, max_tests=10, configs_to_test=3)

    @skip_if_libnatten_is_not_supported()
    @skip_if_hopper_kernels_not_supported()
    def test_randsweep_3d_against_cutlass_2x_quick(self):
        self._test_randsweep_against_cutlass_2x(3, max_tests=10, configs_to_test=3)

    @skip_if_not_running_extended_tests()
    @skip_if_libnatten_is_not_supported()
    @skip_if_hopper_kernels_not_supported()
    def test_randsweep_1d_against_cutlass_2x(self):
        self._test_randsweep_against_cutlass_2x(1, max_tests=RAND_SWEEP_TESTS)

    @skip_if_not_running_extended_tests()
    @skip_if_libnatten_is_not_supported()
    @skip_if_hopper_kernels_not_supported()
    def test_randsweep_2d_against_cutlass_2x(self):
        self._test_randsweep_against_cutlass_2x(2, max_tests=RAND_SWEEP_TESTS)

    @skip_if_not_running_extended_tests()
    @skip_if_libnatten_is_not_supported()
    @skip_if_hopper_kernels_not_supported()
    def test_randsweep_3d_against_cutlass_2x(self):
        self._test_randsweep_against_cutlass_2x(3, max_tests=RAND_SWEEP_TESTS)


class HopperFNAComputeDeltaRangeTest(unittest.TestCase):
    """The Hopper backward's `dO * O` reduction over FP16 out of half's range.

    FmhaKernelBwdSumOdO accumulates delta in FP32 but forms each product in the
    input element type, so an `O * dO` past 65504 became `Inf` in FP16, and the
    row turned into `NaN` as soon as a masked position contributed `0 * Inf`.
    """

    def setUp(self):
        _reset_everything()

    def tearDown(self):
        _reset_everything()

    @skip_if_libnatten_is_not_supported()
    @skip_if_hopper_kernels_not_supported()
    def test_backward_with_out_of_half_range_products(self):
        if not supports_float16(torch.device("cuda")):
            self.skipTest("float16 is not supported on this device.")

        torch.set_default_device("cuda")
        # An extent of a whole KV tile: at these magnitudes a partial last KV
        # tile hits a separate FP16 range problem in this backward, at the last
        # `kernel_size // 2` queries of dQ, which this test is not about.
        batch, extent, heads, head_dim = 2, 128, 2, 32
        kernel_size = (7,)
        shape = (batch, extent, heads, head_dim)

        with torch.no_grad():
            q_ = torch.randn(shape, dtype=torch.float32) * 0.1
            k_ = torch.randn(shape, dtype=torch.float32) * 0.1
            # Every output lands near 256 and every incident gradient is 512, so
            # every `O * dO` is around 131072 -- twice FP16's largest value.
            v_ = 256.0 + 16.0 * torch.randn(shape, dtype=torch.float32)
            d_out_ = torch.full(shape, 512.0, dtype=torch.float32)

        def run(dtype, backend):
            q = q_.to(dtype).requires_grad_(True)
            k = k_.to(dtype).requires_grad_(True)
            v = v_.to(dtype).requires_grad_(True)
            out = na1d(q, k, v, kernel_size=kernel_size, backend=backend)
            out.backward(d_out_.to(dtype))
            assert q.grad is not None and k.grad is not None and v.grad is not None
            return (
                out.detach().float(),
                q.grad.float(),
                k.grad.float(),
                v.grad.float(),
            )

        out, dq, dk, dv = run(torch.float16, "hopper-fna")
        out_ref, dq_ref, dk_ref, dv_ref = run(torch.float32, "cutlass-fna")

        # The discriminating assertion: before the fix these are NaN/Inf.
        for name, tensor in (("dQ", dq), ("dK", dk), ("dV", dv)):
            self.assertTrue(
                torch.isfinite(tensor).all(), f"{name} is not finite: {tensor}"
            )

        # A check that they are also the right gradients. Values here span four
        # orders of magnitude, so each tensor is compared against its own scale
        # rather than in absolute terms. dQ and dK get a looser bound than the
        # output and dV: they carry `dP - delta`, a difference of two quantities
        # around 4e6 that mostly cancels, so what FP16 inputs leave of it is
        # good to a few percent, while the output and dV never form it.
        for name, tensor, reference, atol in (
            ("out", out, out_ref, 1e-2),
            ("dV", dv, dv_ref, 1e-2),
            ("dQ", dq, dq_ref, 5e-2),
            ("dK", dk, dk_ref, 5e-2),
        ):
            scale = reference.abs().max().clamp(min=1.0)
            torch.testing.assert_close(
                tensor / scale,
                reference / scale,
                atol=atol,
                rtol=atol,
                msg=lambda m, name=name: f"{name}: {m}",
            )


class HopperFNAPartialKVTileRangeTest(unittest.TestCase):
    """The Hopper FP16 backward over a partial last KV tile: dQ turns to NaN.

    Known issue, luowyang/NATTEN#<待填>. Same family as the `dO * O` reduction
    overflow above -- a `0 * Inf` once an intermediate leaves FP16's range --
    but a different site: this one is inside the Hopper backward itself, and
    the FP32 conversion in the reduction does not reach it.

    With a `kernel_size` of 3 or 7 the NaNs land on the last `kernel_size // 2`
    queries of dQ, and only when the extent is not a whole number of KV tiles;
    dK and dV stay finite, and `cutlass-fna` is unaffected. Two independent
    changes make it disappear: an extent of exactly one KV tile (128 for this
    configuration, whose backward KV tile is 128), and magnitudes whose
    `O * dO` stays inside FP16 (`v = 128`, `dO = 256`, product 3.3e4). That is
    why HopperFNAComputeDeltaRangeTest above uses an extent of 128.
    """

    def setUp(self):
        _reset_everything()

    def tearDown(self):
        _reset_everything()

    @pytest.mark.xfail(
        reason="Hopper FP16 backward leaves half's range on a partial last KV "
        "tile; luowyang/NATTEN#<待填>.",
        strict=True,
    )
    @skip_if_libnatten_is_not_supported()
    @skip_if_hopper_kernels_not_supported()
    def test_backward_over_a_partial_last_kv_tile(self):
        if not supports_float16(torch.device("cuda")):
            self.skipTest("float16 is not supported on this device.")

        torch.set_default_device("cuda")
        # 64 is half a KV tile for this configuration, so the last tile is
        # partial.
        batch, extent, heads, head_dim = 2, 64, 2, 32
        kernel_size = (7,)
        shape = (batch, extent, heads, head_dim)

        with torch.no_grad():
            q_ = torch.randn(shape, dtype=torch.float32) * 0.1
            k_ = torch.randn(shape, dtype=torch.float32) * 0.1
            # Every output lands on 256 and every incident gradient is 512, so
            # every `O * dO` is 131072 -- twice FP16's largest value.
            v_ = torch.full(shape, 256.0, dtype=torch.float32)
            d_out_ = torch.full(shape, 512.0, dtype=torch.float32)

        q = q_.half().requires_grad_(True)
        k = k_.half().requires_grad_(True)
        v = v_.half().requires_grad_(True)
        out = na1d(q, k, v, kernel_size=kernel_size, backend="hopper-fna")
        out.backward(d_out_.half())
        assert q.grad is not None

        dq = q.grad.float()
        nan_queries = (
            torch.isnan(dq)
            .any(dim=0)
            .any(dim=-1)
            .any(dim=-1)
            .nonzero()
            .flatten()
            .tolist()
        )
        self.assertTrue(
            torch.isfinite(dq).all(),
            f"dQ is not finite; NaN at queries {nan_queries}",
        )


if __name__ == "__main__":
    unittest.main()
