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

import itertools
import math
from dataclasses import dataclass
from typing import cast, Optional, Tuple, Union

import natten
import torch
from natten.backends import cutlass_fna_generic
from natten.backends.reference import reference_fna_generic
from natten.functional import neighborhood_attention_generic
from natten.types import CausalArgType, DimensionType, KernelSchedule
from natten.utils import log
from natten.utils.checks import check_all_args
from torch import Tensor

logger = log.get_logger("natten_tests")


def reset_torch_compile(cache_size_limit, recompile_limit: int | None = None):
    recompile_limit = recompile_limit or cache_size_limit * 4
    # Torch compile reset and sensible settings for unit testing
    logger.debug(
        f"Resetting torch compile cache: {cache_size_limit=}, {recompile_limit=}."
    )
    torch.compiler.reset()
    torch._dynamo.config.cache_size_limit = cache_size_limit
    torch._dynamo.config.accumulated_recompile_limit = recompile_limit
    torch._dynamo.config.fail_on_recompile_limit_hit = True


# Runs one backend once as a reference, and may run another backend multiple times
# with different configurations.
class NattenBackendTester:
    def __init__(
        self,
        batch: int,
        heads: int,
        head_dim: int,
        input_shape: DimensionType,
        kernel_size: DimensionType,
        stride: DimensionType,
        dilation: DimensionType,
        is_causal: CausalArgType,
        test_backprop: bool,
        reference_backend: str,
        reference_fmha_backend: str,
        dtype: torch.dtype,
        head_dim_v: Optional[int] = None,
        heads_kv: Optional[int] = None,
        reference_q_tile_shape: Optional[DimensionType] = None,
        reference_kv_tile_shape: Optional[DimensionType] = None,
        reference_backward_q_tile_shape: Optional[DimensionType] = None,
        reference_backward_kv_tile_shape: Optional[DimensionType] = None,
        reference_backward_kv_splits: Optional[DimensionType] = None,
        reference_backward_use_pt_reduction: bool = False,
    ):
        assert isinstance(input_shape, tuple)
        na_dim = len(input_shape)
        assert na_dim in [1, 2, 3], "Only supports NA1D, 2D, 3D."

        self.batch = batch
        self.heads = heads
        self.heads_kv = heads_kv or heads
        self.head_dim = head_dim
        self.head_dim_v = head_dim_v or head_dim
        self.input_shape = input_shape
        self.kernel_size, self.stride, self.dilation, self.is_causal = check_all_args(
            na_dim, kernel_size, stride, dilation, is_causal
        )

        self.test_backprop = test_backprop
        self.reference_backend = reference_backend
        self.reference_fmha_backend = reference_fmha_backend

        with torch.no_grad():
            orig_dtype = dtype
            if dtype in [torch.float8_e5m2, torch.float8_e4m3fn]:
                dtype = torch.float16

            q_ref, k_ref, v_ref, d_out_ref = (
                torch.randn(
                    (self.batch, *self.input_shape, self.heads, self.head_dim),
                    device="cuda",
                    dtype=dtype,
                ),
                torch.randn(
                    (self.batch, *self.input_shape, self.heads_kv, self.head_dim),
                    device="cuda",
                    dtype=dtype,
                ),
                torch.randn(
                    (self.batch, *self.input_shape, self.heads_kv, self.head_dim_v),
                    device="cuda",
                    dtype=dtype,
                ),
                torch.randn(
                    (self.batch, *self.input_shape, self.heads, self.head_dim_v),
                    device="cuda",
                    dtype=dtype,
                )
                * 0.05,
            )

            if dtype != orig_dtype:
                q_ref = q_ref.to(orig_dtype)
                k_ref = k_ref.to(orig_dtype)
                v_ref = v_ref.to(orig_dtype)
                d_out_ref = d_out_ref.to(orig_dtype)

            self.q, self.k, self.v, self.d_out = (
                q_ref.clone(),
                k_ref.clone(),
                v_ref.clone(),
                d_out_ref.clone(),
            )

        # Reference
        torch.cuda.synchronize()

        q_ref.requires_grad_(True)
        k_ref.requires_grad_(True)
        v_ref.requires_grad_(True)
        d_out_ref.requires_grad_(True)
        if reference_backend is None or reference_backend == "reference":
            out_ref_ = reference_fna_generic(
                q_ref,
                k_ref,
                v_ref,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                is_causal=is_causal,
                return_lse=False,
            )

        else:
            # TODO: don't rely on `neighborhood_attention_generic` finding the right backend
            # and explicitly call the backend fns.
            out_ref_ = neighborhood_attention_generic(
                q_ref,
                k_ref,
                v_ref,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                is_causal=is_causal,
                backend=reference_backend,
                q_tile_shape=reference_q_tile_shape,
                kv_tile_shape=reference_kv_tile_shape,
                backward_q_tile_shape=reference_backward_q_tile_shape,
                backward_kv_tile_shape=reference_backward_kv_tile_shape,
                backward_kv_splits=reference_backward_kv_splits,
                backward_use_pt_reduction=reference_backward_use_pt_reduction,
                attention_kwargs={
                    "backend": reference_fmha_backend,
                },
            )

        self.out_ref = out_ref_.data.clone().float()  # type: ignore[union-attr]

        self.dq_ref, self.dk_ref, self.dv_ref = None, None, None
        if test_backprop:
            out_ref_.backward(d_out_ref)  # type: ignore[union-attr]
            with torch.no_grad():
                assert q_ref.grad is not None
                assert k_ref.grad is not None
                assert v_ref.grad is not None
                self.dq_ref, self.dk_ref, self.dv_ref = (
                    q_ref.grad.clone().float(),
                    k_ref.grad.clone().float(),
                    v_ref.grad.clone().float(),
                )

        torch.cuda.synchronize()

    def test(
        self,
        eps: Union[
            float, Tuple[float, float], Tuple[float, Tuple[float, float, float]]
        ],
        dtype: torch.dtype,
        target_backend: str,
        target_fmha_backend: str = "cutlass-fmha",
        q_tile_shape: Optional[DimensionType] = None,
        kv_tile_shape: Optional[DimensionType] = None,
        backward_q_tile_shape: Optional[DimensionType] = None,
        backward_kv_tile_shape: Optional[DimensionType] = None,
        backward_kv_splits: Optional[DimensionType] = None,
        backward_use_pt_reduction: bool = False,
        run_persistent_kernel: bool = True,
        kernel_schedule: Optional[KernelSchedule] = None,
        torch_compile: bool = False,
        test_backprop: Optional[bool] = None,
    ):
        batch = self.batch
        heads = self.heads
        heads_kv = self.heads_kv
        head_dim = self.head_dim
        head_dim_v = self.head_dim_v
        input_shape = self.input_shape
        kernel_size = self.kernel_size
        stride = self.stride
        dilation = self.dilation
        is_causal = self.is_causal
        reference_backend = self.reference_backend
        test_backprop_safe: bool = (
            self.test_backprop if test_backprop is None else test_backprop
        )

        logger.debug(
            f"Testing {target_backend} against {reference_backend}:\n"
            f"{batch=}, {heads=}, {heads_kv=}, {head_dim=}, {head_dim_v=}, {input_shape=}, {dtype=}\n"
            f"{kernel_size=}, {stride=}, {dilation=}, {is_causal=},\n"
            f"{q_tile_shape=}, {kv_tile_shape=}, {run_persistent_kernel=}, {kernel_schedule=}, "
            f"{torch_compile=}"
            + (
                f"\n{backward_q_tile_shape=}, {backward_kv_tile_shape=}, "
                f"{backward_kv_splits=}, {backward_use_pt_reduction=}."
                if test_backprop_safe
                else "."
            )
        )

        q, k, v, d_out = (
            self.q.clone().to(dtype),
            self.k.clone().to(dtype),
            self.v.clone().to(dtype),
            self.d_out.clone().to(dtype),
        )
        q.requires_grad_(test_backprop_safe)
        k.requires_grad_(test_backprop_safe)
        v.requires_grad_(test_backprop_safe)
        d_out.requires_grad_(test_backprop_safe)

        torch.cuda.synchronize()

        out_: torch.Tensor = neighborhood_attention_generic(  # type: ignore[assignment]
            q,
            k,
            v,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            is_causal=is_causal,
            backend=target_backend,
            q_tile_shape=q_tile_shape,
            kv_tile_shape=kv_tile_shape,
            backward_q_tile_shape=backward_q_tile_shape,
            backward_kv_tile_shape=backward_kv_tile_shape,
            backward_kv_splits=backward_kv_splits,
            backward_use_pt_reduction=backward_use_pt_reduction,
            run_persistent_kernel=run_persistent_kernel,
            kernel_schedule=kernel_schedule,
            torch_compile=torch_compile,
            attention_kwargs={"backend": target_fmha_backend},
        )
        out = out_.data.clone().float()

        if test_backprop_safe:
            dq, dk, dv = None, None, None
            out_.backward(d_out)
            with torch.no_grad():
                assert q.grad is not None
                assert k.grad is not None
                assert v.grad is not None
                dq, dk, dv = (
                    q.grad.clone().float(),
                    k.grad.clone().float(),
                    v.grad.clone().float(),
                )

        if isinstance(eps, tuple):
            eps_forward, eps_backward = eps
        else:
            eps_forward, eps_backward = eps, eps

        torch.cuda.synchronize()

        torch.testing.assert_close(out, self.out_ref, atol=eps_forward, rtol=0)

        if test_backprop_safe:
            if isinstance(eps_backward, tuple):
                assert len(eps_backward) == 3
                eps_dq, eps_dk, eps_dv = eps_backward
            else:
                assert isinstance(eps_backward, float)
                eps_dq, eps_dk, eps_dv = eps_backward, eps_backward, eps_backward

            torch.testing.assert_close(dq, self.dq_ref, atol=eps_dq, rtol=0)
            torch.testing.assert_close(dk, self.dk_ref, atol=eps_dk, rtol=0)
            torch.testing.assert_close(dv, self.dv_ref, atol=eps_dv, rtol=0)


# Variable-length FNA test harness, shared across tests/test_varlen_api.py
# and tests/test_fna_varlen.py.

Dimension = DimensionType
PairwiseRow = Tuple[int, torch.dtype, str, str, str, str, str, str, bool]


@dataclass(frozen=True)
class VarlenCase:
    name: str
    layouts: Tuple[Dimension, ...]
    kernel_size: Dimension
    stride: Dimension
    dilation: Dimension
    is_causal: CausalArgType
    dtype: torch.dtype
    heads: int
    heads_kv: int
    head_dim: int
    head_dim_v: int
    deterministic: bool = True
    split_cap: Optional[Dimension] = None
    q_tile_shape: Optional[Dimension] = None
    kv_tile_shape: Optional[Dimension] = None
    backward_q_tile_shape: Optional[Dimension] = None
    backward_kv_tile_shape: Optional[Dimension] = None
    backward_use_pt_reduction: bool = False
    scale: Optional[float] = None

    @property
    def rank(self) -> int:
        return len(self.layouts[0])


DEFAULT_CASES = (
    VarlenCase(
        "r1-prod-causal-fp16",
        ((49,), (121,)),
        (5,),
        (1,),
        (1,),
        (True,),
        torch.float16,
        3,
        3,
        64,
        64,
        q_tile_shape=(32,),
        kv_tile_shape=(128,),
        backward_q_tile_shape=(64,),
        backward_kv_tile_shape=(64,),
    ),
    VarlenCase(
        "r1-even-gqa-bf16",
        ((65,), (130,)),
        (4,),
        (2,),
        (1,),
        (False,),
        torch.bfloat16,
        8,
        1,
        64,
        80,
        False,
        None,
    ),
    VarlenCase(
        "r1-dilated-mla-fp32",
        ((9,), (13,)),
        (3,),
        (1,),
        (2,),
        (False,),
        torch.float32,
        4,
        4,
        80,
        64,
    ),
    VarlenCase(
        "r1-even-near-full-fp16",
        ((9,), (17,)),
        (8,),
        (8,),
        (1,),
        (False,),
        torch.float16,
        12,
        3,
        128,
        64,
    ),
    VarlenCase(
        "r1-stride-boundary-bf16",
        ((6,), (64,), (129,)),
        (5,),
        (5,),
        (1,),
        (False,),
        torch.bfloat16,
        1,
        1,
        64,
        64,
    ),
    VarlenCase(
        "r1-even-causal-fp32",
        ((8,), (17,)),
        (4,),
        (2,),
        (1,),
        (True,),
        torch.float32,
        2,
        2,
        64,
        128,
    ),
    VarlenCase(
        "r2-prod-even-fp16",
        ((64, 64), (32, 32)),
        (8, 8),
        (1, 1),
        (1, 1),
        (False, False),
        torch.float16,
        12,
        3,
        64,
        64,
        q_tile_shape=(16, 2),
        kv_tile_shape=(64, 2),
        backward_q_tile_shape=(32, 2),
        backward_kv_tile_shape=(2, 32),
    ),
    VarlenCase(
        "r2-mixed-causal-bf16",
        ((25, 32), (17, 24)),
        (5, 8),
        (1, 4),
        (1, 1),
        (True, False),
        torch.bfloat16,
        3,
        3,
        64,
        80,
    ),
    VarlenCase(
        "r2-dilated-fp32",
        ((7, 9), (11, 13)),
        (3, 4),
        (3, 1),
        (2, 2),
        (False, False),
        torch.float32,
        4,
        1,
        80,
        64,
    ),
    VarlenCase(
        "r2-thin-transpose-fp16",
        ((8, 65), (65, 8)),
        (8, 5),
        (8, 1),
        (1, 1),
        (False, True),
        torch.float16,
        8,
        8,
        64,
        128,
    ),
    VarlenCase(
        "r2-gqa-odd-bf16",
        ((17, 19), (33, 15)),
        (5, 3),
        (2, 3),
        (1, 1),
        (False, False),
        torch.bfloat16,
        16,
        8,
        128,
        64,
    ),
    VarlenCase(
        "r2-kv-split-fp32",
        ((80, 72), (17, 19)),
        (4, 5),
        (2, 1),
        (1, 1),
        (True, False),
        torch.float32,
        2,
        2,
        64,
        80,
        False,
        None,
        q_tile_shape=(16, 2),
        kv_tile_shape=(64, 2),
        backward_q_tile_shape=(32, 2),
        backward_kv_tile_shape=(2, 32),
        backward_use_pt_reduction=True,
    ),
    VarlenCase(
        "r3-prod-even-bf16",
        ((25, 16, 16), (61, 8, 8)),
        (5, 8, 8),
        (1, 1, 1),
        (1, 1, 1),
        (True, False, False),
        torch.bfloat16,
        16,
        16,
        64,
        64,
        q_tile_shape=(8, 2, 2),
        kv_tile_shape=(32, 2, 2),
        backward_q_tile_shape=(16, 2, 2),
        backward_kv_tile_shape=(2, 2, 16),
    ),
    VarlenCase(
        "r3-small-fp32",
        ((3, 4, 5), (4, 5, 6)),
        (3, 4, 3),
        (1, 2, 3),
        (1, 1, 1),
        (False, True, False),
        torch.float32,
        1,
        1,
        64,
        80,
    ),
    VarlenCase(
        "r3-dilated-fp16",
        ((7, 9, 11), (9, 9, 13)),
        (3, 4, 5),
        (3, 1, 5),
        (2, 2, 2),
        (False, False, False),
        torch.float16,
        8,
        1,
        80,
        64,
    ),
    VarlenCase(
        "r3-thin-gqa-bf16",
        ((8, 8, 33), (8, 16, 17)),
        (8, 4, 5),
        (4, 4, 1),
        (1, 1, 1),
        (False, False, True),
        torch.bfloat16,
        12,
        3,
        64,
        128,
    ),
    VarlenCase(
        "r3-tile-remainder-fp16",
        ((17, 12, 10), (13, 8, 14)),
        (4, 3, 5),
        (2, 3, 1),
        (1, 1, 1),
        (True, False, True),
        torch.float16,
        3,
        3,
        128,
        64,
    ),
    VarlenCase(
        "r3-kv-split-fp32",
        ((17, 16, 16), (9, 8, 12)),
        (5, 4, 4),
        (1, 2, 4),
        (1, 1, 1),
        (False, True, False),
        torch.float32,
        4,
        4,
        64,
        64,
        False,
        None,
        q_tile_shape=(8, 2, 2),
        kv_tile_shape=(32, 2, 2),
        backward_q_tile_shape=(16, 2, 2),
        backward_kv_tile_shape=(2, 2, 16),
    ),
    VarlenCase(
        "r1-kv-split-pt-reduction-fp16",
        ((128,), (160,)),
        (4,),
        (2,),
        (1,),
        (False,),
        torch.float16,
        2,
        2,
        64,
        64,
        False,
        None,
        backward_q_tile_shape=(64,),
        backward_kv_tile_shape=(64,),
        backward_use_pt_reduction=True,
    ),
    VarlenCase(
        "r1-kv-split-pt-reduction-bf16",
        ((128,), (160,)),
        (4,),
        (2,),
        (1,),
        (False,),
        torch.bfloat16,
        2,
        2,
        64,
        64,
        False,
        None,
        backward_q_tile_shape=(64,),
        backward_kv_tile_shape=(64,),
        backward_use_pt_reduction=True,
    ),
    VarlenCase(
        "r3-kv-split-pt-reduction-fp16",
        ((17, 16, 16), (9, 8, 12)),
        (5, 4, 4),
        (1, 2, 4),
        (1, 1, 1),
        (False, True, False),
        torch.float16,
        4,
        4,
        64,
        64,
        False,
        None,
        backward_q_tile_shape=(16, 2, 2),
        backward_kv_tile_shape=(2, 2, 16),
        backward_use_pt_reduction=True,
    ),
    VarlenCase(
        "r3-kv-split-pt-reduction-bf16",
        ((17, 16, 16), (9, 8, 12)),
        (5, 4, 4),
        (1, 2, 4),
        (1, 1, 1),
        (False, True, False),
        torch.bfloat16,
        4,
        4,
        64,
        64,
        False,
        None,
        backward_q_tile_shape=(16, 2, 2),
        backward_kv_tile_shape=(2, 2, 16),
        backward_use_pt_reduction=True,
    ),
    VarlenCase(
        "r2-production-head-signature-fp16",
        ((17, 19), (33, 15)),
        (5, 3),
        (1, 1),
        (1, 1),
        (False, False),
        torch.float16,
        24,
        24,
        128,
        128,
        scale=0.03125,
    ),
)


NONDETERMINISTIC_CONTROL_CASE = VarlenCase(
    "r1-nondeterministic-control-fp16",
    ((129,), (257,)),
    (5,),
    (1,),
    (1,),
    (False,),
    torch.float16,
    8,
    1,
    64,
    80,
    False,
)


PAIRWISE_ROWS: Tuple[PairwiseRow, ...] = (
    (3, torch.bfloat16, "mha", "v-wide", "odd", "middle", "dilated", "all", False),
    (
        1,
        torch.float32,
        "gqa",
        "qk-wide",
        "even",
        "kernel",
        "unit",
        "alternating",
        True,
    ),
    (2, torch.float16, "mqa", "equal", "even", "unit", "unit", "none", False),
    (2, torch.bfloat16, "mqa", "qk-wide", "odd", "kernel", "dilated", "none", True),
    (
        1,
        torch.float16,
        "gqa",
        "equal",
        "odd",
        "middle",
        "dilated",
        "alternating",
        True,
    ),
    (3, torch.float32, "mha", "equal", "even", "unit", "dilated", "all", True),
    (
        1,
        torch.bfloat16,
        "gqa",
        "v-wide",
        "odd",
        "unit",
        "unit",
        "alternating",
        False,
    ),
    (3, torch.float32, "mqa", "v-wide", "odd", "middle", "unit", "none", False),
    (1, torch.float16, "mha", "qk-wide", "even", "kernel", "unit", "all", False),
    (2, torch.float16, "gqa", "v-wide", "even", "middle", "dilated", "all", True),
    (
        3,
        torch.bfloat16,
        "gqa",
        "equal",
        "even",
        "kernel",
        "unit",
        "alternating",
        False,
    ),
    (
        2,
        torch.float32,
        "mha",
        "equal",
        "even",
        "unit",
        "dilated",
        "alternating",
        False,
    ),
    (
        3,
        torch.float16,
        "mqa",
        "qk-wide",
        "even",
        "unit",
        "dilated",
        "alternating",
        False,
    ),
    (1, torch.float32, "mha", "v-wide", "even", "kernel", "dilated", "none", False),
    (1, torch.bfloat16, "mqa", "qk-wide", "odd", "middle", "dilated", "all", True),
    (2, torch.float16, "gqa", "equal", "odd", "unit", "dilated", "none", False),
)


def _case_from_pairwise_row(index: int, row: PairwiseRow) -> VarlenCase:
    (
        rank,
        dtype,
        head_mode,
        dim_mode,
        kernel_mode,
        stride_mode,
        dilation_mode,
        causal_mode,
        deterministic,
    ) = row
    rank = int(rank)
    kernel_size = cast(Dimension, ((5,) if kernel_mode == "odd" else (4,)) * rank)
    dilation = cast(
        Dimension,
        tuple(1 if dilation_mode == "unit" or axis % 2 else 2 for axis in range(rank)),
    )
    if stride_mode == "unit":
        stride = cast(Dimension, (1,) * rank)
    elif stride_mode == "kernel":
        stride = kernel_size
    else:
        stride = cast(Dimension, tuple(max(1, kernel // 2) for kernel in kernel_size))
    if causal_mode == "none":
        is_causal = cast(CausalArgType, (False,) * rank)
    elif causal_mode == "all":
        is_causal = cast(CausalArgType, (True,) * rank)
    else:
        is_causal = cast(CausalArgType, tuple(axis % 2 == 0 for axis in range(rank)))
    heads, heads_kv = {
        "mha": (3, 3),
        "mqa": (8, 1),
        "gqa": (12, 3),
    }[str(head_mode)]
    head_dim, head_dim_v = {
        "equal": (64, 64),
        "v-wide": (64, 80),
        "qk-wide": (80, 64),
    }[str(dim_mode)]
    split_minimum = {1: 129, 2: 17, 3: 9}[rank]
    layouts = []
    expanded_minimum = {1: 513, 2: 65, 3: 33}[rank] if index < 3 else 0
    for document in range(2):
        layout = []
        for axis, (kernel, dil) in enumerate(zip(kernel_size, dilation)):
            minimum = kernel * dil
            if not deterministic:
                minimum = max(minimum, split_minimum)
            minimum = max(minimum, expanded_minimum)
            layout.append(minimum + ((index + document + axis) % 3))
        layouts.append(cast(Dimension, tuple(layout)))
    return VarlenCase(
        name=f"pairwise-{index:02d}",
        layouts=tuple(layouts),
        kernel_size=kernel_size,
        stride=stride,
        dilation=dilation,
        is_causal=is_causal,
        dtype=dtype,
        heads=heads,
        heads_kv=heads_kv,
        head_dim=head_dim,
        head_dim_v=head_dim_v,
        deterministic=bool(deterministic),
        split_cap=None if deterministic else cast(Dimension, (2,) * rank),
    )


PAIRWISE_CASES = tuple(
    _case_from_pairwise_row(index, row) for index, row in enumerate(PAIRWISE_ROWS)
)


def _prod(values: Dimension) -> int:
    return math.prod(values)


def _dtype_is_supported(dtype: torch.dtype) -> bool:
    if dtype != torch.bfloat16:
        return True
    major, _ = torch.cuda.get_device_capability()
    return major >= 8


def _tolerances(dtype: torch.dtype) -> Tuple[float, float]:
    if dtype == torch.float32:
        return (2e-4, 2e-4)
    if dtype == torch.float16:
        return (2e-2, 2e-2)
    return (4e-2, 4e-2)


def _dq_tolerances(dtype: torch.dtype) -> Tuple[float, float]:
    tolerance = torch.finfo(dtype).eps
    if dtype == torch.float32:
        tolerance *= 8
    return tolerance, tolerance


# _independent_reference's per-document compute is O(tokens^2); documents above this
# cap are skipped via its `covered` mask instead of paying quadratic reference cost
# on every case in the broad-spectrum sweep.
_INDEPENDENT_REFERENCE_TOKEN_CAP = 8192

# The reference kernel (fna_reference_forward.hpp/fna_reference_backward.hpp)
# hard-caps head_dim at MaxDimSupported=1024; cases beyond it would crash into the
# reference kernel instead of comparing against it, so they're skipped via the same
# `covered` mask as the token-count cap above.
_INDEPENDENT_REFERENCE_HEAD_DIM_CAP = 1024


def _independent_tolerances(dtype: torch.dtype) -> float:
    if dtype == torch.float32:
        return 1e-4
    if dtype == torch.float16:
        return 1e-2
    return 5e-2


def _set_deterministic(enabled: bool) -> bool:
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(enabled)
    return previous


def _make_layout(case: VarlenCase) -> "natten.VarlenLayout":
    return natten.VarlenLayout(case.layouts, device="cuda")


def _resolved_state(layout: "natten.VarlenLayout"):
    # Test-only introspection: after exactly one geometry has been resolved
    # against `layout` (the standard one-call-per-layout pattern this suite
    # uses), grab its cached build state straight off the private memo, so
    # tests do not have to rebuild the memo key that
    # _neighborhood_attention_varlen_generic assembles internally.
    assert len(layout._memo) == 1, (
        "expected exactly one resolved geometry on this layout, got "
        f"{len(layout._memo)}; call the varlen op once before requesting "
        "introspection"
    )
    return next(iter(layout._memo.values()))


def _regular_reference(
    case: VarlenCase,
    layout: "natten.VarlenLayout",
    query: Tensor,
    key: Tensor,
    value: Tensor,
) -> Tuple[Tensor, Tensor]:
    outputs = []
    logsumexps = []
    start = 0
    state = _resolved_state(layout)
    for doc_layout in case.layouts:
        end = start + _prod(doc_layout)
        output, logsumexp = cutlass_fna_generic(
            query[start:end].reshape(1, *doc_layout, case.heads, case.head_dim),
            key[start:end].reshape(1, *doc_layout, case.heads_kv, case.head_dim),
            value[start:end].reshape(1, *doc_layout, case.heads_kv, case.head_dim_v),
            kernel_size=case.kernel_size,
            stride=case.stride,
            dilation=case.dilation,
            is_causal=case.is_causal,
            q_tile_shape=state.forward_config[0],
            kv_tile_shape=state.forward_config[1],
            backward_q_tile_shape=state.backward_config[0],
            backward_kv_tile_shape=state.backward_config[1],
            backward_kv_splits=cast(Dimension, (1,) * case.rank),
            backward_use_pt_reduction=case.backward_use_pt_reduction,
            scale=case.scale,
            return_lse=True,
        )
        outputs.append(output.reshape(-1, case.heads, case.head_dim_v))
        logsumexps.append(logsumexp.reshape(-1, case.heads))
        start = end
    output = torch.cat(outputs, dim=0)
    logsumexp = torch.cat(logsumexps, dim=0)
    return output, logsumexp


def _independent_reference(
    case: VarlenCase,
    query: Tensor,
    key: Tensor,
    value: Tensor,
) -> Tuple[Tensor, Tensor, Tensor]:
    outputs = []
    logsumexps = []
    covered_flags = []
    start = 0
    head_dim_exceeds_reference = (
        case.head_dim > _INDEPENDENT_REFERENCE_HEAD_DIM_CAP
        or case.head_dim_v > _INDEPENDENT_REFERENCE_HEAD_DIM_CAP
    )
    for doc_layout in case.layouts:
        length = _prod(doc_layout)
        end = start + length
        if length > _INDEPENDENT_REFERENCE_TOKEN_CAP or head_dim_exceeds_reference:
            outputs.append(query.new_zeros(length, case.heads, case.head_dim_v))
            # Logsumexp is always fp32 in both the varlen op and the reference
            # backend, regardless of QKV dtype; match that here so an all-skipped
            # case's empty comparison doesn't trip assert_close's dtype check.
            logsumexps.append(
                torch.zeros(
                    length, case.heads, dtype=torch.float32, device=query.device
                )
            )
            covered_flags.append(
                torch.zeros(length, dtype=torch.bool, device=query.device)
            )
            start = end
            continue
        with torch.no_grad():
            output, logsumexp = reference_fna_generic(
                query[start:end].reshape(1, *doc_layout, case.heads, case.head_dim),
                key[start:end].reshape(1, *doc_layout, case.heads_kv, case.head_dim),
                value[start:end].reshape(
                    1, *doc_layout, case.heads_kv, case.head_dim_v
                ),
                kernel_size=case.kernel_size,
                stride=case.stride,
                dilation=case.dilation,
                is_causal=case.is_causal,
                scale=case.scale,
                return_lse=True,
            )
        outputs.append(output.reshape(-1, case.heads, case.head_dim_v))
        logsumexps.append(logsumexp.reshape(-1, case.heads))
        covered_flags.append(torch.ones(length, dtype=torch.bool, device=query.device))
        start = end
    output = torch.cat(outputs, dim=0)
    logsumexp = torch.cat(logsumexps, dim=0)
    covered = torch.cat(covered_flags, dim=0)
    return output, logsumexp, covered


def _window_positions(
    index: int,
    extent: int,
    kernel_size: int,
    stride: int,
    dilation: int,
    causal: bool,
) -> Tuple[int, ...]:
    residue = index % dilation
    local_index = index // dilation
    local_extent = (extent - residue + dilation - 1) // dilation
    if causal:
        leader = min((local_index // stride) * stride + stride - 1, local_extent - 1)
        start = max(leader - kernel_size + 1, 0)
        end = min(local_index + 1, local_extent)
    else:
        leader = min((local_index // stride) * stride + stride // 2, local_extent - 1)
        radius_left = kernel_size // 2
        radius_right = kernel_size // 2 + (kernel_size % 2 - 1)
        start = max(leader - radius_left, 0)
        if leader + radius_right >= local_extent:
            start += local_extent - radius_right - leader - 1
        end = start + kernel_size
    return tuple(residue + local * dilation for local in range(start, end))


def _flatten_index(coord: Dimension, layout: Dimension) -> int:
    index = 0
    for axis, extent in zip(coord, layout):
        index = index * extent + axis
    return index


def _explicit_oracle(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    layout: Dimension,
    kernel_size: Dimension,
    stride: Dimension,
    dilation: Dimension,
    is_causal: Tuple[bool, ...],
    scale: Optional[float] = None,
) -> Tuple[Tensor, Tensor]:
    outputs = []
    logsumexps = []
    heads = query.shape[-2]
    heads_kv = key.shape[-2]
    repeats = heads // heads_kv
    scale = query.shape[-1] ** -0.5 if scale is None else scale
    for coord in itertools.product(*(range(extent) for extent in layout)):
        key_axes = tuple(
            _window_positions(i, n, k, s, d, c)
            for i, n, k, s, d, c in zip(
                coord, layout, kernel_size, stride, dilation, is_causal
            )
        )
        key_indices = tuple(
            _flatten_index(key_coord, layout)
            for key_coord in itertools.product(*key_axes)
        )
        query_index = _flatten_index(coord, layout)
        output_heads = []
        lse_heads = []
        for head in range(heads):
            kv_head = head // repeats
            keys = key[key_indices, kv_head].float()
            values = value[key_indices, kv_head].float()
            scores = keys @ query[query_index, head].float() * scale
            output_heads.append(torch.softmax(scores, dim=0) @ values)
            lse_heads.append(torch.logsumexp(scores, dim=0))
        outputs.append(torch.stack(output_heads))
        logsumexps.append(torch.stack(lse_heads))
    return torch.stack(outputs), torch.stack(logsumexps)
