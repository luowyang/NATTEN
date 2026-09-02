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
"""Exercises fna_forward.cu/fna_backward.cu's varlen_fna_generic_forward/backward
check battery directly via torch.ops.natten.varlen_na{1,2,3}d_{forward,backward},
bypassing na{1,2,3}d_varlen's Python-side validation (and VarlenLayout)
entirely. Everywhere else in the suite, illegal inputs are intercepted in
Python before the raw ops see them, so the C++ checks would otherwise have
no coverage of their own.

SCOPE EXCLUSION: CheckArgsAgainstDim's per-document VALUE check -- kernel_size
* dilation <= each document's actual extent -- lives ONLY in
natten.backends.varlen_fna._build_varlen_fna_state (the Python build path,
host tuples already in hand, zero sync). Reading token_layouts' per-document
values on the C++ side to re-check this would need a device-to-host sync the
whole design exists to avoid, so the raw-op boundary below does NOT
re-validate it: a token_layouts tensor with well-formed shape/dtype whose
VALUES understate a document's real extent is silently trusted by the raw
op. This is a deliberate, documented sharp edge, not a bug -- do not add a
test here expecting the raw op to catch it.
"""

import unittest
from typing import Any, Callable, cast, Dict

import natten  # noqa: F401 -- ensures natten.libnatten (and torch.ops.natten.*) is loaded
import torch
from natten.types import DimensionType
from natten.utils.testing import skip_if_libnatten_is_not_supported

_VARLEN_FN_BY_RANK: Dict[int, Callable[..., Any]] = {
    1: natten.na1d_varlen,
    2: natten.na2d_varlen,
    3: natten.na3d_varlen,
}


def _valid_forward_metadata(na_dim: int, device="cuda"):
    """A genuinely valid (layout, tensors) pair for na_dim, built through the
    real production path (VarlenLayout + na{n}d_varlen), not hand-assembled
    -- so every mutation test below starts from a baseline already proven
    correct, and each case changes exactly one property of it.
    """
    # Two DIFFERENT document shapes -- a non-uniform layout -- so this fixture
    # keeps exercising the varlen schedule/memo path this file's raw-op
    # boundary tests are about; a uniform layout (every document sharing one
    # shape) dispatches straight to the fixed-shape kernels instead
    # (natten.backends.varlen_fna's uniform-dispatch branch), building no
    # memo entry for this fixture to read.
    doc_shape_a = cast(DimensionType, tuple(4 + axis for axis in range(na_dim)))
    doc_shape_b = cast(DimensionType, tuple(5 + axis for axis in range(na_dim)))
    layouts = (doc_shape_a, doc_shape_b)
    total_tokens = sum(_prod(s) for s in layouts)
    heads, head_dim = 2, 8
    query = torch.randn(
        total_tokens, heads, head_dim, device=device, dtype=torch.float16
    )
    key = query.clone()
    value = query.clone()
    layout = natten.VarlenLayout(layouts, device=device)
    varlen_fn = _VARLEN_FN_BY_RANK[na_dim]
    kernel_size = tuple(3 for _ in range(na_dim))
    varlen_fn(query, key, value, layout, kernel_size=kernel_size)  # warm the memo
    state = next(iter(layout._memo.values()))
    return {
        "query": query,
        "key": key,
        "value": value,
        "cumulative_seqlens": layout.cu_seqlens,
        "token_layouts": layout.token_layouts,
        "forward_worklist": state.forward_worklist,
        "kernel_size": list(kernel_size),
        "stride": [1] * na_dim,
        "dilation": [1] * na_dim,
        "is_causal": [False] * na_dim,
        "scale": None,
        "q_tile_shape": list(state.forward_config[0]),
        "kv_tile_shape": list(state.forward_config[1]),
        "forward_work_count": state.forward_work_count,
    }


def _prod(shape):
    result = 1
    for x in shape:
        result *= x
    return result


def _call_forward(na_dim: int, **overrides):
    args = _valid_forward_metadata(na_dim)
    args.update(overrides)
    op = getattr(torch.ops.natten, f"varlen_na{na_dim}d_forward")
    return op(
        args["query"],
        args["key"],
        args["value"],
        args["cumulative_seqlens"],
        args["token_layouts"],
        args["forward_worklist"],
        args["kernel_size"],
        args["stride"],
        args["dilation"],
        args["is_causal"],
        args["scale"] or args["query"].shape[-1] ** -0.5,
        args["q_tile_shape"],
        args["kv_tile_shape"],
        args["forward_work_count"],
    )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
class VarlenRawOpForwardValidationTests(unittest.TestCase):
    @skip_if_libnatten_is_not_supported()
    def test_valid_baseline_succeeds(self):
        # Confirms _valid_forward_metadata itself is genuinely valid (the
        # premise every other case in this file depends on): if this fails,
        # every "should raise" case below is meaningless -- a negative-
        # control check must itself be able to read as "no effect" cleanly.
        output, logsumexp = _call_forward(1)
        self.assertTrue(torch.isfinite(output).all())

    @skip_if_libnatten_is_not_supported()
    def test_query_wrong_rank_raises(self):
        args = _valid_forward_metadata(1)
        with self.assertRaisesRegex(RuntimeError, "3-D"):
            _call_forward(1, query=args["query"].unsqueeze(0))

    @skip_if_libnatten_is_not_supported()
    def test_query_key_shape_mismatch_raises(self):
        args = _valid_forward_metadata(1)
        with self.assertRaisesRegex(RuntimeError, "shapes must match"):
            _call_forward(1, key=args["key"][:, :1].contiguous())

    @skip_if_libnatten_is_not_supported()
    def test_query_value_token_count_mismatch_raises(self):
        args = _valid_forward_metadata(1)
        with self.assertRaisesRegex(RuntimeError, "token/head dimensions must match"):
            _call_forward(1, value=args["value"][:-1].contiguous())

    @skip_if_libnatten_is_not_supported()
    def test_head_dim_zero_raises(self):
        with self.assertRaisesRegex(RuntimeError, "head dimension"):
            _call_forward(
                1,
                query=torch.randn(6, 2, 0, device="cuda", dtype=torch.float16),
                key=torch.randn(6, 2, 0, device="cuda", dtype=torch.float16),
                value=torch.randn(6, 2, 8, device="cuda", dtype=torch.float16),
                scale=1.0,  # head_dim=0 makes the default (head_dim**-0.5) undefined
            )

    @skip_if_libnatten_is_not_supported()
    def test_query_key_value_noncontiguous_raises_or_is_made_contiguous(self):
        # The Python wrapper (torch_wrappers.py) calls maybe_contiguous() on
        # Q/K/V before the C++ check, so a non-contiguous Q/K/V does not
        # reach CHECK_CONTIGUOUS -- documenting that this specific input is
        # NOT part of the raw-op's own enforced contract (the wrapper's
        # job), unlike the metadata tensors below (passed through as-is).
        args = _valid_forward_metadata(1)
        noncontig_query = args["query"].transpose(0, 1).transpose(0, 1)
        output, _ = _call_forward(1, query=noncontig_query)
        self.assertTrue(torch.isfinite(output).all())

    @skip_if_libnatten_is_not_supported()
    def test_cumulative_seqlens_noncontiguous_raises(self):
        args = _valid_forward_metadata(1)
        padded = torch.zeros(
            args["cumulative_seqlens"].shape[0] * 2,
            dtype=torch.int32,
            device=args["cumulative_seqlens"].device,
        )
        padded[::2] = args["cumulative_seqlens"]
        noncontig = padded[::2]
        self.assertFalse(noncontig.is_contiguous())
        with self.assertRaisesRegex(RuntimeError, "contiguous"):
            _call_forward(1, cumulative_seqlens=noncontig)

    @skip_if_libnatten_is_not_supported()
    def test_token_layouts_wrong_device_raises(self):
        args = _valid_forward_metadata(1)
        if torch.cuda.device_count() < 2:
            self.skipTest("Fewer than 2 GPUs are available.")
        with self.assertRaisesRegex(RuntimeError, "same CUDA device"):
            _call_forward(1, token_layouts=args["token_layouts"].to("cuda:1"))

    @skip_if_libnatten_is_not_supported()
    def test_cumulative_seqlens_wrong_dtype_raises(self):
        args = _valid_forward_metadata(1)
        with self.assertRaisesRegex(RuntimeError, "int32"):
            _call_forward(
                1, cumulative_seqlens=args["cumulative_seqlens"].to(torch.int64)
            )

    @skip_if_libnatten_is_not_supported()
    def test_token_layouts_wrong_dtype_raises(self):
        args = _valid_forward_metadata(1)
        with self.assertRaisesRegex(RuntimeError, "int32"):
            _call_forward(1, token_layouts=args["token_layouts"].to(torch.int64))

    @skip_if_libnatten_is_not_supported()
    def test_forward_worklist_wrong_dtype_raises(self):
        args = _valid_forward_metadata(1)
        with self.assertRaisesRegex(RuntimeError, "int32"):
            _call_forward(1, forward_worklist=args["forward_worklist"].to(torch.int64))

    @skip_if_libnatten_is_not_supported()
    def test_cumulative_seqlens_wrong_rank_raises(self):
        args = _valid_forward_metadata(1)
        with self.assertRaisesRegex(RuntimeError, r"shape \[B \+ 1\]"):
            _call_forward(1, cumulative_seqlens=args["cumulative_seqlens"].unsqueeze(0))

    @skip_if_libnatten_is_not_supported()
    def test_token_layouts_wrong_width_raises(self):
        # rank-1 token_layouts must be [B, 1]; feed it a [B, 2] tensor
        # instead (structurally wrong for na1d, not a value-level issue --
        # squarely inside this file's scope, unlike the exclusion above).
        args = _valid_forward_metadata(1)
        wrong_width = torch.cat([args["token_layouts"], args["token_layouts"]], dim=1)
        with self.assertRaisesRegex(RuntimeError, r"shape \[B, rank\]"):
            _call_forward(1, token_layouts=wrong_width)

    @skip_if_libnatten_is_not_supported()
    def test_token_layouts_wrong_document_count_raises(self):
        args = _valid_forward_metadata(1)
        with self.assertRaisesRegex(RuntimeError, r"shape \[B, rank\]"):
            _call_forward(1, token_layouts=args["token_layouts"][:1].contiguous())

    @skip_if_libnatten_is_not_supported()
    def test_forward_work_count_not_positive_raises(self):
        with self.assertRaisesRegex(
            RuntimeError, "forward_work_count must be positive"
        ):
            _call_forward(1, forward_work_count=0)

    @skip_if_libnatten_is_not_supported()
    def test_forward_worklist_capacity_too_small_raises(self):
        args = _valid_forward_metadata(1)
        with self.assertRaisesRegex(RuntimeError, "forward_worklist must have shape"):
            _call_forward(1, forward_worklist=args["forward_worklist"][:1].contiguous())

    @skip_if_libnatten_is_not_supported()
    def test_rank2_token_layouts_shape_enforced(self):
        # Same-family check at a different rank, confirming kNADim is
        # actually threaded through per entry point, not hardcoded to 1.
        args = _valid_forward_metadata(2)
        with self.assertRaisesRegex(RuntimeError, r"shape \[B, rank\]"):
            _call_forward(2, token_layouts=args["token_layouts"][:, :1].contiguous())


def _valid_backward_metadata(na_dim: int, device="cuda"):
    fwd = _valid_forward_metadata(na_dim, device=device)
    fwd["query"].requires_grad_(True)
    fwd["key"].requires_grad_(True)
    fwd["value"].requires_grad_(True)
    layout = natten.VarlenLayout(
        [
            tuple(fwd["token_layouts"][i].tolist())
            for i in range(fwd["token_layouts"].shape[0])
        ],
        device=device,
    )
    varlen_fn = _VARLEN_FN_BY_RANK[na_dim]
    kernel_size = tuple(fwd["kernel_size"])
    output, logsumexp = varlen_fn(
        fwd["query"],
        fwd["key"],
        fwd["value"],
        layout,
        kernel_size=kernel_size,
        return_lse=True,
    )
    state = next(iter(layout._memo.values()))
    d_output = torch.randn_like(output)
    return {
        "query": fwd["query"].detach(),
        "key": fwd["key"].detach(),
        "value": fwd["value"].detach(),
        "output": output.detach(),
        "d_output": d_output,
        "logsumexp": logsumexp.detach(),
        "cumulative_seqlens": layout.cu_seqlens,
        "token_layouts": layout.token_layouts,
        "backward_kv_splits": state.backward_kv_splits,
        "backward_worklist": state.backward_worklist,
        "backward_q_tile_offsets": state.backward_q_tile_offsets,
        "backward_kv_split_offsets": state.backward_kv_split_offsets,
        "kernel_size": list(kernel_size),
        "stride": [1] * na_dim,
        "dilation": [1] * na_dim,
        "is_causal": [False] * na_dim,
        "scale": fwd["query"].shape[-1] ** -0.5,
        "q_tile_shape": list(state.backward_config[0]),
        "kv_tile_shape": list(state.backward_config[1]),
        "backward_work_count": state.backward_work_count,
        "total_backward_q_tiles": state.total_backward_q_tiles,
        "compute_delta_with_torch": False,
        "deterministic": True,
    }


def _call_backward(na_dim: int, **overrides):
    args = _valid_backward_metadata(na_dim)
    args.update(overrides)
    op = getattr(torch.ops.natten, f"varlen_na{na_dim}d_backward")
    return op(
        args["query"],
        args["key"],
        args["value"],
        args["output"],
        args["d_output"],
        args["logsumexp"],
        args["cumulative_seqlens"],
        args["token_layouts"],
        args["backward_kv_splits"],
        args["backward_worklist"],
        args["backward_q_tile_offsets"],
        args["backward_kv_split_offsets"],
        args["kernel_size"],
        args["stride"],
        args["dilation"],
        args["is_causal"],
        args["scale"],
        args["q_tile_shape"],
        args["kv_tile_shape"],
        args["backward_work_count"],
        args["total_backward_q_tiles"],
        args["compute_delta_with_torch"],
        args["deterministic"],
    )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
class VarlenRawOpBackwardValidationTests(unittest.TestCase):
    @skip_if_libnatten_is_not_supported()
    def test_valid_baseline_succeeds(self):
        d_query, d_key, d_value = _call_backward(1)
        self.assertTrue(torch.isfinite(d_query).all())

    @skip_if_libnatten_is_not_supported()
    def test_logsumexp_wrong_dtype_raises(self):
        args = _valid_backward_metadata(1)
        with self.assertRaisesRegex(RuntimeError, "float32"):
            _call_backward(1, logsumexp=args["logsumexp"].to(torch.float16))

    @skip_if_libnatten_is_not_supported()
    def test_logsumexp_wrong_device_raises(self):
        if torch.cuda.device_count() < 2:
            self.skipTest("Fewer than 2 GPUs are available.")
        args = _valid_backward_metadata(1)
        with self.assertRaisesRegex(RuntimeError, "same"):
            _call_backward(1, logsumexp=args["logsumexp"].to("cuda:1"))

    @skip_if_libnatten_is_not_supported()
    def test_backward_kv_splits_wrong_dtype_raises(self):
        args = _valid_backward_metadata(1)
        with self.assertRaises(RuntimeError):
            _call_backward(
                1, backward_kv_splits=args["backward_kv_splits"].to(torch.int64)
            )

    @skip_if_libnatten_is_not_supported()
    def test_backward_q_tile_offsets_wrong_dtype_raises(self):
        args = _valid_backward_metadata(1)
        with self.assertRaises(RuntimeError):
            _call_backward(
                1,
                backward_q_tile_offsets=args["backward_q_tile_offsets"].to(torch.int32),
            )

    @skip_if_libnatten_is_not_supported()
    def test_output_dtype_mismatch_raises(self):
        args = _valid_backward_metadata(1)
        with self.assertRaises(RuntimeError):
            _call_backward(1, output=args["output"].float())


if __name__ == "__main__":
    unittest.main()
