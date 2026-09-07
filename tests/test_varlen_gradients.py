"""Exact gradient controls in the dtype used by the attention implementation."""

import natten
import pytest
import torch

from .mixed_batch_utils import MixedCase, reference


def _check_uniform_gradients(attention, rank, dtype):
    shapes = (
        (1,) * rank,
        (2,) + (1,) * (rank - 1),
        (2,) * rank,
        (4,) + (2,) * (rank - 1),
    )
    case = MixedCase(
        "uniform-gradients",
        "nonzero gradients",
        shapes,
        (2,) * rank,
        (False,) * rank,
        dtype=str(dtype).removeprefix("torch."),
        heads=4,
        kv_heads=2,
        dim=16,
        vdim=16,
        scale=1.0,
    )
    total = sum(case.lengths)
    q = torch.ones(total, case.heads, case.dim, device="cuda", dtype=dtype)
    q[..., 1::2] = -1
    k = torch.empty(total, case.kv_heads, case.dim, device="cuda", dtype=dtype)
    v = torch.empty(total, case.kv_heads, case.vdim, device="cuda", dtype=dtype)
    grad = torch.zeros(total, case.heads, case.vdim, device="cuda", dtype=dtype)
    for lo, hi in zip(case.offsets, case.offsets[1:]):
        signs = (1 - 2 * (torch.arange(hi - lo, device="cuda") % 2)).to(dtype)
        k[lo:hi] = signs[:, None, None]
        v[lo:hi] = signs[:, None, None]
        grad[lo] = 1.0 / case.vdim
    inputs = [x.requires_grad_() for x in (q, k, v)]
    expected = reference(inputs, grad, case)

    # Q is orthogonal to K. Uniform weights over 1/2/4/8 keys, inputs,
    # upstream gradients and their sums are all exactly representable.
    for name in ("dq", "dk", "dv"):
        assert expected[name].count_nonzero() > 0, name
        assert torch.equal(expected[name], expected[name].to(dtype).double()), name
    layout = natten.VarlenLayout(shapes, device="cuda")
    output = attention(*inputs, layout, kernel_size=case.kernel, scale=case.scale)
    gradients = torch.autograd.grad(output, inputs, grad)
    for name, actual in zip(("out", "dq", "dk", "dv"), (output, *gradients)):
        torch.testing.assert_close(
            actual.detach().cpu().double(), expected[name], atol=0, rtol=0, msg=name
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("rank", [1, 2, 3])
def test_zero_logits_have_nonzero_qkv_gradients(rank, dtype):
    attention = getattr(natten, f"na{rank}d_varlen")
    _check_uniform_gradients(attention, rank, dtype)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("missing", [0, 1, 2])
def test_gradient_checker_rejects_missing_bf16_gradient(missing):
    def wrong_attention(q, k, v, *args, **kwargs):
        inputs = [q, k, v]
        x = inputs[missing]
        inputs[missing] = x.detach() + x * 0
        return natten.na1d_varlen(*inputs, *args, **kwargs)

    with pytest.raises(AssertionError, match=("dq", "dk", "dv")[missing]):
        _check_uniform_gradients(wrong_attention, 1, torch.bfloat16)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "trainable",
    [
        (True, False, False),
        (False, True, False),
        (False, False, True),
        (True, True, True),
    ],
)
@pytest.mark.parametrize("return_lse", [False, True])
def test_single_key_identity_with_large_inputs_and_partial_gradients(
    dtype, trainable, return_lse
):
    layout = natten.VarlenLayout(((2, 3), (1, 4)))
    # Different Q/K and V widths, GQA, and noncontiguous input storage.
    q = torch.full((10, 4, 16, 2), 1024, device="cuda", dtype=dtype)[..., 0]
    k = torch.full((10, 2, 16, 2), -1024, device="cuda", dtype=dtype)[..., 0]
    v = torch.full((10, 2, 24, 2), 16384, device="cuda", dtype=dtype)[..., 0]
    inputs = [x.requires_grad_(flag) for x, flag in zip((q, k, v), trainable)]
    actual = natten.na2d_varlen(
        *inputs, layout, kernel_size=(1, 1), scale=0.25, return_lse=return_lse
    )
    output = actual[0] if return_lse else actual
    torch.testing.assert_close(output, v.repeat_interleave(2, dim=1), atol=0, rtol=0)
    grad = torch.arange(4, device="cuda", dtype=dtype)[None, :, None].expand_as(output)
    expected = [
        torch.zeros_like(q),
        torch.zeros_like(k),
        grad.reshape(10, 2, 2, 24).sum(2),
    ]
    active = [x for x in inputs if x.requires_grad]
    if return_lse:
        lse = actual[1]
        torch.testing.assert_close(lse, torch.full_like(lse, -(2**22)), atol=0, rtol=0)
        observed = torch.autograd.grad(
            (output, lse), active, (grad, torch.ones_like(lse))
        )
    else:
        observed = torch.autograd.grad(output, active, grad)
    for actual_grad, reference_grad in zip(
        observed, [x for x, flag in zip(expected, trainable) if flag]
    ):
        torch.testing.assert_close(actual_grad, reference_grad, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_identity_lse_fullgraph_backward():
    def call(q, k, v, layout):
        return natten.na1d_varlen(q, k, v, layout, kernel_size=1, return_lse=True)

    torch.compiler.reset()
    try:
        compiled = torch.compile(call, fullgraph=True)
        layout = natten.VarlenLayout(((1,), (7,)))
        q = torch.full(
            (8, 4, 16), 1024, device="cuda", dtype=torch.float16, requires_grad=True
        )
        k = torch.full(
            (8, 2, 16), -1024, device="cuda", dtype=torch.float16, requires_grad=True
        )
        v = torch.full(
            (8, 2, 24), 16384, device="cuda", dtype=torch.float16, requires_grad=True
        )
        out, lse = compiled(q, k, v, layout)
        torch.testing.assert_close(out, v.repeat_interleave(2, dim=1), atol=0, rtol=0)
        torch.testing.assert_close(lse, torch.full_like(lse, -(2**22)), atol=0, rtol=0)
        (out.float().sum() + lse.sum()).backward()
        for grad, expected in (
            (q.grad, torch.zeros_like(q)),
            (k.grad, torch.zeros_like(k)),
            (v.grad, torch.full_like(v, 2)),
        ):
            torch.testing.assert_close(grad, expected, atol=0, rtol=0)
    finally:
        torch.compiler.reset()
