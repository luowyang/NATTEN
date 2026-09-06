"""Known-answer and fault-injection qualification of the mixed batch bench."""

import json
from dataclasses import replace

import torch

from .mixed_batch_utils import (
    checks_pass,
    coordinate_mask,
    dense_attention,
    document_adjudication,
    document_comparison_rule,
    error_metrics,
    lowering_differs_from_pack,
    make_inputs,
    matmul_3xtf32,
    MixedCase,
    pack_takes_identity_path,
    reference,
    structural_checks,
    tf32_split,
)
from .varlen_numerics import (
    axis_neighbors,
    prefix_sum_bound,
    single_key_scalar_bound,
    ulp32,
)


def test_handwritten_neighborhoods():
    assert [axis_neighbors(i, 5, 3, causal=True) for i in range(5)] == [
        (0,),
        (0, 1),
        (0, 1, 2),
        (1, 2, 3),
        (2, 3, 4),
    ]
    assert [axis_neighbors(i, 6, 4) for i in range(6)] == [(0, 1, 2, 3)] * 3 + [
        (1, 2, 3, 4)
    ] + [(2, 3, 4, 5)] * 2
    assert axis_neighbors(4, 7, 4, stride=3) == (2, 3, 4, 5)
    assert axis_neighbors(6, 7, 4, stride=3) == (3, 4, 5, 6)
    assert axis_neighbors(3, 9, 3, dilation=2, causal=True) == (1, 3)
    assert axis_neighbors(8, 9, 3, dilation=2) == (4, 6, 8)
    assert axis_neighbors(1, 2, 5) == (0, 1)
    assert axis_neighbors(3, 4, 1, dilation=17) == (3,)


def test_cartesian_mask_and_empty_seams():
    case = MixedCase("oracle", "mask", ((0, 3), (2, 3), (1, 2)), (3, 3), (True, False))
    mask = coordinate_mask(case)
    assert mask[0].nonzero().flatten().tolist() == [0, 1, 2]
    assert mask[3].nonzero().flatten().tolist() == [0, 1, 2, 3, 4, 5]
    assert mask[6].nonzero().flatten().tolist() == [6, 7]
    assert not mask[:6, 6:].any()


def test_singleton_gqa_known_answer():
    case = MixedCase(
        "oracle",
        "identity",
        ((1,), (3,)),
        (3,),
        (True,),
        heads=4,
        kv_heads=1,
        focus=(0, 0),
        seed=17,
    )
    data, grad = make_inputs(case)
    result = reference(data, grad, case)
    assert checks_pass(structural_checks(result, data, grad, case))


def singleton_kernel_fixture(multiplier, break_rank_one=False):
    """A single-key row inside a non-degenerate document, with a rank-1 dQ/dK residual.

    Q and K on the focus row are powers of two, so `c * x` needs no rounding and the
    least-squares fit recovers the planted scalar exactly; the residual size is the
    only thing the multiplier changes.
    """
    case = MixedCase(
        "oracle",
        "singleton",
        ((5,),),
        (3,),
        (True,),
        dtype="bfloat16",
        heads=2,
        kv_heads=2,
        dim=8,
        vdim=8,
        focus=(0, 0),
        seed=101,
    )
    data, grad = make_inputs(case)
    powers = torch.tensor([1.0, 2.0, 0.5, -1.0, -2.0, 4.0, 0.25, -0.5])
    data[0][0] = powers
    data[1][0] = powers
    scale = case.dim**-0.5
    result = {
        "out": data[2].clone(),
        "lse": torch.zeros(5, case.heads),
        "dq": torch.zeros_like(data[0]),
        "dk": torch.zeros_like(data[1]),
        "dv": torch.zeros_like(data[2]),
    }
    result["dv"][0] = grad[0]
    for head in range(case.heads):
        bound = single_key_scalar_bound(grad[0, head], data[2][0, head], data[0].dtype)
        c = torch.tensor([multiplier * bound * scale], dtype=torch.float64).to(
            data[0].dtype
        )
        result["dq"][0, head] = (c.double() * data[1][0, head].double()).to(
            data[0].dtype
        )
        result["dk"][0, head] = (c.double() * data[0][0, head].double()).to(
            data[1].dtype
        )
    if break_rank_one:
        result["dq"][0, 0, 3] = -result["dq"][0, 0, 3]
    return structural_checks(result, data, grad, case)


def test_single_key_kernel_gate_bounds_the_rank_one_residual():
    inside = singleton_kernel_fixture(0.5)
    assert inside["singleton_row"]["path"] == "cuda-kernel"
    assert checks_pass(inside)
    assert inside["singleton_dq_zero"]["heads"][0]["reconstruction"]["matches"] == 8
    outside = singleton_kernel_fixture(500.0)
    assert not checks_pass(outside)
    assert not outside["singleton_dq_zero"]["pass"]
    assert all(
        h["reconstruction"]["pass"] and not h["within_bound"]
        for h in outside["singleton_dq_zero"]["heads"]
    )
    tilted = singleton_kernel_fixture(0.5, break_rank_one=True)
    assert not checks_pass(tilted)
    assert not tilted["singleton_dq_zero"]["heads"][0]["reconstruction"]["pass"]


def test_uniform_fully_degenerate_pack_keeps_the_identity_gate():
    case = MixedCase(
        "oracle", "identity", ((1,), (1,)), (3,), (True,), focus=(0, 0), seed=17
    )
    assert pack_takes_identity_path(case)
    data, grad = make_inputs(case)
    checks = structural_checks(reference(data, grad, case), data, grad, case)
    assert checks["singleton_row"]["path"] == "python-identity"
    assert "exact" in checks["singleton_out_is_v"]
    assert checks_pass(checks)


def test_heterogeneous_fully_degenerate_document_is_kernel_gated():
    case = MixedCase(
        "oracle", "identity", ((1,), (3,)), (3,), (True,), focus=(0, 0), seed=17
    )
    assert not pack_takes_identity_path(case)
    data, grad = make_inputs(case)
    checks = structural_checks(reference(data, grad, case), data, grad, case)
    assert checks["singleton_row"]["path"] == "cuda-kernel"
    assert checks["singleton_dq_zero"]["heads"]
    assert checks_pass(checks)


def test_document_comparison_rules_follow_the_single_launch_lowering():
    mixed = MixedCase(
        "oracle",
        "mixed",
        ((1, 3, 4), (7, 3, 4), (3, 4, 3)),
        (5, 3, 3),
        (True, False, False),
    )
    assert [document_comparison_rule(mixed, s) for s in mixed.shapes] == [
        "reference-interval",
        "bitwise",
        "bitwise",
    ]
    single_token = MixedCase("oracle", "mixed", ((1,), (9,)), (5,), (True,))
    assert [document_comparison_rule(single_token, s) for s in single_token.shapes] == [
        "single-key-rows",
        "bitwise",
    ]
    uniform = MixedCase(
        "oracle", "identity", ((1, 1, 1),) * 2, (3, 3, 3), (True, False, False)
    )
    assert pack_takes_identity_path(uniform)
    assert [document_comparison_rule(uniform, s) for s in uniform.shapes] == [
        "bitwise",
        "bitwise",
    ]
    # An extent-1 axis the caller already declares kernel_size = 1 on is lowered
    # for the whole pack, so it is not a packed/isolated difference.
    folded = MixedCase("oracle", "folded", ((1, 7), (4, 7)), (1, 3), (False, False))
    assert not lowering_differs_from_pack(folded, (1, 7))
    # A zero-token document is no shape for the pack to agree on: the lowering
    # judges uniformity over the documents that carry tokens, so inserting one
    # leaves both the route and every document's rule where they were.
    for shapes in (((1,), (1,)), ((1,), (1,), (0,)), ((0,), (1,), (1,))):
        padded = MixedCase("oracle", "identity", shapes, (3,), (True,))
        assert pack_takes_identity_path(padded)
        assert [document_comparison_rule(padded, s) for s in shapes] == [
            "bitwise"
        ] * len(shapes)
    # With no token anywhere there is no shape to clamp against and nothing to
    # answer: such a call belongs to the generic all-empty fast path, which
    # returns before the identity path is reached.
    for kernel in ((1,), (3,)):
        empty = MixedCase("oracle", "all-empty", ((0,), (0,)), kernel, (True,))
        assert not pack_takes_identity_path(empty)


def test_document_comparison_rule_treats_a_uniform_partial_clamp_as_bitwise():
    """A pack's host-side clamp answers the packed and isolated calls the same
    way whenever its token-carrying documents share one shape, even where the
    clamp does not reduce every axis to 1 -- not just the fully-degenerate
    (identity-path) packs the test above covers. A "T = 1" image packed among
    "T > 1" video documents is the case that motivates this.
    """
    uniform_images = MixedCase(
        "oracle",
        "uniform-images",
        ((1, 8, 8), (1, 8, 8)),
        (3, 3, 3),
        (True, False, False),
    )
    assert not pack_takes_identity_path(uniform_images)
    assert not lowering_differs_from_pack(uniform_images, (1, 8, 8))
    assert [
        document_comparison_rule(uniform_images, s) for s in uniform_images.shapes
    ] == ["bitwise", "bitwise"]
    # A zero-token document contributes no shape to agree on, so inserting one
    # leaves the uniform pack, and every real document's rule, unchanged.
    padded_images = MixedCase(
        "oracle",
        "uniform-images",
        ((1, 8, 8), (0, 0, 0), (1, 8, 8)),
        (3, 3, 3),
        (True, False, False),
    )
    assert not pack_takes_identity_path(padded_images)
    assert [
        document_comparison_rule(padded_images, s) for s in padded_images.shapes
    ] == ["bitwise", "bitwise", "bitwise"]
    # The same document loses that guarantee once a sibling disagrees with its
    # shape: the pack is no longer uniform, so the host-side clamp is not
    # defined and this document's isolated call may lower differently.
    heterogeneous = MixedCase(
        "oracle",
        "mixed-images",
        ((1, 8, 8), (5, 8, 8)),
        (3, 3, 3),
        (True, False, False),
    )
    assert document_comparison_rule(heterogeneous, (1, 8, 8)) == "reference-interval"
    # ...and when every axis of that document's own effective kernel clamps to
    # 1, the heterogeneous rule reaches for the rank-1 residual gate instead.
    single_key_heterogeneous = MixedCase(
        "oracle",
        "mixed-images",
        ((1, 1, 1), (5, 8, 8)),
        (3, 3, 3),
        (True, False, False),
    )
    assert (
        document_comparison_rule(single_key_heterogeneous, (1, 1, 1))
        == "single-key-rows"
    )


def test_document_adjudication_separates_bitwise_and_lowered_documents():
    case = MixedCase(
        "oracle",
        "mixed",
        ((1, 3, 4), (7, 3, 4)),
        (5, 3, 3),
        (False, False, False),
        seed=17,
    )
    data, grad = make_inputs(case)
    ref = reference(data, grad, case)
    exact = {k: v.to(torch.float32) for k, v in ref.items()}
    rows = document_adjudication(exact, exact, ref, data, grad, case)
    assert [r["rule"] for r in rows] == ["reference-interval", "bitwise"]
    assert all(r["pass"] for r in rows)
    for index, expected in ((0, True), (1, False)):
        drifted = {k: v.clone() for k, v in exact.items()}
        row = case.offsets[index]
        drifted["out"][row, 0, 0] = torch.nextafter(
            drifted["out"][row, 0, 0], torch.tensor(torch.inf)
        )
        verdict = document_adjudication(drifted, exact, ref, data, grad, case)
        assert verdict[index]["pass"] is expected


def test_single_key_document_gate_bounds_the_rank_one_residual():
    case = MixedCase(
        "oracle", "single-key-document", ((1,), (9,)), (5,), (True,), seed=17
    )
    data, grad = make_inputs(case)
    ref = reference(data, grad, case)
    exact = {k: v.to(torch.float32) for k, v in ref.items()}
    rows = document_adjudication(exact, exact, ref, data, grad, case)
    assert [r["rule"] for r in rows] == ["single-key-rows", "bitwise"]
    assert all(r["pass"] for r in rows)
    scale = case.dim**-0.5
    # float32 pays the 3xTF32 product error on top of the reassociation term.
    assert single_key_scalar_bound(
        grad[0, 0], data[2][0, 0], torch.float32
    ) > 8 * ulp32(prefix_sum_bound(grad[0, 0], data[2][0, 0]))
    for multiplier, expected in ((0.5, True), (500.0, False)):
        planted = {k: v.clone() for k, v in exact.items()}
        for head in range(case.heads):
            bound = single_key_scalar_bound(
                grad[0, head], data[2][0, head], torch.float32
            )
            coefficient = multiplier * bound * scale
            planted["dq"][0, head] = (coefficient * data[1][0, head].double()).float()
            planted["dk"][0, head] = (coefficient * data[0][0, head].double()).float()
        verdict = document_adjudication(planted, exact, ref, data, grad, case)
        assert verdict[0]["pass"] is expected


def test_3xtf32_emulation_splits_and_stays_exact_on_representable_products():
    values = torch.tensor(
        [[1.2345678, -3.14159265, 1e-8, 0.0, 65504.0]], dtype=torch.float32
    )
    big, small = tf32_split(values)
    assert torch.equal(
        matmul_3xtf32(torch.ones(1, 1), values), (big.double() + small.double()).float()
    )
    left = torch.tensor([[1.5, -2.25]], dtype=torch.float32)
    right = torch.tensor([[2.25], [4.0]], dtype=torch.float32)
    assert torch.equal(matmul_3xtf32(left, right), left @ right)
    assert torch.equal(tf32_split(left)[1], torch.zeros_like(left))


def test_cotangent_scale_selection_lifts_subnormal_scores():
    from .mixed_batch_properties import cotangent_scale_exponent

    subnormal = torch.tensor([2.0**-16, 0.0, 1.0])
    assert cotangent_scale_exponent(subnormal, 1.0, torch.float16) == 3
    assert cotangent_scale_exponent(subnormal, 1.0, torch.bfloat16) == 0
    assert cotangent_scale_exponent(torch.tensor([1.0, -0.5]), 1.0, torch.float16) == 0
    unreachable = torch.tensor([2.0**-40])
    assert cotangent_scale_exponent(unreachable, 1.0, torch.float16) is None
    assert cotangent_scale_exponent(subnormal, 2.0**13, torch.float16) is None


def test_uniform_mean_and_gradients():
    case = MixedCase(
        "oracle", "uniform", ((4,),), (4,), (False,), heads=1, kv_heads=1, dim=1, vdim=1
    )
    q, k = torch.zeros(4, 1, 1), torch.zeros(4, 1, 1)
    v = torch.tensor([1.0, 3.0, 5.0, 7.0]).reshape(4, 1, 1)
    result = reference([q, k, v], torch.ones_like(v), case)
    assert torch.equal(result["out"], torch.full_like(v, 4))
    assert not result["dq"].any() and not result["dk"].any()
    assert torch.equal(result["dv"], torch.ones_like(v))


def test_reference_backward_finite_difference():
    case = MixedCase(
        "oracle",
        "gradcheck",
        ((2, 3), (3, 2)),
        (3, 3),
        (True, False),
        heads=2,
        kv_heads=1,
        dim=2,
        vdim=3,
        seed=23,
    )
    data, _ = make_inputs(case)
    inputs = tuple(x.double().requires_grad_() for x in data)
    assert torch.autograd.gradcheck(
        lambda *xs: dense_attention(xs, case)[0],
        inputs,
        eps=1e-6,
        atol=1e-5,
        rtol=1e-3,
        fast_mode=True,
    )


def test_future_edge_mutation_is_detected():
    case = MixedCase("oracle", "causal", ((5,),), (3,), (True,), focus=(0, 0), seed=41)
    data, grad = make_inputs(case)
    correct = reference(data, grad, case)
    assert checks_pass(structural_checks(correct, data, grad, case))
    wrong = reference(
        data, grad, case, mask=coordinate_mask(replace(case, causal=(False,)))
    )
    checks = structural_checks(wrong, data, grad, case)
    assert checks["unreachable_key_dv_zero"]["nonzero"] > 0
    assert not checks_pass(checks)


def test_cross_document_edge_mutation_is_detected():
    case = MixedCase(
        "oracle", "isolation", ((2,), (3,)), (3,), (False,), focus=(0, 0), seed=53
    )
    data, grad = make_inputs(case)
    correct = reference(data, grad, case)
    assert checks_pass(structural_checks(correct, data, grad, case))
    wrong = reference(data, grad, case, mask=torch.ones(5, 5, dtype=torch.bool))
    assert not checks_pass(structural_checks(wrong, data, grad, case))


def test_output_order_mutation_is_detected():
    case = MixedCase("oracle", "order", ((2,), (2,)), (1,), (False,), seed=67)
    data, grad = make_inputs(case)
    result = reference(data, grad, case)
    wrong = result["out"].roll(2, dims=0)
    assert error_metrics(result["out"], result["out"])["exact"]
    assert not error_metrics(wrong, result["out"])["exact"]


def test_finite_inputs_with_low_precision_product_overflow():
    x = torch.full((16,), 256, dtype=torch.float16)
    exact = (x.double() * x.double()).sum()
    wrong = (x * x).float().sum()
    correct = (x.float() * x.float()).sum()
    assert exact == 1048576 and torch.isfinite(x).all()
    assert error_metrics(wrong, exact)["nonfinite"] == 1
    assert error_metrics(correct, exact)["exact"]


def test_tiny_products_and_empty_reference():
    x = torch.full((16,), 2**-14, dtype=torch.float16)
    assert (x.double() * x.double()).sum() == 2**-24
    assert (x * x).float().sum() == 0
    case = MixedCase("oracle", "empty", ((0,), (0,)), (3,), (True,))
    data, grad = make_inputs(case)
    result = reference(data, grad, case)
    assert result["out"].shape == (0, 2, 16)
    assert checks_pass(structural_checks(result, data, grad, case))


def test_nonfinite_errors_remain_serializable_and_fail():
    values = torch.tensor([float("nan"), float("inf")])
    metrics = error_metrics(values, values)
    assert not metrics["exact"] and metrics["nonfinite"] == 2
    json.dumps(metrics, allow_nan=False)


def test_paired_timing_statistics_known_effects():
    from scripts.bench_mixed_batch import paired_summary

    null = [{"a": {"cuda_ms": x}, "b": {"cuda_ms": x}} for x in (1, 2, 3, 4)]
    positive = [{"a": {"cuda_ms": x}, "b": {"cuda_ms": x + 2}} for x in (1, 2, 3, 4)]
    assert paired_summary(null)["paired_mean_ci95_ms"] == [0, 0]
    assert paired_summary(positive)["paired_mean_ci95_ms"] == [2, 2]


def test_virtual_gpu_client_identity_and_competition():
    from scripts.bench_mixed_batch import identify_client, parse_clients

    import pytest

    assert identify_client(set(), {87}, 87) == 87
    assert identify_client({51}, {51, 92}, 87) == 92
    with pytest.raises(RuntimeError):
        identify_client({51}, {51, 92, 93}, 87)
    assert parse_clients("GPU-a, 51\nGPU-b, 99\nGPU-a, 92\n", "GPU-a") - {92} == {51}


def test_graph_fingerprint_rejects_distinct_error_classes():
    from .mixed_batch_properties import fingerprint_checks, fingerprint_pass

    expected = torch.tensor([[0.5, 0.5, 0.0], [0.0, 0.5, 0.5]])
    radius = torch.zeros_like(expected)
    assert fingerprint_pass(fingerprint_checks(expected, expected, radius))
    missing = expected.clone()
    missing[0, 1] = 0
    extra = expected.clone()
    extra[0, 2] = 0.5
    for wrong in (missing, extra, expected.roll(1, 1), expected * 2, -expected):
        assert not fingerprint_pass(fingerprint_checks(wrong, expected, radius))


def test_directional_derivative_checker_detects_wrong_backward():
    from .mixed_batch_properties import ad_interval

    assert ad_interval(2, 2, 2, 2, 0)["pass"]
    assert ad_interval(2, 2.01, 2, 2, 0)["pass"]
    assert not ad_interval(0, 2.01, 2, 2, 0)["pass"]
    assert not ad_interval(-2, 2.01, 2, 2, 0)["pass"]


def test_generated_graph_corpus_exercises_different_structures():
    from .mixed_batch_properties import generated_graphs, graph_coverage

    graphs = generated_graphs()
    coverage = graph_coverage(graphs)
    for capability in (
        "rank-1",
        "rank-2",
        "rank-3",
        "dilation",
        "stride",
        "isolated-axis",
        "empty-document",
        "single-key-row",
        "multiple-key-row",
        "shared-kv-heads",
        "noncontiguous",
    ):
        assert coverage[capability] > 0
    assert len({c.seed for c in graphs}) == len(graphs)


def test_scaling_law_accounts_for_final_subnormal_rounding():
    from .mixed_batch_properties import rounded_scaling_check

    unit = 2.0**-24
    unrounded = torch.tensor([0.4 * unit], dtype=torch.float64)
    original = unrounded.half()
    doubled = (unrounded * 2).half()
    result = rounded_scaling_check(original, doubled)
    assert not result["bitwise"] and result["pass"]
    assert not rounded_scaling_check(
        original, torch.tensor([4 * unit], dtype=torch.float16)
    )["pass"]


def test_derivative_interval_on_independent_fp32_and_wrong_backward():
    from .mixed_batch_properties import generated_graphs, forward_backward_consistency

    def fp32_evaluate(data, grad, case):
        return reference(data, grad, case, dtype=torch.float32)

    for case in generated_graphs(count=9, seed=77239):
        correct = forward_backward_consistency(
            case, dense_attention, fp32_evaluate, device="cpu"
        )
        assert correct["status"] == "PASS", correct

    def wrong_evaluate(data, grad, case):
        result = fp32_evaluate(data, grad, case)
        result["dv"].zero_()
        return result

    wrong = forward_backward_consistency(
        case, dense_attention, wrong_evaluate, device="cpu"
    )
    assert wrong["status"] == "FAIL"
