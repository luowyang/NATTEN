"""Generated attention graphs and implementation-independent properties."""

import math
import random
from dataclasses import replace

import torch

from .mixed_batch_utils import (
    MixedCase,
    coordinate_mask,
    dense_attention,
    dense_backward_3xtf32,
    error_metrics,
    make_inputs,
    reference,
    single_key_scalar_bound,
)
from .varlen_numerics import dtype_rounding_radius, dtype_spacing


def generated_graphs(count=24, seed=41011):
    """A bounded graph grammar; the corpus is reproducible from each seed."""
    cases = []
    for index in range(count):
        case_seed = seed + 104729 * index
        rng = random.Random(case_seed)
        rank = 1 + index % 3
        kernel = tuple(rng.randint(1, 4) for _ in range(rank))
        dilation = [1] * rank
        if index % 4 == 0:
            axis = rng.randrange(rank)
            dilation[axis] = 2 if kernel[axis] > 1 else 1
        for _ in range(1000):
            shapes = []
            for doc in range(rng.randint(2, 4)):
                shape = tuple(
                    rng.randint(k * d, k * d + 2) if d > 1 else rng.randint(1, 4)
                    for k, d in zip(kernel, dilation)
                )
                if index % 5 == 0 and doc == 0:
                    axis = rng.randrange(rank)
                    shape = tuple(0 if i == axis else n for i, n in enumerate(shape))
                shapes.append(shape)
            if 1 < sum(math.prod(s) for s in shapes) <= 96:
                break
        else:
            raise RuntimeError(
                "Graph grammar failed to satisfy the declared token bound"
            )
        kv_heads = 1 + index % 2
        cases.append(
            MixedCase(
                name=f"graph-{case_seed}",
                ability="generated attention graph",
                shapes=tuple(shapes),
                kernel=kernel,
                causal=tuple(bool(rng.getrandbits(1)) for _ in range(rank)),
                stride=tuple(rng.randint(1, k) for k in kernel),
                dilation=tuple(dilation),
                dtype=("float32", "float16", "bfloat16")[(index // 3) % 3],
                heads=kv_heads * (1 + index % 3),
                kv_heads=kv_heads,
                seed=case_seed,
                noncontiguous=index % 4 == 1,
            )
        )
    return tuple(cases)


def graph_coverage(cases):
    counters = {}
    for case in cases:
        mask = coordinate_mask(case)
        labels = [f"rank-{case.rank}", case.dtype]
        labels += ["causal" if any(case.causal) else "noncausal"]
        labels += ["shared-kv-heads" if case.heads > case.kv_heads else "equal-heads"]
        if any(d > 1 for d in case.dilations):
            labels.append("dilation")
        if any(s > 1 for s in case.strides):
            labels.append("stride")
        if any(k == 1 for k in case.kernel):
            labels.append("isolated-axis")
        if any(
            n < k
            for shape in case.shapes
            if math.prod(shape)
            for n, k in zip(shape, case.kernel)
        ):
            labels.append("clamped-axis")
        if 0 in case.lengths:
            labels.append("empty-document")
        if bool((mask.sum(1) == 1).any()):
            labels.append("single-key-row")
        if bool((mask.sum(1) > 1).any()):
            labels.append("multiple-key-row")
        if case.noncontiguous:
            labels.append("noncontiguous")
        for label in labels:
            counters[label] = counters.get(label, 0) + 1
    return counters


def fingerprint_checks(actual, expected, radius):
    a, b = actual.detach().cpu().double(), expected.detach().cpu().double()
    allowed = b.ne(0)
    wrong_support = a.ne(0) != allowed
    outside = a[~allowed]
    return {
        "finite": bool(torch.isfinite(a).all()),
        "forbidden_entries_zero": bool(outside.eq(0).all()),
        "allowed_entries_positive": bool((a[allowed] > 0).all()),
        "coefficient_calibration": bool(((a - b).abs() <= radius).all()),
        "wrong_support_count": int(wrong_support.sum()),
        "first_wrong_support": wrong_support.nonzero()[:12].tolist(),
        "metrics": error_metrics(a, b),
    }


def fingerprint_pass(checks):
    return all(
        checks[k]
        for k in (
            "finite",
            "forbidden_entries_zero",
            "allowed_entries_positive",
            "coefficient_calibration",
        )
    )


def connectivity(case, call_fn):
    """Encode every key as a V basis vector, then every query as a dO basis."""
    n = sum(case.lengths)
    width = ((n * case.kv_heads + 7) // 8) * 8
    case = replace(case, vdim=width, noncontiguous=False)
    dtype = getattr(torch, case.dtype)
    mask = coordinate_mask(case)
    weights = mask.double() / mask.sum(1, keepdim=True)
    # Calibrate the single reciprocal plus final dtype rounding separately.
    divided = mask.to(device="cuda", dtype=torch.float32) / mask.sum(
        1, keepdim=True
    ).to("cuda")
    scalar_reference = divided.to(dtype).double().cpu()
    reconstructed = (
        mask.to("cuda")
        * (
            -mask.sum(1, keepdim=True).to(device="cuda", dtype=torch.float32).log()
        ).exp()
    ).to(dtype)
    spacing = dtype_spacing(divided, dtype)
    exp_spacing = dtype_spacing(reconstructed, dtype)
    radius = torch.maximum(
        (scalar_reference - weights).abs(),
        (reconstructed.double().cpu() - weights).abs(),
    ) + torch.maximum(spacing, exp_spacing)
    radius[~mask] = 0
    q = torch.zeros(
        n, case.heads, case.dim, dtype=dtype, device="cuda", requires_grad=True
    )
    k = torch.zeros(
        n, case.kv_heads, case.dim, dtype=dtype, device="cuda", requires_grad=True
    )
    v = torch.zeros(n, case.kv_heads, width, dtype=dtype, device="cuda")
    for head in range(case.kv_heads):
        v[:, head, head * n : (head + 1) * n] = torch.eye(n, dtype=dtype, device="cuda")
    v.requires_grad_()
    out, _ = call_fn([q, k, v], case)
    expected = torch.zeros(n, case.heads, width, dtype=torch.float64)
    bound = torch.zeros_like(expected)
    for head in range(case.heads):
        kv_head = head // (case.heads // case.kv_heads)
        expected[:, head, kv_head * n : (kv_head + 1) * n] = weights
        bound[:, head, kv_head * n : (kv_head + 1) * n] = radius
    forward = fingerprint_checks(out, expected, bound)
    backward = []
    for head in range(case.heads):
        grad = torch.zeros_like(out)
        grad[:, head, :n] = torch.eye(n, dtype=dtype, device="cuda")
        dq, dk, dv = torch.autograd.grad(
            out, (q, k, v), grad, retain_graph=head + 1 < case.heads
        )
        expected_v = torch.zeros(n, case.kv_heads, width, dtype=torch.float64)
        bound_v = torch.zeros_like(expected_v)
        kv_head = head // (case.heads // case.kv_heads)
        expected_v[:, kv_head, :n] = weights.T
        bound_v[:, kv_head, :n] = radius.T
        check = fingerprint_checks(dv, expected_v, bound_v)
        check["zero_qk_gradients"] = bool(dq.eq(0).all() and dk.eq(0).all())
        backward.append(check)
    passed = fingerprint_pass(forward) and all(
        fingerprint_pass(c) and c["zero_qk_gradients"] for c in backward
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "graph": case.descriptor(),
        "nodes": n,
        "allowed_edges": int(mask.sum()),
        "forbidden_edges": int((~mask).sum()),
        "forward": forward,
        "backward_by_query_head": backward,
        "calibration": "Same degrees and dtype, independent CUDA reciprocal and exp(-log(degree)), then cast; one output ULP allowance. Zero support is exact.",
    }


def cotangent_scores(data, grad, case, mask=None):
    """float64 dS*scale, the quantity the backward stores in shared memory.

    delta is taken as rowsum(P*dP), which is rowsum(dO*O) rearranged; evaluating it
    that way makes a one-hot row cancel exactly instead of leaving an fp64 rounding
    residue that would look like a genuinely tiny score.
    """
    mask = coordinate_mask(case) if mask is None else mask
    repeats = case.heads // case.kv_heads
    q = data[0].detach().cpu().double()
    k = data[1].detach().cpu().double().repeat_interleave(repeats, dim=1)
    v = data[2].detach().cpu().double().repeat_interleave(repeats, dim=1)
    upstream = grad.detach().cpu().double()
    scale = float(case.scale if case.scale is not None else case.dim**-0.5)
    rows = []
    for head in range(case.heads):
        scores = (q[:, head] @ k[:, head].t()) * scale
        weights = scores.masked_fill(~mask, -torch.inf).softmax(-1)
        dp = upstream[:, head] @ v[:, head].t()
        delta = (weights * dp).sum(-1, keepdim=True)
        rows.append(weights * (dp - delta) * scale)
    return torch.stack(rows)


def cotangent_scale_exponent(scores, magnitude, dtype, span=(-8, 12)):
    """Smallest k >= 0 keeping every nonzero dS*scale out of the fp16 subnormals.

    fp16 stores dS*scale in shared memory, and subnormal rounding is not equivariant
    under doubling. bf16 and float32 share fp32's exponent range and need no shift.
    """
    if dtype != torch.float16:
        return 0
    nonzero = scores.abs()[scores != 0]
    smallest = float(nonzero.min()) if nonzero.numel() else torch.inf
    for exponent in range(max(0, span[0]), span[1] + 1):
        if (
            smallest * 2.0**exponent >= 2.0**-13
            and magnitude * 2.0 ** (exponent + 1) <= 2.0**14
        ):
            return exponent
    return None


def single_key_residual_bounds(data, grad, case, mask):
    """Per-position bound on the dQ/dK a single-key row can carry.

    Such a row's dS is the residual of one dot product taken over two summation
    paths, so its gradient is the rank-1 image of that scalar and obeys the same
    bound the single-key regression gate uses, `single_key_scalar_bound`. Everywhere
    else the exact gradient is zero only when no edge contributes, and the kernel
    must return exactly zero.
    """
    dtype = getattr(torch, case.dtype)
    repeats = case.heads // case.kv_heads
    q, k, v = (x.detach().cpu() for x in data)
    upstream = grad.detach().cpu()
    scale = float(case.scale if case.scale is not None else case.dim**-0.5)
    tokens = q.shape[0]
    bounds = {
        "dq": torch.zeros(tokens, case.heads, case.dim, dtype=torch.float64),
        "dk": torch.zeros(tokens, case.kv_heads, case.dim, dtype=torch.float64),
        "dv": torch.zeros(tokens, case.kv_heads, case.vdim, dtype=torch.float64),
    }
    for row in (mask.sum(1) == 1).nonzero().flatten().tolist():
        key = int(mask[row].nonzero()[0])
        for head in range(case.heads):
            kv_head = head // repeats
            coefficient = scale * single_key_scalar_bound(
                upstream[row, head], v[key, kv_head], dtype
            )
            bounds["dq"][row, head] += coefficient * k[key, kv_head].double().abs()
            bounds["dk"][key, kv_head] += coefficient * q[row, head].double().abs()
    return bounds


def structural_zero_residual(original, doubled, selection, bound):
    """Positions whose exact gradient is zero carry only the single-key dot residual."""
    values = original.detach().cpu().double().abs()
    twice = doubled.detach().cpu().double().abs()
    allowed = bound + dtype_rounding_radius(original, original.dtype)
    allowed_twice = 2 * bound + dtype_rounding_radius(doubled, doubled.dtype)
    outside = ((values > allowed) | (twice > allowed_twice)) & selection
    ratio = torch.maximum(values / allowed, twice / allowed_twice)[selection]
    return {
        "positions": int(selection.sum()),
        "pass": not bool(outside.any()),
        "outside": int(outside.sum()),
        "max_original": float(values[selection].max()) if selection.any() else 0.0,
        "max_doubled": float(twice[selection].max()) if selection.any() else 0.0,
        "max_bound_ratio": float(ratio.max()) if selection.any() else 0.0,
        "first_outside": outside.nonzero()[:12].tolist(),
    }


def vjp_linearity(case, evaluate_fn):
    data, grad = make_inputs(case, "cuda")
    repeats = case.heads // case.kv_heads
    if repeats > 1:
        # Expanded to per-Q-head KV: the GQA sum happens in Python and turns a
        # sub-ulp perturbation of one head into a whole ulp of the summed dK.
        data = [data[0]] + [x.repeat_interleave(repeats, dim=1) for x in data[1:]]
        case = replace(case, kv_heads=case.heads)
    dtype = getattr(torch, case.dtype)
    ref = reference(data, grad, case)
    magnitude = max(
        [float(grad.abs().max()) if grad.numel() else 0.0]
        + [float(ref[n].abs().max()) if ref[n].numel() else 0.0 for n in ("dq", "dk", "dv")]
    )
    mask = coordinate_mask(case)
    scores = cotangent_scores(data, grad, case, mask)
    exponent = cotangent_scale_exponent(scores, magnitude, dtype)
    nonzero = scores.abs()[scores != 0]
    record = {
        "head_expanded": repeats > 1,
        "cotangent_exponent": exponent,
        "smallest_nonzero_score": float(nonzero.min()) if nonzero.numel() else 0.0,
        "reference_magnitude": magnitude,
    }
    if exponent is None:
        return {
            "status": "NOT_APPLICABLE",
            "reason": "float16 dynamic range: no cotangent scale 2^k with k in [0, 12] "
            "keeps every nonzero dS*scale at or above 2^-13 with the doubled outputs "
            "below 2^14",
            **record,
        }
    grad = grad * 2.0**exponent
    bounds = single_key_residual_bounds(data, grad, case, mask)
    original = evaluate_fn(data, grad, case)
    zero = evaluate_fn(data, torch.zeros_like(grad), case)
    opposite = evaluate_fn(data, -grad, case)
    doubled = evaluate_fn(data, grad * 2, case)
    rows = {}
    for name in ("dq", "dk", "dv"):
        exact_zero = ref[name].eq(0)
        scaling = rounded_scaling_check(original[name], doubled[name], ~exact_zero)
        residual = structural_zero_residual(
            original[name], doubled[name], exact_zero, bounds[name]
        )
        rows[name] = {
            "zero_cotangent": bool(zero[name].eq(0).all()),
            "opposite_cotangent": torch.equal(opposite[name], -original[name]),
            "power_of_two_cotangent": scaling["pass"],
            "scaling_metrics": scaling,
            "structural_zero_bound": residual["pass"],
            "structural_zero_residual": residual,
            "finite": bool(
                torch.isfinite(original[name]).all()
                and torch.isfinite(doubled[name]).all()
            ),
        }
    return {
        "status": "PASS"
        if all(
            all(
                c[k]
                for k in (
                    "zero_cotangent",
                    "opposite_cotangent",
                    "power_of_two_cotangent",
                    "structural_zero_bound",
                    "finite",
                )
            )
            for c in rows.values()
        )
        else "FAIL",
        **record,
        "checks": rows,
    }


def random_vjp_support(case, call_fn):
    """Probe every output query with nonzero Q/K and a random positive loss."""
    data, _ = make_inputs(case, "cuda")
    data = [x.detach().requires_grad_() for x in data]
    mask = coordinate_mask(case)
    output, _ = call_fn(data, case)
    n = output.shape[0]
    generator = torch.Generator().manual_seed(case.seed + 619)
    rows = []
    for row in range(n):
        grad = torch.zeros_like(output)
        grad[row] = (torch.rand(output[row].shape, generator=generator) + 1).to(output)
        grads = torch.autograd.grad(output, data, grad, retain_graph=row + 1 < n)
        dq, dk, dv = [g.detach().cpu() for g in grads]
        other_queries = torch.arange(n) != row
        forbidden = ~mask[row]
        checks = {
            "other_query_dq_zero": bool(dq[other_queries].eq(0).all()),
            "forbidden_dk_zero": bool(dk[forbidden].eq(0).all()),
            "forbidden_dv_zero": bool(dv[forbidden].eq(0).all()),
            "finite": all(bool(torch.isfinite(g).all()) for g in (dq, dk, dv)),
        }
        rows.append(
            {"query": row, "allowed_keys": int(mask[row].sum()), "checks": checks}
        )
    return {
        "status": "PASS" if all(all(r["checks"].values()) for r in rows) else "FAIL",
        "queries": rows,
        "scope": "One independent positive loss projection per query with random nonzero Q/K/V. V basis probes separately check every allowed edge.",
    }


def rounded_scaling_check(original, doubled, include=None):
    difference = (doubled.double() - original.double() * 2).abs()
    radius = 2 * dtype_rounding_radius(
        original, original.dtype
    ) + dtype_rounding_radius(doubled, doubled.dtype)
    failed = difference > radius
    if include is not None:
        failed = failed & include.cpu()
        difference = difference[include.cpu()]
    return {
        "pass": bool(torch.isfinite(difference).all() and not failed.any()),
        "bitwise": torch.equal(doubled, original * 2),
        "compared_entries": int(difference.numel()),
        "outside_rounding_intervals": int(failed.sum()),
        "max_abs": float(difference.max())
        if difference.numel() and torch.isfinite(difference).all()
        else None,
        "first_outside": failed.nonzero()[:12].tolist(),
    }


def ad_interval(
    actual_ad,
    actual_fd,
    reference_ad,
    reference_fd,
    gradient_rounding,
    backward_arithmetic=0.0,
):
    # Every term is measured independently on the perturbed inputs; the last one is
    # the FP32 backward's own 3xTF32 arithmetic, measured by re-running the dense
    # backward with the same GEMM emulation.
    radius = (
        abs(actual_fd - reference_fd)
        + abs(reference_fd - reference_ad)
        + gradient_rounding
        + backward_arithmetic
    )
    residual = abs(actual_ad - actual_fd)
    return {
        "actual_ad": actual_ad,
        "actual_fd": actual_fd,
        "reference_ad": reference_ad,
        "forward_numerical_term": abs(actual_fd - reference_fd),
        "finite_difference_term": abs(reference_fd - reference_ad),
        "gradient_rounding_term": gradient_rounding,
        "backward_arithmetic_term": backward_arithmetic,
        "radius": radius,
        "residual": residual,
        "pass": residual <= radius,
    }


def forward_backward_consistency(case, call_fn, evaluate_fn, device="cuda"):
    case = replace(case, dtype="float32", noncontiguous=False)
    data, grad = make_inputs(case, device)
    actual = evaluate_fn(data, grad, case)
    ref = reference(data, grad, case)
    emulated = dense_backward_3xtf32(data, grad, case)
    generator = torch.Generator().manual_seed(case.seed + 179)
    rows = []
    for axis, grad_name in enumerate(("dq", "dk", "dv")):
        direction = torch.randn(
            data[axis].shape, generator=generator, dtype=torch.float64
        )
        direction /= direction.norm()
        actual_ad = float((actual[grad_name].double() * direction).sum())
        reference_ad = float((ref[grad_name] * direction).sum())
        emulated_ad = float((emulated[grad_name].double() * direction).sum())
        backward_arithmetic = 2 * abs(emulated_ad - reference_ad)
        g = actual[grad_name]
        spacing = dtype_spacing(g, g.dtype)
        gradient_rounding = float((spacing * direction.abs()).sum())
        for step in (0.125, 0.0625):
            outputs, refs = [], []
            for sign in (-1, 1):
                perturbed = [x.detach().clone() for x in data]
                perturbed[axis] = (
                    data[axis].double() + sign * step * direction.to(device)
                ).float()
                with torch.no_grad():
                    outputs.append(call_fn(perturbed, case)[0].cpu().double())
                    refs.append(
                        dense_attention([x.cpu().double() for x in perturbed], case)[0]
                    )
            actual_fd = float(
                ((outputs[1] - outputs[0]) * grad.cpu().double()).sum() / (2 * step)
            )
            reference_fd = float(
                ((refs[1] - refs[0]) * grad.cpu().double()).sum() / (2 * step)
            )
            rows.append(
                {
                    "input": grad_name[1:],
                    "step": step,
                    "emulated_3xtf32_ad": emulated_ad,
                    **ad_interval(
                        actual_ad,
                        actual_fd,
                        reference_ad,
                        reference_fd,
                        gradient_rounding,
                        backward_arithmetic,
                    ),
                }
            )
    return {
        "status": "PASS" if all(r["pass"] for r in rows) else "FAIL",
        "probes": rows,
        "scope": "FP32 local derivative consistency; the FP32 kernels run 3xTF32, so the interval carries a measured term for the backward's own arithmetic. Lower precision and general forward error are separate checks.",
    }


def representability(case, evaluate_fn):
    base_data, grad = make_inputs(case, "cuda")
    dtype = getattr(torch, case.dtype)
    exponent = min(20, math.floor(math.log2(torch.finfo(dtype).max)) - 1)
    grad = torch.ones_like(grad) * 4
    rows = []
    for power in (-12, 0, exponent):
        data = [x.clone() for x in base_data]
        data[2].fill_(2.0**power)
        actual = evaluate_fn(data, grad, case)
        ref = reference(data, grad, case)
        expected_out = torch.full_like(actual["out"], 2.0**power)
        finite = all(bool(torch.isfinite(x).all()) for x in actual.values())
        rows.append(
            {
                "value_exponent": power,
                "finite": finite,
                "out_constant_error": error_metrics(actual["out"], expected_out),
                "algebraic_zero_dq": error_metrics(
                    actual["dq"], torch.zeros_like(actual["dq"])
                ),
                "algebraic_zero_dk": error_metrics(
                    actual["dk"], torch.zeros_like(actual["dk"])
                ),
                "dv_reference": error_metrics(actual["dv"], ref["dv"]),
                "expected_dv_representable": bool(
                    torch.isfinite(ref["dv"].to(dtype)).all()
                ),
            }
        )
    return {
        "status": "PASS"
        if all(r["finite"] and r["expected_dv_representable"] for r in rows)
        else "FAIL",
        "scales": rows,
        "scope": "Finite representable constant-value attention; algebraic cancellation residuals are reported, not treated as missing graph edges.",
    }
