#!/usr/bin/env python3
"""Run capability checks and paired performance measurements for mixed NA batches."""

import argparse
import collections
import hashlib
import json
import os
import pathlib
import re
import shutil
import socket
import subprocess
import sys
import time
import traceback
from dataclasses import replace

# Capture GPU clients before imports can initialize a CUDA context.
STARTUP_GPU_CLIENTS = None
if __name__ == "__main__":
    probe = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
    )
    if probe.returncode == 0:
        STARTUP_GPU_CLIENTS = probe.stdout

import torch  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
GPU_IDENTITY = None

import natten  # noqa: E402
from tests.mixed_batch_utils import (  # noqa: E402
    capability_cases,
    checks_pass,
    coordinate_mask,
    document_adjudication,
    error_metrics,
    make_inputs,
    MixedCase,
    reference,
    structural_checks,
)
from tests.mixed_batch_properties import (  # noqa: E402
    connectivity,
    forward_backward_consistency,
    generated_graphs,
    graph_coverage,
    representability,
    random_vjp_support,
    vjp_linearity,
)
from tests.varlen_numerics import dtype_spacing  # noqa: E402


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def command_output(args):
    return subprocess.run(
        args, capture_output=True, text=True, check=True
    ).stdout.strip()


def environment():
    libdir = pathlib.Path(natten.__file__).parent
    props = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
    return {
        "host": socket.gethostname(),
        "argv": sys.argv,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "natten": natten.__version__,
        "natten_file": natten.__file__,
        "library_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in libdir.glob("*.so")
        },
        "python_sha256": {
            str(p.relative_to(libdir)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in libdir.rglob("*.py")
        },
        "repo_head": command_output(["git", "-C", str(REPO), "rev-parse", "HEAD"]),
        "runtime_capability": [props.major, props.minor] if props else None,
        "sm_count": props.multi_processor_count if props else None,
        "total_memory": props.total_memory if props else None,
        "device_uuid": str(props.uuid) if props and hasattr(props, "uuid") else None,
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_process_identity": GPU_IDENTITY,
        "deterministic": torch.are_deterministic_algorithms_enabled(),
        "tf32": torch.backends.cuda.matmul.allow_tf32,
    }


def call(data, case, layout=None, return_lse=True):
    if layout is None:
        layout = natten.VarlenLayout(case.shapes, device=data[0].device)
    return getattr(natten, "na%dd_varlen" % case.rank)(
        *data,
        layout,
        kernel_size=case.kernel,
        stride=case.strides,
        dilation=case.dilations,
        is_causal=case.causal,
        scale=case.scale,
        backend="cutlass-fna",
        return_lse=return_lse,
    )


def evaluate(data, grad, case, layout=None):
    leaves = [x.detach().requires_grad_() for x in data]
    out, lse = call(leaves, case, layout)
    grads = torch.autograd.grad(out, leaves, grad)
    return dict(
        zip(
            ("out", "lse", "dq", "dk", "dv"),
            (x.detach().cpu() for x in (out, lse, *grads)),
        )
    )


def compare(a, b):
    return {key: error_metrics(a[key], b[key]) for key in a}


def exact_outputs(metrics):
    return all(metrics[key]["exact"] for key in ("out", "dq", "dk", "dv"))


def ulp_difference(a, b):
    difference = (a.detach().double() - b.detach().double()).abs()
    ratio = difference / dtype_spacing(a, a.dtype)
    return {
        "bitwise": torch.equal(a, b),
        "differing": int((difference != 0).sum()),
        "max_abs": float(difference.max()) if difference.numel() else 0.0,
        "max_ulp": float(ratio.max()) if ratio.numel() else 0.0,
    }


def nondeterministic_parity(packed, split):
    """A/A repeats calibrate the split-KV summation noise the cross-mode dQ must respect.

    Only dQ crosses the atomically accumulated fp32 workspace, so out/LSE/dK/dV stay
    bitwise stable; dQ is judged against the noise its own mode reproduces.
    """
    stable = ("out", "lse", "dk", "dv")
    tensors = ("out", "lse", "dq", "dk", "dv")
    pairs = {"packed_aa": (packed[0], packed[1]), "split_aa": (split[0], split[1])}
    for i, p in enumerate(packed):
        for j, s in enumerate(split):
            pairs["cross_%d%d" % (i + 1, j + 1)] = (p, s)
    rows = {
        name: {t: ulp_difference(a[t], b[t]) for t in tensors}
        for name, (a, b) in pairs.items()
    }
    noise = max(rows["packed_aa"]["dq"]["max_ulp"], rows["split_aa"]["dq"]["max_ulp"])
    cross = max(v["dq"]["max_ulp"] for k, v in rows.items() if k.startswith("cross"))
    stable_bitwise = all(rows[name][t]["bitwise"] for name in rows for t in stable)
    return {
        "correctness_status": "PASS"
        if stable_bitwise and cross <= noise + 1
        else "REVIEW",
        "stable_tensors_bitwise": stable_bitwise,
        "dq_aa_max_ulp": noise,
        "dq_cross_max_ulp": cross,
        "dq_allowed_max_ulp": noise + 1,
        "runs": rows,
    }


def per_document(metrics_a, metrics_b, case):
    return [
        compare(
            {k: v[lo:hi] for k, v in metrics_a.items()},
            {k: v[lo:hi] for k, v in metrics_b.items()},
        )
        for lo, hi in zip(case.offsets, case.offsets[1:])
    ]


def core_case(case):
    data, grad = make_inputs(case, "cuda")
    layout = natten.VarlenLayout(case.shapes, device=data[0].device)
    packed = evaluate(data, grad, case, layout)
    repeat = evaluate(data, grad, case, layout)
    ref64 = reference(data, grad, case)
    ref32 = reference(data, grad, case, dtype=torch.float32)
    isolated, isolated_repeat = [], []
    for shape, lo, hi in zip(case.shapes, case.offsets, case.offsets[1:]):
        single = replace(case, shapes=(shape,))
        sliced = [x[lo:hi] for x in data]
        isolated.append(evaluate(sliced, grad[lo:hi], single))
        isolated_repeat.append(evaluate(sliced, grad[lo:hi], single))
    split = {k: torch.cat([r[k] for r in isolated]) for k in packed}
    split_repeat = {k: torch.cat([r[k] for r in isolated_repeat]) for k in packed}
    checks = structural_checks(packed, data, grad, case)
    finite = all(bool(torch.isfinite(x).all()) for x in packed.values())
    adjudication = document_adjudication(packed, split, ref64, data, grad, case)
    comparisons = {
        "packed_repeat": compare(packed, repeat),
        "isolated_repeat": compare(split, split_repeat),
        "packed_vs_isolated": compare(packed, split),
        "packed_vs_isolated_by_document": per_document(packed, split, case),
        "packed_vs_isolated_adjudication": adjudication,
        "fp64_reference_by_document": per_document(packed, ref64, case),
        "fp32_reference_roundoff_by_document": per_document(ref32, ref64, case),
    }
    gates = {"finite": finite, "structural": checks_pass(checks)}
    for key in ("packed_repeat", "isolated_repeat"):
        gates[key] = exact_outputs(comparisons[key])
    gates["packed_vs_isolated"] = all(row["pass"] for row in adjudication)

    order = tuple(reversed(range(len(case.shapes))))
    permutation = torch.cat(
        [
            torch.arange(case.offsets[i], case.offsets[i + 1], device="cuda")
            for i in order
        ]
    )
    inv = permutation.argsort().cpu()
    perm_case = replace(case, shapes=tuple(case.shapes[i] for i in order))
    perm_result = evaluate([x[permutation] for x in data], grad[permutation], perm_case)
    comparisons["reorder_restore"] = compare(
        packed, {k: v[inv] for k, v in perm_result.items()}
    )
    gates["reorder_restore"] = exact_outputs(comparisons["reorder_restore"])

    empty_shape = (0,) + (1,) * (case.rank - 1)
    empty_case = replace(case, shapes=(empty_shape,) + case.shapes + (empty_shape,))
    with_empty = evaluate(data, grad, empty_case)
    comparisons["insert_empty"] = compare(packed, with_empty)
    gates["insert_empty"] = exact_outputs(comparisons["insert_empty"])

    nonempty = [i for i, n in enumerate(case.lengths) if n]
    if len(nonempty) > 1:
        doc = nonempty[-1]
        lo, hi = case.offsets[doc : doc + 2]
        changed = [x.clone() for x in data]
        for index, tensor in enumerate(changed):
            tensor[lo:hi] += index + 1
        mutated = evaluate(changed, grad, case)
        outside = torch.ones(sum(case.lengths), dtype=torch.bool)
        outside[lo:hi] = False
        comparisons["other_document_perturbation"] = compare(
            {k: v[outside] for k, v in packed.items()},
            {k: v[outside] for k, v in mutated.items()},
        )
        gates["other_document_perturbation"] = exact_outputs(
            comparisons["other_document_perturbation"]
        )
        gates["perturbation_positive_control"] = not torch.equal(
            packed["out"][lo:hi], mutated["out"][lo:hi]
        )

    if case.focus is not None:
        row = case.offsets[case.focus[0]] + case.focus[1]
        prohibited = ~coordinate_mask(case)[row]
        changed = [x.clone() for x in data]
        changed[1][prohibited.to("cuda")] += 3
        changed[2][prohibited.to("cuda")] += 7
        mutated = evaluate(changed, grad, case)
        gates["unreachable_input_perturbation"] = torch.equal(
            packed["out"][row], mutated["out"][row]
        )

    if case.noncontiguous:
        contiguous = evaluate([x.contiguous() for x in data], grad, case)
        comparisons["contiguous_values"] = compare(packed, contiguous)
        gates["contiguous_values"] = exact_outputs(comparisons["contiguous_values"])

    return {
        "status": "PASS" if all(gates.values()) else "FAIL",
        "gates": gates,
        "structural": checks,
        "comparisons": comparisons,
        "numerical_adjudication": "FP64 errors are reported independently; strict PASS does not certify their harmlessness.",
    }


def delta_known_answers():
    from natten._libnatten import compute_delta

    rows = []
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        for name, value in (
            ("unit", 1),
            ("large-product", 256),
            ("tiny-product", 2**-14),
        ):
            out = torch.full((1, 1, 1, 16), value, dtype=dtype, device="cuda")
            delta = torch.empty((1, 1, 1), device="cuda", dtype=torch.float32)
            compute_delta(out, out, delta)
            expected = (out.double() * out.double()).sum(-1)
            metrics = error_metrics(delta, expected)
            rows.append(
                {
                    "dtype": str(dtype),
                    "case": name,
                    "status": "PASS" if metrics["exact"] else "FAIL",
                    "metrics": metrics,
                }
            )
    return {
        "status": "PASS" if all(r["status"] == "PASS" for r in rows) else "FAIL",
        "cases": rows,
    }


def layout_reuse():
    layout = natten.VarlenLayout(((1, 7, 9), (9, 9, 7)), device="cuda")
    base = MixedCase(
        "reuse",
        "same layout across layers",
        ((1, 7, 9), (9, 9, 7)),
        (5, 3, 3),
        (True, False, False),
    )
    phases = [
        replace(base, name="temporal", kernel=(5, 1, 1)),
        replace(base, name="spatial", kernel=(1, 3, 3)),
        replace(base, name="3d-bf16", dtype="bfloat16"),
        replace(
            base,
            name="strided-noncausal",
            causal=(False, False, False),
            stride=(2, 2, 2),
        ),
        replace(base, name="spatial-dilation", dilation=(1, 2, 2)),
        replace(base, name="temporal-again", kernel=(5, 1, 1)),
    ]
    rows = []
    for case in phases:
        data, grad = make_inputs(case, "cuda")
        cached = evaluate(data, grad, case, layout)
        fresh = evaluate(data, grad, case)
        metrics = compare(cached, fresh)
        rows.append(
            {
                "case": case.descriptor(),
                "status": "PASS" if exact_outputs(metrics) else "FAIL",
                "metrics": metrics,
            }
        )
    return {
        "status": "PASS" if all(r["status"] == "PASS" for r in rows) else "FAIL",
        "cases": rows,
    }


LEGACY_TESTS = {
    "gradient-values": ["tests/test_varlen_gradients.py"],
    "layout-api": [
        "tests/test_varlen_layout.py",
        "tests/test_varlen_api.py",
        "tests/test_varlen_raw_op_validation.py",
    ],
    "execution": [
        "tests/test_fna_varlen.py::VarlenFnaGpuTests::test_cross_stream_schedule_build_and_consumption",
        "tests/test_fna_varlen.py::VarlenFnaGpuTests::test_correctness_on_non_default_cuda_device",
        "tests/test_fna_varlen.py::VarlenFnaGpuTests::test_noncontiguous_input_backward_with_gqa_and_vdim",
        "tests/test_fna_varlen.py::VarlenFnaGpuTests::test_partial_gradient_and_output_only_paths",
        "tests/test_fna_varlen.py::VarlenFnaGpuTests::test_deterministic_repeated_runs_are_bitwise_equal",
        "tests/test_fna_varlen.py::VarlenFnaGpuTests::test_forward_captures_deterministic_state_for_backward",
        "tests/test_fna_varlen.py::VarlenFnaGpuTests::test_fullgraph_and_aot_autograd",
        "tests/test_fna_varlen.py::VarlenFnaGpuTests::test_fullgraph_compile_empty_documents",
        "tests/test_fna_varlen.py::VarlenFnaGpuTests::test_cold_miss_default_budget_correct_and_bounded_recompile",
        "tests/test_fna_varlen.py::VarlenFnaGpuTests::test_distinct_layouts_amortize_under_torch_compile",
    ],
    "degenerate-compile": ["tests/test_varlen_degenerate_axes.py", "-k", "compile"],
    "uniform-dispatch": ["tests/test_varlen_uniform_dispatch.py"],
    "layout-lifecycle": [
        "tests/test_fna_varlen.py::VarlenFnaGpuTests::test_extended_many_tiny_docs_leak_loop"
    ],
}


def pytest_group(name, paths, output):
    xml = output / (name + ".xml")
    command = [sys.executable, "-m", "pytest", *paths, "-q", "--junitxml=" + str(xml)]
    overrides = (
        {"NATTEN_RUN_EXTENDED_TESTS": "1", "PYTORCH_NO_CUDA_MEMORY_CACHING": "1"}
        if name == "layout-lifecycle"
        else {}
    )
    with (output / (name + ".log")).open("w") as log:
        proc = subprocess.run(
            command,
            cwd=REPO,
            env={**os.environ, **overrides},
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    cases = []
    if xml.exists():
        import xml.etree.ElementTree as ET

        for item in ET.parse(xml).iter("testcase"):
            status = "PASS"
            for tag, state in (
                ("skipped", "SKIP"),
                ("failure", "FAIL"),
                ("error", "ERROR"),
            ):
                if item.find(tag) is not None:
                    status = state
            cases.append(
                {
                    "name": item.attrib.get("classname", "")
                    + "::"
                    + item.attrib["name"],
                    "status": status,
                }
            )
    return {
        "status": "PASS"
        if proc.returncode == 0 and cases
        else "ERROR"
        if not cases
        else "FAIL",
        "returncode": proc.returncode,
        "command": command,
        "environment_overrides": overrides,
        "cases": cases,
        "counts": dict(collections.Counter(x["status"] for x in cases)),
    }


def performance_cases():
    C = MixedCase
    common = dict(dtype="bfloat16", heads=8, kv_heads=8, dim=64, vdim=64)
    images, videos = ((1, 16, 16),) * 6, ((9, 16, 16),) * 2
    return (
        C(
            "perf-uniform",
            "uniform batched dispatch",
            ((9, 16, 16),) * 4,
            (5, 4, 4),
            (True, False, False),
            **common,
        ),
        C(
            "perf-images-grouped",
            "image-heavy composition",
            images + videos,
            (5, 4, 4),
            (True, False, False),
            **common,
        ),
        C(
            "perf-images-interleaved",
            "same composition, different order",
            (images[0], videos[0], *images[1:4], videos[1], *images[4:]),
            (5, 4, 4),
            (True, False, False),
            **common,
        ),
        C(
            "perf-video-heavy",
            "video-heavy composition",
            ((1, 16, 16), (17, 16, 16), (9, 16, 16), (5, 16, 16)),
            (5, 4, 4),
            (True, False, False),
            **common,
        ),
        C(
            "perf-long-tail",
            "length imbalance",
            ((1, 16, 16), (3, 16, 16), (65, 16, 16)),
            (5, 4, 4),
            (True, False, False),
            **common,
        ),
        C(
            "perf-many-short",
            "metadata and launch overhead",
            ((1,), (3,), (5,), (9,)) * 16,
            (5,),
            (True,),
            **common,
        ),
        C(
            "perf-spatial",
            "spatial-only mixed batch",
            ((1, 24, 32), (9, 32, 32), (5, 16, 48)),
            (1, 4, 4),
            (False, False, False),
            **common,
        ),
        C(
            "perf-temporal",
            "temporal-only mixed batch",
            ((1, 24, 32), (9, 32, 32), (5, 16, 48)),
            (5, 1, 1),
            (True, False, False),
            **common,
        ),
        C(
            "perf-downstream-3d",
            "downstream documented shapes",
            ((1, 24, 32), (9, 32, 32), (5, 16, 48)),
            (5, 4, 4),
            (True, False, False),
            **common,
        ),
        C(
            "perf-thin-spatial",
            "mixed spatial degeneracy",
            ((9, 1, 32), (9, 32, 1), (9, 16, 16)),
            (5, 4, 4),
            (True, False, False),
            **common,
        ),
        C(
            "perf-downstream-throughput",
            "nondeterministic KV split policy",
            ((1, 24, 32), (9, 32, 32), (5, 16, 48)),
            (5, 4, 4),
            (True, False, False),
            deterministic=False,
            **common,
        ),
        C(
            "perf-gqa",
            "head expansion memory and throughput",
            ((1, 24, 32), (9, 32, 32), (5, 16, 48)),
            (5, 4, 4),
            (True, False, False),
            **{**common, "kv_heads": 2},
        ),
    )


def parse_clients(raw, uuid):
    return {int(line.split(",")[1]) for line in raw.splitlines() if uuid in line}


def identify_client(before, after, local_pid):
    if local_pid in after:
        return local_pid
    added = after - before
    if len(added) != 1:
        raise RuntimeError(
            "GPU client identity is ambiguous across CUDA initialization"
        )
    return next(iter(added))


def initialize_gpu_identity():
    global GPU_IDENTITY
    if STARTUP_GPU_CLIENTS is None:
        return
    torch.empty(1, device="cuda")
    uuid = str(torch.cuda.get_device_properties(0).uuid)
    after = command_output(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ]
    )
    before_pids, after_pids = (
        parse_clients(STARTUP_GPU_CLIENTS, uuid),
        parse_clients(after, uuid),
    )
    try:
        client = identify_client(before_pids, after_pids, os.getpid())
    except RuntimeError as exc:
        GPU_IDENTITY = {
            "error": str(exc),
            "before": sorted(before_pids),
            "after": sorted(after_pids),
        }
        return
    GPU_IDENTITY = {
        "local_pid": os.getpid(),
        "gpu_client_pid": client,
        "before": sorted(before_pids),
        "after": sorted(after_pids),
        "uuid": uuid,
    }


def gpu_processes():
    if not GPU_IDENTITY or "gpu_client_pid" not in GPU_IDENTITY:
        raise RuntimeError("GPU process identity could not be verified")
    props = torch.cuda.get_device_properties(0)
    uuid = str(props.uuid)
    visible = command_output(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader,nounits"]
    )
    if uuid not in visible:
        raise RuntimeError("Cannot match the CUDA device UUID to the process inventory")
    raw = command_output(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ]
    )
    return [
        line.strip()
        for line in raw.splitlines()
        if uuid in line and int(line.split(",")[1]) != GPU_IDENTITY["gpu_client_pid"]
    ]


def timed(fn):
    torch.cuda.synchronize()
    start, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    wall = time.perf_counter()
    start.record()
    fn()
    end.record()
    end.synchronize()
    return {
        "cuda_ms": start.elapsed_time(end),
        "wall_ms": (time.perf_counter() - wall) * 1000,
    }


def paired_summary(samples, key="cuda_ms"):
    a = torch.tensor([s["a"][key] for s in samples], dtype=torch.float64)
    b = torch.tensor([s["b"][key] for s in samples], dtype=torch.float64)
    delta = b - a
    gen = torch.Generator().manual_seed(883)
    bootstrap = delta[
        torch.randint(len(delta), (2000, len(delta)), generator=gen)
    ].mean(1)
    lo, hi = torch.quantile(
        bootstrap, torch.tensor([0.025, 0.975], dtype=torch.float64)
    ).tolist()
    return {
        "a_median_ms": float(a.median()),
        "b_median_ms": float(b.median()),
        "mean_b_minus_a_ms": float(delta.mean()),
        "paired_mean_ci95_ms": [lo, hi],
    }


def pairs(a, b, count):
    samples = []
    for index in range(count):
        if index % 2:
            bv, av = timed(b), timed(a)
        else:
            av, bv = timed(a), timed(b)
        samples.append({"order": "BA" if index % 2 else "AB", "a": av, "b": bv})
    return samples


def profile_call(fn):
    from torch.profiler import profile, ProfilerActivity

    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    cuda = collections.Counter(
        e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA
    )
    copies = {
        e.key: e.count
        for e in prof.key_averages()
        if e.key
        in (
            "aten::cat",
            "aten::index_select",
            "aten::contiguous",
            "aten::copy_",
            "aten::clone",
        )
    }
    return {
        "live_allocated_bytes": base,
        "peak_increment_bytes": peak - base,
        "cuda_event_count": sum(cuda.values()),
        "cuda_events": dict(cuda),
        "tensor_copy_operations": copies,
    }


def performance_case(case, samples, smoke=False):
    previous = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(case.deterministic)
        return _performance_case(case, samples, smoke)
    finally:
        torch.use_deterministic_algorithms(previous)


def _performance_case(case, samples, smoke=False):
    before = gpu_processes()
    if before and not smoke:
        return {
            "status": "NOT_RUN",
            "reason": "GPU has other compute processes",
            "processes": before,
        }
    if case.name == "perf-images-interleaved":
        grouped_case = next(
            c for c in performance_cases() if c.name == "perf-images-grouped"
        )
        original, original_grad = make_inputs(grouped_case, "cuda")
        order = (0, 6, 1, 2, 3, 7, 4, 5)
        permutation = torch.cat(
            [
                torch.arange(
                    grouped_case.offsets[i], grouped_case.offsets[i + 1], device="cuda"
                )
                for i in order
            ]
        )
        data, grad = [x[permutation] for x in original], original_grad[permutation]
    else:
        data, grad = make_inputs(case, "cuda")
    leaves = [x.detach().requires_grad_() for x in data]
    layout = natten.VarlenLayout(case.shapes, device="cuda")
    singles = [replace(case, shapes=(shape,)) for shape in case.shapes]
    single_layouts = [natten.VarlenLayout(c.shapes, device="cuda") for c in singles]
    parts = [[x[a:b] for a, b in zip(case.offsets, case.offsets[1:])] for x in leaves]

    def packed(cold=False, packing=False, backward=False):
        current = [torch.cat(x) for x in parts] if packing else leaves
        chosen = natten.VarlenLayout(case.shapes, device="cuda") if cold else layout
        out = call(current, case, chosen, return_lse=False)
        if backward:
            torch.autograd.grad(out, leaves, grad)
        return out

    def split(cold=False, packing=False, backward=False):
        current = [torch.cat(x) for x in parts] if packing else leaves
        outputs = [
            call(
                [x[a:b] for x in current],
                c,
                natten.VarlenLayout(c.shapes, device="cuda") if cold else single_layout,
                return_lse=False,
            )
            for c, single_layout, a, b in zip(
                singles, single_layouts, case.offsets, case.offsets[1:]
            )
        ]
        out = torch.cat(outputs)
        if backward:
            torch.autograd.grad(out, leaves, grad)
        return out

    def packed_result():
        return evaluate(data, grad, case, layout)

    def split_result():
        parts = [
            evaluate([x[a:b] for x in data], grad[a:b], c, single_layout)
            for c, single_layout, a, b in zip(
                singles, single_layouts, case.offsets, case.offsets[1:]
            )
        ]
        return {k: torch.cat([r[k] for r in parts]) for k in parts[0]}

    a_result = packed_result()
    b_result = split_result()
    parity = compare(a_result, b_result)
    if case.deterministic:
        correctness = {
            "correctness_status": "PASS" if exact_outputs(parity) else "FAIL"
        }
    else:
        correctness = nondeterministic_parity(
            (a_result, packed_result()), (b_result, split_result())
        )
    measurements = {}
    phases = (
        ("warm_forward", {}),
        ("warm_forward_backward", {"backward": True}),
        ("cold_layout_forward", {"cold": True}),
        ("pack_forward_backward", {"packing": True, "backward": True}),
    )
    for name, kwargs in phases:

        def a():
            with torch.set_grad_enabled(kwargs.get("backward", False)):
                return packed(**kwargs)

        def b():
            with torch.set_grad_enabled(kwargs.get("backward", False)):
                return split(**kwargs)

        for _ in range(5):
            a()
            b()
        ab = pairs(a, b, samples)
        aa = pairs(a, a, samples)
        measurements[name] = {
            "ab_samples": ab,
            "aa_samples": aa,
            "ab_cuda": paired_summary(ab),
            "ab_wall": paired_summary(ab, "wall_ms"),
            "aa_cuda": paired_summary(aa),
            "aa_wall": paired_summary(aa, "wall_ms"),
            "packed_profile": profile_call(a),
            "split_profile": profile_call(b),
        }
        measurements[name]["packed_tokens_per_second"] = (
            sum(case.lengths) * 1000 / measurements[name]["ab_cuda"]["a_median_ms"]
        )
    after = gpu_processes()
    return {
        "status": "MEASURED" if not (before or after or smoke) else "SMOKE_ONLY",
        "before_processes": before,
        "after_processes": after,
        **correctness,
        "bitwise_packed_vs_isolated": exact_outputs(parity),
        "correctness_scope": "Packed/isolated compatibility only. Deterministic loads keep bitwise acceptance; nondeterministic loads calibrate the dQ split-KV noise with same-mode repeats. Neither is independent mathematical certification.",
        "parity": parity,
        "tokens": sum(case.lengths),
        "a": "mixed",
        "b": "per-document",
        "measurements": measurements,
    }


def performance_selfcheck(samples):
    before = gpu_processes()
    if before:
        return {
            "status": "NOT_RUN",
            "reason": "GPU has other compute processes",
            "processes": before,
        }
    x = torch.ones((1024, 1024), device="cuda")

    def a():
        return x @ x

    def b():
        for _ in range(4):
            a()

    for _ in range(5):
        a()
        b()
    aa, ab = pairs(a, a, samples), pairs(a, b, samples)
    aa_summary, ab_summary = paired_summary(aa), paired_summary(ab)
    detected = ab_summary["paired_mean_ci95_ms"][0] > max(
        0, aa_summary["paired_mean_ci95_ms"][1]
    )
    after = gpu_processes()
    return {
        "status": "PASS" if detected and not after else "FAIL",
        "aa_samples": aa,
        "ab_samples": ab,
        "aa_summary": aa_summary,
        "ab_summary": ab_summary,
        "after_processes": after,
    }


def sanitizer_check(tool, output):
    executable = (
        shutil.which("compute-sanitizer") or "/usr/local/cuda/bin/compute-sanitizer"
    )
    if not pathlib.Path(executable).exists():
        return {"status": "NOT_RUN", "reason": "compute-sanitizer unavailable"}
    command = [
        executable,
        "--tool",
        tool,
        "--target-processes",
        "all",
        "--error-exitcode",
        "86",
        sys.executable,
        str(pathlib.Path(__file__).resolve()),
        "--mode",
        "regressions",
        "--case",
        "all-degenerate-axis-patterns",
        "--case",
        "dilation-2d",
        "--case",
        "mqa-noncontiguous",
        "--output",
        str(output / tool),
    ]
    log_path = output / (tool + ".log")
    with log_path.open("w") as log:
        proc = subprocess.run(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
    log_text = log_path.read_text()
    errors = [int(x) for x in re.findall(r"ERROR SUMMARY: (\d+) error", log_text)]
    functional_path = output / tool / "summary.json"
    functional = (
        json.loads(functional_path.read_text()) if functional_path.exists() else None
    )
    valid = (
        bool(errors)
        and not any(errors)
        and proc.returncode in (0, 1)
        and functional is not None
        and not functional["not_completed"]
    )
    return {
        "status": "PASS" if valid else "FAIL",
        "tool": tool,
        "command": command,
        "returncode": proc.returncode,
        "error_summaries": errors,
        "functional_counts": functional["counts"] if functional else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument(
        "--mode",
        choices=(
            "properties",
            "regressions",
            "contracts",
            "performance",
            "sanitizer",
            "all",
        ),
        default="properties",
    )
    parser.add_argument(
        "--case", action="append", help="Exact case name; repeat to select several"
    )
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--graph-seed", type=int, default=41011)
    parser.add_argument(
        "--property",
        action="append",
        choices=(
            "connectivity",
            "batch-algebra",
            "vjp-linearity",
            "forward-backward",
            "representability",
            "vjp-support",
        ),
    )
    parser.add_argument("--performance-smoke", action="store_true")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    if args.list:
        print(
            json.dumps(
                {
                    "properties": [
                        c.descriptor() for c in generated_graphs(seed=args.graph_seed)
                    ],
                    "regressions": [c.descriptor() for c in capability_cases()],
                    "performance": [c.descriptor() for c in performance_cases()],
                    "contracts": LEGACY_TESTS,
                },
                indent=2,
            )
        )
        return 0
    if not args.output:
        parser.error("--output is required")
    if args.samples < 4:
        parser.error("--samples must be at least 4")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    initialize_gpu_identity()
    write_json(args.output / "environment.json", environment())
    (args.output / "source.diff").write_text(
        command_output(["git", "-C", str(REPO), "diff", "HEAD"])
    )
    source_files = [
        pathlib.Path(__file__),
        REPO / "tests/varlen_numerics.py",
        REPO / "tests/mixed_batch_utils.py",
        REPO / "tests/test_mixed_batch_bench.py",
        REPO / "tests/mixed_batch_properties.py",
        REPO / "tests/test_varlen_gradients.py",
        REPO / "tests/test_varlen_layout.py",
        REPO / "tests/test_varlen_raw_op_validation.py",
        REPO / "tests/test_varlen_uniform_dispatch.py",
        REPO / "tests/test_compute_delta.py",
        REPO / "docs/mixed-batch-bench.md",
    ]
    write_json(
        args.output / "bench-sha256.json",
        {
            str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in source_files
        },
    )
    for path in source_files:
        target = args.output / "source" / path.relative_to(REPO)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    work = [
        (
            "oracle-selfcheck",
            lambda: pytest_group(
                "oracle-selfcheck", ["tests/test_mixed_batch_bench.py"], args.output
            ),
        )
    ]
    if args.mode == "properties":
        work.append(
            (
                "gradient-values",
                lambda: pytest_group(
                    "gradient-values", LEGACY_TESTS["gradient-values"], args.output
                ),
            )
        )
    descriptors = {}
    if args.mode in ("properties", "all"):
        graphs = tuple(
            c
            for c in generated_graphs(seed=args.graph_seed)
            if not args.case or c.name in args.case
        )
        write_json(args.output / "graph-coverage.json", graph_coverage(graphs))
        for case in graphs:
            if args.case and case.name not in args.case:
                continue
            descriptors[case.name] = case.descriptor()
            property_work = (
                (case.name + "/connectivity", lambda c=case: connectivity(c, call)),
                (case.name + "/batch-algebra", lambda c=case: core_case(c)),
                (
                    case.name + "/vjp-linearity",
                    lambda c=case: vjp_linearity(c, evaluate),
                ),
                (
                    case.name + "/forward-backward",
                    lambda c=case: forward_backward_consistency(c, call, evaluate),
                ),
                (
                    case.name + "/representability",
                    lambda c=case: representability(c, evaluate),
                ),
                (
                    case.name + "/vjp-support",
                    lambda c=case: random_vjp_support(c, call),
                ),
            )
            work.extend(
                (name, fn)
                for name, fn in property_work
                if not args.property or name.rsplit("/", 1)[1] in args.property
            )
    if args.mode in ("regressions", "all"):
        for case in capability_cases():
            if args.case and case.name not in args.case:
                continue
            work.append((case.name, lambda c=case: core_case(c)))
            descriptors[case.name] = case.descriptor()
        work.append(("delta-known-answers", delta_known_answers))
        work.append(("layout-reuse-across-layers", layout_reuse))
    if args.mode in ("contracts", "all"):
        for name, nodes in LEGACY_TESTS.items():
            if args.case and name not in args.case:
                continue
            work.append((name, lambda n=name, p=nodes: pytest_group(n, p, args.output)))
            descriptors[name] = {"tests": nodes}
    if args.mode in ("performance", "all"):
        work.append(
            ("performance-selfcheck", lambda: performance_selfcheck(args.samples))
        )
        for case in performance_cases():
            if args.case and case.name not in args.case:
                continue
            work.append(
                (
                    case.name,
                    lambda c=case: performance_case(
                        c, args.samples, args.performance_smoke
                    ),
                )
            )
            descriptors[case.name] = case.descriptor()
    if args.mode == "sanitizer":
        for tool in ("memcheck", "synccheck"):
            work.append((tool, lambda t=tool: sanitizer_check(t, args.output)))
    if args.case and not set(args.case) <= set(descriptors):
        parser.error("Some --case names do not belong to the selected mode")
    write_json(
        args.output / "plan.json",
        {"items": [name for name, _ in work], "cases": descriptors},
    )
    rows = []
    with (args.output / "results.jsonl").open("w", buffering=1) as stream:
        for name, fn in work:
            start = time.time()
            try:
                result = fn()
                json.dumps(result, allow_nan=False)
            except Exception:
                result = {"status": "ERROR", "traceback": traceback.format_exc()}
            row = {"name": name, "seconds": time.time() - start, **result}
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            rows.append(row)
            print(name, row["status"], flush=True)
            if name == "oracle-selfcheck" and row["status"] != "PASS":
                break
    summary = {
        "planned": len(work),
        "completed": len(rows),
        "counts": dict(collections.Counter(r["status"] for r in rows)),
        "not_completed": [n for n, _ in work if n not in {r["name"] for r in rows}],
        "results": rows,
        "coverage_limits": [
            "Only the recorded runtime architecture/device was executed",
            "General FP64 discrepancy and algebraic cancellation metrics require numerical adjudication",
            "compute-sanitizer and large multi-GPU tests require separate runs",
        ],
    }
    write_json(args.output / "summary.json", summary)
    return int(
        bool(summary["not_completed"])
        or any(
            r["status"] in ("FAIL", "ERROR", "NOT_RUN")
            or r.get("correctness_status") in ("FAIL", "REVIEW")
            for r in rows
        )
    )


if __name__ == "__main__":
    sys.exit(main())
