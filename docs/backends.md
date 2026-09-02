# Backends

In this page, we list our available implementations for standard attention (FMHA), and neighborhood
attention (FNA).

## Note on torch.compile support

Starting in version `0.21.5`, NATTEN offers full support for `torch.compile`.
However, this has only been verified for `torch >= 2.8`.
Use `torch.compile` in earlier versions at your own risk.

## CUTLASS FNA / FMHA

**Supported features**

- [x] Inference (forward pass)
- [x] Training (backward pass)
- [x] GQA/MQA support (with tensor repeats)
- [x] MLA support (head_dim != head_dim_v)
- [x] torch.compile support without graph breaks

**FNA-specific features**

- [x] Causal masking
- [x] Variable length with sequence-packed format

**FMHA-specific features**

- [x] Causal masking
- [x] Variable length with sequence-packed format

![FNA visualization](assets/ampere-fna-viz.png){ width="80%" }
/// caption
Visualization of FNA, as proposed in
[Faster Neighborhood Attention (2024)](https://arxiv.org/abs/2403.04690).
///

Based on
[xFormers FMHA](https://github.com/NVIDIA/cutlass/tree/main/examples/41_fused_multi_head_attention)
(a.k.a. _memory-efficient attention_), this kernel is based on the CUTLASS 2.X API, and targets
multiple architectures: SM50 (Maxwell), SM70 (Volta), SM75 (Turing), and SM80 (Ampere).
You can use these kernels on any NVIDIA GPU with compute capability >= 5.0, and
both for training and inference. FP16 and FP32 inputs require compute capability
>= 5.0; BF16 inputs require compute capability >= 8.0.

Variable-length FNA packs independent documents into QKV tensors shaped
`[total_tokens, heads, head_dim]` (no batch dimension, no capacity padding --
`total_tokens` must equal the layout's total exactly). Build a
[`VarlenLayout`][natten.VarlenLayout] once, then reuse it from eager or
compiled code, across layers, geometries, and AMP dtypes.

[`from_tensor_list`][natten.VarlenLayout.from_tensor_list] is the supported
construction path: it builds the layout and packs a list of per-document
tensors in one call, guaranteeing layout/data consistency at construction
time:

```python
layout, query = natten.VarlenLayout.from_tensor_list([doc0_q, doc1_q])
```

Advanced construction / interoperability: a layout can also be built
directly from already-known per-document shapes. This validates only
metadata-local invariants -- the caller is then responsible for keeping
every packed tensor used with the layout in the same document order and
token layout, since NATTEN cannot infer that from the packed tensor alone:

```python
layout = natten.VarlenLayout(
    [(25, 16, 16), (61, 8, 8)],
    device=query.device,
)
output = natten.na3d_varlen(
    query,
    key,
    value,
    layout,
    kernel_size=(5, 8, 8),
    backend="cutlass-fna",
)
```

A document may be empty (zero tokens along any axis); an empty document is an
exact no-op, and a layout whose documents are all empty returns correctly-shaped
empty outputs without a kernel launch. A document narrower than `kernel_size`
on some axis attends over its whole extent on that axis instead
(`effective_kernel_size = min(kernel_size, extent)`), as long as `dilation == 1`
on that axis; axes with `dilation > 1` still require the document to fit
`kernel_size * dilation`. All documents in one layout share the
same spatial rank; `kernel_size`, `stride`, `dilation`, and the causal mask are
per-call arguments, not bound to the layout, so the same layout can be reused
across different geometries. A `VarlenLayout` defines the document order for
the packed tensors used with it. Its derived metadata is built on the stream
current at first use of a geometry; consuming it from a different stream
afterwards is the caller's responsibility to order, same as any other tensor
shared across streams.

!!! warning
    A mismatched packing computes attention over the wrong neighborhoods
    without an error. Prefer `from_tensor_list` over direct construction
    when the packed data isn't already fixed elsewhere -- it structurally
    rules this out.

Resolving a geometry while deterministic algorithms are enabled selects a
single (serialized) backward KV split per document for that geometry,
matching fixed FNA; resolving the same geometry again under a different
determinism setting builds its own memo entry instead of reusing or
rejecting the earlier one. As with fixed FNA, grouped-query and multi-query
attention repeat packed key/value heads internally and therefore use memory
proportional to the expanded query head count.

Some newer architectures such as Hopper (SM90), and Blackwell DC-class (SM100, SM103) have much
more performant dedicated kernels.

This implementation fuses multi-dimensional tiling directly into the kernel, but at the same time
may suffer from additional overhead of software predication.
To read more about this, we refer you to our
[Generalized Neighborhood Attention](https://arxiv.org/abs/2504.16922) paper, in which we also
proposed solutions such as Token Permutation, which we use to build our
[Hopper](#hopper-fna-fmha) and [Blackwell](#blackwell-fna-fmha) kernels.

### Design notes

[`VarlenLayout`][natten.VarlenLayout]'s construction and
[`split`][natten.VarlenLayout.split]/[`from_tensor_list`][natten.VarlenLayout.from_tensor_list]
mirror xFormers' `BlockDiagonalMask`. Unlike that class, a `VarlenLayout` also owns **mutable,
per-object derived schedule state** (worklists, KV-split selection, resolved tile configs), one
entry per neighborhood-attention geometry (kernel size, dilation, query head count, backward
KV-split cap, deterministic flag, the KV-parallelism switch, the grid bound the memory-usage
preference resolves to, resolved forward and backward tile configs, dtype, device -- stride and
the causal mask are not part of the key: they act per-call inside the kernel and do not influence
how this metadata is built). Derived state lives on the layout, rather than a separate plan
object, because its derivation inputs (kernel size, dilation, dtype, ...) only arrive per call --
derivation is necessarily lazy and per-geometry, and holding the resulting cache on the
caller-owned object follows the same pattern as FlashInfer's wrapper objects and xFormers'
`BlockDiagonalMask` (derived state lives on an object the caller already holds, and is reclaimed
with it); a separate plan type would push a second object into every call site without changing
what is cached or when. Building that metadata is the expensive part of variable-length FNA, so
the same `VarlenLayout` should be reused across calls -- and across layers, geometries, and AMP
dtypes, since the object only fixes document composition, not any of those -- rather than
reconstructed every step. Derived state lives only on this object and only for its lifetime:
distinct instances never share entries, and a discarded instance's state is reclaimed with it.
Growth is one entry per distinct geometry used on this object; recreate the layout if that set
must be unbounded.

### Finding configurations

You can use [profiler dry runs](profiler.md#dry-run) to find configurations for any of our
backends, and also find backends compatible with your device and use case. You can also use the
following functions in your code.

??? tip "Finding configurations for CUTLASS FMHA/FNA"
    ::: natten
        options:
              heading_level: 4
              show_signature: true
              separate_signature: true
              show_object_full_path: true
              members:
                  - get_configs_for_cutlass_fmha
                  - get_bwd_configs_for_cutlass_fmha
                  - get_configs_for_cutlass_fna
                  - get_bwd_configs_for_cutlass_fna


## Hopper FNA / FMHA

**Supported features**

- [x] Inference (forward pass)
- [x] Training (backward pass)
- [x] GQA/MQA support (with tensor repeats)
- [ ] MLA support (head_dim != head_dim_v)
- [x] torch.compile support without graph breaks

**FMHA-specific features**

- [x] Causal masking
- [x] Variable length with sequence-packed format

![Hopper FNA performance sample](assets/hopper-fna-perf.png){ width="80%" }
/// caption
Performance levels of Hopper FNA (forward pass) as of version `0.20.0`.
///

Based on CUTLASS's
[Hopper FMHA kernel](https://github.com/NVIDIA/cutlass/tree/main/examples/88_hopper_fmha)
(3.X API), this backend offers non-persistent,
warp-specialized cooperative, and warp-specialized ping-ponging kernels, similar to
[Flash Attention 3](https://arxiv.org/abs/2407.08608). This kernel exhibits similar forward pass
(inference) performance to Flash Attention 3.
Backward pass is more limited compared to Flash Attention 3 and cuDNN's Hopper FMHA for now, but we
plan to improve the performance to those levels in future releases.

This backend does not fuse multi-dimensional tiling into the kernel, and instead uses
[Token Permutation](https://arxiv.org/abs/2504.16922).

### Finding configurations

You can use [profiler dry runs](profiler.md#dry-run) to find configurations for any of our
backends, and also find backends compatible with your device and use case. You can also use the
following functions in your code.

??? tip "Finding configurations for Hopper FMHA/FNA"
    ::: natten
        options:
              heading_level: 4
              show_signature: true
              separate_signature: true
              show_object_full_path: true
              members:
                  - get_configs_for_cutlass_hopper_fmha
                  - get_bwd_configs_for_cutlass_hopper_fmha
                  - get_configs_for_cutlass_hopper_fna
                  - get_bwd_configs_for_cutlass_hopper_fna


## Blackwell FNA / FMHA

**Supported features**

- [x] Inference (forward pass)
- [x] Training (backward pass)
- [x] GQA/MQA support
- [ ] MLA support (head_dim != head_dim_v)
- [x] torch.compile support without graph breaks

**FMHA-specific features**

- [x] Causal masking
- [x] Variable length with sequence-packed format

![Blackwell FNA performance sample](assets/blackwell-fna-perf.png){ width="80%" }
/// caption
Performance levels of Blackwell FNA (forward pass) as of version `0.20.0` (also
reported in
[Generalized Neighborhood Attention (2025)](https://arxiv.org/abs/2504.16922)).
///

Based on CUTLASS's
[Blackwell FMHA kernel](https://github.com/NVIDIA/cutlass/tree/main/examples/77_blackwell_fmha)
(3.X API), this backend offers incredible forward pass and backward pass performance, which is
comparable with cuDNN's Blackwell FMHA.

This backend does not fuse multi-dimensional tiling into the kernel, and instead uses
[Token Permutation](https://arxiv.org/abs/2504.16922).

### Finding configurations

You can use [profiler dry runs](profiler.md#dry-run) to find configurations for any of our
backends, and also find backends compatible with your device and use case. You can also use the
following functions in your code.

??? tip "Finding configurations for Blackwell FMHA/FNA"
    ::: natten
        options:
              heading_level: 4
              show_signature: true
              separate_signature: true
              show_object_full_path: true
              members:
                  - get_configs_for_cutlass_blackwell_fmha
                  - get_bwd_configs_for_cutlass_blackwell_fmha
                  - get_configs_for_cutlass_blackwell_fna
                  - get_bwd_configs_for_cutlass_blackwell_fna


## Flex FNA / FMHA

!!! warning
    This backend is experimental.

!!! info inline end
    This backend requires PyTorch >= 2.7.

**Supported features**

- [x] Inference (forward pass)
- [x] Training (backward pass)
- [x] GQA/MQA support
- [ ] MLA support (head_dim != head_dim_v)
- [ ] torch.compile support without graph breaks

**FMHA-specific features**

- [ ] Causal masking
- [ ] Variable length with sequence-packed format

This backend is PyTorch-native, and supports some non-NVIDIA devices as well (CPU and ROCm).
It is based on 
[Flex Attention](https://docs.pytorch.org/docs/stable/nn.attention.flex_attention.html#module-torch.nn.attention.flex_attention).

Since this backend is implemented in PyTorch, fusion of multi-dimensional tiling is not possible.
This backend however does support [Token Permutation](https://arxiv.org/abs/2504.16922), similar to
the Hopper and Blackwell backends.

By default, if tile shapes are not specified, token permutation will be disabled, and our
legacy Flex mask will be used.
If tile shapes are specified, it will use token permutation.

### Finding configurations

You can use [profiler dry runs](profiler.md#dry-run) to find configurations for any of our
backends, and also find backends compatible with your device and use case. You can also use the
following functions in your code.

??? tip "Finding configurations for Flex FMHA/FNA"
    ::: natten
        options:
              heading_level: 4
              show_signature: true
              separate_signature: true
              show_object_full_path: true
              members:
                  - get_configs_for_flex_fmha
                  - get_configs_for_flex_fna
