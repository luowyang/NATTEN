# Operations

In this page we list our PyTorch Autograd-compatible operations.
These operations come with performance knobs (configurations), some of which are specific to
certain [backends](backends.md).

Changing those knobs is completely optional, and NATTEN will continue to be functionally correct in
all cases. However, to squeeze out the maximum performance achievable, we highly recommend looking
at [backends](backends.md), or just using our [profiler toolkit](profiler.md) and its
[dry run feature](profiler.md#dry-run) to navigate through
available backends and their valid configurations for your specific use case and GPU architecture.
You can also use the profiler's [optimize](profiler.md#optimize) feature to search and find the
best configuration.


## Neighborhood Attention

A `kernel_size` entry may be `1`: that axis mixes nothing (each query attends only to the token
sharing its coordinate on that axis), and `is_causal`/`dilation` have no effect there (`stride`
is then forced to `1`, since it can never exceed `kernel_size`). Such an axis is lowered away in
Python -- folded (zero-copy, for a leading run) or permuted (gather in, scatter back, for the
rest) -- before reaching a backend kernel, which only ever sees `kernel_size >= 2`; if every axis
is `1`, the call short-circuits to `output = value`, `logsumexp = scale * (query *
key).sum(-1)`, with no kernel launch. Explicit tile shapes, `backward_kv_splits`,
`additional_keys`/`additional_values`, and `attention_kwargs` are not supported together with a
`kernel_size = 1` axis, since lowering changes the call's rank (or bypasses the kernel
altogether).

::: natten
    options:
          heading_level: 3
          show_object_full_path: true
          members:
              - na1d
              - na2d
              - na3d

## Standard Attention

::: natten
    options:
          heading_level: 3
          show_object_full_path: true
          members:
              - attention
              - merge_attentions
