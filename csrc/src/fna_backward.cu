/***************************************************************************************************
 * Copyright (c) 2022 - 2026 Ali Hassani.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 *all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 **************************************************************************************************/
/*! \file
    \brief FNA backward interface
*/

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include <algorithm>
#include <limits>

#include <natten/compute_delta.h>
#include <natten/helpers.h>
#include <natten/natten.h>

#ifdef NATTEN_WITH_CUTLASS
#include <natten_autogen/cuda/fna/interface.h>
#include <natten/cuda/fna/fna_backward.cuh>
#endif

namespace natten {

template <typename StdTuple>
auto tuple_product(StdTuple a) {
  static_assert(
      std::tuple_size_v<StdTuple> > 0 && std::tuple_size_v<StdTuple> < 4);

  if constexpr (std::tuple_size_v<StdTuple> == 1) {
    return std::get<0>(a);
  } else if constexpr (std::tuple_size_v<StdTuple> == 2) {
    return std::get<0>(a) * std::get<1>(a);
  } else {
    return std::get<0>(a) * std::get<1>(a) * std::get<2>(a);
  }
}

template <class StdNADim, class StdCausal>
void fna_generic_backward(
    at::Tensor& grad_query,
    at::Tensor& grad_key,
    at::Tensor& grad_value,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& out,
    const at::Tensor& grad_out,
    const at::Tensor& logsumexp,
    const StdNADim& kernel_size,
    const StdNADim& stride,
    const StdNADim& dilation,
    const StdCausal& is_causal,
    float attn_scale,
    const StdNADim& qkv_shape,
    const StdNADim& query_tile_size,
    const StdNADim& key_tile_size,
    const StdNADim& num_splits_key,
    bool compute_delta_with_torch) {
  static_assert(
      std::tuple_size_v<StdNADim> > 0 && std::tuple_size_v<StdNADim> < 4);
  static constexpr int kNADim = std::tuple_size_v<StdNADim>;
  static_assert(std::tuple_size_v<StdCausal> == kNADim);

#ifdef NATTEN_WITH_CUTLASS
  AssertDimsAre128BitAligned(query, value);

  CHECK_CUDA(query);
  CHECK_CUDA(key);
  CHECK_CUDA(value);
  CHECK_CUDA(out);
  CHECK_CUDA(grad_query);
  CHECK_CUDA(grad_key);
  CHECK_CUDA(grad_value);
  CHECK_CUDA(grad_out);
  CHECK_CUDA(logsumexp);

  at::cuda::OptionalCUDAGuard device_guard(query.device());

  CHECK_CONTIGUOUS(query);
  CHECK_CONTIGUOUS(key);
  CHECK_CONTIGUOUS(value);
  CHECK_CONTIGUOUS(grad_query);
  CHECK_CONTIGUOUS(grad_key);
  CHECK_CONTIGUOUS(grad_value);
  CHECK_CONTIGUOUS(out);
  CHECK_CONTIGUOUS(grad_out);
  CHECK_CONTIGUOUS(logsumexp);

  CheckArgs(kernel_size, stride, dilation);
  CheckIfPropertiesMatch(query, key, value);
  CheckIfPropertiesMatch(grad_value, grad_out, out);
  CheckIfPropertiesMatch(grad_query, grad_key, grad_value);
  CheckIfPropertiesMatch(grad_query, query, value);

  CheckIfTensorShapesMatch<kNADim>(query, key);
  CheckIfTensorShapesMatchExceptHeadDim<kNADim>(query, value);
  CheckIfTensorShapesMatch<kNADim>(out, value);
  CheckIfTensorShapesMatch<kNADim>(grad_query, query);
  CheckIfTensorShapesMatch<kNADim>(grad_key, key);
  CheckIfTensorShapesMatch<kNADim>(grad_value, value);
  CheckIfTensorShapesMatch<kNADim>(grad_out, out);

  CheckLogSumExp<kNADim>(out, logsumexp);

  int batch_size = query.size(0);
  int heads = query.size(kNADim + 1);
  int dim = query.size(kNADim + 2);
  int dim_value = value.size(kNADim + 2);
  auto seqlen = tuple_product(qkv_shape);
  CheckArgsAgainstDim(qkv_shape, kernel_size, dilation);

  at::Tensor workspace;
  auto alloc_bytes = [&workspace, &query](
                         void** ptr, int64_t bytes, bool zfill) {
    workspace = at::empty({bytes}, query.options().dtype(at::ScalarType::Byte));
    if (zfill) {
      workspace.zero_();
    }
    *ptr = static_cast<void*>(workspace.data_ptr());
  };
  at::Tensor delta;
  if (compute_delta_with_torch) {
    delta = (grad_out.to(at::kFloat) * out.to(at::kFloat))
                .flatten(1, kNADim)
                .sum(-1);
  } else {
    delta = torch::empty(
        {batch_size, seqlen, heads}, query.options().dtype(at::kFloat));
    auto out_ = torch::flatten(out, /*start_dim=*/1, /*end_dim=*/kNADim);
    auto grad_out_ =
        torch::flatten(grad_out, /*start_dim=*/1, /*end_dim=*/kNADim);
    compute_delta(out_, grad_out_, delta);
  }
  TORCH_CHECK(delta.size(0) == batch_size);
  TORCH_CHECK(delta.size(1) == seqlen);
  TORCH_CHECK(delta.size(2) == heads);
  if (at::globalContext().deterministicAlgorithms()) {
    TORCH_CHECK(
        not compute_delta_with_torch,
        "Computing delta with PyTorch is not guaranteed to be deterministic!");
    TORCH_CHECK(
        natten::flatten(num_splits_key) <= 1,
        "FNA-backward was called with KV parallelism, "
        "which makes it algorithm non-deterministic, "
        "but PyTorch's deterministic mode is enabled. "
        "NATTEN Python API should have avoided this; which means "
        "you're probably calling the C function directly.");
  }

  cudaDeviceProp* device_props =
      at::cuda::getDeviceProperties(query.device().index());
  const int cc = device_props->major * 10 + device_props->minor;
  const size_t max_smem = device_props->sharedMemPerBlockOptin;

  if (cc >= 80 || (cc >= 50 && query.scalar_type() != torch::kBFloat16)) {
    natten::cuda::fna::fna_backward_generic(
        query.scalar_type(),
        cc,
        max_smem,
        at::cuda::getCurrentCUDAStream(query.device().index()),
        alloc_bytes,
        static_cast<void*>(grad_out.data_ptr()),
        static_cast<void*>(query.data_ptr()),
        static_cast<void*>(key.data_ptr()),
        static_cast<void*>(value.data_ptr()),
        static_cast<void*>(logsumexp.data_ptr()),
        static_cast<void*>(delta.data_ptr()),
        static_cast<void*>(out.data_ptr()),
        static_cast<void*>(grad_query.data_ptr()),
        static_cast<void*>(grad_key.data_ptr()),
        static_cast<void*>(grad_value.data_ptr()),
        batch_size,
        qkv_shape,
        heads,
        dim,
        dim_value,
        kernel_size,
        stride,
        dilation,
        is_causal,
        attn_scale,
        query_tile_size,
        key_tile_size,
        num_splits_key);
  } else {
    NATTEN_FAILURE(
        "Fused kernels are only available on devices with "
        "compute capability >= 50 for FP16/FP32 inputs, and devices with "
        "compute capability >= 80 for FP32, BF16, and FP16 inputs.");
  }
#else
  TORCH_CHECK(false, "libnatten not compiled with CUTLASS.");
#endif
}

void na1d_backward(
    at::Tensor& grad_query,
    at::Tensor& grad_key,
    at::Tensor& grad_value,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& out,
    const at::Tensor& grad_out,
    const at::Tensor& logsumexp,
    const std::tuple<int32_t>& kernel_size,
    const std::tuple<int32_t>& stride,
    const std::tuple<int32_t>& dilation,
    const std::tuple<bool>& is_causal,
    float attn_scale,
    const std::tuple<int32_t>& query_tile_size,
    const std::tuple<int32_t>& key_tile_size,
    const std::tuple<int32_t>& num_splits_key,
    bool compute_delta_with_torch) {
  TORCH_CHECK(query.dim() == 4, "Tensors must be 4-D.");

  fna_generic_backward(
      grad_query,
      grad_key,
      grad_value,
      query,
      key,
      value,
      out,
      grad_out,
      logsumexp,
      kernel_size,
      stride,
      dilation,
      is_causal,
      attn_scale,
      {query.size(1)},
      query_tile_size,
      key_tile_size,
      num_splits_key,
      compute_delta_with_torch);
}

template <class StdNADim, class StdCausal>
void varlen_fna_generic_backward(
    at::Tensor& grad_query,
    at::Tensor& grad_key,
    at::Tensor& grad_value,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& out,
    const at::Tensor& grad_out,
    const at::Tensor& logsumexp,
    const at::Tensor& cumulative_seqlens,
    const at::Tensor& token_layouts,
    const at::Tensor& backward_kv_splits,
    const at::Tensor& backward_worklist,
    const at::Tensor& backward_q_tile_offsets,
    const at::Tensor& backward_kv_split_offsets,
    const StdNADim& kernel_size,
    const StdNADim& stride,
    const StdNADim& dilation,
    const StdCausal& is_causal,
    float attn_scale,
    const StdNADim& query_tile_size,
    const StdNADim& key_tile_size,
    int64_t backward_work_count,
    int64_t total_backward_q_tiles,
    bool compute_delta_with_torch,
    bool deterministic) {
  static_assert(
      std::tuple_size_v<StdNADim> > 0 && std::tuple_size_v<StdNADim> < 4);
  static constexpr int kNADim = std::tuple_size_v<StdNADim>;
  static_assert(std::tuple_size_v<StdCausal> == kNADim);
#ifdef NATTEN_WITH_CUTLASS
  TORCH_CHECK(query.dim() == 3, "Varlen FNA tensors must be 3-D.");
  TORCH_CHECK(key.dim() == 3, "Varlen FNA tensors must be 3-D.");
  TORCH_CHECK(value.dim() == 3, "Varlen FNA tensors must be 3-D.");
  TORCH_CHECK(out.dim() == 3, "Varlen FNA tensors must be 3-D.");
  TORCH_CHECK(grad_out.dim() == 3, "Varlen FNA tensors must be 3-D.");
  TORCH_CHECK(query.sizes() == key.sizes(), "Query and key shapes must match.");
  TORCH_CHECK(
      query.size(0) == value.size(0) && query.size(1) == value.size(1),
      "Query and value token/head dimensions must match.");
  TORCH_CHECK(
      out.sizes() == value.sizes(), "Output and value shapes must match.");
  TORCH_CHECK(
      grad_out.sizes() == out.sizes(),
      "Output gradient and output shapes must match.");
  TORCH_CHECK(
      grad_query.sizes() == query.sizes(),
      "Query gradient and query shapes must match.");
  TORCH_CHECK(
      grad_key.sizes() == key.sizes(),
      "Key gradient and key shapes must match.");
  TORCH_CHECK(
      grad_value.sizes() == value.sizes(),
      "Value gradient and value shapes must match.");
  TORCH_CHECK(
      query.size(0) > 0 && query.size(0) <= std::numeric_limits<int32_t>::max(),
      "Packed token count must fit in int32 and be positive.");
  TORCH_CHECK(query.size(1) > 0, "Head count must be positive.");
  TORCH_CHECK(
      query.size(2) > 0 && query.size(2) <= (int64_t{1} << 16) &&
          key.size(2) > 0 && key.size(2) <= (int64_t{1} << 16),
      "Query and key head dimensions must be in [1, 65536] for varlen FNA.");
  TORCH_CHECK(
      value.size(2) > 0 && value.size(2) <= (int64_t{1} << 16),
      "Value head dimension must be in [1, 65536] for varlen FNA.");
  const int64_t query_head_stride = CheckedPositiveTupleProduct(
      std::make_tuple(query.size(1), query.size(2)), "Query/key head stride");
  const int64_t value_head_stride = CheckedPositiveTupleProduct(
      std::make_tuple(value.size(1), value.size(2)), "Value head stride");
  TORCH_CHECK(
      query_head_stride <= std::numeric_limits<int32_t>::max(),
      "Query/key heads * head dimension must fit in int32.");
  TORCH_CHECK(
      value_head_stride <= std::numeric_limits<int32_t>::max(),
      "Value heads * head dimension must fit in int32.");

  AssertDimsAre128BitAligned(query, value);
  CHECK_CUDA(query);
  CHECK_CUDA(key);
  CHECK_CUDA(value);
  CHECK_CUDA(out);
  CHECK_CUDA(grad_out);
  CHECK_CUDA(grad_query);
  CHECK_CUDA(grad_key);
  CHECK_CUDA(grad_value);
  CHECK_CUDA(logsumexp);
  CHECK_CUDA(cumulative_seqlens);
  CHECK_CUDA(token_layouts);
  CHECK_CUDA(backward_kv_splits);
  CHECK_CUDA(backward_worklist);
  CHECK_CUDA(backward_q_tile_offsets);
  CHECK_CUDA(backward_kv_split_offsets);
  CHECK_CONTIGUOUS(query);
  CHECK_CONTIGUOUS(key);
  CHECK_CONTIGUOUS(value);
  CHECK_CONTIGUOUS(out);
  CHECK_CONTIGUOUS(grad_out);
  CHECK_CONTIGUOUS(grad_query);
  CHECK_CONTIGUOUS(grad_key);
  CHECK_CONTIGUOUS(grad_value);
  CHECK_CONTIGUOUS(logsumexp);
  CHECK_CONTIGUOUS(cumulative_seqlens);
  CHECK_CONTIGUOUS(token_layouts);
  CHECK_CONTIGUOUS(backward_kv_splits);
  CHECK_CONTIGUOUS(backward_worklist);
  CHECK_CONTIGUOUS(backward_q_tile_offsets);
  CHECK_CONTIGUOUS(backward_kv_split_offsets);
  CheckIfPropertiesMatch(query, key, value);
  CheckIfPropertiesMatch(grad_value, grad_out, out);
  CheckIfPropertiesMatch(grad_query, grad_key, grad_value);
  CheckIfPropertiesMatch(grad_query, query);
  TORCH_CHECK(
      query.device() == key.device() && query.device() == value.device() &&
          query.device() == out.device() &&
          query.device() == grad_out.device() &&
          query.device() == grad_query.device() &&
          query.device() == grad_key.device() &&
          query.device() == grad_value.device() &&
          query.device() == logsumexp.device() &&
          query.device() == cumulative_seqlens.device() &&
          query.device() == token_layouts.device() &&
          query.device() == backward_kv_splits.device() &&
          query.device() == backward_worklist.device() &&
          query.device() == backward_q_tile_offsets.device() &&
          query.device() == backward_kv_split_offsets.device(),
      "All varlen FNA tensors must be on the same CUDA device.");

  TORCH_CHECK(
      logsumexp.scalar_type() == torch::kFloat && logsumexp.dim() == 2 &&
          logsumexp.size(0) == query.size(0) &&
          logsumexp.size(1) == query.size(1),
      "Varlen FNA logsumexp must be float32 [N_total, heads].");
  TORCH_CHECK(
      cumulative_seqlens.scalar_type() == torch::kInt,
      "cumulative_seqlens must have dtype int32.");
  TORCH_CHECK(
      token_layouts.scalar_type() == torch::kInt,
      "token_layouts must have dtype int32.");
  TORCH_CHECK(
      backward_kv_splits.scalar_type() == torch::kInt,
      "backward_kv_splits must have dtype int32.");
  TORCH_CHECK(
      backward_worklist.scalar_type() == torch::kInt,
      "backward_worklist must have dtype int32.");
  TORCH_CHECK(
      backward_q_tile_offsets.scalar_type() == torch::kLong &&
          backward_kv_split_offsets.scalar_type() == torch::kLong,
      "Compact backward workspace offsets must have dtype int64.");
  TORCH_CHECK(
      cumulative_seqlens.dim() == 1 && cumulative_seqlens.size(0) > 1,
      "cumulative_seqlens must have shape [B + 1] with B > 0.");
  const int64_t batch_size_64 = cumulative_seqlens.size(0) - 1;
  TORCH_CHECK(
      batch_size_64 <= std::numeric_limits<int32_t>::max(),
      "Logical document count must fit in int32.");
  TORCH_CHECK(
      token_layouts.dim() == 2 && token_layouts.size(0) == batch_size_64 &&
          token_layouts.size(1) == kNADim,
      "token_layouts must have shape [B, rank] for varlen FNA.");
  TORCH_CHECK(
      backward_kv_splits.dim() == 2 &&
          backward_kv_splits.size(0) == batch_size_64 &&
          backward_kv_splits.size(1) == kNADim,
      "backward_kv_splits must have shape [B, rank] for varlen FNA.");
  TORCH_CHECK(backward_work_count > 0, "backward_work_count must be positive.");
  TORCH_CHECK(
      total_backward_q_tiles > 0, "total_backward_q_tiles must be positive.");
  TORCH_CHECK(
      backward_worklist.dim() == 2 && backward_worklist.size(1) == 2 &&
          backward_worklist.size(0) >= backward_work_count,
      "backward_worklist must have shape [capacity, 2] with capacity >= "
      "backward_work_count.");
  TORCH_CHECK(
      backward_q_tile_offsets.dim() == 1 &&
          backward_q_tile_offsets.size(0) == batch_size_64 + 1 &&
          backward_kv_split_offsets.dim() == 1 &&
          backward_kv_split_offsets.size(0) == batch_size_64 + 1,
      "Compact backward workspace offsets must each have shape [B + 1].");

  CheckArgs(kernel_size, stride, dilation);
  const int64_t query_tile_product =
      CheckedPositiveTupleProduct(query_tile_size, "Query tile shape");
  const int64_t key_tile_product =
      CheckedPositiveTupleProduct(key_tile_size, "Key tile shape");
  TORCH_CHECK(
      query_tile_product <= std::numeric_limits<int32_t>::max(),
      "Query tile shape product must fit in int32.");
  TORCH_CHECK(
      key_tile_product <= std::numeric_limits<int32_t>::max(),
      "Key tile shape product must fit in int32.");
  if (deterministic) {
    TORCH_CHECK(
        not compute_delta_with_torch,
        "Computing delta with PyTorch is not guaranteed to be deterministic!");
    // Each document contributes at most one backward work item under
    // determinism; zero-token documents contribute none, so a mixed
    // empty/non-empty layout legitimately has backward_work_count <
    // batch_size_64.
    TORCH_CHECK(
        backward_work_count <= batch_size_64,
        "Varlen FNA backward requires at most one work item per document "
        "under deterministic algorithms (KV parallelism is incompatible "
        "with deterministic algorithms).");
  }

  at::cuda::OptionalCUDAGuard device_guard(query.device());
  grad_query.zero_();
  grad_key.zero_();
  grad_value.zero_();
  cudaDeviceProp* device_props =
      at::cuda::getDeviceProperties(query.device().index());
  TORCH_CHECK(
      backward_work_count <= device_props->maxGridSize[0],
      "Varlen FNA backward grid.x exceeds the device limit.");
  const int64_t dilation_product = CheckedTupleProduct(dilation, "dilation");
  TORCH_CHECK(
      query.size(1) <= std::numeric_limits<int64_t>::max() / dilation_product,
      "Varlen FNA backward grid.y exceeds int64.");
  const int64_t grid_y = query.size(1) * dilation_product;
  TORCH_CHECK(
      grid_y <= device_props->maxGridSize[1],
      "Varlen FNA backward grid.y exceeds the device limit.");

  at::Tensor workspace;
  auto alloc_bytes = [&workspace, &query](
                         void** ptr, int64_t bytes, bool zfill) {
    workspace = at::empty({bytes}, query.options().dtype(at::ScalarType::Byte));
    if (zfill) {
      workspace.zero_();
    }
    *ptr = static_cast<void*>(workspace.data_ptr());
  };

  at::Tensor delta;
  auto active_out = out.narrow(0, 0, query.size(0));
  auto active_grad_out = grad_out.narrow(0, 0, query.size(0));
  if (compute_delta_with_torch) {
    delta =
        (active_grad_out.to(at::kFloat) * active_out.to(at::kFloat)).sum(-1);
  } else {
    auto out_batched = active_out.unsqueeze(0);
    auto grad_out_batched = active_grad_out.unsqueeze(0);
    auto delta_batched = torch::empty(
        {1, query.size(0), query.size(1)}, query.options().dtype(at::kFloat));
    compute_delta(out_batched, grad_out_batched, delta_batched);
    delta = delta_batched.view({query.size(0), query.size(1)});
  }
  TORCH_CHECK(
      delta.dim() == 2 && delta.size(0) == query.size(0) &&
          delta.size(1) == query.size(1),
      "Varlen FNA delta must have shape [N_active, heads].");

  const int cc = device_props->major * 10 + device_props->minor;
  const size_t max_smem = device_props->sharedMemPerBlockOptin;
  natten::cuda::fna::VarlenFnaBackwardMeta varlen_meta{
      static_cast<const int32_t*>(cumulative_seqlens.data_ptr()),
      static_cast<const int32_t*>(token_layouts.data_ptr()),
      static_cast<const int32_t*>(backward_kv_splits.data_ptr()),
      static_cast<const int32_t*>(backward_worklist.data_ptr()),
      static_cast<const int64_t*>(backward_q_tile_offsets.data_ptr()),
      static_cast<const int64_t*>(backward_kv_split_offsets.data_ptr()),
      backward_work_count,
      total_backward_q_tiles};
  if (cc >= 80 || (cc >= 50 && query.scalar_type() != torch::kBFloat16)) {
    natten::cuda::fna::fna_backward_generic(
        query.scalar_type(),
        cc,
        max_smem,
        at::cuda::getCurrentCUDAStream(query.device().index()),
        alloc_bytes,
        static_cast<void*>(grad_out.data_ptr()),
        static_cast<void*>(query.data_ptr()),
        static_cast<void*>(key.data_ptr()),
        static_cast<void*>(value.data_ptr()),
        static_cast<void*>(logsumexp.data_ptr()),
        static_cast<void*>(delta.data_ptr()),
        static_cast<void*>(out.data_ptr()),
        static_cast<void*>(grad_query.data_ptr()),
        static_cast<void*>(grad_key.data_ptr()),
        static_cast<void*>(grad_value.data_ptr()),
        static_cast<int32_t>(batch_size_64),
        StdNADim{},
        query.size(1),
        query.size(2),
        value.size(2),
        kernel_size,
        stride,
        dilation,
        is_causal,
        attn_scale,
        query_tile_size,
        key_tile_size,
        StdNADim{},
        &varlen_meta);
  } else {
    NATTEN_FAILURE(
        "Fused kernels require compute capability >= 50 for FP16/FP32 "
        "inputs and compute capability >= 80 for BF16 inputs.");
  }
#else
  TORCH_CHECK(false, "libnatten not compiled with CUTLASS.");
#endif
}

void varlen_na1d_backward(
    at::Tensor& grad_query,
    at::Tensor& grad_key,
    at::Tensor& grad_value,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& out,
    const at::Tensor& grad_out,
    const at::Tensor& logsumexp,
    const at::Tensor& cumulative_seqlens,
    const at::Tensor& token_layouts,
    const at::Tensor& backward_kv_splits,
    const at::Tensor& backward_worklist,
    const at::Tensor& backward_q_tile_offsets,
    const at::Tensor& backward_kv_split_offsets,
    const std::tuple<int32_t>& kernel_size,
    const std::tuple<int32_t>& stride,
    const std::tuple<int32_t>& dilation,
    const std::tuple<bool>& is_causal,
    float attn_scale,
    const std::tuple<int32_t>& query_tile_size,
    const std::tuple<int32_t>& key_tile_size,
    int64_t backward_work_count,
    int64_t total_backward_q_tiles,
    bool compute_delta_with_torch,
    bool deterministic) {
  varlen_fna_generic_backward(
      grad_query,
      grad_key,
      grad_value,
      query,
      key,
      value,
      out,
      grad_out,
      logsumexp,
      cumulative_seqlens,
      token_layouts,
      backward_kv_splits,
      backward_worklist,
      backward_q_tile_offsets,
      backward_kv_split_offsets,
      kernel_size,
      stride,
      dilation,
      is_causal,
      attn_scale,
      query_tile_size,
      key_tile_size,
      backward_work_count,
      total_backward_q_tiles,
      compute_delta_with_torch,
      deterministic);
}

void varlen_na2d_backward(
    at::Tensor& grad_query,
    at::Tensor& grad_key,
    at::Tensor& grad_value,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& out,
    const at::Tensor& grad_out,
    const at::Tensor& logsumexp,
    const at::Tensor& cumulative_seqlens,
    const at::Tensor& token_layouts,
    const at::Tensor& backward_kv_splits,
    const at::Tensor& backward_worklist,
    const at::Tensor& backward_q_tile_offsets,
    const at::Tensor& backward_kv_split_offsets,
    const std::tuple<int32_t, int32_t>& kernel_size,
    const std::tuple<int32_t, int32_t>& stride,
    const std::tuple<int32_t, int32_t>& dilation,
    const std::tuple<bool, bool>& is_causal,
    float attn_scale,
    const std::tuple<int32_t, int32_t>& query_tile_size,
    const std::tuple<int32_t, int32_t>& key_tile_size,
    int64_t backward_work_count,
    int64_t total_backward_q_tiles,
    bool compute_delta_with_torch,
    bool deterministic) {
  varlen_fna_generic_backward(
      grad_query,
      grad_key,
      grad_value,
      query,
      key,
      value,
      out,
      grad_out,
      logsumexp,
      cumulative_seqlens,
      token_layouts,
      backward_kv_splits,
      backward_worklist,
      backward_q_tile_offsets,
      backward_kv_split_offsets,
      kernel_size,
      stride,
      dilation,
      is_causal,
      attn_scale,
      query_tile_size,
      key_tile_size,
      backward_work_count,
      total_backward_q_tiles,
      compute_delta_with_torch,
      deterministic);
}

void varlen_na3d_backward(
    at::Tensor& grad_query,
    at::Tensor& grad_key,
    at::Tensor& grad_value,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& out,
    const at::Tensor& grad_out,
    const at::Tensor& logsumexp,
    const at::Tensor& cumulative_seqlens,
    const at::Tensor& token_layouts,
    const at::Tensor& backward_kv_splits,
    const at::Tensor& backward_worklist,
    const at::Tensor& backward_q_tile_offsets,
    const at::Tensor& backward_kv_split_offsets,
    const std::tuple<int32_t, int32_t, int32_t>& kernel_size,
    const std::tuple<int32_t, int32_t, int32_t>& stride,
    const std::tuple<int32_t, int32_t, int32_t>& dilation,
    const std::tuple<bool, bool, bool>& is_causal,
    float attn_scale,
    const std::tuple<int32_t, int32_t, int32_t>& query_tile_size,
    const std::tuple<int32_t, int32_t, int32_t>& key_tile_size,
    int64_t backward_work_count,
    int64_t total_backward_q_tiles,
    bool compute_delta_with_torch,
    bool deterministic) {
  varlen_fna_generic_backward(
      grad_query,
      grad_key,
      grad_value,
      query,
      key,
      value,
      out,
      grad_out,
      logsumexp,
      cumulative_seqlens,
      token_layouts,
      backward_kv_splits,
      backward_worklist,
      backward_q_tile_offsets,
      backward_kv_split_offsets,
      kernel_size,
      stride,
      dilation,
      is_causal,
      attn_scale,
      query_tile_size,
      key_tile_size,
      backward_work_count,
      total_backward_q_tiles,
      compute_delta_with_torch,
      deterministic);
}

void na2d_backward(
    at::Tensor& grad_query,
    at::Tensor& grad_key,
    at::Tensor& grad_value,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& out,
    const at::Tensor& grad_out,
    const at::Tensor& logsumexp,
    const std::tuple<int32_t, int32_t>& kernel_size,
    const std::tuple<int32_t, int32_t>& stride,
    const std::tuple<int32_t, int32_t>& dilation,
    const std::tuple<bool, bool>& is_causal,
    float attn_scale,
    const std::tuple<int32_t, int32_t>& query_tile_size,
    const std::tuple<int32_t, int32_t>& key_tile_size,
    const std::tuple<int32_t, int32_t>& num_splits_key,
    bool compute_delta_with_torch) {
  TORCH_CHECK(query.dim() == 5, "Tensors must be 5-D.");

  fna_generic_backward(
      grad_query,
      grad_key,
      grad_value,
      query,
      key,
      value,
      out,
      grad_out,
      logsumexp,
      kernel_size,
      stride,
      dilation,
      is_causal,
      attn_scale,
      {query.size(1), query.size(2)},
      query_tile_size,
      key_tile_size,
      num_splits_key,
      compute_delta_with_torch);
}

void na3d_backward(
    at::Tensor& grad_query,
    at::Tensor& grad_key,
    at::Tensor& grad_value,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& out,
    const at::Tensor& grad_out,
    const at::Tensor& logsumexp,
    const std::tuple<int32_t, int32_t, int32_t>& kernel_size,
    const std::tuple<int32_t, int32_t, int32_t>& stride,
    const std::tuple<int32_t, int32_t, int32_t>& dilation,
    const std::tuple<bool, bool, bool>& is_causal,
    float attn_scale,
    const std::tuple<int32_t, int32_t, int32_t>& query_tile_size,
    const std::tuple<int32_t, int32_t, int32_t>& key_tile_size,
    const std::tuple<int32_t, int32_t, int32_t>& num_splits_key,
    bool compute_delta_with_torch) {
  TORCH_CHECK(query.dim() == 6, "Tensors must be 6-D.");

  fna_generic_backward(
      grad_query,
      grad_key,
      grad_value,
      query,
      key,
      value,
      out,
      grad_out,
      logsumexp,
      kernel_size,
      stride,
      dilation,
      is_causal,
      attn_scale,
      {query.size(1), query.size(2), query.size(3)},
      query_tile_size,
      key_tile_size,
      num_splits_key,
      compute_delta_with_torch);
}

} // namespace natten
