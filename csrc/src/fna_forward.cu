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
    \brief FNA forward interface
*/

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include <algorithm>
#include <limits>

#include <natten/helpers.h>
#include <natten/natten.h>

#ifdef NATTEN_WITH_CUTLASS
#include <natten_autogen/cuda/fna/interface.h>
#include <natten/cuda/fna/fna_forward.cuh>
#endif

namespace natten {

template <class StdNADim, class StdCausal>
void fna_generic_forward(
    at::Tensor& out,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::optional<at::Tensor>& logsumexp,
    const StdNADim& kernel_size,
    const StdNADim& stride,
    const StdNADim& dilation,
    const StdCausal& is_causal,
    float attn_scale,
    const StdNADim& qkv_shape,
    const StdNADim& query_tile_size,
    const StdNADim& key_tile_size) {
  static_assert(
      std::tuple_size_v<StdNADim> > 0 && std::tuple_size_v<StdNADim> < 4);
  static constexpr int kNADim = std::tuple_size_v<StdNADim>;
  static_assert(std::tuple_size_v<StdCausal> == kNADim);

#ifdef NATTEN_WITH_CUTLASS
  AssertDimsAre128BitAligned(query, value);

  CHECK_CONTIGUOUS(query);
  CHECK_CONTIGUOUS(key);
  CHECK_CONTIGUOUS(value);
  CHECK_CONTIGUOUS(out);

  CHECK_CUDA(query);
  CHECK_CUDA(key);
  CHECK_CUDA(value);
  CHECK_CUDA(out);

  at::cuda::OptionalCUDAGuard device_guard(query.device());

  CheckArgs(kernel_size, stride, dilation);
  CheckIfPropertiesMatch(query, key, value);
  CheckIfTensorShapesMatch<kNADim>(query, key);
  CheckIfTensorShapesMatchExceptHeadDim<kNADim>(query, value);
  CheckIfTensorShapesMatch<kNADim>(out, value);

  int batch_size = query.size(0);
  int heads = query.size(kNADim + 1);
  int dim = query.size(kNADim + 2);
  int dim_value = value.size(kNADim + 2);
  CheckArgsAgainstDim(qkv_shape, kernel_size, dilation);
  if (logsumexp.has_value()) {
    CheckLogSumExp<kNADim>(out, logsumexp.value());
    CHECK_CUDA(logsumexp.value());
  }

  at::Tensor workspace;
  auto alloc_bytes = [&workspace, &query](
                         void** ptr, int64_t bytes, bool zfill) {
    workspace = at::empty({bytes}, query.options().dtype(at::ScalarType::Byte));
    if (zfill) {
      workspace.zero_();
    }
    *ptr = static_cast<void*>(workspace.data_ptr());
  };

  cudaDeviceProp* device_props =
      at::cuda::getDeviceProperties(query.device().index());
  const int cc = device_props->major * 10 + device_props->minor;
  const size_t max_smem = device_props->sharedMemPerBlockOptin;

  if (cc >= 80 || (cc >= 50 && query.scalar_type() != torch::kBFloat16)) {
    natten::cuda::fna::fna_forward_generic(
        query.scalar_type(),
        cc,
        max_smem,
        at::cuda::getCurrentCUDAStream(query.device().index()),
        alloc_bytes,
        static_cast<void*>(query.data_ptr()),
        static_cast<void*>(key.data_ptr()),
        static_cast<void*>(value.data_ptr()),
        static_cast<void*>(out.data_ptr()),
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
        logsumexp.has_value() ? static_cast<void*>(logsumexp.value().data_ptr())
                              : nullptr,
        query_tile_size,
        key_tile_size);
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

template <class StdNADim, class StdCausal>
void varlen_fna_generic_forward(
    at::Tensor& out,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::optional<at::Tensor>& logsumexp,
    const at::Tensor& cumulative_seqlens,
    const at::Tensor& token_layouts,
    const at::Tensor& forward_worklist,
    const StdNADim& kernel_size,
    const StdNADim& stride,
    const StdNADim& dilation,
    const StdCausal& is_causal,
    float attn_scale,
    const StdNADim& query_tile_size,
    const StdNADim& key_tile_size,
    int64_t forward_work_count) {
  static_assert(
      std::tuple_size_v<StdNADim> > 0 && std::tuple_size_v<StdNADim> < 4);
  static constexpr int kNADim = std::tuple_size_v<StdNADim>;
  static_assert(std::tuple_size_v<StdCausal> == kNADim);
#ifdef NATTEN_WITH_CUTLASS
  TORCH_CHECK(query.dim() == 3, "Varlen FNA tensors must be 3-D.");
  TORCH_CHECK(key.dim() == 3, "Varlen FNA tensors must be 3-D.");
  TORCH_CHECK(value.dim() == 3, "Varlen FNA tensors must be 3-D.");
  TORCH_CHECK(out.dim() == 3, "Varlen FNA tensors must be 3-D.");
  TORCH_CHECK(query.sizes() == key.sizes(), "Query and key shapes must match.");
  TORCH_CHECK(
      query.size(0) == value.size(0) && query.size(1) == value.size(1),
      "Query and value token/head dimensions must match.");
  TORCH_CHECK(
      out.sizes() == value.sizes(), "Output and value shapes must match.");
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
  CHECK_CONTIGUOUS(query);
  CHECK_CONTIGUOUS(key);
  CHECK_CONTIGUOUS(value);
  CHECK_CONTIGUOUS(out);
  CHECK_CONTIGUOUS(cumulative_seqlens);
  CHECK_CONTIGUOUS(token_layouts);
  CHECK_CONTIGUOUS(forward_worklist);
  CHECK_CUDA(query);
  CHECK_CUDA(key);
  CHECK_CUDA(value);
  CHECK_CUDA(out);
  CHECK_CUDA(cumulative_seqlens);
  CHECK_CUDA(token_layouts);
  CHECK_CUDA(forward_worklist);
  CheckIfPropertiesMatch(query, key, value);
  TORCH_CHECK(
      out.scalar_type() == value.scalar_type(),
      "Output and value must match in dtype.");
  TORCH_CHECK(
      query.device() == key.device() && query.device() == value.device() &&
          query.device() == out.device() &&
          query.device() == cumulative_seqlens.device() &&
          query.device() == token_layouts.device() &&
          query.device() == forward_worklist.device() &&
          (!logsumexp.has_value() ||
           query.device() == logsumexp.value().device()),
      "All varlen FNA tensors must be on the same CUDA device.");

  TORCH_CHECK(
      cumulative_seqlens.scalar_type() == torch::kInt,
      "cumulative_seqlens must have dtype int32.");
  TORCH_CHECK(
      token_layouts.scalar_type() == torch::kInt,
      "token_layouts must have dtype int32.");
  TORCH_CHECK(
      forward_worklist.scalar_type() == torch::kInt,
      "forward_worklist must have dtype int32.");
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
  TORCH_CHECK(forward_work_count > 0, "forward_work_count must be positive.");
  TORCH_CHECK(
      forward_worklist.dim() == 2 && forward_worklist.size(1) == 2 &&
          forward_worklist.size(0) >= forward_work_count,
      "forward_worklist must have shape [capacity, 2] with capacity >= "
      "forward_work_count.");

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

  at::cuda::OptionalCUDAGuard device_guard(query.device());
  if (logsumexp.has_value()) {
    const auto& lse = logsumexp.value();
    CHECK_CONTIGUOUS(lse);
    CHECK_CUDA(lse);
    TORCH_CHECK(
        lse.device() == query.device(),
        "Varlen FNA logsumexp must be on the same CUDA device as query.");
    TORCH_CHECK(
        lse.scalar_type() == torch::kFloat,
        "Varlen FNA logsumexp must have dtype float32.");
    TORCH_CHECK(
        lse.dim() == 2 && lse.size(0) == query.size(0) &&
            lse.size(1) == query.size(1),
        "Varlen FNA logsumexp must have shape [N_total, heads].");
  }
  cudaDeviceProp* device_props =
      at::cuda::getDeviceProperties(query.device().index());
  TORCH_CHECK(
      forward_work_count <= device_props->maxGridSize[0],
      "Varlen FNA forward grid.x exceeds the device limit.");
  const int64_t dilation_product = CheckedTupleProduct(dilation, "dilation");
  TORCH_CHECK(
      query.size(1) <= std::numeric_limits<int64_t>::max() / dilation_product,
      "Varlen FNA forward grid.y exceeds int64.");
  const int64_t grid_y = query.size(1) * dilation_product;
  TORCH_CHECK(
      grid_y <= device_props->maxGridSize[1],
      "Varlen FNA forward grid.y exceeds the device limit.");

  at::Tensor workspace;
  auto alloc_bytes = [&workspace, &query](
                         void** ptr, int64_t bytes, bool zfill) {
    workspace = at::empty({bytes}, query.options().dtype(at::ScalarType::Byte));
    if (zfill) {
      workspace.zero_();
    }
    *ptr = static_cast<void*>(workspace.data_ptr());
  };

  const int cc = device_props->major * 10 + device_props->minor;
  const size_t max_smem = device_props->sharedMemPerBlockOptin;
  natten::cuda::fna::VarlenFnaForwardMeta varlen_meta{
      static_cast<const int32_t*>(cumulative_seqlens.data_ptr()),
      static_cast<const int32_t*>(token_layouts.data_ptr()),
      static_cast<const int32_t*>(forward_worklist.data_ptr()),
      forward_work_count};
  if (cc >= 80 || (cc >= 50 && query.scalar_type() != torch::kBFloat16)) {
    natten::cuda::fna::fna_forward_generic(
        query.scalar_type(),
        cc,
        max_smem,
        at::cuda::getCurrentCUDAStream(query.device().index()),
        alloc_bytes,
        static_cast<void*>(query.data_ptr()),
        static_cast<void*>(key.data_ptr()),
        static_cast<void*>(value.data_ptr()),
        static_cast<void*>(out.data_ptr()),
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
        logsumexp.has_value() ? static_cast<void*>(logsumexp.value().data_ptr())
                              : nullptr,
        query_tile_size,
        key_tile_size,
        &varlen_meta,
        query.size(0));
  } else {
    NATTEN_FAILURE(
        "Fused kernels require compute capability >= 50 for FP16/FP32 "
        "inputs and compute capability >= 80 for BF16 inputs.");
  }
#else
  TORCH_CHECK(false, "libnatten not compiled with CUTLASS.");
#endif
}

void varlen_na1d_forward(
    at::Tensor& out,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::optional<at::Tensor>& logsumexp,
    const at::Tensor& cumulative_seqlens,
    const at::Tensor& token_layouts,
    const at::Tensor& forward_worklist,
    const std::tuple<int32_t>& kernel_size,
    const std::tuple<int32_t>& stride,
    const std::tuple<int32_t>& dilation,
    const std::tuple<bool>& is_causal,
    float attn_scale,
    const std::tuple<int32_t>& query_tile_size,
    const std::tuple<int32_t>& key_tile_size,
    int64_t forward_work_count) {
  varlen_fna_generic_forward(
      out,
      query,
      key,
      value,
      logsumexp,
      cumulative_seqlens,
      token_layouts,
      forward_worklist,
      kernel_size,
      stride,
      dilation,
      is_causal,
      attn_scale,
      query_tile_size,
      key_tile_size,
      forward_work_count);
}

void varlen_na2d_forward(
    at::Tensor& out,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::optional<at::Tensor>& logsumexp,
    const at::Tensor& cumulative_seqlens,
    const at::Tensor& token_layouts,
    const at::Tensor& forward_worklist,
    const std::tuple<int32_t, int32_t>& kernel_size,
    const std::tuple<int32_t, int32_t>& stride,
    const std::tuple<int32_t, int32_t>& dilation,
    const std::tuple<bool, bool>& is_causal,
    float attn_scale,
    const std::tuple<int32_t, int32_t>& query_tile_size,
    const std::tuple<int32_t, int32_t>& key_tile_size,
    int64_t forward_work_count) {
  varlen_fna_generic_forward(
      out,
      query,
      key,
      value,
      logsumexp,
      cumulative_seqlens,
      token_layouts,
      forward_worklist,
      kernel_size,
      stride,
      dilation,
      is_causal,
      attn_scale,
      query_tile_size,
      key_tile_size,
      forward_work_count);
}

void varlen_na3d_forward(
    at::Tensor& out,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::optional<at::Tensor>& logsumexp,
    const at::Tensor& cumulative_seqlens,
    const at::Tensor& token_layouts,
    const at::Tensor& forward_worklist,
    const std::tuple<int32_t, int32_t, int32_t>& kernel_size,
    const std::tuple<int32_t, int32_t, int32_t>& stride,
    const std::tuple<int32_t, int32_t, int32_t>& dilation,
    const std::tuple<bool, bool, bool>& is_causal,
    float attn_scale,
    const std::tuple<int32_t, int32_t, int32_t>& query_tile_size,
    const std::tuple<int32_t, int32_t, int32_t>& key_tile_size,
    int64_t forward_work_count) {
  varlen_fna_generic_forward(
      out,
      query,
      key,
      value,
      logsumexp,
      cumulative_seqlens,
      token_layouts,
      forward_worklist,
      kernel_size,
      stride,
      dilation,
      is_causal,
      attn_scale,
      query_tile_size,
      key_tile_size,
      forward_work_count);
}

void na1d_forward(
    at::Tensor& out,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::optional<at::Tensor>& logsumexp,
    const std::tuple<int32_t>& kernel_size,
    const std::tuple<int32_t>& stride,
    const std::tuple<int32_t>& dilation,
    const std::tuple<bool>& is_causal,
    float attn_scale,
    const std::tuple<int32_t>& query_tile_size,
    const std::tuple<int32_t>& key_tile_size) {
  TORCH_CHECK(query.dim() == 4, "Tensors must be 4-D.");

  fna_generic_forward(
      out,
      query,
      key,
      value,
      logsumexp,
      kernel_size,
      stride,
      dilation,
      is_causal,
      attn_scale,
      {query.size(1)},
      query_tile_size,
      key_tile_size);
}

void na2d_forward(
    at::Tensor& out,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::optional<at::Tensor>& logsumexp,
    const std::tuple<int32_t, int32_t>& kernel_size,
    const std::tuple<int32_t, int32_t>& stride,
    const std::tuple<int32_t, int32_t>& dilation,
    const std::tuple<bool, bool>& is_causal,
    float attn_scale,
    const std::tuple<int32_t, int32_t>& query_tile_size,
    const std::tuple<int32_t, int32_t>& key_tile_size) {
  TORCH_CHECK(query.dim() == 5, "Tensors must be 5-D.");

  fna_generic_forward(
      out,
      query,
      key,
      value,
      logsumexp,
      kernel_size,
      stride,
      dilation,
      is_causal,
      attn_scale,
      {query.size(1), query.size(2)},
      query_tile_size,
      key_tile_size);
}

void na3d_forward(
    at::Tensor& out,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::optional<at::Tensor>& logsumexp,
    const std::tuple<int32_t, int32_t, int32_t>& kernel_size,
    const std::tuple<int32_t, int32_t, int32_t>& stride,
    const std::tuple<int32_t, int32_t, int32_t>& dilation,
    const std::tuple<bool, bool, bool>& is_causal,
    float attn_scale,
    const std::tuple<int32_t, int32_t, int32_t>& query_tile_size,
    const std::tuple<int32_t, int32_t, int32_t>& key_tile_size) {
  TORCH_CHECK(query.dim() == 6, "Tensors must be 6-D.");

  fna_generic_forward(
      out,
      query,
      key,
      value,
      logsumexp,
      kernel_size,
      stride,
      dilation,
      is_causal,
      attn_scale,
      {query.size(1), query.size(2), query.size(3)},
      query_tile_size,
      key_tile_size);
}

} // namespace natten
