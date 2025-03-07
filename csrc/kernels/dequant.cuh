// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "util/cuda_utils.cuh"

namespace vptq::kernels {

template <typename scalar_t, int IDXBITS, int ResidualBits, int GROUPSIZE,
          bool Return_OUF_x_INF>
__global__ void DequantizeWithOutliers_PackIndice(
    scalar_t* out, const int32_t* q_indice, const int16_t* q_indice_outliers,
    const scalar_t* centroids, const scalar_t* residual_centroids,
    const scalar_t* outliers_centroids, const uint16_t* invert_perm,
    const scalar_t* weight_scale, const scalar_t* weight_bias, int out_features,
    int in_features, int outliers_infeatures, int OL_GroupSize,
    const int index_stride_0, const int index_stride_1,
    const int centroids_stride_0, const int group_nums) {
  int bid = blockIdx.x;
  int tid = (bid * cuda::kBlockSize + threadIdx.x);
  int in_x = tid % in_features;
  int in_y = tid / in_features;
  using VecType = typename cuda::TypeVec2<scalar_t>::type;

  uint16_t mapped_index_x = invert_perm ? invert_perm[in_x] : in_x;
  const scalar_t scale = weight_scale[in_x];
  const scalar_t bias = weight_bias[in_x];

  if (mapped_index_x < outliers_infeatures) {
    const int n_outlisers_groups_in_normalgroup = GROUPSIZE / OL_GroupSize;
    q_indice_outliers +=
        in_y * n_outlisers_groups_in_normalgroup * outliers_infeatures +
        mapped_index_x;
#pragma unroll(3)
    for (int i = 0; i < n_outlisers_groups_in_normalgroup; ++i) {
      if (in_y * n_outlisers_groups_in_normalgroup + i >=
          out_features / OL_GroupSize)
        return;
      const uint16_t outliers_ind = q_indice_outliers[(i)*outliers_infeatures];
      const scalar_t* outliers_centroids_start =
          outliers_centroids + outliers_ind * OL_GroupSize;
      const int gi = in_y * GROUPSIZE + i * OL_GroupSize;
#pragma unroll(4)
      for (int j = 0; j < OL_GroupSize; ++j) {
        if ((gi + j) >= out_features) {
          return;
        }
        out[(gi + j) * in_features + in_x] =
            FMA(outliers_centroids_start[j], scale, bias);
      }
    }
    return;
  }

  const int inliers_infeatures_in_group =
      (in_features - outliers_infeatures) / group_nums;

  const int mapped_inliers_inx = (mapped_index_x - outliers_infeatures);
  const int code_books_id = mapped_inliers_inx / inliers_infeatures_in_group;
  const int mappped_inx_in_a_codebook =
      mapped_inliers_inx % inliers_infeatures_in_group;

  if (group_nums > 1) {  // has multiple codebooks
    q_indice += code_books_id * index_stride_0;
    centroids += code_books_id * centroids_stride_0;
    residual_centroids += code_books_id * centroids_stride_0;
  }
  q_indice += in_y * index_stride_1;
  uint32_t merged_ind = cuda::iterator_packed_tensor<IDXBITS + ResidualBits>(
      (const uint32_t*)q_indice, mappped_inx_in_a_codebook);

  const uint16_t base_ind = merged_ind & ((1 << IDXBITS) - 1);
  VecType base[GROUPSIZE / 2];
  const scalar_t* centroids_start = centroids + base_ind * GROUPSIZE;
  cuda::ldg_vec_x<GROUPSIZE>((base), (const uint32_t*)(centroids_start));

  if constexpr (ResidualBits > 0) {
    VecType residual[GROUPSIZE / 2];
    merged_ind >>= IDXBITS;
    const uint16_t res_ind = merged_ind & ((1 << ResidualBits) - 1);
    const scalar_t* residual_centroids_start =
        residual_centroids + res_ind * GROUPSIZE;
    cuda::ldg_vec_x<GROUPSIZE>((residual),
                               (const uint32_t*)(residual_centroids_start));
#pragma unroll
    for (int i = 0; i < GROUPSIZE / 2; ++i) {
      base[i] = ADD2(*(((VecType*)base) + i), *(((VecType*)residual) + i));
    }
  }

  VecType hres[GROUPSIZE / 2];
  VecType scale2 = VecType{scale, scale};
  VecType bias2 = VecType{bias, bias};
#pragma unroll
  for (int i = 0; i < GROUPSIZE / 2; ++i) {
    hres[i] = FMA2(base[i], scale2, bias2);
  }
  scalar_t* res = (scalar_t*)hres;
  const int group_step = in_y * GROUPSIZE;
  if constexpr (!Return_OUF_x_INF) {
    out += in_x * out_features + group_step;
  } else {
    out += (group_step)*in_features + in_x;
  }
#pragma unroll
  for (int i = 0; i < GROUPSIZE; ++i) {
    if ((group_step + i) < out_features) {
      if constexpr (Return_OUF_x_INF) {
        out[i * in_features] = res[i];
      } else {
        out[i] = res[i];
      }
    }
  }
}

}  // namespace vptq::kernels
