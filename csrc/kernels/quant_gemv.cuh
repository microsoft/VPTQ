// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "config.cuh"
#include "kernels/convert.cuh"
#include "util/cuda_utils.cuh"

namespace vptq::kernels {

template <typename scalar_t, int IDXBITS, int ResidualBits, int GROUPSIZE,
          int OL_GroupSize, int Do_Reduce>
__global__ void WqA16WithOutliers_PackIndice(
    scalar_t* out, const scalar_t* input_data, const int32_t* q_indice,
    const uint16_t* q_indice_outliers, const scalar_t* __restrict__ centroids,
    const scalar_t* __restrict__ residual_centroids,
    const scalar_t* outliers_centroids, const uint16_t* invert_perm,
    const scalar_t* weight_scale, const scalar_t* weight_bias,
    const scalar_t* bias, int out_features, int in_features,
    int outliers_infeatures, const int index_stride_0, const int index_stride_1,
    const int centroids_stride_0, const int group_nums) {
  static_assert((GROUPSIZE & 1) == 0, "GROUPSIZE must be even.");

  int bidx = blockIdx.x;  // out_features//base_groupsize
  int bidy = blockIdx.y;  // batch
  int bidz = blockIdx.z;  // segment in_features
  int tidx = threadIdx.x;
  using VecType = typename cuda::TypeVec2<scalar_t>::type;
  if constexpr (Do_Reduce > 0) {
    tidx += bidz * cuda::kBlockSize * Do_Reduce;
  }
  int in_y = bidx;
  __shared__ float shared_output[GROUPSIZE][cuda::kBlockSize / WARP_SIZE + 1];
  scalar_t tmp_output[GROUPSIZE];
  const scalar_t zero_value = ZERO_VALUE(scalar_t());

#pragma unroll
  for (int i = 0; i < GROUPSIZE; ++i) {
    tmp_output[i] = zero_value;
  }
  input_data = input_data + in_features * bidy;
  out = out + out_features * bidy * gridDim.z;
  if (tidx >= in_features) {
    return;
  }
  scalar_t base[GROUPSIZE];
  const int inliers_infeatures_in_group =
      (in_features - outliers_infeatures) / group_nums;
  const int col_end =
      Do_Reduce ? min((bidz + 1) * cuda::kBlockSize * Do_Reduce, in_features)
                : in_features;
  for (int col = tidx; col < col_end; col += cuda::kBlockSize) {
    const int w_col = Do_Reduce ? (invert_perm ? invert_perm[col] : col) : 0;
    const scalar_t input_col_v = input_data[w_col];
    const scalar_t bias = input_col_v * weight_bias[w_col];
    scalar_t input_v = input_col_v * weight_scale[w_col];
    VecType input_v2 = VecType{input_v, input_v};
    VecType bias2 = VecType{bias, bias};

    int32_t mapped_index_x = col;
    if (mapped_index_x < outliers_infeatures) {
      // outliers
      constexpr int n_outlisers_groups_in_normalgroup =
          GROUPSIZE / OL_GroupSize;
#pragma unroll
      for (int i = 0; i < n_outlisers_groups_in_normalgroup; ++i) {
        if (in_y * n_outlisers_groups_in_normalgroup + i >=
            out_features / OL_GroupSize)
          continue;
        const uint16_t outliers_ind =
            q_indice_outliers[(in_y * n_outlisers_groups_in_normalgroup + i) *
                                  outliers_infeatures +
                              mapped_index_x];
        const scalar_t* outliers_centroids_start =
            (outliers_centroids) + outliers_ind * OL_GroupSize;
        const int gi = i * OL_GroupSize;
        const int out_y = in_y * GROUPSIZE + gi;
        scalar_t* tmp_output_off_p = tmp_output + gi;
        scalar_t scalar_weight[OL_GroupSize];
        if (out_y < out_features) {
          cuda::ldg_vec_x<OL_GroupSize>(
              (scalar_weight), (const uint32_t*)outliers_centroids_start);
          VecType* weight_h2 = (VecType*)scalar_weight;
          VecType* tmp_output_off_h2 = (VecType*)tmp_output_off_p;
          tmp_output_off_h2[0] =
              FMA2(weight_h2[0], input_v2, tmp_output_off_h2[0]);
          tmp_output_off_h2[1] =
              FMA2(weight_h2[1], input_v2, tmp_output_off_h2[1]);
          tmp_output_off_h2[0] = ADD2(tmp_output_off_h2[0], bias2);
          tmp_output_off_h2[1] = ADD2(tmp_output_off_h2[1], bias2);
        }
      }
    } else {
      const int mapped_inliers_inx = (mapped_index_x - outliers_infeatures);
      int mappped_inx_in_a_codebook = mapped_inliers_inx;
      const scalar_t* centroids_cb = centroids;
      const scalar_t* residual_centroids_cb = residual_centroids;
      const uint32_t* q_indice_cb = (const uint32_t*)q_indice;
      if (group_nums > 1) {  // has multiple codebooks
        mappped_inx_in_a_codebook =
            mapped_inliers_inx % inliers_infeatures_in_group;
        const int code_books_id =
            mapped_inliers_inx / inliers_infeatures_in_group;
        q_indice_cb += code_books_id * index_stride_0;
        centroids_cb += code_books_id * centroids_stride_0;
        residual_centroids_cb += code_books_id * centroids_stride_0;
      }

      uint32_t merged_ind =
          cuda::iterator_packed_tensor<IDXBITS + ResidualBits>(
              q_indice_cb + in_y * index_stride_1, mappped_inx_in_a_codebook);
      const uint32_t base_ind = merged_ind & ((1 << IDXBITS) - 1);

      const scalar_t* centroids_start = (centroids_cb) + base_ind * GROUPSIZE;
      cuda::ldg_vec_x<GROUPSIZE>((base), (const uint32_t*)(centroids_start));

      VecType* hres_ptr = nullptr;
      if constexpr (ResidualBits > 0) {
        scalar_t residual[GROUPSIZE];
        const uint32_t res_ind =
            (merged_ind >> IDXBITS) & ((1 << ResidualBits) - 1);
        const scalar_t* residual_centroids_start =
            (residual_centroids_cb) + res_ind * GROUPSIZE;
        cuda::ldg_vec_x<GROUPSIZE>((residual),
                                   (const uint32_t*)(residual_centroids_start));

        VecType hres[GROUPSIZE / 2];
        hres_ptr = hres;
#pragma unroll
        for (int i = 0; i < GROUPSIZE / 2; ++i) {
          hres[i] = ADD2(*(((VecType*)base) + i), *(((VecType*)residual) + i));
        }
      } else {
        hres_ptr = (VecType*)base;
      }

      VecType* h2_tmp_output = (VecType*)tmp_output;
#pragma unroll
      for (int gi = 0; gi < GROUPSIZE / 2; ++gi) {
        h2_tmp_output[gi] = FMA2(hres_ptr[gi], input_v2, h2_tmp_output[gi]);
        h2_tmp_output[gi] = ADD2(h2_tmp_output[gi], bias2);
      }
    }
  }

  int warpid = threadIdx.x / WARP_SIZE;  // at most 8 warp = 256 / WARP_SIZE
  int landid = threadIdx.x % WARP_SIZE;
#pragma unroll
  for (int gi = 0; gi < GROUPSIZE; ++gi) {
    float reduce_out = 0.f;
    reduce_out = to_float(tmp_output[gi]);
    reduce_out = cuda::warpReduceSum<WARP_SIZE>(reduce_out);
    if (landid == 0) {
      shared_output[gi][warpid] = reduce_out;
    }
  }

  if constexpr (Do_Reduce > 0) {
    out += (in_y * GROUPSIZE) * gridDim.z + bidz;
    bias += (bias == nullptr ? 0 : (in_y * GROUPSIZE) + bidz);
  } else {
    out += in_y * GROUPSIZE;
    bias += (bias == nullptr ? 0 : GROUPSIZE);
  }

  __syncthreads();
  if (landid < cuda::kBlockSize / WARP_SIZE) {
#pragma unroll
    for (int wid = warpid; wid < GROUPSIZE;
         wid += cuda::kBlockSize / WARP_SIZE) {
      float reduce_out = shared_output[wid][landid];
      reduce_out =
          cuda::warpReduceSum<cuda::kBlockSize / WARP_SIZE>(reduce_out);
      if (landid == 0 && (in_y * GROUPSIZE + wid) < out_features) {
        if constexpr (Do_Reduce) {
          out[(wid)*gridDim.z] =
              from_float<scalar_t>(reduce_out, zero_value) +
              ((bidz == 0 && bias != 0) ? bias[wid] : zero_value);
        } else {
          out[wid] = from_float<scalar_t>(reduce_out, zero_value) +
                     ((bias != 0) ? bias[wid] : zero_value);
        }
      }
    }
  }
}

}  // namespace vptq::kernels
