// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cuda_bf16.h>
#include <cmath>
#include <math_constants.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include "common.h"
#include "utils.cuh"

template <typename T> 
struct C10ToNvType {
   typedef __nv_bfloat16 type;
};

template <> 
struct C10ToNvType<c10::Half> {
   typedef __half type;
};

template <typename scalar_t, int IDXBITS, int GROUPSIZE,  int OL_GroupSize, int Do_Reduce, bool Has_Residual>
__global__ void WqA16WithOutliers_PackIndice(
    scalar_t *out, const scalar_t* input_data, const int32_t *q_indice, const int16_t *q_indice_outliers,
    const scalar_t* __restrict__ centroids, const scalar_t* __restrict__ residual_centroids, const scalar_t* outliers_centroids, 
    const uint16_t *invert_perm,  const scalar_t* weight_scale, const scalar_t* weight_bias,
    int out_features, int in_features, int outliers_infeatures, const int index_stride_0, const int index_stride_1,
    const int centroids_stride_0, const int group_nums) {
  int bidx = blockIdx.x;// out_features//groupsize
  int bidy = blockIdx.y;// batch
  int bidz = blockIdx.z;// segment in_features
  int tidx = threadIdx.x;
  using VecType = typename cuda::TypeVec2<scalar_t>::type;
  if constexpr (Do_Reduce > 0) {
    tidx += bidz*cuda::kBlockSize*Do_Reduce;
  }
  int in_y = bidx;
  extern  __shared__ scalar_t shared_memory[];// 3xin_features, dynamic
  scalar_t* shared_input = shared_memory;// in_features, dynamic
  //scalar_t* shared_w_scales = shared_memory+in_features;// in_features, dynamic
  scalar_t* shared_w_bias = shared_memory+in_features;// in_features, dynamic
  __shared__ float shared_output[GROUPSIZE][cuda::kBlockSize/32+1];
  scalar_t tmp_output[GROUPSIZE] = {0};
#pragma unroll
  for(int i=0;i<GROUPSIZE;i++){
    tmp_output[i]=scalar_t(0);
  }
  input_data = input_data+in_features*bidy;
  out = out+out_features*bidy*gridDim.z;
  if constexpr (Do_Reduce == 0) {
    for(int i=tidx;i<in_features;i+=cuda::kBlockSize){
      int w_col = invert_perm?invert_perm[i]:i;
      shared_input[i] = input_data[w_col]*weight_scale[w_col];
      shared_w_bias[i] = input_data[w_col]*weight_bias[w_col];
    }
    __syncthreads();
  }
  if (tidx >= in_features){return;}
  scalar_t base[GROUPSIZE];
  const int inliers_infeatures_in_group = (in_features-outliers_infeatures)/group_nums;
  const int col_end = Do_Reduce ? min((bidz+1)*cuda::kBlockSize*Do_Reduce, in_features):in_features;
  for(int col=tidx;col<col_end;col+=cuda::kBlockSize){
    //const scalar_t scale = shared_w_scales[col];
    const int w_col = Do_Reduce?(invert_perm?invert_perm[col]:col):0;
    const scalar_t input_col_v = input_data[w_col];
    const scalar_t bias = Do_Reduce?input_col_v*weight_bias[w_col]:shared_w_bias[col];
    scalar_t input_v = Do_Reduce?input_col_v*weight_scale[w_col]:shared_input[col];
    VecType input_v2 = VecType(input_v, input_v);
    VecType bias2 = VecType(bias, bias);

    int32_t mapped_index_x = col;
    if (mapped_index_x < outliers_infeatures) {
      // outliers
      constexpr int n_outlisers_groups_in_normalgroup = GROUPSIZE/OL_GroupSize;
      #pragma unroll
      for (int i=0;i<n_outlisers_groups_in_normalgroup;i++){
        if (in_y*n_outlisers_groups_in_normalgroup+i >= out_features/OL_GroupSize)continue;
        const uint16_t outliers_ind = q_indice_outliers[(in_y*n_outlisers_groups_in_normalgroup+i)*outliers_infeatures+mapped_index_x];
        const scalar_t* outliers_centroids_start = (outliers_centroids)+outliers_ind*OL_GroupSize;
        const int gi = i*OL_GroupSize;
        const int out_y = in_y*GROUPSIZE+gi;
        scalar_t* tmp_output_off_p = tmp_output+gi;
        scalar_t scalar_weight[OL_GroupSize];
        if (out_y < out_features) {
          cuda::ldg_vec_x<OL_GroupSize>(reinterpret_cast<uint32_t*>(scalar_weight), (const uint32_t*)outliers_centroids_start);
          VecType *weight_h2 = (VecType*)scalar_weight;
          VecType *tmp_output_off_h2 = (VecType*)tmp_output_off_p;
          tmp_output_off_h2[0] = __hfma2(weight_h2[0], input_v2, tmp_output_off_h2[0]);
          tmp_output_off_h2[1] = __hfma2(weight_h2[1], input_v2, tmp_output_off_h2[1]);
          tmp_output_off_h2[0] = __hadd2(tmp_output_off_h2[0], bias2);
          tmp_output_off_h2[1] = __hadd2(tmp_output_off_h2[1], bias2);
        }
      }
    }else{
      const int mapped_inliers_inx = (mapped_index_x-outliers_infeatures);
      int mappped_inx_in_a_codebook = mapped_inliers_inx;
      const scalar_t* centroids_cb  = centroids;
      const scalar_t* residual_centroids_cb  = residual_centroids;
      const uint32_t* q_indice_cb = (const uint32_t*)q_indice;
      if (group_nums>1){//has multi-ple codebooks
        mappped_inx_in_a_codebook = mapped_inliers_inx%inliers_infeatures_in_group;
        const int code_books_id = mapped_inliers_inx/inliers_infeatures_in_group;
        q_indice_cb += code_books_id*index_stride_0;
        centroids_cb += code_books_id*centroids_stride_0;
        residual_centroids_cb += code_books_id*centroids_stride_0;
      }
      
      uint32_t merged_ind = cuda::iterator_packed_tensor<IDXBITS*(Has_Residual+1)>(q_indice_cb+in_y*index_stride_1, mappped_inx_in_a_codebook);
      const uint32_t base_ind = merged_ind&((1<<IDXBITS)-1);

      const scalar_t* centroids_start = (centroids_cb)+base_ind*GROUPSIZE;
      cuda::ldg_vec_x<GROUPSIZE>(reinterpret_cast<uint32_t*>(base), (const uint32_t*)(centroids_start));


      VecType *hres_ptr = nullptr;
      if constexpr (Has_Residual) {
        scalar_t residual[GROUPSIZE];
        const uint32_t res_ind = (merged_ind>>IDXBITS)&((1<<IDXBITS)-1);
        const scalar_t* residual_centroids_start = (residual_centroids_cb)+res_ind*GROUPSIZE;
        cuda::ldg_vec_x<GROUPSIZE>(reinterpret_cast<uint32_t*>(residual), (const uint32_t*)(residual_centroids_start));

        VecType hres[GROUPSIZE/2];
        hres_ptr = hres;
        #pragma unroll
        for (int i=0;i<GROUPSIZE/2;i++){
            hres[i] = __hadd2(*(((VecType*)base)+i), *(((VecType*)residual)+i));
            //hres[i] = __hfma2(hres[i], scale2, bias2);
        }
      }else{
        hres_ptr = (VecType*)base;
      }
      //scalar_t* res = (scalar_t*)hres;
      //#pragma unroll
      //for (int gi=0;gi<GROUPSIZE;gi++){
      //  tmp_output[gi] = __hfma(res[gi], input_v, tmp_output[gi]);
      //  tmp_output[gi] += bias;
      //}
      VecType *h2_tmp_output = (VecType*)tmp_output;
      #pragma unroll
      for (int gi=0;gi<GROUPSIZE/2;gi++){
        h2_tmp_output[gi] = __hfma2(hres_ptr[gi], input_v2, h2_tmp_output[gi]);
        h2_tmp_output[gi] = __hadd2(h2_tmp_output[gi], bias2);
      }
    }
  }

  // warp_size = 32
  int warpid = threadIdx.x/32; // at most 8 warp= 256/32
  int landid = threadIdx.x%32;
  #pragma unroll
  for (int gi=0;gi<GROUPSIZE;gi++){
    float reduce_out = 0.f;
    if constexpr (!std::is_same_v<scalar_t, c10::BFloat16>) {
      reduce_out = __half2float(tmp_output[gi]);
    }else{
      reduce_out = __bfloat162float(tmp_output[gi]);
    }
    reduce_out = cuda::warpReduceSum<32>(reduce_out);
    if (landid == 0){
      shared_output[gi][warpid] = reduce_out;
    }
  }
  
  if constexpr(Do_Reduce>0){
    out += (in_y*GROUPSIZE)*gridDim.z+bidz;
  }else{
    out += in_y*GROUPSIZE;
  }
  __syncthreads();
  if (landid < cuda::kBlockSize/32) {
    #pragma unroll
    for(int wid=warpid;wid<GROUPSIZE;wid+=cuda::kBlockSize/32){
      float reduce_out = shared_output[wid][landid];
      reduce_out = cuda::warpReduceSum<cuda::kBlockSize/32>(reduce_out);
      if (landid == 0 && (in_y*GROUPSIZE+wid) < out_features){
        if constexpr(Do_Reduce){
          out[(wid)*gridDim.z] = cuda::ConvertFromFloat<scalar_t>(reduce_out);
        }else{
          out[wid] = cuda::ConvertFromFloat<scalar_t>(reduce_out);
        }
      }
    }
  }
}

template <typename scalar_t, int IDXBITS, int GROUPSIZE, bool Return_OUF_x_INF, bool Has_Residual>
__global__ void DequantizeWithOutliers_PackIndice(
    scalar_t *out, const int32_t *q_indice, const int16_t *q_indice_outliers,
    const scalar_t* centroids,const scalar_t* residual_centroids, const scalar_t* outliers_centroids, const uint16_t *invert_perm,  const scalar_t* weight_scale, const scalar_t* weight_bias,
    int out_features, int in_features, int outliers_infeatures, int OL_GroupSize, const int index_stride_0, const int index_stride_1,
    const int centroids_stride_0, const int group_nums) {
  int bid = blockIdx.x;
  int tid = (bid * cuda::kBlockSize + threadIdx.x);
  int in_x = tid % in_features;
  int in_y = tid / in_features;

  uint16_t mapped_index_x = invert_perm?invert_perm[in_x]:in_x;
  const scalar_t scale = weight_scale[in_x];
  const scalar_t bias = weight_bias[in_x];

  if (mapped_index_x<outliers_infeatures){
    const int n_outlisers_groups_in_normalgroup = GROUPSIZE/OL_GroupSize;
    q_indice_outliers += in_y*n_outlisers_groups_in_normalgroup*outliers_infeatures+mapped_index_x;
    #pragma unroll(3)
    for (int i=0;i<n_outlisers_groups_in_normalgroup;i++){
      if (in_y*n_outlisers_groups_in_normalgroup+i >= out_features/OL_GroupSize)return;
      //const uint16_t outliers_ind = q_indice_outliers[(in_y*n_outlisers_groups_in_normalgroup+i)*outliers_infeatures+mapped_index_x];
      const uint16_t outliers_ind = q_indice_outliers[(i)*outliers_infeatures];
      const scalar_t* outliers_centroids_start = outliers_centroids+outliers_ind*OL_GroupSize;
      const int gi = in_y*GROUPSIZE+i*OL_GroupSize;
      #pragma unroll(4)
      for (int j=0;j<OL_GroupSize;j++){
        
        if ((gi+j) >= out_features){
            return;
        }
        out[(gi+j)*in_features+in_x] = outliers_centroids_start[j]*scale+bias;
      }
    }
    return;    
  }

  const int inliers_infeatures_in_group = (in_features-outliers_infeatures)/group_nums;

  const int mapped_inliers_inx = (mapped_index_x-outliers_infeatures);
  const int code_books_id = mapped_inliers_inx/inliers_infeatures_in_group;
  const int mappped_inx_in_a_codebook = mapped_inliers_inx%inliers_infeatures_in_group;

  if (group_nums > 1){//has multi-ple codebooks
    q_indice += code_books_id*index_stride_0;
    centroids += code_books_id*centroids_stride_0;
    residual_centroids += code_books_id*centroids_stride_0;
  }
  q_indice += in_y*index_stride_1;
  uint32_t merged_ind = cuda::iterator_packed_tensor<IDXBITS*(Has_Residual+1)>((const uint32_t*)q_indice, mappped_inx_in_a_codebook);

  const uint16_t base_ind = merged_ind&((1<<IDXBITS)-1);
  __half2 base[GROUPSIZE/2];
  const scalar_t* centroids_start = centroids+base_ind*GROUPSIZE;
  cuda::ldg_vec_x<GROUPSIZE>((uint32_t*)(base), (const uint32_t*)(centroids_start));


  if constexpr (Has_Residual) {
    __half2 residual[GROUPSIZE/2];
    merged_ind >>= IDXBITS;
    const uint16_t res_ind = merged_ind&((1<<GROUPSIZE)-1);
    const scalar_t* residual_centroids_start = residual_centroids+res_ind*GROUPSIZE;
    cuda::ldg_vec_x<GROUPSIZE>((uint32_t*)(residual), (const uint32_t*)(residual_centroids_start));
    #pragma unroll
    for (int i=0;i<GROUPSIZE/2;i++){
        base[i] = __hadd2(*(((__half2*)base)+i), *(((__half2*)residual)+i));
    }
  }

  __half2 hres[GROUPSIZE/2];
  __half2 scale2 = __half2(scale, scale);
  __half2 bias2 = __half2(bias, bias);
  #pragma unroll
  for (int i=0;i<GROUPSIZE/2;i++){
      hres[i] = __hfma2(base[i], scale2, bias2);
  }
  scalar_t* res = (scalar_t*)hres;
  const int group_step = in_y*GROUPSIZE;
  if constexpr (!Return_OUF_x_INF){
    out += in_x*out_features+group_step;
  }else{
    out += (group_step)*in_features+in_x;
  }
  #pragma unroll
  for (int i=0;i<GROUPSIZE;i++){
    if ((group_step+i) < out_features){
      if constexpr (Return_OUF_x_INF){
        out[i*in_features] = res[i];
      } else {
        out[i] = res[i];
      }
    }
  }
}



torch::Tensor lauch_deqantize_outliers_cuda_packkernel(const int* outf_x_inf, //[out_f, in_f]
                                const torch::Tensor &q_indice,  //[num_cen, o_c_size, in_inf]
                                const torch::Tensor &centroids, //[num_c, c_size, vec_len]
                                const c10::optional<torch::Tensor>& q_indice_residual,//[num_cen, o_c_size, in_inf]
                                const c10::optional<torch::Tensor>& residual_centroids,//[num_c, c_size, vec_len]
                                const c10::optional<torch::Tensor>& outliers_indices, //[num_cen, c_size, ol_in_f]
                                const c10::optional<torch::Tensor>& outliers_centroids, //[num_c, c_size, out_vec_len]
                                const c10::optional<torch::Tensor>& perm,
                                const torch::Tensor &weight_scale,
                                const torch::Tensor &weight_bias
                                ) {
  int groupsize = centroids.size(-1);
  int index_bits = log2(centroids.size(1));
  auto out_size = outf_x_inf;
  dim3 blocks(cuda::ceil_div<int>(cuda::ceil_div<int>(out_size[0], groupsize)*out_size[1], cuda::kBlockSize));
  dim3 threads(cuda::kBlockSize);
  torch::Tensor output;
  constexpr bool out_ouf_inf = true;//why =flase is 10 times slow?
  if (out_ouf_inf) {// out_ouf_inf
    output = at::empty({out_size[0], out_size[1]}, centroids.options());
  } else {
    output = at::empty({out_size[1], out_size[0]}, centroids.options());
  }
  int outliers_indices_size_n1 = outliers_indices.has_value()?outliers_indices.value().size(-1):0;
  int outliers_centroids_size_n1 = outliers_centroids.has_value()?outliers_centroids.value().size(-1):1;
  const uint16_t* perm_ptr = perm.has_value()?(const uint16_t*)(perm.value().data_ptr<int16_t>()):nullptr;
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  using scalar_t = at::Half;
  #define callDequantWithOutliers(IDXBITS, GP, OUT_OUF_INF, Has_Residual)\
        DequantizeWithOutliers_PackIndice<scalar_t, IDXBITS, GP, OUT_OUF_INF, Has_Residual><<<blocks, threads,0,stream>>>(output.data_ptr<scalar_t>(),\
        q_indice.data_ptr<int32_t>(),\
        outliers_indices.has_value()?outliers_indices.value().data_ptr<int16_t>():nullptr,\
        centroids.data_ptr<scalar_t>(),\
        residual_centroids.has_value()? residual_centroids.value().data_ptr<scalar_t>():nullptr,\
        outliers_centroids.has_value()? outliers_centroids.value().data_ptr<scalar_t>():nullptr,\
        perm_ptr,\
        weight_scale.data_ptr<scalar_t>(),\
        weight_bias.data_ptr<scalar_t>(),\
        out_size[0],out_size[1],\
        outliers_indices_size_n1, outliers_centroids_size_n1,\
        q_indice.stride(0), q_indice.stride(1),\
        centroids.stride(0), q_indice.size(0));
  #define callDequantWithOutliers_bits(GP, OUT_OUF_INF, Has_Residual)\
    switch (index_bits) {\
      case 16:\
      callDequantWithOutliers(16, GP, OUT_OUF_INF, Has_Residual);\
      break;\
      case 12:\
      callDequantWithOutliers(12, GP, OUT_OUF_INF, Has_Residual);\
      break;\
      case 8:\
      callDequantWithOutliers(8, GP, OUT_OUF_INF, Has_Residual);\
      break;\
      case 4:\
      callDequantWithOutliers(4, GP, OUT_OUF_INF, Has_Residual);\
      break;\
    default:\
    TORCH_CHECK(false, "unspportetd index_bits:"+std::to_string(index_bits));\
    }
  #define DispatchDequantWithOutliers(GP, OUT_OUF_INF)\
    if (residual_centroids.has_value()){\
          callDequantWithOutliers_bits(GP, OUT_OUF_INF, true);\
    }else {\
          callDequantWithOutliers_bits(GP, OUT_OUF_INF, false);\
    }
  switch (groupsize){
    case 16:
        DispatchDequantWithOutliers(16, out_ouf_inf);
    break;
    case 12:
        DispatchDequantWithOutliers(12, out_ouf_inf);
    break;
    case 8:
        DispatchDequantWithOutliers(8, out_ouf_inf);
    break;
    case 6:
        DispatchDequantWithOutliers(6, out_ouf_inf);
    break;
    case 4:
        DispatchDequantWithOutliers(4, out_ouf_inf);
    case 2:
        DispatchDequantWithOutliers(2, out_ouf_inf);
    break;
    default:
    TORCH_CHECK(false, "unspportetd groupsize:"+std::to_string(groupsize));
  }
  if (out_ouf_inf){
    return output;
  }else{
    return output.t();
  }
}


torch::Tensor lauch_gemv_outliers_cuda_packkernel(const int out_features,
                                const torch::Tensor& input,
                                const torch::Tensor &q_indice,  //[num_cen, o_c_size, in_inf]
                                const torch::Tensor &centroids, //[num_c, c_size, vec_len]
                                const c10::optional<torch::Tensor>& q_indice_residual,//[num_cen, o_c_size, in_inf]
                                const c10::optional<torch::Tensor>& residual_centroids,//[num_c, c_size, vec_len]
                                const c10::optional<torch::Tensor>& outliers_indices, //[num_cen, c_size, ol_in_f]
                                const c10::optional<torch::Tensor>& outliers_centroids, //[num_c, c_size, out_vec_len]
                                const c10::optional<torch::Tensor>& perm,
                                const torch::Tensor &weight_scale,
                                const torch::Tensor &weight_bias) {
  const int groupsize = centroids.size(-1);
  int index_bits = log2(centroids.size(1));

  const int in_features = input.size(-1);
  //const int out_features = output.size(-1);
  auto output_shape = input.sizes().vec();
  output_shape[input.dim()-1] = out_features;
  torch::Tensor output;
  //  blocks = (out_features, batch)
  dim3 blocks(cuda::ceil_div(out_features, groupsize), input.numel()/in_features);
  dim3 threads(cuda::kBlockSize);
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  // using scalar_t = c10::Half;
  //c10::BFloat16
  int shared_memory_size = 2*in_features*2;
  const int outliers_indices_size_n1 = outliers_indices.has_value()?outliers_indices.value().size(-1):0;

  if (outliers_centroids.has_value()){
    TORCH_CHECK(outliers_centroids.value().size(-1)==4, "only support 4 out_vec_len");
  }
  const uint16_t* perm_ptr = perm.has_value()?(const uint16_t*)(perm.value().data_ptr<int16_t>()):nullptr;
  #define CallWqA16kernel(scalar_t, out_buf, IDXBITS, G, Do_Reduce, Has_Residual)\
        WqA16WithOutliers_PackIndice<scalar_t, IDXBITS, G, 4, Do_Reduce, Has_Residual><<<blocks, threads, shared_memory_size, stream>>>(\
        out_buf.data_ptr<scalar_t>(),\
        input.data_ptr<scalar_t>(),\
        q_indice.data_ptr<int32_t>(),\
        outliers_indices.has_value()?outliers_indices.value().data_ptr<int16_t>():nullptr,\
        centroids.data_ptr<scalar_t>(),\
        residual_centroids.has_value()? residual_centroids.value().data_ptr<scalar_t>():nullptr,\
        outliers_centroids.has_value()?outliers_centroids.value().data_ptr<scalar_t>():nullptr,\
        perm_ptr,\
        weight_scale.data_ptr<scalar_t>(),\
        weight_bias.data_ptr<scalar_t>(),\
        out_features,in_features, outliers_indices_size_n1,\
        q_indice.stride(0), q_indice.stride(1),\
        centroids.stride(0), q_indice.size(0));
  #define CallWqA16kernel_dtype(out_buf, IDXBITS, G, Do_Reduce, Has_Residual)\
    if (input.dtype() == at::ScalarType::Half) {\
      using scalar_t = c10::Half;\
      CallWqA16kernel(scalar_t, out_buf, IDXBITS, G, Do_Reduce, Has_Residual);\
    } else {\
      using scalar_t = c10::Half;\
      CallWqA16kernel(scalar_t, out_buf, IDXBITS, G, Do_Reduce, Has_Residual);\
    }
  #define CallWqA16kernel_bits(out_buf, G, Do_Reduce, Has_Residual)\
    switch (index_bits) {\
      case 16:\
      CallWqA16kernel_dtype(out_buf, 16, G, Do_Reduce, Has_Residual);\
      break;\
      case 12:\
      CallWqA16kernel_dtype(out_buf, 12, G, Do_Reduce, Has_Residual);\
      break;\
      case 8:\
      CallWqA16kernel_dtype(out_buf, 8, G, Do_Reduce, Has_Residual);\
      break;\
      case 4:\
      CallWqA16kernel_dtype(out_buf, 4, G, Do_Reduce, Has_Residual);\
      break;\
    default:\
    TORCH_CHECK(false, "unspportetd index_bits:"+std::to_string(index_bits));\
    }

  #define DispatchWqA16Kernel(out_buf, G, Do_Reduce)\
    if (residual_centroids.has_value()){\
          CallWqA16kernel_bits(out_buf, G, Do_Reduce, true);\
    }else {\
          CallWqA16kernel_bits(out_buf, G, Do_Reduce, false);\
    }
  if (in_features <= cuda::kBlockSize){
    //output = at::empty(output_shape, centroids.options());
    //switch (groupsize){
    //  case 16:
    //      gpuErrchk(cudaFuncSetAttribute(WqA16WithOutliers<scalar_t, 16, 4, false>,
    //                cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory_size));
    //      DispatchWqA16Kernel(output, 16, false);
    //  break;
    //  case 12:
    //      gpuErrchk(cudaFuncSetAttribute(WqA16WithOutliers<scalar_t, 12, 4, false>,
    //                cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory_size));
    //      DispatchWqA16Kernel(output, 12, false);
    //  break;
    //  default:
    //      TORCH_CHECK(false, "unspportetd groupsize:"+std::to_string(groupsize));
    //}
    TORCH_CHECK(false, "unspportetd yet");
  }else{
    constexpr int do_reduce = 4;
    shared_memory_size = 0;
    auto tmp_output_shape = output_shape;
    tmp_output_shape.push_back(cuda::ceil_div(in_features, cuda::kBlockSize*do_reduce));
    torch::Tensor tmp_output = at::empty(tmp_output_shape, centroids.options());
    blocks.z = tmp_output_shape.back();
    switch (groupsize){
      case 16:
          DispatchWqA16Kernel(tmp_output, 16, do_reduce);
      break;
      case 12:
          DispatchWqA16Kernel(tmp_output, 12, do_reduce);
      break;
      case 8:
          DispatchWqA16Kernel(tmp_output, 8, do_reduce);
      break;
      case 6:
          DispatchWqA16Kernel(tmp_output, 6, do_reduce);
      break;
      case 4:
          DispatchWqA16Kernel(tmp_output, 4, do_reduce);
      case 2:
          DispatchWqA16Kernel(tmp_output, 2, do_reduce);
      break;
      default:
      TORCH_CHECK(false, "unspportetd groupsize:"+std::to_string(groupsize));
    }
    output = tmp_output.sum(-1);
  }
  return output;
}
