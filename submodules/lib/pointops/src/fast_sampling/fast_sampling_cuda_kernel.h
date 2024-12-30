#ifndef _FAST_SAMPLING_CUDA_KERNEL
#define _FAST_SAMPLING_CUDA_KERNEL
#include <vector>
#include <torch/serialize/tensor.h>
#include <ATen/cuda/CUDAContext.h>

#define MAX_BLOCKS 82

void farthestsampling_cuda(int b, int n, int m, at::Tensor xyz_tensor, at::Tensor offset_tensor, at::Tensor new_offset_tensor, at::Tensor tmp_tensor, at::Tensor idx_tensor);

#ifdef __cplusplus
extern "C" {
#endif

void farthestsampling_cuda_launcher(int b, int n, int m, const float *xyz, const int *offset, const int *new_offset, float *tmp, int *idx);

#ifdef __cplusplus
}
#endif
#endif
