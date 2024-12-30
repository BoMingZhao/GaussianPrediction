#include <vector>
// #include <THC/THC.h>
#include <torch/serialize/tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include "fast_sampling_cuda_kernel.h"


void farthestsampling_cuda(int b, int n, int m, at::Tensor xyz_tensor, at::Tensor offset_tensor, at::Tensor new_offset_tensor, at::Tensor tmp_tensor, at::Tensor idx_tensor)
{
    const float *xyz = xyz_tensor.data_ptr<float>();
    const int *offset = offset_tensor.data_ptr<int>();
    const int *new_offset = new_offset_tensor.data_ptr<int>();
    float *tmp = tmp_tensor.data_ptr<float>();
    int *idx = idx_tensor.data_ptr<int>();
    farthestsampling_cuda_launcher(b, n, m, xyz, offset, new_offset, tmp, idx);
}
