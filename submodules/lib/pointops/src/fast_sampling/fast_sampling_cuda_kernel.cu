#include <cstdlib>
#include <stdio.h>
#include <cooperative_groups.h>
#include "../cuda_utils.h"
#include "fast_sampling_cuda_kernel.h"

struct extra_info {
    float *dev_dists;
    float *dev_dists_i;
    unsigned int *max_block_per_batch;

    extra_info(int batch_size, unsigned int blocks_per_grid, unsigned int max_block_per_batch) {
        HANDLE_ERROR( cudaMalloc((void **)&(this->max_block_per_batch), sizeof(unsigned int)) );
        HANDLE_ERROR( cudaMemcpy(this->max_block_per_batch, &max_block_per_batch, sizeof(unsigned int), cudaMemcpyHostToDevice) );
        HANDLE_ERROR( cudaMalloc((void **)&dev_dists, blocks_per_grid * sizeof(float)) );
        HANDLE_ERROR( cudaMalloc((void **)&dev_dists_i, blocks_per_grid * sizeof(float)) );
        HANDLE_ERROR( cudaMemset((void *)dev_dists, 0, blocks_per_grid * sizeof(float)) );
    }
    ~extra_info() {
        cudaFree(dev_dists);
        cudaFree(dev_dists_i);
        cudaFree(max_block_per_batch);
    }
};


__device__ static void __fupdate(float *dists, int *dists_i, int idx1, int idx2) {
    const float v1 = dists[idx1], v2 = dists[idx2];
    const int i1 = dists_i[idx1], i2 = dists_i[idx2];
    dists[idx1] = max(v1, v2);
    dists_i[idx1] = v2 > v1 ? i2 : i1;
}


// input xyz: (n, 3), tmp: (b, n_max)
// ouput idx (m)
template <unsigned int block_size>
__global__ void farthestsampling_cuda_kernel(const float *xyz, const int *offset, const int *new_offset, float *tmp, int *idx, int* e_max_block_per_batch, float *e_dists, int *e_dists_i, int *e_m_max)
{
    __shared__ float dists[block_size];
    __shared__ int dists_i[block_size];

    cooperative_groups::grid_group g = cooperative_groups::this_grid();

    __shared__ float d[block_size]; // use dynamic
    __shared__ int d_i[block_size];

    int max_block_per_batch = *(e_max_block_per_batch);
    int max_block_per_batch_pow_2 = 1 << (int)ceil(log2(max_block_per_batch));
    int bid = blockIdx.x / max_block_per_batch; // batch id
    int blockid_base = bid * max_block_per_batch_pow_2;
    int start_n, end_n, start_m, end_m, old, i_m;
    int m_max = *(e_m_max);

    if (bid == 0) {
        start_n = 0;
        end_n = offset[0];
        start_m = 0;
        end_m = new_offset[0];
        i_m = 1;
        old = 0;
    }
    else {
        start_n = offset[bid - 1];
        end_n = offset[bid];
        start_m = new_offset[bid - 1];
        end_m = new_offset[bid];
        i_m = new_offset[bid - 1] + 1;
        old = offset[bid - 1];
    }


    const int stride = blockDim.x * max_block_per_batch;
    int tid = threadIdx.x + (blockIdx.x % max_block_per_batch) * blockDim.x;
    int tid_in_block = threadIdx.x;
    int tid_batch = tid_in_block + blockid_base;
    int grid_size_pow_2 = 1 << (int)ceil(log2(gridDim.x));
    int blockid_pow_2 = blockIdx.x % max_block_per_batch + blockIdx.x / max_block_per_batch * max_block_per_batch_pow_2;
    if (tid == 0) idx[start_m] = start_n;

    __syncthreads();
    // for (int j = start_m + 1; j < end_m; j++) // loop for farthest point set
    for (int j = 0; j < m_max; j++)
    {
        int besti = start_n;
        float best = -1;
        float x1 = xyz[old * 3 + 0];
        float y1 = xyz[old * 3 + 1];
        float z1 = xyz[old * 3 + 2];
        for (int k = start_n + tid; k < end_n; k += stride)
        {
            float x2 = xyz[k * 3 + 0];
            float y2 = xyz[k * 3 + 1];
            float z2 = xyz[k * 3 + 2];
            float d = (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) + (z2 - z1) * (z2 - z1);

            float d2 = min(d, tmp[k]);
            tmp[k] = d2;
            besti = d2 > best ? k : besti;
            best = d2 > best ? d2 : best;
        }
        dists[tid_in_block] = best;
        dists_i[tid_in_block] = besti;
        __syncthreads();


        // if (tid == 0 && (j == 3)) {
        //     printf("before block reduce\n");
        //     for(int i = 0; i < block_size; i++) {
        //         printf("%f\t", dists[i]);
        //         if ((i + 1) % 10 == 0)
        //             printf("\n");
        //     }
        //     printf("\n");

        //     for(int i = 0; i < block_size; i++) {
        //         printf("%d\t", dists_i[i]);
        //         if ((i + 1) % 10 == 0)
        //             printf("\n");
        //     }
        //     printf("\n");
        // }

        for (int i = blockDim.x / 2; i > 0; i >>= 1) { // max dists (reduce in block)
            if (tid_in_block < i) {
                __fupdate(dists, dists_i, tid_in_block, tid_in_block + i);
            }
            __syncthreads();
        }
        
        
        if (tid_in_block == 0) {
            e_dists[blockid_pow_2] = dists[0];
            e_dists_i[blockid_pow_2] = dists_i[0];
        }
        
        g.sync(); // sync threads in group

        if (tid_in_block < max_block_per_batch_pow_2) {
            d[tid_in_block] = e_dists[tid_batch];
            d_i[tid_in_block] = e_dists_i[tid_batch];
        }
        __syncthreads();

        // if (tid == 0 && (j == 3)) {
        //     printf("before reduce\n");
        //     for(int i = 0; i < max_block_per_batch_pow_2; i++)
        //         printf("%f\t", d[i]);
        //     printf("\n");
            
        //     for(int i = 0; i < max_block_per_batch_pow_2; i++)
        //         printf("%d\t", d_i[i]);
        //     printf("\n");
        // }

        for (int i = max_block_per_batch_pow_2 / 2; i > 0; i >>= 1) { // each block should do reduce as `old` is responsible for each block
            if (tid_in_block < i) {
                __fupdate(d, d_i, tid_in_block, tid_in_block + i);
            }
            __syncthreads();
            // if (tid == 0 && j == 3) {
            //     printf("after reduce\n");
            //     for(int i = 0; i < max_block_per_batch_pow_2; i++)
            //         printf("%f\t", d[i]);
            //     printf("\n");
                
            //     for(int i = 0; i < max_block_per_batch_pow_2; i++)
            //         printf("%d\t", d_i[i]);
            //     printf("\n");
            // }
        }

        old = d_i[0];

        if (tid == 0) {
            if (i_m < end_m) {
                idx[i_m] = old;
                i_m++;
            }
            // idx[j] = old;
        }
    }
    // g.sync();
    // int ttid = threadIdx.x + blockIdx.x * blockDim.x;
    // if (ttid == 0) {
    //     printf("idx\n");
    //     for (int i = 0; i < 5; i++) {
    //         printf("%d ", idx[i]);
    //     }
    //     printf("\n");
    // } 
}

void farthestsampling_cuda_launcher(int b, int n, int m, const float *xyz, const int *offset, const int *new_offset, float *tmp, int *idx)
{   
	unsigned int n_threads = opt_n_threads(n);
    unsigned int max_block_per_batch = (n + n_threads - 1) / n_threads;
    // int numBlocksPerSm = 1;
    // int dev = 2;
    // cudaDeviceProp deviceProp;
    // cudaGetDeviceProperties(&deviceProp, 2);
    // cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, farthestsampling_cuda_kernel<1024>, n_threads, 0);
    // printf("multiProcessorCount: %d %d\n", deviceProp.multiProcessorCount, numBlocksPerSm);
    unsigned int blocks_per_grid = std::min(MAX_BLOCKS, (int)max_block_per_batch * b); // TODO: optimize size of block
    max_block_per_batch = blocks_per_grid / b;
    blocks_per_grid = max_block_per_batch * b;
    unsigned int max_block_per_batch_pow_2 = 1 << (int)ceil(log2(max_block_per_batch));
    unsigned int blocks_per_grid_pow_2 = max_block_per_batch_pow_2 * b;
    // extra_info info(b, blocks_per_grid, blocks_per_grid / b);
    // extra_info *dev_info;

    // printf("max blocks: %u, threads: %d, max block per batch: %d, batch: %d\n", blocks_per_grid, n_threads, max_block_per_batch, b);
    // printf("dev_dists size: %d\n", blocks_per_grid_pow_2);
    // HANDLE_ERROR( cudaMalloc((void **)&dev_info, sizeof(struct extra_info)) );
    // HANDLE_ERROR( cudaMemcpy(dev_info, &info, sizeof(struct extra_info), cudaMemcpyHostToDevice) );
    int *dev_max_block_per_batch;
    float *dev_dists; 
    int *dev_dists_i;
    int *dev_m_max;

    HANDLE_ERROR( cudaMalloc((void **)&(dev_max_block_per_batch), sizeof(unsigned int)) );
    HANDLE_ERROR( cudaMemcpy(dev_max_block_per_batch, &max_block_per_batch, sizeof(unsigned int), cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMalloc((void **)&dev_dists, blocks_per_grid_pow_2 * sizeof(float)) );
    HANDLE_ERROR( cudaMalloc((void **)&dev_dists_i, blocks_per_grid_pow_2 * sizeof(int)) );
    HANDLE_ERROR( cudaMemset((void *)dev_dists, 0, blocks_per_grid_pow_2 * sizeof(float)) );
    HANDLE_ERROR( cudaMalloc((void **)&(dev_m_max), sizeof(int)) );
    HANDLE_ERROR( cudaMemcpy(dev_m_max, &m, sizeof(int), cudaMemcpyHostToDevice) );


    // launch
    void *kernelArgs[] = {&xyz, &offset, &new_offset, &tmp, &idx, &dev_max_block_per_batch, &dev_dists, &dev_dists_i, &dev_m_max};
    dim3 dimBlock(n_threads, 1, 1);
    // dim3 dimGrid(deviceProp.multiProcessorCount * numBlocksPerSm, 1, 1);
    dim3 dimGrid(blocks_per_grid, 1, 1);
    // cudaLaunchCooperativeKernel((void*)farthestsampling_cuda_kernel, dimGrid, dimBlock, kernelArgs);



	switch (n_threads) {
        case 1024:
            // farthestsampling_cuda_kernel<1024><<<blocks_per_grid, n_threads, 0>>>(xyz, offset, new_offset, tmp, idx, dev_max_block_per_batch, dev_dists, dev_dists_i);
            cudaLaunchCooperativeKernel((void*)farthestsampling_cuda_kernel<1024>, dimGrid, dimBlock, kernelArgs);
            break;
        case 512:
            // farthestsampling_cuda_kernel<512><<<blocks_per_grid, n_threads, 0>>>(xyz, offset, new_offset, tmp, idx, dev_max_block_per_batch, dev_dists, dev_dists_i);
            cudaLaunchCooperativeKernel((void*)farthestsampling_cuda_kernel<512>, dimGrid, dimBlock, kernelArgs);
            break;
        case 256:
            // farthestsampling_cuda_kernel<256><<<blocks_per_grid, n_threads, 0>>>(xyz, offset, new_offset, tmp, idx, dev_max_block_per_batch, dev_dists, dev_dists_i);
            cudaLaunchCooperativeKernel((void*)farthestsampling_cuda_kernel<256>, dimGrid, dimBlock, kernelArgs);
            break;
        case 128:
            // farthestsampling_cuda_kernel<128><<<blocks_per_grid, n_threads, 0>>>(xyz, offset, new_offset, tmp, idx, dev_max_block_per_batch, dev_dists, dev_dists_i);
            cudaLaunchCooperativeKernel((void*)farthestsampling_cuda_kernel<128>, dimGrid, dimBlock, kernelArgs);
            break;
        case 64:
            // farthestsampling_cuda_kernel<64><<<blocks_per_grid, n_threads, 0>>>(xyz, offset, new_offset, tmp, idx, dev_max_block_per_batch, dev_dists, dev_dists_i);
            cudaLaunchCooperativeKernel((void*)farthestsampling_cuda_kernel<64>, dimGrid, dimBlock, kernelArgs);
            break;
        case 32:
            // farthestsampling_cuda_kernel<32><<<blocks_per_grid, n_threads, 0>>>(xyz, offset, new_offset, tmp, idx, dev_max_block_per_batch, dev_dists, dev_dists_i);
            cudaLaunchCooperativeKernel((void*)farthestsampling_cuda_kernel<32>, dimGrid, dimBlock, kernelArgs);
            break;
        case 16:
            // farthestsampling_cuda_kernel<16><<<blocks_per_grid, n_threads, 0>>>(xyz, offset, new_offset, tmp, idx, dev_max_block_per_batch, dev_dists, dev_dists_i);
            cudaLaunchCooperativeKernel((void*)farthestsampling_cuda_kernel<16>, dimGrid, dimBlock, kernelArgs);
            break;
        case 8:
            // farthestsampling_cuda_kernel<8><<<blocks_per_grid, n_threads, 0>>>(xyz, offset, new_offset, tmp, idx, dev_max_block_per_batch, dev_dists, dev_dists_i);
            cudaLaunchCooperativeKernel((void*)farthestsampling_cuda_kernel<8>, dimGrid, dimBlock, kernelArgs);
            break;
        case 4:
            // farthestsampling_cuda_kernel<4><<<blocks_per_grid, n_threads, 0>>>(xyz, offset, new_offset, tmp, idx, dev_max_block_per_batch, dev_dists, dev_dists_i);
            cudaLaunchCooperativeKernel((void*)farthestsampling_cuda_kernel<4>, dimGrid, dimBlock, kernelArgs);
            break;
        case 2:
            // farthestsampling_cuda_kernel<2><<<blocks_per_grid, n_threads, 0>>>(xyz, offset, new_offset, tmp, idx, dev_max_block_per_batch, dev_dists, dev_dists_i);
            cudaLaunchCooperativeKernel((void*)farthestsampling_cuda_kernel<2>, dimGrid, dimBlock, kernelArgs);
            break;
        case 1:
            // farthestsampling_cuda_kernel<1><<<blocks_per_grid, n_threads, 0>>>(xyz, offset, new_offset, tmp, idx, dev_max_block_per_batch, dev_dists, dev_dists_i);
            cudaLaunchCooperativeKernel((void*)farthestsampling_cuda_kernel<1>, dimGrid, dimBlock, kernelArgs);
            break;
        default:
            // farthestsampling_cuda_kernel<512><<<blocks_per_grid, n_threads, 0>>>(xyz, offset, new_offset, tmp, idx, dev_max_block_per_batch, dev_dists, dev_dists_i);
            cudaLaunchCooperativeKernel((void*)farthestsampling_cuda_kernel<512>, dimGrid, dimBlock, kernelArgs);
    }
    // printf("%s\n", cudaGetErrorString(cudaGetLastError()));

    HANDLE_ERROR(cudaFree(dev_dists));
    HANDLE_ERROR(cudaFree(dev_dists_i));
    HANDLE_ERROR(cudaFree(dev_max_block_per_batch));
    HANDLE_ERROR(cudaFree(dev_m_max));
}
