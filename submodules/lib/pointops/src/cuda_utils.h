#ifndef _CUDA_UTILS_H
#define _CUDA_UTILS_H
#include <cstdio>
#include <cmath>
#include <algorithm>

#define TOTAL_THREADS 1024
#define THREADS_PER_BLOCK 256
#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))

inline void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))


inline int opt_n_threads(int work_size) {
    const int pow_2 = std::ceil(std::log(static_cast<double>(work_size)) / std::log(2.0));
    // const int pow_2 = std::log(static_cast<double>(work_size)) / std::log(2.0);
    return std::max(std::min(1 << pow_2, TOTAL_THREADS), 1);
}

inline dim3 opt_block_config(int x, int y) {
    const int x_threads = opt_n_threads(x);
    const int y_threads = std::max(std::min(opt_n_threads(y), TOTAL_THREADS / x_threads), 1);
    dim3 block_config(x_threads, y_threads, 1);
    return block_config;
}

#endif
