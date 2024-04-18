#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <cuda.h>
#include "timer.hpp"

__host__ inline void checkCudaError(cudaError_t code, const char *file, int line)
{
    if (code != cudaSuccess)
    {
        printf("CUDA error: %s %s %d\n", cudaGetErrorString(code), file, line);
    }
}

#define CHECKCUDAERR(ans)                          \
    {                                              \
        checkCudaError((ans), __FILE__, __LINE__); \
    }


__global__ void addOneToEachElement(uint64_t *data, uint64_t N) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        data[idx] = __brev(data[idx]);
    }
}

void process(uint64_t *data, uint64_t N, cudaStream_t ss) {
    uint64_t *d_data;
    cudaMalloc(&d_data, N * sizeof(uint64_t));

    CHECKCUDAERR(cudaMemcpyAsync(d_data, data, N * sizeof(uint64_t), cudaMemcpyHostToDevice, ss));

    int blockSize = 64;
    int numBlocks = (N + blockSize - 1) / blockSize;
    addOneToEachElement<<<numBlocks, blockSize, 0, ss>>>(d_data, N);

    CHECKCUDAERR(cudaMemcpyAsync(data, d_data, N * sizeof(uint64_t), cudaMemcpyDeviceToHost, ss));
    CHECKCUDAERR(cudaGetLastError());
    CHECKCUDAERR(cudaStreamSynchronize(ss));
    for (uint64_t i = 0; i < 2; ++i) {
        printf("data: %lu\n", data[i]);
    }

    cudaFree(d_data);
}

#define MAX_GPUS 16

int main() {
    int deviceCount;
    CHECKCUDAERR(cudaGetDeviceCount(&deviceCount));
    printf("Detected %d CUDA Capable device(s).\n", deviceCount);

    cudaStream_t ss[MAX_GPUS];
    for (uint64_t d = 0; d < deviceCount; ++d) {
        cudaSetDevice(d);
        CHECKCUDAERR(cudaStreamCreate(&ss[d]));
    }

    uint64_t N = (uint64_t(1)<<32);
    uint64_t *data = (uint64_t *)malloc(N * sizeof(uint64_t));

    // 初始化host memory数据
    for (uint64_t i = 0; i < N; ++i) {
        data[i] = i;
    }

    for (uint64_t i = 0; i < 8; ++i) {
        printf("data: %lu\n", data[i]);
    }

    uint64_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    printf("1. free_mem: %lu, total_mem: %lu\n", free_mem>>20, total_mem>>20);

    uint64_t n_per_device = N / deviceCount;
    uint64_t parallel_n = 64;
    uint64_t n_per_piece = N / parallel_n;

    uint64_t *h_data = (uint64_t *)malloc(N * sizeof(uint64_t));
#pragma omp parallel for
    for (uint64_t i = 0; i < parallel_n; ++i) {
        memcpy(h_data + i * n_per_piece, data + i * n_per_piece, n_per_piece * sizeof(uint64_t));
    }

    TimerStart(malloc_test);
#pragma omp parallel for
    for (uint64_t d = 0; d < deviceCount; ++d) {
        cudaSetDevice(d);
        process(h_data + d * n_per_piece, n_per_piece, ss[d]);
    }
    TimerStopAndLog(malloc_test);
    free(h_data);

    uint64_t *h_data1;
    cudaMallocHost(&h_data1, N * sizeof(uint64_t));
#pragma omp parallel for
    for (uint64_t i = 0; i < parallel_n; ++i) {
        memcpy(h_data1 + i * n_per_piece, data + i * n_per_piece, n_per_piece * sizeof(uint64_t));
    }
    TimerStart(cudaMallocHost_test);
#pragma omp parallel for
    for (uint64_t d = 0; d < deviceCount; ++d) {
        cudaSetDevice(d);
        process(h_data1 + d * n_per_piece, n_per_piece, ss[d]);
    }
    TimerStopAndLog(cudaMallocHost_test);
    cudaFreeHost(h_data1);

    uint64_t *h_data2;
    cudaMallocManaged(&h_data2, N * sizeof(uint64_t));
#pragma omp parallel for
    for (uint64_t i = 0; i < parallel_n; ++i) {
        memcpy(h_data1 + i * n_per_piece, data + i * n_per_piece, n_per_piece * sizeof(uint64_t));
    }
    TimerStart(cudaMallocManaged_test);
#pragma omp parallel for
    for (uint64_t d = 0; d < deviceCount; ++d) {
        cudaSetDevice(d);
        process(h_data2 + d * n_per_piece, n_per_piece, ss[d]);
    }
    TimerStopAndLog(cudaMallocManaged_test);
    cudaFree(h_data2);

    cudaFree(data);
    CHECKCUDAERR(cudaStreamDestroy(ss));

    return 0;
}
