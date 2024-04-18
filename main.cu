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

void process(uint64_t *data, uint64_t N) {
    uint64_t *d_data;
    cudaMalloc(&d_data, N * sizeof(uint64_t));

    cudaMemcpy(d_data, data, N * sizeof(uint64_t), cudaMemcpyHostToDevice);

    int blockSize = 64;
    int numBlocks = (N + blockSize - 1) / blockSize;
    addOneToEachElement<<<numBlocks, blockSize>>>(d_data, N);
    CHECKCUDAERR(cudaGetLastError());

    cudaMemcpy(data, d_data, N * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    for (uint64_t i = 0; i < 8; ++i) {
        printf("h_data: %lu\n", data[i]);
    }

    cudaFree(d_data);
}

int main() {
    int deviceCount;
    CHECKCUDAERR(cudaGetDeviceCount(&deviceCount));
    printf("Detected %d CUDA Capable device(s).\n", deviceCount);

    cudaStream_t ss;
    cudaSetDevice(0);
    CHECKCUDAERR(cudaStreamCreate(&ss));

    uint64_t N = (1<<31);
    uint64_t *data;
    cudaMallocManaged(&data, N * sizeof(uint64_t), cudaMemAttachHost);


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

    int blockSize = 64;
    int numBlocks = (N + blockSize - 1) / blockSize;
    addOneToEachElement<<<numBlocks, blockSize, 0, ss>>>(data, N);
    CHECKCUDAERR(cudaGetLastError());
    CHECKCUDAERR(cudaStreamSynchronize(ss));

    for (uint64_t i = 0; i < 8; ++i) {
        printf("data: %lu\n", data[i]);
    }

    cudaMemGetInfo(&free_mem, &total_mem);
    printf("2. free_mem: %lu, total_mem: %lu\n", free_mem>>20, total_mem>>20);

    CHECKCUDAERR(cudaMemPrefetchAsync(data, N * sizeof(uint64_t), cudaCpuDeviceId, ss));
    CHECKCUDAERR(cudaStreamSynchronize(ss));

    cudaMemGetInfo(&free_mem, &total_mem);
    printf("3. free_mem: %lu, total_mem: %lu\n", free_mem>>20, total_mem>>20);

    uint64_t *h_data = (uint64_t *)malloc(N * sizeof(uint64_t));
    memcpy(h_data, data, N * sizeof(uint64_t));
    TimerStart(malloc_test);
    process(h_data, N);
    TimerStopAndLog(malloc_test);
    free(h_data);

    uint64_t *h_data1 = (uint64_t *)cudaMallocHost(N * sizeof(uint64_t));
    memcpy(h_data1, data, N * sizeof(uint64_t));
    TimerStart(cudaMallocHost_test);
    process(h_data1, N);
    TimerStopAndLog(cudaMallocHost_test);
    cudaFreeHost(h_data1);

    uint64_t *h_data2 = (uint64_t *)cudaMallocManaged(N * sizeof(uint64_t));
    memcpy(h_data2, data, N * sizeof(uint64_t));
    TimerStart(cudaMallocManaged_test);
    process(h_data2, N);
    TimerStopAndLog(cudaMallocManaged_test);
    cudaFreeHost(h_data2);

    cudaFree(data);

    return 0;
}
