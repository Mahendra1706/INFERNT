#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(err)                                                        \
    do {                                                                        \
        cudaError_t e = (err);                                                  \
        if (e != cudaSuccess) {                                                 \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,       \
                    cudaGetErrorString(e));                                      \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

struct GPUInfo {
    int device_id;
    char name[256];
    size_t total_memory;
    size_t free_memory;
    int compute_major;
    int compute_minor;
    int sm_count;
    int max_threads_per_block;
    int warp_size;
};

void cuda_init(int device_id = 0);
GPUInfo cuda_get_device_info(int device_id = 0);
void cuda_print_device_info(const GPUInfo& info);
size_t cuda_get_free_memory();
