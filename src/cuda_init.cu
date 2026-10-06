#include "cuda_utils.h"
#include <cuda_runtime.h>

void cuda_init(int device_id) {
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        fprintf(stderr, "No CUDA-capable GPU detected\n");
        exit(EXIT_FAILURE);
    }
    if (device_id >= device_count) {
        fprintf(stderr, "Requested device %d but only %d devices available\n",
                device_id, device_count);
        exit(EXIT_FAILURE);
    }
    CUDA_CHECK(cudaSetDevice(device_id));
}

GPUInfo cuda_get_device_info(int device_id) {
    GPUInfo info = {};
    info.device_id = device_id;

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));

    snprintf(info.name, sizeof(info.name), "%s", prop.name);
    info.total_memory = prop.totalGlobalMem;
    info.compute_major = prop.major;
    info.compute_minor = prop.minor;
    info.sm_count = prop.multiProcessorCount;
    info.max_threads_per_block = prop.maxThreadsPerBlock;
    info.warp_size = prop.warpSize;

    size_t free_mem, total_mem;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    info.free_memory = free_mem;

    return info;
}

void cuda_print_device_info(const GPUInfo& info) {
    printf("=== GPU Device Info ===\n");
    printf("  Device ID:           %d\n", info.device_id);
    printf("  Name:                %s\n", info.name);
    printf("  Total Memory:        %.2f GB\n",
           info.total_memory / (1024.0 * 1024.0 * 1024.0));
    printf("  Free Memory:         %.2f GB\n",
           info.free_memory / (1024.0 * 1024.0 * 1024.0));
    printf("  Compute Capability:  %d.%d\n",
           info.compute_major, info.compute_minor);
    printf("  SM Count:            %d\n", info.sm_count);
    printf("  Max Threads/Block:   %d\n", info.max_threads_per_block);
    printf("  Warp Size:           %d\n", info.warp_size);
    printf("=======================\n");
}

size_t cuda_get_free_memory() {
    size_t free_mem, total_mem;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    return free_mem;
}
